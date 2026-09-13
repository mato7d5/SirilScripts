#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SirilSync - replay the processing of the loaded image onto a folder of images.

The script reads the FITS HISTORY of the image currently loaded in Siril,
translates every entry back into the Siril command that produced it, and lets
you replay the selected commands on every image found in a chosen folder.

Requires Siril >= 1.4 with the Python (sirilpy) interface.
Copy this file into your Siril scripts directory to get it in the script menu.
"""

import os
import re
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import sirilpy as s

s.ensure_installed("ttkthemes")

from ttkthemes import ThemedTk           # noqa: E402
from sirilpy import tksiril              # noqa: E402


# --------------------------------------------------------------------------- #
#  History -> command translation
# --------------------------------------------------------------------------- #

# Siril writes numbers with formats such as %6.1lf, so the values in the
# history are often padded with spaces: allow leading whitespace everywhere.
_F = r"\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
_I = r"\s*([-+]?[0-9]+)"

EXACT = "exact"      # every parameter could be recovered from the history
PARTIAL = "partial"  # command recognised, but some parameters are not recorded
NONE = "none"        # no equivalent single-image command


def _bool(text):
    return text.strip().lower() in ("true", "yes", "1", "on")


def _scnr(m):
    types = {
        "average neutral": "0",
        "maximum neutral": "1",
        "maximum mask": "2",
        "additive mask": "3",
    }
    parts = ["rmgreen"]
    if not _bool(m.group(3)):
        parts.append("-nopreserve")
    parts.append(types.get(m.group(1).strip().lower(), "0"))
    parts.append(m.group(2))
    return " ".join(parts)


def _rotate(m):
    parts = ["rotate", m.group(1)]
    if not _bool(m.group(2)):
        parts.append("-nocrop")
    if not _bool(m.group(3)):
        parts.append("-noclamp")
    return " ".join(parts)


def _resample(m):
    return "resample %s" % m.group(1)


def _banding(m):
    sigma = m.group(3) if m.group(3) else "0"
    return "fixbanding %s %s" % (m.group(2), sigma)


def _binning(m):
    parts = ["binxy", m.group(1)]
    if "sum" in m.group(2).lower():
        parts.append("-sum")
    return " ".join(parts)


def _autostretch(m):
    parts = ["autostretch"]
    if "unlinked" not in m.group(3).lower():
        parts.append("-linked")
    parts += [m.group(1), m.group(2)]
    return " ".join(parts)


def _autoghs(m):
    parts = ["autoghs"]
    prefix = m.group(1).lower()
    if "linked" in prefix and "unlinked" not in prefix:
        parts.append("-linked")
    parts += [m.group(2), m.group(3), "-b=%s" % m.group(4),
              "-lp=%s" % m.group(5), "-hp=%s" % m.group(6)]
    return " ".join(parts)


def _ghs(command):
    def build(m):
        return "%s -D=%s -B=%s -SP=%s -LP=%s -HP=%s" % (
            command, m.group(2), m.group(3), m.group(1), m.group(4), m.group(5))
    return build


def _ghs_asinh(command):
    def build(m):
        return "%s -D=%s -SP=%s -LP=%s -HP=%s" % (
            command, m.group(2), m.group(1), m.group(3), m.group(4))
    return build


# (pattern, builder, confidence, note)
_RULES = [
    # --- geometry --------------------------------------------------------- #
    (r"^Crop \(x=%s, y=%s, w=%s, h=%s\)" % (_I, _I, _I, _I),
     lambda m: "crop %s %s %s %s" % m.group(1, 2, 3, 4), EXACT, ""),
    (r"^Rotation \(%sdeg, cropped=\s*(\w+), clamped=\s*(\w+)\)" % _F,
     _rotate, EXACT, ""),
    (r"^Rotation \(%s ?deg\)" % _I,
     lambda m: "rotate %s -nocrop" % m.group(1), EXACT, ""),
    (r"^(Mirror ?X|Horizontal mirror|Top-bottom mirror)",
     lambda m: "mirrorx", EXACT, ""),
    (r"^(Mirror ?Y|Vertical mirror|Left-right mirror)",
     lambda m: "mirrory", EXACT, ""),
    (r"^Resample \(%s - %s\)" % (_F, _F),
     _resample, PARTIAL, "non-uniform scaling is not fully recorded"),
    (r"^Binning x%s \((\w+)\)" % _I,
     _binning, EXACT, ""),

    # --- stretching ------------------------------------------------------- #
    (r"^Histogram Transf\. \(mid=%s, lo=%s, hi=%s\)" % (_F, _F, _F),
     lambda m: "mtf %s %s %s" % m.group(2, 1, 3), EXACT, ""),
    (r"^Asinh Transformation: \(stretch=%s, bp=%s\)" % (_F, _F),
     lambda m: "asinh %s %s" % m.group(1, 2), EXACT, ""),
    (r"^Asinh stretch \(amount:%s, offset:%s, human:\s*(\w+)\)" % (_F, _F),
     lambda m: "asinh %s%s %s" % ("-human " if _bool(m.group(3)) else "",
                                  m.group(1), m.group(2)), EXACT, ""),
    (r"^Autostretch \(shadows:%s, target bg:%s,\s*(\w+)\)" % (_F, _F),
     _autostretch, EXACT, ""),
    (r"^AutoGHS \((.*?)k\.sigma:%s, amount:%s, local:%s \[%s,%s\]\)"
     % (_F, _F, _F, _F, _F),
     _autoghs, EXACT, ""),
    (r"^GHS \(pivot:%s, amount:%s, local:%s \[%s,%s\]\)" % (_F, _F, _F, _F, _F),
     _ghs("ght"), EXACT, ""),
    (r"^Inverse GHS \(pivot:%s, amount:%s, local:%s \[%s,%s\]\)"
     % (_F, _F, _F, _F, _F),
     _ghs("invght"), EXACT, ""),
    (r"^GHS asinh \(pivot:%s, amount:%s \[%s,%s\]\)" % (_F, _F, _F, _F),
     _ghs_asinh("modasinh"), EXACT, ""),
    (r"^GHS inverse asinh \(pivot:%s, amount:%s \[%s,%s\]\)" % (_F, _F, _F, _F),
     _ghs_asinh("invmodasinh"), EXACT, ""),
    (r"^GHS BP shift \(new BP:%s\)" % _F,
     lambda m: "linstretch -BP=%s" % m.group(1), EXACT, ""),
    (r"^DDP stretch, threshold:%s, multiplier:%s, sigma:%s" % (_F, _F, _F),
     lambda m: "ddp %s %s %s" % m.group(1, 2, 3), EXACT, ""),
    (r"^Negative Transformation",
     lambda m: "neg", EXACT, ""),

    # --- colour ----------------------------------------------------------- #
    (r"^SCNR \(type=\s*(.+?), amount=%s, preserve=\s*(\w+)\)" % _F,
     _scnr, EXACT, ""),
    (r"^Saturation enhancement \(amount=%s\)" % _F,
     lambda m: "satu %s" % m.group(1), EXACT, ""),
    (r"^Unpurple filter: \(thresh=%s, mod_b=%s\)" % (_F, _F),
     lambda m: "unpurple -thresh=%s -mod_b=%s" % m.group(1, 2), EXACT, ""),
    (r"^Photometric CC",
     lambda m: "pcc", PARTIAL, "catalogue and options are not recorded"),
    (r"^Color Calibration",
     lambda m: "", NONE, "manual colour calibration has no command equivalent"),

    # --- filters / noise -------------------------------------------------- #
    (r"^Gaussian filtering, sigma:%s" % _F,
     lambda m: "gauss %s" % m.group(1), EXACT, ""),
    (r"^Unsharp filtering, sigma:%s, coefficient:%s" % (_F, _F),
     lambda m: "unsharp %s %s" % m.group(1, 2), EXACT, ""),
    (r"^Median Filter \(filter=%sx\d+ px\)" % _I,
     lambda m: "fmedian %s 1" % m.group(1), PARTIAL,
     "modulation is not recorded, 1 is assumed"),
    (r"^Bilateral filter: \(d=%s, sigma_col=%s, sigma_spatial=%s\)"
     % (_F, _F, _F),
     lambda m: "epf -d=%s -si=%s -ss=%s" % m.group(1, 2, 3), EXACT, ""),
    (r"^Guided filtering, d:%s, sigma:%s, modulation:%s" % (_F, _F, _F),
     lambda m: "epf -guided -d=%s -si=%s -mod=%s" % m.group(1, 2, 3),
     EXACT, ""),
    (r"^CLAHE \(size=%s, clip=%s\)" % (_I, _F),
     lambda m: "clahe %s %s" % m.group(2, 1), EXACT, ""),
    (r"^NL-Bayes denoise \(mod=%s" % _F,
     lambda m: "denoise -mod=%s" % m.group(1), PARTIAL,
     "the secondary denoise options are not recorded"),
    (r"^(Canon )?Banding Reduction \(amount=%s(?:, Protect=TRUE, invsigma=%s)?\)"
     % (_F, _F),
     _banding, EXACT, ""),
    (r"^RGradient: \(dR=%s, dA=%s, xc=%s, yc=%s\)" % (_F, _F, _F, _F),
     lambda m: "rgradient %s %s %s %s" % m.group(3, 4, 1, 2), EXACT, ""),
    (r"^Cosmetic Correction",
     lambda m: "find_cosme 3 3", PARTIAL, "the sigma values are not recorded"),

    # --- background / gradient -------------------------------------------- #
    (r"^Background extraction \(Correction:\s*(\w+)\)",
     lambda m: "subsky 1", PARTIAL,
     "degree / RBF settings and samples are not recorded"),

    # --- stars ------------------------------------------------------------ #
    (r"^Synthetic stars:",
     lambda m: "synthstar", EXACT, ""),
    (r"^Deconvolution",
     lambda m: "", NONE, "the deconvolution algorithm and PSF are not recorded"),
    (r"^(StarNet|Starnet)",
     lambda m: "starnet", PARTIAL, "the StarNet options are not recorded"),
    (r"^Linear Match",
     lambda m: "", NONE, "the reference image is not recorded"),

    # --- not replayable on a single image --------------------------------- #
    (r"^(Stacking|Registration|Calibration|Calibrated|Conversion|Convert"
     r"|Debayer|Bayer|Split CFA|Merge CFA|Extract |Plate ?[Ss]olve"
     r"|Solved by|Astrometric|Pixel Math|Wavelets Transformation"
     r"|Assigned ICC|Converted to ICC|ICC profile)",
     lambda m: "", NONE, "not a replayable single-image operation"),
]

_COMPILED = [(re.compile(p), b, c, n) for p, b, c, n in _RULES]


def translate(entry):
    """Return (command, confidence, note) for a single HISTORY entry."""
    for regex, builder, level, note in _COMPILED:
        match = regex.match(entry)
        if not match:
            continue
        try:
            command = builder(match)
        except Exception:
            return "", NONE, "could not rebuild the command from this entry"
        if not command:
            return "", NONE, note
        return command, level, note
    return "", NONE, "unrecognised history entry"


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

FITS_EXTS = (".fit", ".fits", ".fts")
IMAGE_EXTS = FITS_EXTS + (
    ".fit.fz", ".fits.fz", ".fts.fz",
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp",
    ".ppm", ".pgm", ".pnm", ".xisf",
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".raf", ".rw2", ".pef",
)


def split_ext(path):
    """Split a path, keeping '.fits.fz' style double extensions together."""
    base, ext = os.path.splitext(path)
    if ext.lower() == ".fz":
        base2, ext2 = os.path.splitext(base)
        if ext2.lower() in FITS_EXTS:
            return base2, ext2.lower() + ext.lower()
    return base, ext.lower()


def list_images(folder, recursive):
    found = []
    try:
        if recursive:
            for root, _dirs, files in os.walk(folder):
                for name in files:
                    if split_ext(name)[1] in IMAGE_EXTS:
                        found.append(os.path.join(root, name))
        else:
            for name in os.listdir(folder):
                full = os.path.join(folder, name)
                if os.path.isfile(full) and split_ext(name)[1] in IMAGE_EXTS:
                    found.append(full)
    except OSError:
        return []
    return sorted(found)


def read_fits_history(path):
    """Read the HISTORY cards straight from an uncompressed FITS file.

    This is used to tell apart the history already stored on disk from the
    changes made since the image was loaded. Returns None when it cannot be
    read (compressed FITS, non-FITS formats, unsaved images).
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(6) != b"SIMPLE":
                return None
            handle.seek(0)
            history = []
            for _block in range(512):
                block = handle.read(2880)
                if len(block) < 2880:
                    return history
                text = block.decode("ascii", "replace")
                for i in range(36):
                    card = text[i * 80:(i + 1) * 80]
                    key = card[:8].strip()
                    if key == "HISTORY":
                        history.append(card[8:].strip())
                    elif key == "END":
                        return history
            return history
    except Exception:
        return None


def quote(path):
    return '"%s"' % path.replace("\\", "/")


def save_command(out_base, ext):
    if ext in (".tif", ".tiff"):
        return "savetif %s" % quote(out_base)
    if ext == ".png":
        return "savepng %s" % quote(out_base)
    if ext in (".jpg", ".jpeg"):
        return "savejpg %s 95" % quote(out_base)
    return "save %s" % quote(out_base)


# --------------------------------------------------------------------------- #
#  One row of the change list
# --------------------------------------------------------------------------- #

class ChangeRow:

    def __init__(self, parent, number, text, command, level, note,
                 pre_existing):
        self.level = level
        self.note = note
        self.pre_existing = pre_existing

        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(1, weight=1)

        self.enabled = tk.BooleanVar(value=bool(command) and not pre_existing)
        self.command = tk.StringVar(value=command)

        self.check = ttk.Checkbutton(self.frame, variable=self.enabled)
        self.check.grid(row=0, column=0, rowspan=2, sticky="n", padx=(0, 4))

        if pre_existing:
            tag = "   [already stored in the file]"
        elif level == PARTIAL:
            tag = "   [incomplete]"
        elif level == NONE:
            tag = "   [no command]"
        else:
            tag = ""

        ttk.Label(self.frame, text="%d. %s%s" % (number, text, tag),
                  wraplength=560, justify="left").grid(
                      row=0, column=1, sticky="w")

        row = ttk.Frame(self.frame)
        row.grid(row=1, column=1, sticky="ew", pady=(2, 0))
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="command:").grid(row=0, column=0, padx=(0, 4))
        self.entry = ttk.Entry(row, textvariable=self.command)
        self.entry.grid(row=0, column=1, sticky="ew")

        if note:
            ttk.Label(self.frame, text=note, wraplength=560,
                      justify="left").grid(row=2, column=1, sticky="w")

        ttk.Separator(self.frame, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=6)

        self.command.trace_add("write", self._on_command_change)
        self._on_command_change()

    def _on_command_change(self, *_args):
        if self.command.get().strip():
            self.check.state(["!disabled"])
        else:
            self.enabled.set(False)
            self.check.state(["disabled"])

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def is_active(self):
        return self.enabled.get() and bool(self.command.get().strip())


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #

class SirilSync:

    def __init__(self, root, siril):
        self.root = root
        self.siril = siril
        self.rows = []
        self.source_path = None
        self.messages = queue.Queue()
        self.worker = None
        self.cancel = threading.Event()

        root.title("SirilSync")
        root.minsize(760, 640)

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)

        self.source_label = ttk.Label(outer, text="", wraplength=700,
                                      justify="left")
        self.source_label.pack(fill="x", pady=(0, 8))

        self._build_target(outer)
        self._build_changes(outer)
        self._build_output(outer)
        self._build_status(outer)
        self._build_buttons(outer)

        self.load_history()
        self.root.after(120, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -- widgets ----------------------------------------------------------- #

    def _build_target(self, parent):
        box = ttk.LabelFrame(parent, text="Images to process", padding=8)
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Folder:").grid(row=0, column=0, sticky="w")
        self.folder = tk.StringVar()
        ttk.Entry(box, textvariable=self.folder).grid(
            row=0, column=1, sticky="ew", padx=6)
        ttk.Button(box, text="Browse...", command=self.pick_folder).grid(
            row=0, column=2)
        self.folder.trace_add("write", lambda *_a: self.refresh_count())

        self.recursive = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Include subfolders",
                        variable=self.recursive,
                        command=self.refresh_count).grid(
                            row=1, column=1, sticky="w", padx=6, pady=(6, 0))

        self.count_label = ttk.Label(box, text="No folder selected.")
        self.count_label.grid(row=2, column=1, sticky="w", padx=6, pady=(4, 0))

    def _build_changes(self, parent):
        box = ttk.LabelFrame(parent, text="Changes to apply", padding=8)
        box.pack(fill="both", expand=True, pady=(10, 0))

        bar = ttk.Frame(box)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(bar, text="Select all",
                   command=lambda: self.select_all(True)).pack(side="left")
        ttk.Button(bar, text="Select none",
                   command=lambda: self.select_all(False)).pack(
                       side="left", padx=6)
        ttk.Button(bar, text="Reload from image",
                   command=self.load_history).pack(side="left")

        self.scroller = tksiril.ScrollableFrame(box)
        self.scroller.pack(fill="both", expand=True)
        self.list_frame = self.scroller.scrollable_frame

    def _build_output(self, parent):
        box = ttk.LabelFrame(parent, text="Output", padding=8)
        box.pack(fill="x", pady=(10, 0))
        box.columnconfigure(1, weight=1)

        self.overwrite = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Overwrite the original images",
                        variable=self.overwrite,
                        command=self.update_output_state).grid(
                            row=0, column=0, columnspan=3, sticky="w")

        self.out_caption = ttk.Label(box, text="Save to:")
        self.out_caption.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.out_folder = tk.StringVar()
        self.out_entry = ttk.Entry(box, textvariable=self.out_folder)
        self.out_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        self.out_button = ttk.Button(box, text="Browse...",
                                     command=self.pick_output)
        self.out_button.grid(row=1, column=2, pady=(6, 0))
        self.update_output_state()

    def _build_status(self, parent):
        box = ttk.Frame(parent)
        box.pack(fill="x", pady=(10, 0))
        self.progress = ttk.Progressbar(box, mode="determinate")
        self.progress.pack(fill="x")
        self.status = ttk.Label(box, text="Ready.")
        self.status.pack(fill="x", pady=(4, 0))

        log_box = ttk.Frame(parent)
        log_box.pack(fill="both", pady=(6, 0))
        self.log = tk.Text(log_box, height=7, wrap="none", state="disabled")
        scroll = ttk.Scrollbar(log_box, orient="vertical",
                               command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

    def _build_buttons(self, parent):
        box = ttk.Frame(parent)
        box.pack(fill="x", pady=(10, 0))
        ttk.Button(box, text="Close", command=self.on_close).pack(side="right")
        self.sync_button = ttk.Button(box, text="Sync", command=self.on_sync)
        self.sync_button.pack(side="right", padx=6)

    # -- history ----------------------------------------------------------- #

    def load_history(self):
        for row in self.rows:
            row.frame.destroy()
        self.rows = []
        for child in self.list_frame.winfo_children():
            child.destroy()

        try:
            if not self.siril.is_image_loaded():
                self.source_label.configure(
                    text="No image is loaded in Siril. Load the image you have "
                         "processed, then press \"Reload from image\".")
                return
            filename = self.siril.get_image_filename()
            history = self.siril.get_image_history() or []
        except s.SirilError as exc:
            self.source_label.configure(
                text="Could not read the loaded image: %s" % exc)
            return

        self.source_path = os.path.abspath(filename) if filename else None
        self.source_label.configure(
            text="Source image: %s" % (filename or "(unsaved image)"))

        baseline = (read_fits_history(self.source_path)
                    if self.source_path else None)
        baseline_len = 0
        if baseline is not None:
            # The on-disk history is a prefix of the in-memory one; anything
            # past it was done since the image was loaded.
            while (baseline_len < len(baseline)
                   and baseline_len < len(history)
                   and baseline[baseline_len] == history[baseline_len]):
                baseline_len += 1

        if not history:
            ttk.Label(self.list_frame,
                      text="This image has no HISTORY entries.").pack(
                          anchor="w", pady=4)
            return

        for index, entry in enumerate(history):
            command, level, note = translate(entry)
            row = ChangeRow(self.list_frame, index + 1, entry, command, level,
                            note, pre_existing=index < baseline_len)
            row.pack(fill="x", expand=True)
            self.rows.append(row)

        if baseline is None:
            self.write_log("The history stored on disk could not be read, so "
                           "every entry is listed - check the selection.")
        else:
            self.write_log("%d of %d history entries were added after the "
                           "image was loaded."
                           % (len(history) - baseline_len, len(history)))

    def select_all(self, value):
        for row in self.rows:
            if value and not row.command.get().strip():
                continue
            row.enabled.set(value)

    # -- folders ----------------------------------------------------------- #

    def pick_folder(self):
        path = filedialog.askdirectory(
            title="Select the folder with the images", parent=self.root)
        if path:
            self.folder.set(os.path.normpath(path))

    def pick_output(self):
        path = filedialog.askdirectory(title="Select the output folder",
                                       parent=self.root)
        if path:
            self.out_folder.set(os.path.normpath(path))

    def update_output_state(self):
        state = "disabled" if self.overwrite.get() else "normal"
        self.out_entry.configure(state=state)
        self.out_button.configure(state=state)
        self.out_caption.configure(state=state)

    def refresh_count(self):
        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder):
            self.count_label.configure(text="No folder selected.")
            return
        files = list_images(folder, self.recursive.get())
        self.count_label.configure(text="%d image(s) found." % len(files))

    # -- logging ----------------------------------------------------------- #

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _pump(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self.write_log(payload)
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "progress":
                    self.progress.configure(value=payload)
                elif kind == "maximum":
                    self.progress.configure(maximum=payload, value=0)
                elif kind == "done":
                    self.on_worker_finished(payload)
        except queue.Empty:
            pass
        self.root.after(120, self._pump)

    def post(self, kind, payload=None):
        self.messages.put((kind, payload))

    # -- sync -------------------------------------------------------------- #

    def on_sync(self):
        if self.worker and self.worker.is_alive():
            return

        commands = [row.command.get().strip() for row in self.rows
                    if row.is_active()]
        if not commands:
            messagebox.showwarning("SirilSync", "No change is selected.",
                                   parent=self.root)
            return

        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning(
                "SirilSync", "Select a valid folder with the images to "
                             "process.", parent=self.root)
            return

        files = list_images(folder, self.recursive.get())
        if not files:
            messagebox.showwarning(
                "SirilSync", "No supported image was found in that folder.",
                parent=self.root)
            return

        # The loaded image is processed like any other file in the folder: the
        # copy on disk still is the unprocessed one, so replaying the commands
        # on it gives the same result as the one currently on screen.
        includes_source = bool(self.source_path) and any(
            os.path.abspath(f) == self.source_path for f in files)

        overwrite = self.overwrite.get()
        out_folder = self.out_folder.get().strip()
        if not overwrite:
            if not out_folder:
                messagebox.showwarning(
                    "SirilSync", "Choose an output folder, or enable "
                                 "\"Overwrite the original images\".",
                    parent=self.root)
                return
            if not os.path.isdir(out_folder):
                try:
                    os.makedirs(out_folder)
                except OSError as exc:
                    messagebox.showerror(
                        "SirilSync", "Cannot create the output folder:\n%s"
                                     % exc, parent=self.root)
                    return
            if os.path.abspath(out_folder) == os.path.abspath(folder):
                messagebox.showerror(
                    "SirilSync", "The output folder is the same as the source "
                                 "folder. Enable overwriting, or pick a "
                                 "different folder.", parent=self.root)
                return

        flagged = [row for row in self.rows
                   if row.is_active() and row.level != EXACT]
        warning = ""
        if flagged:
            warning = ("\n\nThese selected changes could not be fully "
                       "recovered from the history:\n"
                       + "\n".join("    %s" % row.command.get().strip()
                                   for row in flagged)
                       + "\nCheck their commands before continuing.")

        source_note = ""
        if includes_source:
            source_note = ("\n\nThe image currently loaded in Siril is part of "
                           "this batch and will be processed too. Make sure "
                           "its file on disk does not already contain these "
                           "changes, otherwise they are applied twice.")

        target = ("the original files will be overwritten" if overwrite
                  else out_folder)
        if not messagebox.askokcancel(
                "SirilSync",
                "Apply %d change(s) to %d image(s)?\n\nOutput: %s%s%s"
                % (len(commands), len(files), target, source_note, warning),
                parent=self.root):
            return

        self.cancel.clear()
        self.sync_button.configure(state="disabled")
        self.post("maximum", len(files))
        self.worker = threading.Thread(
            target=self._run, args=(files, commands, overwrite, out_folder),
            daemon=True)
        self.worker.start()

    def _run(self, files, commands, overwrite, out_folder):
        siril = self.siril
        done = 0
        failed = 0

        try:
            value = siril.get_siril_config("core", "extension")
            current_ext = str(value).lower() if value else ".fit"
        except Exception:
            current_ext = ".fit"
        original_ext = current_ext

        try:
            for index, path in enumerate(files, start=1):
                if self.cancel.is_set():
                    self.post("log", "Cancelled.")
                    break

                name = os.path.basename(path)
                self.post("status", "[%d/%d] %s" % (index, len(files), name))
                base, ext = split_ext(path)

                try:
                    siril.cmd("load", quote(path))

                    for command in commands:
                        if self.cancel.is_set():
                            raise InterruptedError()
                        siril.cmd(command)

                    if ext.endswith(".fz"):
                        ext = ext[:-3]
                        self.post("log", "%s: saved uncompressed." % name)

                    if ext in FITS_EXTS and ext != current_ext:
                        siril.cmd("setext", ext[1:])
                        current_ext = ext

                    if overwrite:
                        out_base = base
                    else:
                        out_base = os.path.join(out_folder,
                                                os.path.basename(base))
                    siril.cmd(save_command(out_base, ext))
                    done += 1
                except InterruptedError:
                    self.post("log", "Cancelled.")
                    break
                except s.SirilError as exc:
                    failed += 1
                    self.post("log", "%s: FAILED - %s" % (name, exc))

                self.post("progress", index)
        finally:
            if current_ext != original_ext:
                try:
                    siril.cmd("setext", original_ext.lstrip("."))
                except Exception:
                    pass
            if self.source_path and os.path.isfile(self.source_path):
                try:
                    siril.cmd("load", quote(self.source_path))
                except Exception:
                    pass
            self.post("done", (done, failed))

    def on_worker_finished(self, result):
        done, failed = result
        self.sync_button.configure(state="normal")
        message = "Finished: %d processed, %d failed." % (done, failed)
        self.status.configure(text=message)
        self.write_log(message)
        try:
            self.siril.log("SirilSync: " + message)
        except Exception:
            pass
        if failed:
            messagebox.showwarning(
                "SirilSync", message + "\nSee the log for details.",
                parent=self.root)
        else:
            messagebox.showinfo("SirilSync", "%d image(s) processed." % done,
                                parent=self.root)

    # -- closing ----------------------------------------------------------- #

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askokcancel(
                    "SirilSync", "A sync is running. Stop it and close?",
                    parent=self.root):
                return
            self.cancel.set()
            self.worker.join(timeout=15)
        try:
            self.siril.disconnect()
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()


# --------------------------------------------------------------------------- #

def main():
    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilError as exc:
        print("SirilSync: could not connect to Siril: %s" % exc)
        return

    root = ThemedTk()
    try:
        tksiril.match_theme_to_siril(root, siril)
    except Exception:
        pass

    SirilSync(root, siril)

    try:
        tksiril.elevate(root)
    except Exception:
        pass

    root.mainloop()


if __name__ == "__main__":
    main()
