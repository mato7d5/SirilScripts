#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert FITS files between bit depths (Siril 1.4+).

Point the script at a folder of FITS files and pick the target depth:

  32-bit float    -> 16-bit unsigned integer     (smaller files, lossy)
  16-bit integer  -> 32-bit float                (lossless, twice the size)

Siril works internally with exactly two pixel formats - unsigned 16-bit integer
in [0, 65535] and 32-bit float in [0, 1] - so those are the two targets. Each
file is read, its pixel data is converted, and the result is written straight to
disk. The image currently loaded in Siril is never touched, and Siril's own
16/32-bit processing preference is left alone.

Scaling
-------
The two formats use different value ranges, so the values are rescaled to keep
the picture identical:

  float -> integer   value * 65535, rounded, clipped to [0, 65535]
  integer -> float   value / 65535

Float files that are already stored in ADU (maximum well above 1.0) are detected
and rounded without rescaling, otherwise everything above 1.0 would clip to
white. Turn the scaling off with --no-rescale to cast the raw numbers instead.

Going from float to 16-bit is lossy: values outside the range are clipped and the
fine gradation between two integer steps is gone. The script counts the clipped
pixels of every file and reports them, so a bad conversion does not pass
unnoticed. The reverse direction is lossless but does not recover what an earlier
16-bit conversion threw away.

Usage
-----
  A) From the Siril GUI: copy this script into the scripts directory
     (Windows: %LOCALAPPDATA%\\siril\\scripts, Linux/macOS: ~/.siril/scripts)
     and run it from the Scripts menu. A dialog opens.

  B) From a command line, with Siril running and its Python environment active:
     python DepthFITSConversion.py --folder "D:/astro/M31/lights" --target 16 --no-gui

Without arguments the GUI opens. Command line arguments pre-fill the form; with
--no-gui the script runs straight away, without a window.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import numpy as np

try:
    import sirilpy as s
except ImportError:  # pragma: no cover
    sys.exit("Module 'sirilpy' not found - run this script from Siril 1.4+.")

try:
    from sirilpy import LogColor
except ImportError:  # builds of sirilpy without LogColor
    LogColor = None

try:
    from sirilpy import tksiril  # matches the GUI to Siril's theme (1.4.x)
except ImportError:
    tksiril = None


FITS_EXTS = (".fit", ".fits", ".fts")

# target depth -> (numpy dtype, BITPIX written by Siril, label)
TARGET_16 = "16"
TARGET_32 = "32"
TARGETS = {
    TARGET_16: (np.uint16, 16, "16-bit unsigned integer"),
    TARGET_32: (np.float32, -32, "32-bit float"),
}
TARGET_LABELS = {
    TARGET_16: "16-bit unsigned integer (0 - 65535)",
    TARGET_32: "32-bit float (0.0 - 1.0)",
}

# the two ways of choosing what to convert
SOURCE_FOLDER = "folder"
SOURCE_SEQUENCE = "sequence"

USHORT_MAX = 65535.0

# A float image whose maximum sits above this is taken to be stored in ADU
# rather than in Siril's normalised [0, 1] range.
ADU_THRESHOLD = 1.5


class ConvertError(RuntimeError):
    """An error we can explain to the user in plain language."""


# ---------------------------------------------------------------------------
# logging and progress (the GUI registers its own sinks here)
# ---------------------------------------------------------------------------

_LOG_SINKS: list = []
_PROGRESS_SINKS: list = []


def add_log_sink(callback) -> None:
    _LOG_SINKS.append(callback)


def add_progress_sink(callback) -> None:
    _PROGRESS_SINKS.append(callback)


def log(siril, message: str, color: str | None = None) -> None:
    """Write a message to Siril's log and to every registered sink (the GUI)."""
    try:
        if LogColor is not None and color is not None:
            siril.log(message, getattr(LogColor, color.upper(), LogColor.DEFAULT))
        else:
            siril.log(message)
    except Exception:
        print(message)
    for sink in _LOG_SINKS:
        try:
            sink(message, color)
        except Exception:
            pass


def emit_progress(done: int, total: int) -> None:
    for sink in _PROGRESS_SINKS:
        try:
            sink(done, total)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def split_ext(path) -> tuple:
    """Split a name, keeping '.fits.fz' style double extensions together."""
    base, ext = os.path.splitext(str(path))
    if ext.lower() == ".fz":
        base2, ext2 = os.path.splitext(base)
        if ext2.lower() in FITS_EXTS:
            return base2, ext2.lower() + ext.lower()
    return base, ext.lower()


def is_fits(name: str) -> bool:
    ext = split_ext(name)[1]
    return ext in FITS_EXTS or ext.endswith(".fz")


def list_fits(folder: Path, recursive: bool) -> list:
    """Every FITS file in the folder, sorted by path."""
    found = []
    try:
        if recursive:
            for root, _dirs, files in os.walk(folder):
                for name in files:
                    if is_fits(name):
                        found.append(Path(root) / name)
        else:
            for entry in folder.iterdir():
                if entry.is_file() and is_fits(entry.name):
                    found.append(entry)
    except OSError:
        return []
    return sorted(found)


def clean_seq_name(name: str) -> str:
    """'light_.seq' / 'light_' -> 'light_' (the name Siril's seq commands use)."""
    name = (name or "").strip().strip('"')
    if name.lower().endswith(".seq"):
        name = name[:-4]
    return name


def list_sequences(folder: Path) -> list:
    """Sequence names for which a .seq file exists in the folder."""
    try:
        return sorted((p.stem for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() == ".seq"),
                      key=str.lower)
    except OSError:
        return []


def sequence_frames(folder: Path, seq_name: str) -> list:
    """The FITS frames of a regular sequence, in frame-number order.

    Empty for sequences that are not one file per frame (SER, FITSEQ) - there is
    no individual file to convert in those.
    """
    if not seq_name:
        return []
    prefix = seq_name.lower()
    found = []
    try:
        entries = list(folder.iterdir())
    except OSError:
        return []
    for path in entries:
        if not path.is_file():
            continue
        base, ext = split_ext(path.name)
        if ext not in FITS_EXTS and not ext.endswith(".fz"):
            continue
        base = Path(base).name
        if not base.lower().startswith(prefix):
            continue
        digits = base[len(seq_name):]
        if digits.isdigit():
            found.append((int(digits), path))
    return [path for _number, path in sorted(found, key=lambda item: item[0])]


def read_fits_bitpix(path: Path):
    """BITPIX read straight from the primary header, or None.

    Used to tell what is actually stored on disk, and afterwards to confirm that
    the file we wrote really has the depth that was asked for. Compressed FITS
    keeps the real data in an extension, so None is returned for those.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(6) != b"SIMPLE":
                return None
            handle.seek(0)
            for _block in range(64):
                block = handle.read(2880)
                if len(block) < 2880:
                    return None
                text = block.decode("ascii", "replace")
                for index in range(36):
                    card = text[index * 80:(index + 1) * 80]
                    key = card[:8].strip()
                    if key == "BITPIX":
                        value = card[10:].split("/")[0].strip()
                        try:
                            return int(value)
                        except ValueError:
                            return None
                    if key == "END":
                        return None
        return None
    except OSError:
        return None


def depth_name(bitpix) -> str:
    return {8: "8-bit integer", 16: "16-bit signed integer",
            20: "16-bit unsigned integer", 32: "32-bit integer",
            -32: "32-bit float", -64: "64-bit float"}.get(
                bitpix, "unknown" if bitpix is None else str(bitpix))


def matches_target(bitpix, target: str) -> bool:
    """Is a file with this BITPIX already stored at the requested depth?"""
    if bitpix is None:
        return False
    if target == TARGET_16:
        return bitpix in (16, 20)
    return bitpix == -32


# ---------------------------------------------------------------------------
# the conversion itself
# ---------------------------------------------------------------------------

def convert_pixels(data, target: str, rescale: bool) -> tuple:
    """Return (converted array, number of clipped pixels).

    Siril hands the data over as uint16 or float32; those are also the only two
    formats it can write, so the conversion is between exactly those.
    """
    dtype = TARGETS[target][0]
    clipped = 0

    if target == TARGET_16:
        if data.dtype == np.uint16:
            return data, 0                      # already there, nothing to do
        values = data.astype(np.float32)
        if rescale and float(np.nanmax(values)) <= ADU_THRESHOLD:
            values = values * USHORT_MAX        # normalised [0, 1] -> ADU
        clipped = int(np.count_nonzero((values < 0.0) | (values > USHORT_MAX)))
        values = np.clip(values, 0.0, USHORT_MAX)
        return np.rint(values).astype(np.uint16), clipped

    if data.dtype == np.float32:
        return data, 0                          # already there, nothing to do
    values = data.astype(np.float32)
    if rescale:
        values = values / USHORT_MAX            # ADU -> normalised [0, 1]
    return values.astype(np.float32), 0


def output_path(source: Path, folder: Path, out_folder: Path | None,
                overwrite: bool) -> Path:
    """Where the converted file goes; '.fz' inputs are written uncompressed."""
    base, ext = split_ext(source)
    if ext.endswith(".fz"):
        ext = ext[:-3]
    if overwrite:
        return Path(base + ext)
    try:
        relative = source.relative_to(folder).parent
    except ValueError:
        relative = Path()
    target_dir = out_folder / relative
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / (Path(base).name + ext)


def convert_file(siril, source: Path, destination: Path, target: str,
                 rescale: bool) -> tuple:
    """Convert one file. Returns (clipped pixels, total pixels)."""
    ffit = siril.load_image_from_file(str(source), with_pixels=True)
    if ffit is None:
        raise ConvertError("the file could not be read")
    data = ffit.data
    if data is None:
        raise ConvertError("the file carries no pixel data")

    converted, clipped = convert_pixels(data, target, rescale)
    converted = np.ascontiguousarray(converted)

    if not siril.save_image_file(converted, ffit.header or "", str(destination)):
        raise ConvertError("Siril refused to save the file")
    return clipped, int(data.size)


# ---------------------------------------------------------------------------
# the whole run
# ---------------------------------------------------------------------------

def siril_path(path: Path) -> str:
    """A path in the form Siril's parser understands (forward slashes, quotes)."""
    text = Path(path).as_posix()
    return '"' + text + '"' if " " in text else text


def register_sequence(siril, seq_name: str, folder: Path) -> None:
    """(Re)write the .seq for the converted frames.

    create_new_seq() looks in Siril's own working directory, so it is pointed at
    the folder holding the results and put back afterwards. Rewriting it also
    clears the depth an in-place conversion has just made stale.
    """
    previous = None
    try:
        previous = siril.get_siril_wd()
    except Exception:
        previous = None
    try:
        siril.cmd("cd", siril_path(folder))
        if siril.create_new_seq(seq_name):
            log(siril, "  sequence file written: "
                + str(folder / (seq_name + ".seq")), "green")
        else:
            log(siril, "  Siril did not write the .seq for " + seq_name
                + " - use 'Search sequence' in the Sequence tab.", "salmon")
    except Exception as exc:
        log(siril, "  could not write the .seq for " + seq_name + ": "
            + str(exc), "salmon")
    finally:
        if previous:
            try:
                siril.cmd("cd", siril_path(Path(previous)))
            except Exception:
                pass


def collect_inputs(args) -> tuple:
    """Work out what to convert. Returns (folder, files, description).

    The input is either a folder of FITS files or the name of a sequence living
    in that folder; naming a sequence is what picks the second mode.
    """
    folder = Path(args.folder).resolve() if args.folder else None
    if folder is None or not folder.is_dir():
        raise ConvertError("The source folder does not exist: "
                           + str(args.folder))

    seq_name = clean_seq_name(getattr(args, "sequence", None) or "")
    if not seq_name:
        files = list_fits(folder, args.recursive)
        if not files:
            raise ConvertError("No FITS file was found in " + str(folder) + ".")
        description = (str(folder)
                       + ("  (including subfolders)" if args.recursive else ""))
        return folder, files, description

    files = sequence_frames(folder, seq_name)
    if not files:
        if (folder / (seq_name + ".seq")).is_file():
            raise ConvertError(
                "The sequence '" + seq_name + "' has a .seq file but no "
                "individual FITS frames, so it is a SER or FITSEQ sequence. "
                "There is no per-frame file to convert.")
        raise ConvertError(
            "No frame of the sequence '" + seq_name + "' was found in "
            + str(folder) + ".")
    return folder, files, "sequence '" + seq_name + "' in " + str(folder)


def run_pipeline(siril, args, cancel=None) -> tuple:
    """Convert every file. Returns (converted, skipped, failed)."""
    if args.target not in TARGETS:
        raise ConvertError("The target depth must be 16 or 32.")

    folder, files, description = collect_inputs(args)
    seq_name = clean_seq_name(getattr(args, "sequence", None) or "")

    out_folder = None
    if not args.overwrite:
        if not args.output:
            raise ConvertError(
                "Choose an output folder, or enable overwriting.")
        out_folder = Path(args.output).resolve()
        if out_folder == folder:
            raise ConvertError(
                "The output folder is the same as the source folder. Enable "
                "overwriting, or pick a different folder.")
        try:
            out_folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConvertError("Cannot create the output folder: " + str(exc))

    dtype, want_bitpix, label = TARGETS[args.target]

    log(siril, "=" * 62, "green")
    log(siril, "Converting FITS files to " + label, "green")
    log(siril, "=" * 62, "green")
    log(siril, "  source   : " + description)
    log(siril, "  files    : " + str(len(files)))
    log(siril, "  output   : " + ("overwriting the originals" if args.overwrite
                                  else str(out_folder)))
    log(siril, "  scaling  : " + ("on (values are rescaled between the two "
                                  "conventions)" if args.rescale
                                  else "off (raw values are cast)"))
    if args.target == TARGET_16:
        log(siril, "  NOTE: 32-bit float to 16-bit integer is lossy - values "
                   "outside the range are clipped.", "salmon")

    converted = skipped = failed = 0
    clipped_files = []

    for index, source in enumerate(files, start=1):
        if cancel is not None and cancel.is_set():
            log(siril, "Cancelled.", "salmon")
            break

        name = source.name
        stored = read_fits_bitpix(source)

        if args.skip_matching and matches_target(stored, args.target):
            skipped += 1
            log(siril, "  %s: already %s, skipped." % (name, label))
            emit_progress(index, len(files))
            continue

        try:
            destination = output_path(source, folder, out_folder, args.overwrite)
            clipped, total = convert_file(siril, source, destination,
                                          args.target, args.rescale)

            written = read_fits_bitpix(destination)
            if written is not None and written != want_bitpix:
                raise ConvertError(
                    "the file was written as %s instead of %s"
                    % (depth_name(written), label))

            converted += 1
            note = "%s -> %s" % (depth_name(stored), label)
            if clipped:
                share = 100.0 * clipped / max(total, 1)
                clipped_files.append((name, clipped, share))
                log(siril, "  %s: %s, %d pixel(s) clipped (%.3f%%)"
                    % (name, note, clipped, share), "salmon")
            else:
                log(siril, "  %s: %s" % (name, note))
        except Exception as exc:
            failed += 1
            log(siril, "  %s: FAILED - %s" % (name, exc), "salmon")

        emit_progress(index, len(files))

    if seq_name and args.make_seq and converted:
        register_sequence(siril, seq_name,
                          out_folder if out_folder is not None else folder)

    log(siril, "-" * 62, "green")
    log(siril, "DONE. %d converted, %d skipped, %d failed."
        % (converted, skipped, failed),
        "green" if not failed else "salmon")
    if clipped_files:
        log(siril, "  %d file(s) had clipped pixels - check that the source "
                   "really was in the range the scaling assumed."
            % len(clipped_files), "salmon")
    log(siril, "-" * 62, "green")
    return converted, skipped, failed


# ---------------------------------------------------------------------------
# GUI (tkinter + tksiril - the recommended approach for Siril 1.4.x)
# ---------------------------------------------------------------------------

def build_root():
    """The main window; uses the themed variant when ttkthemes is available."""
    try:
        s.ensure_installed("ttkthemes")
        from ttkthemes import ThemedTk
        return ThemedTk()
    except Exception:
        import tkinter as tk
        return tk.Tk()


class BitDepthGUI:
    """Settings dialog; the conversion runs on its own thread."""

    def __init__(self, root, siril, defaults):
        import queue
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.siril = siril
        self.queue = queue.Queue()
        self.running = False
        self.alive = True
        self.pump_id = None
        self.cancel = threading.Event()

        root.title("Convert the bit depth of FITS files")
        root.minsize(720, 640)

        if tksiril is not None:
            try:
                self.style = tksiril.standard_style()
                tksiril.match_theme_to_siril(root, siril)
            except Exception:
                self.style = ttk.Style()
        else:
            self.style = ttk.Style()

        try:
            wd = siril.get_siril_wd() or ""
        except Exception:
            wd = ""

        d = defaults
        self.var_folder = tk.StringVar(value=d.folder or wd)
        self.var_source = tk.StringVar(
            value=SOURCE_SEQUENCE if d.sequence else SOURCE_FOLDER)
        self.var_sequence = tk.StringVar(value=clean_seq_name(d.sequence or ""))
        self.var_make_seq = tk.BooleanVar(value=d.make_seq)
        self.var_recursive = tk.BooleanVar(value=d.recursive)
        self.var_target = tk.StringVar(value=d.target)
        self.var_skip = tk.BooleanVar(value=d.skip_matching)
        self.var_rescale = tk.BooleanVar(value=d.rescale)
        self.var_overwrite = tk.BooleanVar(value=d.overwrite)
        self.var_output = tk.StringVar(value=d.output or "")
        self.var_count = tk.StringVar(value="No folder selected.")
        self.var_status = tk.StringVar(value="Ready.")

        self._build_widgets()
        self.var_folder.trace_add("write", lambda *_a: self._on_folder_change())
        self.var_sequence.trace_add("write", lambda *_a: self._refresh_count())
        self._refresh_sequences()
        self._sync_source()
        self._sync_output()
        self.pump_id = self.root.after(100, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout -------------------------------------------------------------

    def _build_widgets(self) -> None:
        ttk, tk = self.ttk, self.tk
        from tkinter import scrolledtext

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # --- source ---
        box = ttk.LabelFrame(main, text="Files to convert", padding=8)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Folder:").grid(row=0, column=0, sticky="w", pady=2)
        entry = ttk.Entry(box, textvariable=self.var_folder)
        entry.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        self._tip(entry, "Folder holding the FITS files, or the folder the "
                         "sequence lives in.")
        ttk.Button(box, text="...", width=3,
                   command=lambda: self._browse(self.var_folder)).grid(
                       row=0, column=2, pady=2)

        radio = ttk.Radiobutton(box, text="Every FITS file in the folder",
                                variable=self.var_source, value=SOURCE_FOLDER,
                                command=self._sync_source)
        radio.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.chk_recursive = ttk.Checkbutton(box, text="Include subfolders",
                                             variable=self.var_recursive,
                                             command=self._refresh_count)
        self.chk_recursive.grid(row=2, column=1, sticky="w", padx=(20, 0))
        self._tip(self.chk_recursive,
                  "The folder structure is reproduced in the output folder.")

        radio = ttk.Radiobutton(box, text="One sequence:",
                                variable=self.var_source, value=SOURCE_SEQUENCE,
                                command=self._sync_source)
        radio.grid(row=3, column=0, sticky="w", pady=(6, 0))

        self.cmb_seq = ttk.Combobox(box, textvariable=self.var_sequence)
        self.cmb_seq.grid(row=3, column=1, sticky="ew", padx=6, pady=(6, 0))
        self._tip(self.cmb_seq,
                  "Sequence name, e.g. 'light_'. The list holds the .seq files "
                  "found in the folder. Only sequences stored as one file per "
                  "frame can be converted, not SER or FITSEQ.")
        ttk.Button(box, text="Refresh", command=self._refresh_sequences).grid(
            row=3, column=2, pady=(6, 0))

        self.chk_make_seq = ttk.Checkbutton(
            box, text="Write a .seq for the converted sequence",
            variable=self.var_make_seq)
        self.chk_make_seq.grid(row=4, column=1, sticky="w", padx=(20, 0))
        self._tip(self.chk_make_seq,
                  "Rewrites the .seq next to the results, so Siril picks the "
                  "converted frames up without a manual 'Search sequence'.")

        ttk.Label(box, textvariable=self.var_count).grid(
            row=5, column=1, sticky="w", padx=6, pady=(6, 0))

        # --- target ---
        box = ttk.LabelFrame(main, text="Convert to", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        for row, key in enumerate((TARGET_16, TARGET_32)):
            radio = ttk.Radiobutton(box, text=TARGET_LABELS[key],
                                    variable=self.var_target, value=key,
                                    command=self._sync_target)
            radio.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)

        self.lbl_warning = ttk.Label(box, text="", wraplength=640,
                                     justify="left")
        self.lbl_warning.grid(row=2, column=0, columnspan=2, sticky="w",
                              pady=(4, 0))

        chk = ttk.Checkbutton(
            box, text="Skip files that already have the target depth",
            variable=self.var_skip)
        chk.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._tip(chk, "The depth is read from the BITPIX header of each file.")

        chk = ttk.Checkbutton(
            box, text="Rescale the values between the two conventions",
            variable=self.var_rescale)
        chk.grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        self._tip(chk, "Siril stores 16-bit data as 0 - 65535 and 32-bit data "
                       "as 0.0 - 1.0. With this off the raw numbers are cast, "
                       "which changes how the image looks.")

        # --- output ---
        box = ttk.LabelFrame(main, text="Output", padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        chk = ttk.Checkbutton(box, text="Overwrite the original files",
                              variable=self.var_overwrite,
                              command=self._sync_output)
        chk.grid(row=0, column=0, columnspan=3, sticky="w")
        self._tip(chk, "The originals are replaced. There is no undo.")

        self.lbl_out = ttk.Label(box, text="Save to:")
        self.lbl_out.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ent_out = ttk.Entry(box, textvariable=self.var_output)
        self.ent_out.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        self.btn_out = ttk.Button(box, text="...", width=3,
                                  command=lambda: self._browse(self.var_output))
        self.btn_out.grid(row=1, column=2, pady=(6, 0))

        # --- progress ---
        box = ttk.LabelFrame(main, text="Progress", padding=8)
        box.grid(row=3, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(2, weight=1)
        main.rowconfigure(3, weight=1)

        ttk.Label(box, textvariable=self.var_status).grid(row=0, column=0,
                                                          sticky="w")
        self.progress = ttk.Progressbar(box, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(4, 6))
        self.text = scrolledtext.ScrolledText(box, height=12, wrap="none")
        self.text.grid(row=2, column=0, sticky="nsew")
        self.text.configure(state="disabled")

        # --- buttons ---
        buttons = ttk.Frame(main, padding=(0, 8, 0, 0))
        buttons.grid(row=4, column=0, sticky="ew")
        self.btn_run = ttk.Button(buttons, text="Convert", command=self._start)
        self.btn_run.pack(side="right")
        ttk.Button(buttons, text="Close", command=self._on_close).pack(
            side="right", padx=(0, 6))

        self._sync_target()

    def _tip(self, widget, text) -> None:
        if tksiril is not None:
            try:
                tksiril.create_tooltip(widget, text)
            except Exception:
                pass

    # -- widget callbacks ---------------------------------------------------

    def _browse(self, var) -> None:
        from tkinter import filedialog
        initial = var.get() or self.var_folder.get() or os.getcwd()
        chosen = filedialog.askdirectory(initialdir=initial, parent=self.root)
        if chosen:
            var.set(os.path.normpath(chosen))

    def _folder(self):
        text = self.var_folder.get().strip()
        folder = Path(text) if text else None
        return folder if folder is not None and folder.is_dir() else None

    def _on_folder_change(self) -> None:
        self._refresh_sequences()
        self._refresh_count()

    def _refresh_sequences(self) -> None:
        folder = self._folder()
        names = list_sequences(folder) if folder else []
        self.cmb_seq.configure(values=names)
        if not self.var_sequence.get() and len(names) == 1:
            self.var_sequence.set(names[0])
        self._refresh_count()

    def _sync_source(self) -> None:
        """Only the controls of the chosen input mode stay usable."""
        sequence = self.var_source.get() == SOURCE_SEQUENCE
        self.chk_recursive.state(["disabled"] if sequence else ["!disabled"])
        self.cmb_seq.configure(state="normal" if sequence else "disabled")
        self.chk_make_seq.state(["!disabled"] if sequence else ["disabled"])
        self._refresh_count()

    def _refresh_count(self) -> None:
        folder = self._folder()
        if folder is None:
            self.var_count.set("No folder selected.")
            return
        if self.var_source.get() == SOURCE_SEQUENCE:
            name = clean_seq_name(self.var_sequence.get())
            if not name:
                self.var_count.set("No sequence selected.")
                return
            frames = sequence_frames(folder, name)
            if not frames:
                if (folder / (name + ".seq")).is_file():
                    self.var_count.set(
                        "'" + name + "' has a .seq but no separate frames, so "
                        "it is a SER or FITSEQ sequence - there is no per-frame "
                        "file to convert.")
                else:
                    self.var_count.set("No frame of '" + name + "' found here.")
                return
            self.var_count.set("%d frame(s) in the sequence." % len(frames))
            return
        self.var_count.set("%d FITS file(s) found."
                           % len(list_fits(folder, self.var_recursive.get())))

    def _sync_target(self) -> None:
        if self.var_target.get() == TARGET_16:
            self.lbl_warning.configure(
                text="Lossy: float values are rounded to whole steps and "
                     "anything outside the range is clipped. Clipped pixels are "
                     "counted and reported per file.")
        else:
            self.lbl_warning.configure(
                text="Lossless, but it does not bring back precision that an "
                     "earlier 16-bit conversion already threw away. The files "
                     "become twice as large.")

    def _sync_output(self) -> None:
        state = "disabled" if self.var_overwrite.get() else "normal"
        self.ent_out.configure(state=state)
        self.btn_out.configure(state=state)
        self.lbl_out.configure(state=state)

    # -- starting the work --------------------------------------------------

    def _collect(self):
        """Build the same settings object argparse produces, from the form."""
        from tkinter import messagebox

        folder = self._folder()
        if folder is None:
            messagebox.showerror("Error", "The source folder does not exist.",
                                 parent=self.root)
            return None

        sequence = None
        if self.var_source.get() == SOURCE_SEQUENCE:
            sequence = clean_seq_name(self.var_sequence.get())
            if not sequence:
                messagebox.showerror("Error", "Choose a sequence.",
                                     parent=self.root)
                return None
            files = sequence_frames(folder, sequence)
            if not files:
                messagebox.showerror(
                    "Error",
                    "No frame of the sequence '" + sequence + "' was found.\n\n"
                    "SER and FITSEQ sequences keep every frame in one container "
                    "and cannot be converted frame by frame.", parent=self.root)
                return None
        else:
            files = list_fits(folder, self.var_recursive.get())
            if not files:
                messagebox.showerror("Error", "No FITS file was found there.",
                                     parent=self.root)
                return None

        overwrite = self.var_overwrite.get()
        output = self.var_output.get().strip()
        if not overwrite:
            if not output:
                messagebox.showerror(
                    "Error", "Choose an output folder, or enable overwriting.",
                    parent=self.root)
                return None
            if Path(output).resolve() == folder.resolve():
                messagebox.showerror(
                    "Error", "The output folder is the same as the source "
                             "folder.", parent=self.root)
                return None

        args = parse_args([])
        args.folder = str(folder)
        args.sequence = sequence
        args.recursive = self.var_recursive.get()
        args.make_seq = self.var_make_seq.get()
        args.target = self.var_target.get()
        args.skip_matching = self.var_skip.get()
        args.rescale = self.var_rescale.get()
        args.overwrite = overwrite
        args.output = output or None

        label = TARGETS[args.target][2]
        question = ("Convert %d %s to %s?\n\nOutput: %s"
                    % (len(files),
                       "frame(s) of '" + sequence + "'" if sequence
                       else "file(s)",
                       label,
                       "the originals will be overwritten" if overwrite
                       else output))
        if args.target == TARGET_16:
            question += ("\n\nThis is lossy - values outside the range are "
                         "clipped and cannot be recovered.")
        if not messagebox.askokcancel("Convert", question, parent=self.root):
            return None
        return args

    def _start(self) -> None:
        if self.running:
            return
        args = self._collect()
        if args is None:
            return

        self.running = True
        self.cancel.clear()
        self.btn_run.state(["disabled"])
        self.progress.configure(value=0)
        self._clear_log()
        self.var_status.set("Converting...")

        thread = threading.Thread(target=self._worker, args=(args,), daemon=True)
        thread.start()

    def _worker(self, args) -> None:
        """Runs off the GUI thread; every Siril call is issued from here."""
        try:
            result = run_pipeline(self.siril, args, self.cancel)
            self.queue.put(("done", result, None))
        except ConvertError as exc:
            self.queue.put(("done", None, str(exc)))
        except Exception as exc:
            self.queue.put(("done", None,
                            exc.__class__.__name__ + ": " + str(exc)))

    # -- passing messages from the worker thread to the GUI -----------------

    def sink_log(self, message, color) -> None:
        self.queue.put(("log", message, color))

    def sink_progress(self, done, total) -> None:
        self.queue.put(("progress", done, total))

    def _pump(self) -> None:
        import queue as queue_mod
        if not self.alive:
            return
        try:
            while True:
                item = self.queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "progress":
                    done, total = item[1], item[2]
                    self.progress.configure(value=100.0 * done / max(total, 1))
                    self.var_status.set("Converting... %d/%d" % (done, total))
                elif kind == "done":
                    self._finish(item[1], item[2])
        except queue_mod.Empty:
            pass
        self.pump_id = self.root.after(100, self._pump)

    def _append_log(self, message) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", message + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def _finish(self, result, error) -> None:
        from tkinter import messagebox
        self.running = False
        self.btn_run.state(["!disabled"])
        if error:
            self.var_status.set("Conversion failed.")
            self._append_log("ERROR: " + error)
            messagebox.showerror("Conversion failed", error, parent=self.root)
            return

        converted, skipped, failed = result
        message = ("%d converted, %d skipped, %d failed."
                   % (converted, skipped, failed))
        self.var_status.set(message)
        self.progress.configure(value=100)
        self._refresh_count()
        if failed:
            messagebox.showwarning("Finished", message
                                   + "\nSee the log for details.",
                                   parent=self.root)
        else:
            messagebox.showinfo("Finished", message, parent=self.root)

    def _on_close(self) -> None:
        from tkinter import messagebox
        if self.running:
            if not messagebox.askyesno(
                    "Conversion is running",
                    "A conversion is still running. Close the window anyway?",
                    parent=self.root):
                return
            self.cancel.set()
        # stop the queue pump first, otherwise the pending callback fires
        # after the window is gone and Tcl reports an invalid command
        self.alive = False
        if self.pump_id is not None:
            try:
                self.root.after_cancel(self.pump_id)
            except Exception:
                pass
        self.root.destroy()


def launch_gui(siril, defaults) -> int:
    """Open the dialog; returns 0 (errors are reported inside the window)."""
    root = build_root()
    gui = BitDepthGUI(root, siril, defaults)
    add_log_sink(gui.sink_log)
    add_progress_sink(gui.sink_progress)
    try:
        if tksiril is not None:
            tksiril.elevate(root)
    except Exception:
        pass
    root.mainloop()
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert FITS files between 16-bit unsigned integer and "
                    "32-bit float (Siril 1.4+).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-gui", dest="gui", action="store_false",
                   help="Skip the dialog and start straight away.")
    p.add_argument("--folder", default=None,
                   help="Folder with the FITS files, or the folder holding the "
                        "sequence; defaults to Siril's working directory.")
    p.add_argument("--sequence", default=None,
                   help="Convert the frames of this sequence instead of every "
                        "FITS file in the folder, e.g. 'light_'.")
    p.add_argument("--recursive", action="store_true",
                   help="Also convert the FITS files in subfolders. Ignored "
                        "when a sequence is named.")
    p.add_argument("--no-seq", dest="make_seq", action="store_false",
                   help="Do not write a .seq for the converted sequence.")
    p.add_argument("--target", default=TARGET_16, choices=[TARGET_16, TARGET_32],
                   help="Target depth: 16 = unsigned integer, 32 = float.")
    p.add_argument("--output", default=None,
                   help="Folder for the converted files. Required unless "
                        "--overwrite is given.")
    p.add_argument("--overwrite", action="store_true",
                   help="Write the converted files over the originals.")
    p.add_argument("--no-skip", dest="skip_matching", action="store_false",
                   help="Convert every file, even one already stored at the "
                        "target depth.")
    p.add_argument("--no-rescale", dest="rescale", action="store_false",
                   help="Cast the raw values instead of rescaling between the "
                        "0-65535 and 0.0-1.0 conventions.")
    p.set_defaults(gui=True, skip_matching=True, rescale=True, make_seq=True)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    siril = s.SirilInterface()
    try:
        siril.connect()
    except Exception as exc:
        print("Failed to connect to Siril: " + str(exc))
        print("This script must be run from Siril 1.4+ (Scripts menu) "
              "or from its Python environment.")
        return 1

    try:
        if not args.folder:
            try:
                args.folder = siril.get_siril_wd() or None
            except Exception:
                args.folder = None

        if args.gui:
            try:
                return launch_gui(siril, args)
            except Exception as exc:
                log(siril, "Could not start the GUI (" + str(exc)
                    + "), continuing in text mode.", "salmon")

        _converted, _skipped, failed = run_pipeline(siril, args)
        return 1 if failed else 0

    except ConvertError as exc:
        message = str(exc)
        log(siril, "ERROR: " + message, "red")
        try:
            siril.error_messagebox(message[:1000])
        except Exception:
            pass
        return 1
    except Exception as exc:  # unexpected failure - do not fail silently
        message = "Unexpected error: " + exc.__class__.__name__ + ": " + str(exc)
        log(siril, message, "red")
        try:
            siril.error_messagebox(message[:1000])
        except Exception:
            pass
        return 1
    finally:
        try:
            siril.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
