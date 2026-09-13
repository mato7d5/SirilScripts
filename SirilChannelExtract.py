#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract a single colour channel (R, G or B) from an OSC sequence (Siril 1.4+).

Give the script a sequence name. It looks at the first frame, decides whether the
sequence comes from a one-shot colour (OSC) camera, and if it does, offers to
extract the red, green or blue channel into a new sequence.

Two kinds of OSC sequence are recognised:

  CFA (undebayered)   one channel per frame plus a BAYERPAT header.
                      R / B  ->  seqsplit_cfa, keeping the Bayer plane that
                                 carries the requested colour
                      G      ->  seqextract_Green (both green pixels averaged)
                      The result is half width and half height - these are the
                      real sensor pixels, nothing is interpolated.

  RGB (debayered)     three channels per frame.
                      Each frame is loaded and split with the "split" command,
                      and only the requested channel is kept. Full resolution.

A mono sequence has no colour channels to extract, and the script says so
instead of producing something meaningless.

Usage
-----
  A) From the Siril GUI: copy this script into the scripts directory
     (Windows: %LOCALAPPDATA%\\siril\\scripts, Linux/macOS: ~/.siril/scripts)
     and run it from the Scripts menu. A dialog opens.

  B) From a command line, with Siril running and its Python environment active:
     python SirilChannelExtract.py --sequence light_ --channel R --no-gui

Without arguments the GUI opens. Command line arguments pre-fill the form; with
--no-gui the script runs straight away, without a window.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from dataclasses import dataclass, field
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


CHANNELS = ("R", "G", "B")
CHANNEL_LABELS = {"R": "Red", "G": "Green", "B": "Blue"}

# Bayer pattern -> which split_cfa plane carries which colour.
#
# split_cfa cuts the 2x2 Bayer cell into four quarter-size images, numbered in
# reading order: 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right.
# BAYERPAT names the same four positions in the same order, so the mapping is a
# direct lookup. Green appears twice; seqextract_Green is used for it instead,
# because it combines both green pixels.
BAYER_PLANES = {
    "RGGB": {"R": 0, "G": (1, 2), "B": 3},
    "BGGR": {"R": 3, "G": (1, 2), "B": 0},
    "GRBG": {"R": 1, "G": (0, 3), "B": 2},
    "GBRG": {"R": 2, "G": (0, 3), "B": 1},
}

FITS_EXTS = (".fit", ".fits", ".fts")

# sequence kinds reported by analyse_sequence()
KIND_CFA = "cfa"
KIND_RGB = "rgb"
KIND_MONO = "mono"
KIND_UNKNOWN = "unknown"

class ExtractError(RuntimeError):
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


def run(siril, *args, quiet: bool = False) -> None:
    """Run a Siril command, raising ExtractError with context on failure."""
    argv = [str(a) for a in args]
    printable = " ".join(argv)
    if not quiet:
        log(siril, "  > " + printable)
    try:
        siril.cmd(*argv)
    except Exception as exc:
        raise ExtractError(
            "Command failed: " + printable + "\n    -> " + str(exc)
        ) from exc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def quote(text: str) -> str:
    """Quote for Siril's parser if the path contains a space."""
    return '"' + text + '"' if " " in text else text


def siril_path(path: Path) -> str:
    """A path in the form Siril's parser understands (forward slashes, quotes)."""
    return quote(Path(path).as_posix())


def clean_seq_name(name: str) -> str:
    """'light_.seq' / 'light_' -> 'light_' (the name Siril's seq* commands take)."""
    name = (name or "").strip().strip('"')
    if name.lower().endswith(".seq"):
        name = name[:-4]
    return name


def list_sequences(folder: Path) -> list[str]:
    """Sequence names for which a .seq file exists in the folder."""
    try:
        return sorted((p.stem for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() == ".seq"),
                      key=str.lower)
    except OSError:
        return []


def sequence_frames(folder: Path, seq_name: str) -> list[Path]:
    """The individual FITS frames of a regular sequence, in frame-number order.

    Returns an empty list for sequences that are not stored as one file per
    frame (SER, FITSEQ), which the per-frame RGB path cannot handle.
    """
    pattern = re.compile(
        r"^" + re.escape(seq_name) + r"(\d+)$", re.IGNORECASE)
    found = []
    try:
        entries = list(folder.iterdir())
    except OSError:
        return []
    for path in entries:
        if not path.is_file():
            continue
        stem, ext = path.stem, path.suffix.lower()
        if ext == ".fz":  # light_00001.fit.fz -> stem 'light_00001.fit'
            stem, ext = os.path.splitext(stem)
            ext = ext.lower()
        if ext not in FITS_EXTS:
            continue
        match = pattern.match(stem)
        if match:
            found.append((int(match.group(1)), path))
    return [path for _number, path in sorted(found, key=lambda item: item[0])]


def frame_number(path: Path, seq_name: str) -> str:
    """'light_00007.fit' -> '00007' (the digits are kept exactly as written)."""
    stem = path.stem
    if path.suffix.lower() == ".fz":
        stem = os.path.splitext(stem)[0]
    return stem[len(seq_name):]


def normalise_pattern(pattern: str) -> str:
    return re.sub(r"[^A-Za-z]", "", pattern or "").upper()


# ---------------------------------------------------------------------------
# looking at the sequence
# ---------------------------------------------------------------------------

@dataclass
class SeqInfo:
    """What the first frame of a sequence tells us about it."""

    name: str = ""
    kind: str = KIND_UNKNOWN
    frames: list = field(default_factory=list)
    width: int = 0
    height: int = 0
    channels: int = 0
    bayer_pattern: str = ""
    row_order: str = ""
    bayer_offset: tuple = (0, 0)
    note: str = ""

    @property
    def is_osc(self) -> bool:
        return self.kind in (KIND_CFA, KIND_RGB)

    def summary(self) -> str:
        if self.kind == KIND_CFA:
            head = "OSC camera, undebayered CFA"
        elif self.kind == KIND_RGB:
            head = "OSC camera, already debayered (RGB)"
        elif self.kind == KIND_MONO:
            head = "Monochrome - no colour channels to extract"
        else:
            head = "Could not be identified"

        parts = [head]
        if self.frames:
            parts.append("%d frame(s)" % len(self.frames))
        if self.width and self.height:
            parts.append("%dx%d" % (self.width, self.height))
        if self.channels:
            parts.append("%d channel(s)" % self.channels)
        if self.bayer_pattern:
            parts.append("BAYERPAT %s" % self.bayer_pattern)
        if self.row_order:
            parts.append("ROWORDER %s" % self.row_order)
        return "   -   ".join(parts)


def analyse_sequence(siril, folder: Path, seq_name: str) -> SeqInfo:
    """Decide what kind of sequence this is by reading its first frame.

    The frame is read with load_image_from_file(), which does not disturb the
    image currently loaded in Siril.
    """
    info = SeqInfo(name=seq_name)

    if not seq_name:
        raise ExtractError("Enter a sequence name.")
    if not folder.is_dir():
        raise ExtractError("The working directory does not exist: " + str(folder))

    info.frames = sequence_frames(folder, seq_name)
    if not info.frames:
        if (folder / (seq_name + ".seq")).is_file():
            info.note = (
                "A .seq file exists but no individual FITS frames were found, so "
                "this is a SER or FITSEQ sequence. Its type cannot be detected "
                "here; extraction is only offered for CFA data, which Siril's "
                "sequence commands handle themselves.")
            return info
        raise ExtractError(
            "No frames of the sequence '" + seq_name + "' were found in "
            + str(folder) + ".")

    first = info.frames[0]
    try:
        ffit = siril.load_image_from_file(str(first), with_pixels=False)
    except Exception as exc:
        raise ExtractError(
            "Could not read the first frame (" + first.name + "): " + str(exc)
        ) from exc
    if ffit is None:
        raise ExtractError("Could not read the first frame: " + first.name)

    try:
        width, height, channels = ffit.naxes
    except Exception:
        raise ExtractError("The first frame reports no usable dimensions.")
    info.width, info.height, info.channels = int(width), int(height), int(channels)

    keywords = getattr(ffit, "keywords", None)
    info.bayer_pattern = normalise_pattern(getattr(keywords, "bayer_pattern", ""))
    info.row_order = (getattr(keywords, "row_order", "") or "").strip()
    info.bayer_offset = (int(getattr(keywords, "bayer_xoffset", 0) or 0),
                         int(getattr(keywords, "bayer_yoffset", 0) or 0))

    if info.channels >= 3:
        info.kind = KIND_RGB
    elif info.bayer_pattern:
        info.kind = KIND_CFA
        if len(info.bayer_pattern) > 4:
            info.note = (
                "The Bayer pattern is " + info.bayer_pattern + ", which is an "
                "X-Trans matrix, not a 2x2 Bayer cell. split_cfa cannot separate "
                "it into R/G/B planes.")
        elif info.bayer_pattern not in BAYER_PLANES:
            info.note = ("Unknown Bayer pattern '" + info.bayer_pattern
                         + "' - choose the CFA plane manually.")
        elif info.bayer_offset != (0, 0):
            info.note = (
                "BAYERPAT is offset by " + str(info.bayer_offset)
                + ", so the pattern does not start at the first pixel. Check the "
                "result, and use the plane override if the colours are swapped.")
    else:
        info.kind = KIND_MONO
        info.note = ("The frames have one channel and no BAYERPAT header, so this "
                     "is monochrome data.")

    return info


def plane_for_channel(info: SeqInfo, channel: str) -> int | None:
    """Which split_cfa plane carries this colour, or None if it is not known."""
    planes = BAYER_PLANES.get(info.bayer_pattern)
    if not planes:
        return None
    value = planes[channel]
    return value[0] if isinstance(value, tuple) else value


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def register_sequence(siril, root: str) -> None:
    """Ask Siril to write a .seq for the files we produced."""
    try:
        if siril.create_new_seq(root):
            log(siril, "  sequence created: " + root + ".seq", "green")
        else:
            log(siril, "  Siril did not create a .seq for " + root
                + " - use 'Search sequence' in the Sequence tab.", "salmon")
    except Exception as exc:
        log(siril, "  could not create the .seq for " + root + ": " + str(exc),
            "salmon")


def strip_bayer_cards(header: str) -> str:
    """Drop the CFA keywords - an extracted plane is no longer mosaiced."""
    if not header:
        return ""
    dropped = ("BAYERPAT", "XBAYROFF", "YBAYROFF")
    cards = [header[i:i + 80] for i in range(0, len(header), 80)]
    return "".join(card for card in cards
                   if card[:8].strip().upper() not in dropped)


def as_2d(data):
    """Siril hands over planar data; a one-channel image may arrive as (1,h,w)."""
    if data is None:
        raise ExtractError("the frame carries no pixel data")
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    return data


def save_plane(siril, plane, header: str, path: Path) -> None:
    """Write a single 2-D plane as its own FITS file.

    save_image_file() goes straight to disk without touching the image loaded in
    Siril, which is what lets us write one channel and nothing else.
    """
    array = np.ascontiguousarray(plane)
    if array.dtype not in (np.float32, np.uint16):
        array = array.astype(np.float32)
    if not siril.save_image_file(array, header, str(path)):
        raise ExtractError("Siril refused to save " + path.name)


def warn_if_pattern_looks_flipped(siril, data, info: SeqInfo) -> None:
    """Sanity-check the Bayer mapping against the pixels themselves.

    Both green positions carry the same filter, so their statistics are nearly
    identical while red and blue differ. Whichever diagonal of the 2x2 cell holds
    that matching pair is the green one. If that disagrees with BAYERPAT, the
    stored row order has flipped the pattern and red/blue are swapped.
    """
    planes = BAYER_PLANES.get(info.bayer_pattern)
    if not planes:
        return
    try:
        medians = [float(np.median(data[plane // 2::2, plane % 2::2]))
                   for plane in range(4)]
    except Exception:
        return

    anti = abs(medians[1] - medians[2])   # greens of RGGB / BGGR
    main = abs(medians[0] - medians[3])   # greens of GRBG / GBRG
    tight, loose = sorted((anti, main))
    if loose < tight * 3:
        return  # the two pairs are too alike to conclude anything

    if (set(planes["G"]) == {1, 2}) == (anti <= main):
        return

    log(siril, "  WARNING: BAYERPAT puts green on planes " + str(planes["G"])
        + ", but the pixel statistics point at the other diagonal. The stored "
        "row order has probably flipped the pattern, so red and blue come out "
        "swapped - set the CFA plane by hand.", "salmon")


def extract_frames(siril, folder: Path, info: SeqInfo, channel: str,
                   plane: int | None, out_prefix: str, ext: str,
                   make_seq: bool, cancel=None) -> str:
    """Write the requested channel of every frame, and nothing else.

    Siril's own "split" and "split_cfa" commands always write all of the
    channels, so the wanted plane is taken out of the pixel data and saved on its
    own instead.
    """
    out_root = out_prefix + info.name
    written, failed = 0, 0
    checked = False

    log(siril, "  writing %d frame(s) to %s"
        % (len(info.frames), out_root + "NNNNN" + ext))

    for index, frame in enumerate(info.frames, start=1):
        if cancel is not None and cancel.is_set():
            log(siril, "  cancelled.", "salmon")
            break

        target = folder / (out_root + frame_number(frame, info.name) + ext)
        try:
            ffit = siril.load_image_from_file(str(frame), with_pixels=True)
            if ffit is None:
                raise ExtractError("the frame could not be read")
            header = ffit.header or ""

            if info.kind == KIND_RGB:
                data = ffit.data
                if data is None or data.ndim != 3:
                    raise ExtractError("the frame is not a 3-channel image")
                pixels = data[CHANNELS.index(channel)]
            else:
                data = as_2d(ffit.data)
                if not checked:
                    warn_if_pattern_looks_flipped(siril, data, info)
                    checked = True
                pixels = data[plane // 2::2, plane % 2::2]
                header = strip_bayer_cards(header)

            save_plane(siril, pixels, header, target)
            written += 1
        except Exception as exc:
            failed += 1
            log(siril, "  %s: FAILED - %s" % (frame.name, exc), "salmon")

        emit_progress(index, len(info.frames))
        if index % 10 == 0 or index == len(info.frames):
            log(siril, "    [%d/%d] %s" % (index, len(info.frames), frame.name))

    if not written:
        raise ExtractError("No frame could be processed - nothing was written.")
    if failed:
        log(siril, "  %d frame(s) failed and were skipped." % failed, "salmon")

    if make_seq:
        register_sequence(siril, out_root)
    return out_root


# ---------------------------------------------------------------------------
# the whole run
# ---------------------------------------------------------------------------

def run_pipeline(siril, args, info: SeqInfo | None = None, cancel=None) -> str:
    """Extract the requested channel. Raises ExtractError on failure."""
    original_wd = None
    try:
        original_wd = siril.get_siril_wd()
    except Exception:
        original_wd = None

    try:
        run(siril, "requires", "1.4.0")

        folder = Path(args.work_dir).resolve() if args.work_dir \
            else Path(original_wd or ".").resolve()
        if not folder.is_dir():
            raise ExtractError("The working directory does not exist: " + str(folder))

        seq_name = clean_seq_name(args.sequence)
        channel = args.channel.upper()
        if channel not in CHANNELS:
            raise ExtractError("The channel must be one of R, G or B.")

        run(siril, "cd", siril_path(folder))

        if info is None or info.name != seq_name:
            info = analyse_sequence(siril, folder, seq_name)

        try:
            value = siril.get_siril_config("core", "extension")
            ext = str(value).lower() if value else ".fit"
        except Exception:
            ext = ".fit"
        if not ext.startswith("."):
            ext = "." + ext

        out_prefix = args.prefix if args.prefix is not None else channel + "_"
        if not out_prefix:
            raise ExtractError("The output prefix must not be empty.")

        log(siril, "=" * 62, "green")
        log(siril, "Extracting the %s channel from an OSC sequence"
            % CHANNEL_LABELS[channel], "green")
        log(siril, "=" * 62, "green")
        log(siril, "  working directory : " + str(folder))
        log(siril, "  sequence          : " + seq_name)
        log(siril, "  detected          : " + info.summary())
        if info.note:
            log(siril, "  note              : " + info.note, "salmon")

        if info.kind == KIND_MONO:
            raise ExtractError(
                "'" + seq_name + "' is a monochrome sequence - there are no "
                "colour channels to extract.")
        if info.kind == KIND_UNKNOWN and not args.force_cfa:
            raise ExtractError(
                "The type of '" + seq_name + "' could not be detected. If you "
                "know it holds undebayered CFA frames, use --force-cfa.")

        # --- CFA green: Siril's own command already writes one sequence only --
        if info.kind == KIND_CFA and channel == "G" and args.plane is None:
            log(siril, "[1/1] Extracting the green pixels...", "blue")
            out_root = out_prefix + info.name
            run(siril, "seqextract_Green", quote(info.name),
                "-prefix=" + out_prefix)
            log(siril, "  both green pixels are combined into "
                + out_root + "NNNNN" + ext)
            if args.make_seq:
                register_sequence(siril, out_root)

        # --- everything else is taken out of the pixel data, one plane only --
        else:
            plane = None
            if info.kind != KIND_RGB:
                plane = args.plane
                if plane is None:
                    plane = plane_for_channel(info, channel)
                if plane is None:
                    raise ExtractError(
                        "The Bayer pattern '" + (info.bayer_pattern or "?")
                        + "' is not one of RGGB / BGGR / GRBG / GBRG, so the "
                          "plane carrying " + CHANNEL_LABELS[channel]
                        + " is unknown. Choose the CFA plane manually.")
                log(siril, "[1/1] Taking Bayer plane %d (%s)..."
                    % (plane, CHANNEL_LABELS[channel]), "blue")
            else:
                log(siril, "[1/1] Taking the %s channel of every frame..."
                    % CHANNEL_LABELS[channel].lower(), "blue")

            out_root = extract_frames(siril, folder, info, channel, plane,
                                      out_prefix, ext, args.make_seq, cancel)

        log(siril, "-" * 62, "green")
        log(siril, "DONE. New sequence: " + out_root, "green")
        if info.kind == KIND_CFA:
            log(siril, "  CFA extraction halves the width and the height - the "
                       "result is %dx%d." % (info.width // 2, info.height // 2))
        log(siril, "-" * 62, "green")
        return out_root

    finally:
        if original_wd:
            try:
                siril.cmd("cd", siril_path(Path(original_wd)))
            except Exception:
                pass


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


PLANE_AUTO = "Automatic (from BAYERPAT)"


class ChannelExtractGUI:
    """Sequence picker and channel chooser; the work runs on its own thread."""

    def __init__(self, root, siril, defaults):
        import queue
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.siril = siril
        self.defaults = defaults
        self.queue = queue.Queue()
        self.running = False
        self.alive = True
        self.pump_id = None
        self.cancel = threading.Event()
        self.info: SeqInfo | None = None

        root.title("Extract a colour channel from an OSC sequence")
        root.minsize(720, 620)

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
        self.var_work = tk.StringVar(value=d.work_dir or wd)
        self.var_seq = tk.StringVar(value=clean_seq_name(d.sequence or ""))
        self.var_channel = tk.StringVar(value=(d.channel or "R").upper())
        self.var_plane = tk.StringVar(value=PLANE_AUTO)
        self.var_plane_hint = tk.StringVar(value="")
        self.var_prefix = tk.StringVar(value=d.prefix or "")
        self.var_auto_prefix = tk.BooleanVar(value=d.prefix is None)
        self.var_make_seq = tk.BooleanVar(value=d.make_seq)
        self.var_detected = tk.StringVar(value="Choose a sequence.")
        self.var_status = tk.StringVar(value="Ready.")

        self._build_widgets()
        self._refresh_sequences()
        if self.var_seq.get():
            self._analyse()
        else:
            # nothing to analyse yet, but the controls still have to reflect it
            self._sync_channel()
        self._sync_prefix()
        self.pump_id = self.root.after(100, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout -------------------------------------------------------------

    def _build_widgets(self) -> None:
        ttk, tk = self.ttk, self.tk
        from tkinter import scrolledtext

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # --- sequence ---
        box = ttk.LabelFrame(main, text="Sequence", padding=8)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Working directory:").grid(row=0, column=0, sticky="w",
                                                       pady=2)
        entry = ttk.Entry(box, textvariable=self.var_work)
        entry.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        self._tip(entry, "The directory holding the sequence. Defaults to "
                         "Siril's working directory.")
        ttk.Button(box, text="...", width=3, command=self._browse).grid(
            row=0, column=2, pady=2)

        ttk.Label(box, text="Sequence:").grid(row=1, column=0, sticky="w", pady=2)
        self.cmb_seq = ttk.Combobox(box, textvariable=self.var_seq)
        self.cmb_seq.grid(row=1, column=1, sticky="ew", padx=6, pady=2)
        self.cmb_seq.bind("<<ComboboxSelected>>", lambda _e: self._analyse())
        self.cmb_seq.bind("<Return>", lambda _e: self._analyse())
        self._tip(self.cmb_seq, "Name of the sequence, e.g. 'light_'. The list "
                                "holds the .seq files found in the directory.")
        ttk.Button(box, text="Analyse", command=self._analyse).grid(
            row=1, column=2, pady=2)

        self.lbl_detected = ttk.Label(box, textvariable=self.var_detected,
                                      wraplength=640, justify="left")
        self.lbl_detected.grid(row=2, column=0, columnspan=3, sticky="w",
                               pady=(6, 0))

        # --- channel ---
        box = ttk.LabelFrame(main, text="Channel to extract", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(3, weight=1)

        self.radios = []
        for index, channel in enumerate(CHANNELS):
            radio = ttk.Radiobutton(box, text=CHANNEL_LABELS[channel] +
                                    " (" + channel + ")",
                                    variable=self.var_channel, value=channel,
                                    command=self._sync_channel)
            radio.grid(row=0, column=index, sticky="w", padx=(0, 16), pady=2)
            self.radios.append(radio)

        ttk.Label(box, text="CFA plane:").grid(row=1, column=0, sticky="w", pady=2)
        self.cmb_plane = ttk.Combobox(
            box, textvariable=self.var_plane, state="readonly",
            values=[PLANE_AUTO, "0", "1", "2", "3"])
        self.cmb_plane.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6,
                            pady=2)
        self._tip(self.cmb_plane,
                  "Which quarter of the 2x2 Bayer cell to keep. Automatic reads "
                  "it from BAYERPAT; override it if red and blue come out "
                  "swapped. Green normally uses seqextract_Green instead, which "
                  "combines both green pixels - picking a plane here forces a "
                  "single one.")

        # A disabled ttk widget does not fire the tooltip, so the reason it is
        # greyed out has to be visible without hovering.
        ttk.Label(box, textvariable=self.var_plane_hint, wraplength=620,
                  justify="left").grid(row=2, column=0, columnspan=4,
                                       sticky="w", pady=(2, 0))

        # --- output ---
        box = ttk.LabelFrame(main, text="Output", padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Prefix:").grid(row=0, column=0, sticky="w", pady=2)
        self.ent_prefix = ttk.Entry(box, textvariable=self.var_prefix)
        self.ent_prefix.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        self._tip(self.ent_prefix,
                  "The new sequence is <prefix><sequence>, e.g. R_light_.")
        chk = ttk.Checkbutton(box, text="from the channel", command=self._sync_prefix,
                              variable=self.var_auto_prefix)
        chk.grid(row=0, column=2, sticky="w", pady=2)
        self._tip(chk, "Use R_, G_ or B_ according to the selected channel.")

        chk = ttk.Checkbutton(box, text="Create a .seq for the result",
                              variable=self.var_make_seq)
        chk.grid(row=1, column=0, columnspan=3, sticky="w", pady=2)
        self._tip(chk, "Writes the .seq so the new sequence shows up in Siril "
                       "without a manual 'Search sequence'.")

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
        self.btn_run = ttk.Button(buttons, text="Extract", command=self._start)
        self.btn_run.pack(side="right")
        self.btn_close = ttk.Button(buttons, text="Close", command=self._on_close)
        self.btn_close.pack(side="right", padx=(0, 6))
        ttk.Button(buttons, text="Refresh list",
                   command=self._refresh_sequences).pack(side="left")

    def _tip(self, widget, text) -> None:
        if tksiril is not None:
            try:
                tksiril.create_tooltip(widget, text)
            except Exception:
                pass

    # -- widget callbacks ---------------------------------------------------

    def _browse(self) -> None:
        from tkinter import filedialog
        initial = self.var_work.get() or os.getcwd()
        chosen = filedialog.askdirectory(initialdir=initial, parent=self.root)
        if chosen:
            self.var_work.set(chosen)
            self.var_seq.set("")
            self.info = None
            self._refresh_sequences()
            self.var_detected.set("Choose a sequence.")
            self._sync_channel()

    def _folder(self) -> Path | None:
        text = self.var_work.get().strip()
        if not text:
            return None
        path = Path(text)
        return path if path.is_dir() else None

    def _refresh_sequences(self) -> None:
        folder = self._folder()
        names = list_sequences(folder) if folder else []
        self.cmb_seq.configure(values=names)
        if not self.var_seq.get() and len(names) == 1:
            self.var_seq.set(names[0])

    def _analyse(self) -> None:
        folder = self._folder()
        if folder is None:
            self.var_detected.set("The working directory does not exist.")
            self.info = None
            self._sync_channel()
            return

        name = clean_seq_name(self.var_seq.get())
        self.var_seq.set(name)
        if not name:
            self.var_detected.set("Choose a sequence.")
            self.info = None
            self._sync_channel()
            return

        try:
            self.info = analyse_sequence(self.siril, folder, name)
        except ExtractError as exc:
            self.info = None
            self.var_detected.set(str(exc))
            self._sync_channel()
            return

        text = self.info.summary()
        if self.info.note:
            text += "\n" + self.info.note
        if self.info.kind == KIND_CFA:
            text += ("\nExtraction produces a %dx%d sequence - half the width and "
                     "half the height, one real sensor pixel per output pixel."
                     % (self.info.width // 2, self.info.height // 2))
        self.var_detected.set(text)
        self._sync_channel()

    def _sync_channel(self) -> None:
        """Enable only what makes sense for the sequence that was detected."""
        info = self.info
        usable = info is not None and info.is_osc
        for radio in self.radios:
            radio.state(["!disabled"] if usable else ["disabled"])

        cfa = info is not None and info.kind == KIND_CFA
        self.cmb_plane.configure(state="readonly" if cfa else "disabled")
        if not cfa:
            self.var_plane.set(PLANE_AUTO)
        self.var_plane_hint.set(self._plane_hint(info, cfa))

        self.btn_run.state(["!disabled"] if usable and not self.running
                           else ["disabled"])
        self._sync_prefix()

    def _plane_hint(self, info: SeqInfo | None, cfa: bool) -> str:
        """Why the plane chooser is greyed out, or what Automatic will do."""
        if info is None:
            return ("Only for undebayered CFA sequences - select a sequence "
                    "first.")
        if info.kind == KIND_RGB:
            return ("Not applicable: this sequence is already debayered, so R, G "
                    "and B are real channels and there is no Bayer cell to cut.")
        if info.kind == KIND_MONO:
            return "Not applicable: this sequence is monochrome."
        if not cfa:
            return ("Only for undebayered CFA sequences - the type of this one "
                    "could not be detected.")

        channel = self.var_channel.get()
        planes = BAYER_PLANES.get(info.bayer_pattern)
        if not planes:
            return ("BAYERPAT '" + (info.bayer_pattern or "?") + "' is not one of "
                    "RGGB / BGGR / GRBG / GBRG, so Automatic cannot work out the "
                    "plane - choose it yourself.")

        value = planes[channel]
        if isinstance(value, tuple):
            return ("BAYERPAT %s: green sits on planes %d and %d. Automatic uses "
                    "seqextract_Green, which combines both; picking a plane here "
                    "keeps only that one."
                    % (info.bayer_pattern, value[0], value[1]))
        return ("BAYERPAT %s: Automatic takes plane %d for %s. Change it if red "
                "and blue come out swapped."
                % (info.bayer_pattern, value, CHANNEL_LABELS[channel].lower()))

    def _sync_prefix(self) -> None:
        if self.var_auto_prefix.get():
            self.var_prefix.set(self.var_channel.get() + "_")
            self.ent_prefix.configure(state="disabled")
        else:
            self.ent_prefix.configure(state="normal")

    # -- starting the work --------------------------------------------------

    def _collect(self):
        """Build the same settings object argparse produces, from the form."""
        from tkinter import messagebox

        folder = self._folder()
        if folder is None:
            messagebox.showerror("Error", "The working directory does not exist.",
                                 parent=self.root)
            return None
        if self.info is None or not self.info.is_osc:
            messagebox.showerror(
                "Error", "Choose a sequence that holds OSC data first.",
                parent=self.root)
            return None

        prefix = self.var_prefix.get().strip()
        if not prefix:
            messagebox.showerror("Error", "The output prefix must not be empty.",
                                 parent=self.root)
            return None
        if any(character in prefix for character in '\\/:*?"<>|'):
            messagebox.showerror("Error",
                                 "The prefix must not contain path characters.",
                                 parent=self.root)
            return None

        args = parse_args([])
        args.work_dir = str(folder)
        args.sequence = self.var_seq.get().strip()
        args.channel = self.var_channel.get()
        args.prefix = prefix
        args.plane = (None if self.var_plane.get() == PLANE_AUTO
                      else int(self.var_plane.get()))
        args.make_seq = self.var_make_seq.get()

        out_root = prefix + args.sequence
        existing = sequence_frames(folder, out_root)
        if existing and not messagebox.askokcancel(
                "Overwrite?",
                "%d file(s) named %sNNNNN already exist and will be overwritten.\n\n"
                "Continue?" % (len(existing), out_root), parent=self.root):
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
        self.var_status.set("Working...")

        thread = threading.Thread(target=self._worker, args=(args,), daemon=True)
        thread.start()

    def _worker(self, args) -> None:
        """Runs off the GUI thread; every Siril command is issued from here."""
        try:
            out_root = run_pipeline(self.siril, args, self.info, self.cancel)
            self.queue.put(("done", out_root, None))
        except ExtractError as exc:
            self.queue.put(("done", "", str(exc)))
        except Exception as exc:
            self.queue.put(("done", "", exc.__class__.__name__ + ": " + str(exc)))

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
                    if item[1].startswith("["):
                        self.var_status.set(item[1])
                elif kind == "progress":
                    done, total = item[1], item[2]
                    self.progress.configure(value=100.0 * done / max(total, 1))
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

    def _finish(self, out_root, error) -> None:
        from tkinter import messagebox
        self.running = False
        self._sync_channel()
        if error:
            self.var_status.set("Extraction failed.")
            self._append_log("ERROR: " + error)
            messagebox.showerror("Extraction failed", error, parent=self.root)
        else:
            self.var_status.set("Done - new sequence: " + out_root)
            self.progress.configure(value=100)
            self._refresh_sequences()
            messagebox.showinfo("Done", "New sequence: " + out_root,
                                parent=self.root)

    def _on_close(self) -> None:
        from tkinter import messagebox
        if self.running:
            if not messagebox.askyesno(
                    "Extraction is running",
                    "Extraction is still running. Close the window anyway?",
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
    gui = ChannelExtractGUI(root, siril, defaults)
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
        description="Extract the R, G or B channel from a one-shot colour (OSC) "
                    "sequence into a new sequence (Siril 1.4+).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-gui", dest="gui", action="store_false",
                   help="Skip the dialog and start straight away.")
    p.add_argument("--work-dir", default=None,
                   help="Directory holding the sequence; defaults to Siril's "
                        "working directory.")
    p.add_argument("--sequence", default=None,
                   help="Name of the sequence, e.g. 'light_'.")
    p.add_argument("--channel", default="R", choices=["R", "G", "B", "r", "g", "b"],
                   help="Channel to extract.")
    p.add_argument("--prefix", default=None,
                   help="Prefix of the new sequence "
                        "(default: the channel letter, e.g. 'R_').")
    p.add_argument("--plane", type=int, default=None, choices=[0, 1, 2, 3],
                   help="Force a split_cfa plane instead of deriving it from "
                        "BAYERPAT. Only applies to undebayered CFA sequences.")
    p.add_argument("--no-seq", dest="make_seq", action="store_false",
                   help="Do not write a .seq for the result.")
    p.add_argument("--force-cfa", action="store_true",
                   help="Treat the sequence as undebayered CFA even when its "
                        "type could not be detected (SER, FITSEQ).")
    p.set_defaults(gui=True, make_seq=True)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.channel = args.channel.upper()

    siril = s.SirilInterface()
    try:
        siril.connect()
    except Exception as exc:
        print("Failed to connect to Siril: " + str(exc))
        print("This script must be run from Siril 1.4+ (Scripts menu) "
              "or from its Python environment.")
        return 1

    try:
        if args.gui:
            try:
                return launch_gui(siril, args)
            except Exception as exc:
                log(siril, "Could not start the GUI (" + str(exc)
                    + "), continuing in text mode.", "salmon")

        if not args.sequence:
            log(siril, "ERROR: --sequence is required with --no-gui.", "red")
            return 1

        run_pipeline(siril, args)
        return 0

    except ExtractError as exc:
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
