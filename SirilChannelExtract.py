#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Copyright (C) 2026 Martin Mancuska <martin@martin-in.space>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
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

__version__ = "1.0.0"

import argparse
import html
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

# PyQt6 is the Qt binding that ships in Siril's own Python environment
# (Siril 1.4 bundles PyQt6 6.11 / Qt 6.11). Kept tolerant so --no-gui still
# works on an installation without it.
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError:
    try:
        s.ensure_installed("PyQt6")
        from PyQt6 import QtCore, QtGui, QtWidgets
    except Exception:
        QtCore = QtGui = QtWidgets = None


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

# Siril's log colours, in a light and a dark variant so the embedded log stays
# readable whichever theme Siril is set to.
LOG_COLOURS = {
    "light": {"green": "#1b6e2b", "salmon": "#b34a20", "blue": "#14539a",
              "red": "#b3261e"},
    "dark": {"green": "#7fd18c", "salmon": "#ffb08f", "blue": "#7cb6f2",
             "red": "#ff9b94"},
}


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
# GUI (PyQt6 - the Qt binding that ships in Siril's Python environment)
# ---------------------------------------------------------------------------

PLANE_AUTO = "Automatic (from BAYERPAT)"


if QtWidgets is not None:

    class ChannelExtractWindow(QtWidgets.QWidget):
        """Sequence picker and channel chooser; the work runs on its own thread.

        The worker reports back through Qt signals, which Qt delivers on the GUI
        thread, so no widget is touched from the wrong thread.
        """

        log_line = QtCore.pyqtSignal(str, object)
        progress_changed = QtCore.pyqtSignal(int, int)
        run_finished = QtCore.pyqtSignal(object, object)

        def __init__(self, siril, defaults):
            super().__init__()
            self.siril = siril
            self.running = False
            self.cancel = threading.Event()
            self.info = None
            self.log_theme = "dark" if siril_is_dark(siril) else "light"

            self.setWindowTitle(
                "Extract a colour channel from an OSC sequence v" + __version__)
            self._build_widgets(defaults)

            self.log_line.connect(self._append_log)
            self.progress_changed.connect(self._on_progress)
            self.run_finished.connect(self._finish)

            self._refresh_sequences()
            if self.cmb_seq.currentText().strip():
                self._analyse()
            else:
                self._sync_channel()
            self._sync_prefix()
            self._fit_to_screen()

        # -- layout ---------------------------------------------------------

        def _build_widgets(self, d) -> None:
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(8, 8, 8, 8)

            try:
                wd = self.siril.get_siril_wd() or ""
            except Exception:
                wd = ""

            # --- sequence ---
            box = QtWidgets.QGroupBox("Sequence")
            grid = QtWidgets.QGridLayout(box)
            self.ed_work = self._dir_row(
                grid, 0, "Working directory:", d.work_dir or wd,
                "The directory holding the sequence. Defaults to Siril's "
                "working directory.")
            self.ed_work.editingFinished.connect(self._on_folder_change)

            grid.addWidget(QtWidgets.QLabel("Sequence:"), 1, 0)
            self.cmb_seq = QtWidgets.QComboBox()
            self.cmb_seq.setEditable(True)
            self.cmb_seq.setCurrentText(clean_seq_name(d.sequence or ""))
            self.cmb_seq.setToolTip(
                "Name of the sequence, e.g. 'light_'. The list holds the .seq "
                "files found in the directory.")
            self.cmb_seq.lineEdit().editingFinished.connect(self._analyse)
            self.cmb_seq.activated.connect(lambda _i: self._analyse())
            grid.addWidget(self.cmb_seq, 1, 1)
            btn = QtWidgets.QPushButton("Analyse")
            btn.clicked.connect(self._analyse)
            grid.addWidget(btn, 1, 2)

            self.lbl_detected = QtWidgets.QLabel("Choose a sequence.")
            self.lbl_detected.setWordWrap(True)
            grid.addWidget(self.lbl_detected, 2, 0, 1, 3)
            outer.addWidget(box)

            # --- channel ---
            box = QtWidgets.QGroupBox("Channel to extract")
            grid = QtWidgets.QGridLayout(box)
            row = QtWidgets.QHBoxLayout()
            self.channel_group = QtWidgets.QButtonGroup(self)
            self.radios = []
            wanted = (d.channel or "R").upper()
            for channel in CHANNELS:
                radio = QtWidgets.QRadioButton(
                    "%s (%s)" % (CHANNEL_LABELS[channel], channel))
                radio.setProperty("channel", channel)
                radio.setChecked(channel == wanted)
                radio.toggled.connect(self._sync_channel)
                self.channel_group.addButton(radio)
                row.addWidget(radio)
                self.radios.append(radio)
            row.addStretch(1)
            grid.addLayout(row, 0, 0, 1, 2)

            grid.addWidget(QtWidgets.QLabel("CFA plane:"), 1, 0)
            self.cmb_plane = QtWidgets.QComboBox()
            self.cmb_plane.addItems([PLANE_AUTO, "0", "1", "2", "3"])
            if d.plane is not None:
                self.cmb_plane.setCurrentText(str(d.plane))
            self.cmb_plane.setToolTip(
                "Which quarter of the 2x2 Bayer cell to keep. Automatic reads "
                "it from BAYERPAT; override it if red and blue come out "
                "swapped.")
            self.cmb_plane.currentIndexChanged.connect(self._sync_channel)
            grid.addWidget(self.cmb_plane, 1, 1)
            grid.setColumnStretch(1, 1)

            # A disabled widget does not show its tooltip, so the reason it is
            # greyed out has to be visible without hovering.
            self.lbl_plane_hint = QtWidgets.QLabel("")
            self.lbl_plane_hint.setWordWrap(True)
            grid.addWidget(self.lbl_plane_hint, 2, 0, 1, 2)
            outer.addWidget(box)

            # --- output ---
            box = QtWidgets.QGroupBox("Output")
            grid = QtWidgets.QGridLayout(box)
            grid.addWidget(QtWidgets.QLabel("Prefix:"), 0, 0)
            self.ed_prefix = QtWidgets.QLineEdit(d.prefix or "")
            self.ed_prefix.setToolTip(
                "The new sequence is <prefix><sequence>, e.g. R_light_.")
            grid.addWidget(self.ed_prefix, 0, 1)
            self.chk_auto_prefix = QtWidgets.QCheckBox("from the channel")
            self.chk_auto_prefix.setChecked(d.prefix is None)
            self.chk_auto_prefix.setToolTip(
                "Use R_, G_ or B_ according to the selected channel.")
            self.chk_auto_prefix.toggled.connect(self._sync_prefix)
            grid.addWidget(self.chk_auto_prefix, 0, 2)

            self.chk_make_seq = QtWidgets.QCheckBox(
                "Create a .seq for the result")
            self.chk_make_seq.setChecked(d.make_seq)
            self.chk_make_seq.setToolTip(
                "Writes the .seq so the new sequence shows up in Siril without "
                "a manual 'Search sequence'.")
            grid.addWidget(self.chk_make_seq, 1, 0, 1, 3)
            grid.setColumnStretch(1, 1)
            outer.addWidget(box)

            outer.addWidget(self._log_box(), 1)

            buttons = QtWidgets.QHBoxLayout()
            btn = QtWidgets.QPushButton("Refresh list")
            btn.clicked.connect(self._refresh_sequences)
            buttons.addWidget(btn)
            buttons.addStretch(1)
            btn = QtWidgets.QPushButton("Close")
            btn.clicked.connect(self.close)
            buttons.addWidget(btn)
            self.btn_run = QtWidgets.QPushButton("Extract")
            self.btn_run.setDefault(True)
            self.btn_run.clicked.connect(self._start)
            buttons.addWidget(self.btn_run)
            outer.addLayout(buttons)

        # -- the sinks the pipeline writes into ------------------------------

        def sink_log(self, message, color) -> None:
            self.log_line.emit(message, color)

        def sink_progress(self, done, total) -> None:
            self.progress_changed.emit(done, total)

        # -- slots, all on the GUI thread ------------------------------------

        def _append_log(self, message, color=None) -> None:
            colour = LOG_COLOURS[self.log_theme].get(
                (color or "").lower() if color else "")
            if colour:
                self.text.appendHtml(
                    '<span style="color:%s; white-space:pre">%s</span>'
                    % (colour, html.escape(message)))
            else:
                self.text.appendPlainText(message)

        def _on_progress(self, done, total) -> None:
            self.progress.setValue(int(100.0 * done / max(total, 1)))

        def _fit_to_screen(self, min_w=560, min_h=400) -> None:
            """Size the window to its content, never larger than the screen."""
            available = QtGui.QGuiApplication.primaryScreen().availableGeometry()
            hint = self.sizeHint()
            width = min(max(hint.width(), min_w), int(available.width() * 0.92))
            height = min(max(hint.height(), min_h), int(available.height() * 0.85))
            self.setMinimumSize(min(min_w, width), min(min_h, height))
            self.resize(width, height)
            self.move(available.x() + (available.width() - width) // 2,
                      available.y() + (available.height() - height) // 3)

        def _error(self, message: str) -> None:
            QtWidgets.QMessageBox.critical(self, "Error", message)

        def _log_box(self, height=130):
            """The progress group box every script ends with."""
            box = QtWidgets.QGroupBox("Progress")
            inner = QtWidgets.QVBoxLayout(box)
            self.lbl_status = QtWidgets.QLabel("Ready.")
            inner.addWidget(self.lbl_status)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setRange(0, 100)
            inner.addWidget(self.progress)
            self.text = QtWidgets.QPlainTextEdit()
            self.text.setReadOnly(True)
            self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.text.setMinimumHeight(height)
            inner.addWidget(self.text, 1)
            return box

        def _dir_row(self, grid, row, label, value, tip, browse=True):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            edit = QtWidgets.QLineEdit(value)
            edit.setToolTip(tip)
            grid.addWidget(edit, row, 1)
            if browse:
                button = QtWidgets.QPushButton("...")
                button.setFixedWidth(32)
                button.clicked.connect(lambda _=False, e=edit: self._browse(e))
                grid.addWidget(button, row, 2)
            grid.setColumnStretch(1, 1)
            return edit

        # -- widget callbacks -----------------------------------------------

        def _browse(self, edit) -> None:
            start = edit.text().strip() or os.getcwd()
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select a directory", start)
            if chosen:
                edit.setText(os.path.normpath(chosen))
                self.cmb_seq.setCurrentText("")
                self.info = None
                self._on_folder_change()
                self.lbl_detected.setText("Choose a sequence.")
                self._sync_channel()

        def _folder(self):
            text = self.ed_work.text().strip()
            folder = Path(text) if text else None
            return folder if folder is not None and folder.is_dir() else None

        def _on_folder_change(self) -> None:
            self._refresh_sequences()

        def _refresh_sequences(self) -> None:
            folder = self._folder()
            names = list_sequences(folder) if folder else []
            current = self.cmb_seq.currentText()
            self.cmb_seq.blockSignals(True)
            self.cmb_seq.clear()
            self.cmb_seq.addItems(names)
            self.cmb_seq.setCurrentText(
                current or (names[0] if len(names) == 1 else ""))
            self.cmb_seq.blockSignals(False)

        def _channel(self) -> str:
            for radio in self.radios:
                if radio.isChecked():
                    return radio.property("channel")
            return "R"

        def _analyse(self) -> None:
            folder = self._folder()
            if folder is None:
                self.lbl_detected.setText(
                    "The working directory does not exist.")
                self.info = None
                self._sync_channel()
                return

            name = clean_seq_name(self.cmb_seq.currentText())
            if name != self.cmb_seq.currentText():
                self.cmb_seq.setCurrentText(name)
            if not name:
                self.lbl_detected.setText("Choose a sequence.")
                self.info = None
                self._sync_channel()
                return

            try:
                self.info = analyse_sequence(self.siril, folder, name)
            except ExtractError as exc:
                self.info = None
                self.lbl_detected.setText(str(exc))
                self._sync_channel()
                return

            text = self.info.summary()
            if self.info.note:
                text += "\n" + self.info.note
            if self.info.kind == KIND_CFA:
                text += ("\nExtraction produces a %dx%d sequence - half the "
                         "width and half the height, one real sensor pixel per "
                         "output pixel."
                         % (self.info.width // 2, self.info.height // 2))
            self.lbl_detected.setText(text)
            self._sync_channel()

        def _sync_channel(self) -> None:
            """Enable only what makes sense for the sequence that was detected."""
            info = self.info
            usable = info is not None and info.is_osc
            for radio in self.radios:
                radio.setEnabled(usable)

            cfa = info is not None and info.kind == KIND_CFA
            self.cmb_plane.setEnabled(cfa)
            if not cfa and self.cmb_plane.currentText() != PLANE_AUTO:
                self.cmb_plane.setCurrentText(PLANE_AUTO)
            self.lbl_plane_hint.setText(self._plane_hint(info, cfa))

            self.btn_run.setEnabled(usable and not self.running)
            self._sync_prefix()

        def _plane_hint(self, info, cfa: bool) -> str:
            """Why the plane chooser is greyed out, or what Automatic will do."""
            if info is None:
                return ("Only for undebayered CFA sequences - select a sequence "
                        "first.")
            if info.kind == KIND_RGB:
                return ("Not applicable: this sequence is already debayered, so "
                        "R, G and B are real channels and there is no Bayer "
                        "cell to cut.")
            if info.kind == KIND_MONO:
                return "Not applicable: this sequence is monochrome."
            if not cfa:
                return ("Only for undebayered CFA sequences - the type of this "
                        "one could not be detected.")

            channel = self._channel()
            planes = BAYER_PLANES.get(info.bayer_pattern)
            if not planes:
                return ("BAYERPAT '" + (info.bayer_pattern or "?")
                        + "' is not one of RGGB / BGGR / GRBG / GBRG, so "
                          "Automatic cannot work out the plane - choose it "
                          "yourself.")

            value = planes[channel]
            if isinstance(value, tuple):
                return ("BAYERPAT %s: green sits on planes %d and %d. Automatic "
                        "uses seqextract_Green, which combines both; picking a "
                        "plane here keeps only that one."
                        % (info.bayer_pattern, value[0], value[1]))
            return ("BAYERPAT %s: Automatic takes plane %d for %s. Change it if "
                    "red and blue come out swapped."
                    % (info.bayer_pattern, value,
                       CHANNEL_LABELS[channel].lower()))

        def _sync_prefix(self) -> None:
            if self.chk_auto_prefix.isChecked():
                self.ed_prefix.setText(self._channel() + "_")
                self.ed_prefix.setEnabled(False)
            else:
                self.ed_prefix.setEnabled(True)

        # -- starting the work ----------------------------------------------

        def _collect(self):
            """Build the same settings object argparse produces, from the form."""
            folder = self._folder()
            if folder is None:
                self._error("The working directory does not exist.")
                return None
            if self.info is None or not self.info.is_osc:
                self._error("Choose a sequence that holds OSC data first.")
                return None

            prefix = self.ed_prefix.text().strip()
            if not prefix:
                self._error("The output prefix must not be empty.")
                return None
            if any(ch in prefix for ch in r'\/:*?"<>|'):
                self._error("The prefix must not contain path characters.")
                return None

            args = parse_args([])
            args.work_dir = str(folder)
            args.sequence = self.cmb_seq.currentText().strip()
            args.channel = self._channel()
            args.prefix = prefix
            plane = self.cmb_plane.currentText()
            args.plane = None if plane == PLANE_AUTO else int(plane)
            args.make_seq = self.chk_make_seq.isChecked()

            out_root = prefix + args.sequence
            existing = sequence_frames(folder, out_root)
            if existing:
                answer = QtWidgets.QMessageBox.question(
                    self, "Overwrite?",
                    "%d file(s) named %sNNNNN already exist and will be "
                    "overwritten.\n\nContinue?" % (len(existing), out_root))
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
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
            self.btn_run.setEnabled(False)
            self.progress.setValue(0)
            self.text.clear()
            self.lbl_status.setText("Working...")

            thread = threading.Thread(target=self._worker, args=(args,),
                                      daemon=True)
            thread.start()

        def _worker(self, args) -> None:
            """Runs off the GUI thread; every Siril command is issued here."""
            try:
                self.run_finished.emit(
                    run_pipeline(self.siril, args, self.info, self.cancel),
                    None)
            except ExtractError as exc:
                self.run_finished.emit(None, str(exc))
            except Exception as exc:
                self.run_finished.emit(
                    None, exc.__class__.__name__ + ": " + str(exc))

        def _append_log(self, message, color=None) -> None:
            colour = LOG_COLOURS[self.log_theme].get(
                (color or "").lower() if color else "")
            if colour:
                self.text.appendHtml(
                    '<span style="color:%s; white-space:pre">%s</span>'
                    % (colour, html.escape(message)))
            else:
                self.text.appendPlainText(message)
            if message.startswith("["):
                self.lbl_status.setText(message)

        def _finish(self, out_root, error) -> None:
            self.running = False
            self._sync_channel()
            if error:
                self.lbl_status.setText("Extraction failed.")
                self.text.appendPlainText("ERROR: " + error)
                QtWidgets.QMessageBox.critical(self, "Extraction failed", error)
                return

            self.lbl_status.setText("Done - new sequence: " + out_root)
            self.progress.setValue(100)
            self._refresh_sequences()
            QtWidgets.QMessageBox.information(self, "Done",
                                              "New sequence: " + out_root)

        def closeEvent(self, event) -> None:
            if self.running:
                answer = QtWidgets.QMessageBox.question(
                    self, "Extraction is running",
                    "Extraction is still running. Close the window anyway?")
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                self.cancel.set()
            event.accept()


def siril_is_dark(siril) -> bool:
    """Siril's own light/dark preference (gui.theme: 0 dark, 1 light)."""
    try:
        return siril.get_siril_config("gui", "theme") == 0
    except Exception:
        return False


def apply_siril_theme(app, siril) -> None:
    """Match Qt to Siril's light/dark preference."""
    if not siril_is_dark(siril):
        return  # the light theme is Qt's default look

    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    window = QtGui.QColor(53, 53, 53)
    base = QtGui.QColor(35, 35, 35)
    text = QtGui.QColor(220, 220, 220)
    for target, colour in ((role.Window, window), (role.Base, base),
                           (role.AlternateBase, window), (role.Button, window),
                           (role.ToolTipBase, window), (role.WindowText, text),
                           (role.Text, text), (role.ButtonText, text),
                           (role.ToolTipText, text),
                           (role.Highlight, QtGui.QColor(42, 130, 218)),
                           (role.HighlightedText, QtGui.QColor(0, 0, 0))):
        palette.setColor(target, colour)
    disabled = QtGui.QPalette.ColorGroup.Disabled
    for target in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(disabled, target, QtGui.QColor(127, 127, 127))
    app.setPalette(palette)


def launch_gui(siril, defaults) -> int:
    """Open the dialog; returns 0 (errors are reported inside the window)."""
    if QtWidgets is None:
        raise ExtractError(
            "PyQt6 is not available in this Python environment.")

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])
    apply_siril_theme(app, siril)

    window = ChannelExtractWindow(siril, defaults)
    add_log_sink(window.sink_log)
    add_progress_sink(window.sink_progress)
    window.show()
    window.raise_()
    window.activateWindow()

    if owns_app:
        app.exec()
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
    p.add_argument("--version", action="version",
                   version="%(prog)s " + __version__)
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
