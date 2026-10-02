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

__version__ = "1.0.0"

import argparse
import html
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


# Siril's log colours, in a light and a dark variant so the embedded log stays
# readable whichever theme Siril is set to.
LOG_COLOURS = {
    "light": {"green": "#1b6e2b", "salmon": "#b34a20", "blue": "#14539a",
              "red": "#b3261e"},
    "dark": {"green": "#7fd18c", "salmon": "#ffb08f", "blue": "#7cb6f2",
             "red": "#ff9b94"},
}


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
# GUI (PyQt6 - the Qt binding that ships in Siril's Python environment)
# ---------------------------------------------------------------------------

if QtWidgets is not None:

    class BitDepthWindow(QtWidgets.QWidget):
        """Settings dialog; the conversion runs on its own thread.

        The worker talks to the window through Qt signals, which Qt delivers on
        the GUI thread, so no widget is ever touched from the wrong thread.
        """

        log_line = QtCore.pyqtSignal(str, object)
        progress_changed = QtCore.pyqtSignal(int, int)
        run_finished = QtCore.pyqtSignal(object, object)

        def __init__(self, siril, defaults):
            super().__init__()
            self.siril = siril
            self.running = False
            self.cancel = threading.Event()
            self.log_theme = "dark" if siril_is_dark(siril) else "light"

            self.setWindowTitle(
                "Convert the bit depth of FITS files v" + __version__)
            self._build_widgets(defaults)

            self.log_line.connect(self._append_log)
            self.progress_changed.connect(self._on_progress)
            self.run_finished.connect(self._finish)

            self._refresh_sequences()
            self._sync_source()
            self._sync_target()
            self._sync_output()
            self._fit_to_screen()

        # -- layout ---------------------------------------------------------

        def _build_widgets(self, d) -> None:
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(8, 8, 8, 8)

            try:
                wd = self.siril.get_siril_wd() or ""
            except Exception:
                wd = ""

            # --- source ---
            box = QtWidgets.QGroupBox("Files to convert")
            grid = QtWidgets.QGridLayout(box)
            self.ed_folder = self._dir_row(
                grid, 0, "Folder:", d.folder or wd,
                "Folder holding the FITS files, or the folder the sequence "
                "lives in.")
            self.ed_folder.editingFinished.connect(self._on_folder_change)

            self.rb_folder = QtWidgets.QRadioButton(
                "Every FITS file in the folder")
            self.rb_sequence = QtWidgets.QRadioButton("One sequence:")
            self.source_group = QtWidgets.QButtonGroup(self)
            self.source_group.addButton(self.rb_folder)
            self.source_group.addButton(self.rb_sequence)
            (self.rb_sequence if d.sequence else self.rb_folder).setChecked(True)
            self.rb_folder.toggled.connect(self._sync_source)
            grid.addWidget(self.rb_folder, 1, 0, 1, 2)

            self.chk_recursive = QtWidgets.QCheckBox("Include subfolders")
            self.chk_recursive.setChecked(d.recursive)
            self.chk_recursive.setToolTip(
                "The folder structure is reproduced in the output folder.")
            self.chk_recursive.toggled.connect(self._refresh_count)
            indent = QtWidgets.QHBoxLayout()
            indent.addSpacing(20)
            indent.addWidget(self.chk_recursive)
            indent.addStretch(1)
            grid.addLayout(indent, 2, 0, 1, 3)

            grid.addWidget(self.rb_sequence, 3, 0)
            self.cmb_seq = QtWidgets.QComboBox()
            self.cmb_seq.setEditable(True)
            self.cmb_seq.setToolTip(
                "Sequence name, e.g. 'light_'. The list holds the .seq files "
                "found in the folder. Only sequences stored as one file per "
                "frame can be converted, not SER or FITSEQ.")
            self.cmb_seq.setCurrentText(clean_seq_name(d.sequence or ""))
            self.cmb_seq.currentTextChanged.connect(self._refresh_count)
            grid.addWidget(self.cmb_seq, 3, 1)
            btn = QtWidgets.QPushButton("Refresh")
            btn.clicked.connect(self._refresh_sequences)
            grid.addWidget(btn, 3, 2)

            self.chk_make_seq = QtWidgets.QCheckBox(
                "Write a .seq for the converted sequence")
            self.chk_make_seq.setChecked(d.make_seq)
            self.chk_make_seq.setToolTip(
                "Rewrites the .seq next to the results, so Siril picks the "
                "converted frames up without a manual 'Search sequence'.")
            indent = QtWidgets.QHBoxLayout()
            indent.addSpacing(20)
            indent.addWidget(self.chk_make_seq)
            indent.addStretch(1)
            grid.addLayout(indent, 4, 0, 1, 3)

            self.lbl_count = QtWidgets.QLabel("No folder selected.")
            self.lbl_count.setWordWrap(True)
            grid.addWidget(self.lbl_count, 5, 0, 1, 3)
            outer.addWidget(box)

            # --- target ---
            box = QtWidgets.QGroupBox("Convert to")
            inner = QtWidgets.QVBoxLayout(box)
            self.target_group = QtWidgets.QButtonGroup(self)
            self.radios = {}
            for key in (TARGET_16, TARGET_32):
                radio = QtWidgets.QRadioButton(TARGET_LABELS[key])
                radio.setChecked(d.target == key)
                radio.toggled.connect(self._sync_target)
                self.target_group.addButton(radio)
                inner.addWidget(radio)
                self.radios[key] = radio

            self.lbl_warning = QtWidgets.QLabel("")
            self.lbl_warning.setWordWrap(True)
            inner.addWidget(self.lbl_warning)

            self.chk_skip = QtWidgets.QCheckBox(
                "Skip files that already have the target depth")
            self.chk_skip.setChecked(d.skip_matching)
            self.chk_skip.setToolTip(
                "The depth is read from the BITPIX header of each file.")
            inner.addWidget(self.chk_skip)

            self.chk_rescale = QtWidgets.QCheckBox(
                "Rescale the values between the two conventions")
            self.chk_rescale.setChecked(d.rescale)
            self.chk_rescale.setToolTip(
                "Siril stores 16-bit data as 0 - 65535 and 32-bit data as "
                "0.0 - 1.0. With this off the raw numbers are cast, which "
                "changes how the image looks.")
            inner.addWidget(self.chk_rescale)
            outer.addWidget(box)

            # --- output ---
            box = QtWidgets.QGroupBox("Output")
            grid = QtWidgets.QGridLayout(box)
            self.chk_overwrite = QtWidgets.QCheckBox(
                "Overwrite the original files")
            self.chk_overwrite.setChecked(d.overwrite)
            self.chk_overwrite.setToolTip(
                "The originals are replaced. There is no undo.")
            self.chk_overwrite.toggled.connect(self._sync_output)
            grid.addWidget(self.chk_overwrite, 0, 0, 1, 3)
            self.lbl_out = QtWidgets.QLabel("Save to:")
            grid.addWidget(self.lbl_out, 1, 0)
            self.ed_output = QtWidgets.QLineEdit(d.output or "")
            grid.addWidget(self.ed_output, 1, 1)
            self.btn_out = QtWidgets.QPushButton("...")
            self.btn_out.setFixedWidth(32)
            self.btn_out.clicked.connect(lambda: self._browse(self.ed_output))
            grid.addWidget(self.btn_out, 1, 2)
            grid.setColumnStretch(1, 1)
            outer.addWidget(box)

            outer.addWidget(self._log_box(), 1)

            buttons = QtWidgets.QHBoxLayout()
            buttons.addStretch(1)
            btn = QtWidgets.QPushButton("Close")
            btn.clicked.connect(self.close)
            buttons.addWidget(btn)
            self.btn_run = QtWidgets.QPushButton("Convert")
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
            start = edit.text().strip() or self.ed_folder.text().strip() \
                or os.getcwd()
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select a directory", start)
            if chosen:
                edit.setText(os.path.normpath(chosen))
                if edit is self.ed_folder:
                    self._on_folder_change()

        def _folder(self):
            text = self.ed_folder.text().strip()
            folder = Path(text) if text else None
            return folder if folder is not None and folder.is_dir() else None

        def _on_folder_change(self) -> None:
            self._refresh_sequences()
            self._refresh_count()

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
            self._refresh_count()

        def _sync_source(self) -> None:
            sequence = self.rb_sequence.isChecked()
            self.chk_recursive.setEnabled(not sequence)
            self.cmb_seq.setEnabled(sequence)
            self.chk_make_seq.setEnabled(sequence)
            self._refresh_count()

        def _refresh_count(self) -> None:
            folder = self._folder()
            if folder is None:
                self.lbl_count.setText("No folder selected.")
                return
            if self.rb_sequence.isChecked():
                name = clean_seq_name(self.cmb_seq.currentText())
                if not name:
                    self.lbl_count.setText("No sequence selected.")
                    return
                frames = sequence_frames(folder, name)
                if not frames:
                    if (folder / (name + ".seq")).is_file():
                        self.lbl_count.setText(
                            "'" + name + "' has a .seq but no separate frames, "
                            "so it is a SER or FITSEQ sequence - there is no "
                            "per-frame file to convert.")
                    else:
                        self.lbl_count.setText(
                            "No frame of '" + name + "' found here.")
                    return
                self.lbl_count.setText("%d frame(s) in the sequence."
                                       % len(frames))
                return
            self.lbl_count.setText(
                "%d FITS file(s) found."
                % len(list_fits(folder, self.chk_recursive.isChecked())))

        def _target(self) -> str:
            return TARGET_16 if self.radios[TARGET_16].isChecked() else TARGET_32

        def _sync_target(self) -> None:
            if self._target() == TARGET_16:
                self.lbl_warning.setText(
                    "Lossy: float values are rounded to whole steps and "
                    "anything outside the range is clipped. Clipped pixels are "
                    "counted and reported per file.")
            else:
                self.lbl_warning.setText(
                    "Lossless, but it does not bring back precision that an "
                    "earlier 16-bit conversion already threw away. The files "
                    "become twice as large.")

        def _sync_output(self) -> None:
            on = not self.chk_overwrite.isChecked()
            self.ed_output.setEnabled(on)
            self.btn_out.setEnabled(on)
            self.lbl_out.setEnabled(on)

        # -- starting the work ----------------------------------------------

        def _collect(self):
            """Build the same settings object argparse produces, from the form."""
            folder = self._folder()
            if folder is None:
                self._error("The source folder does not exist.")
                return None

            sequence = None
            if self.rb_sequence.isChecked():
                sequence = clean_seq_name(self.cmb_seq.currentText())
                if not sequence:
                    self._error("Choose a sequence.")
                    return None
                files = sequence_frames(folder, sequence)
                if not files:
                    self._error(
                        "No frame of the sequence '" + sequence + "' was "
                        "found.\n\nSER and FITSEQ sequences keep every frame in "
                        "one container and cannot be converted frame by frame.")
                    return None
            else:
                files = list_fits(folder, self.chk_recursive.isChecked())
                if not files:
                    self._error("No FITS file was found there.")
                    return None

            overwrite = self.chk_overwrite.isChecked()
            output = self.ed_output.text().strip()
            if not overwrite:
                if not output:
                    self._error(
                        "Choose an output folder, or enable overwriting.")
                    return None
                if Path(output).resolve() == folder.resolve():
                    self._error("The output folder is the same as the source "
                                "folder.")
                    return None

            args = parse_args([])
            args.folder = str(folder)
            args.sequence = sequence
            args.recursive = self.chk_recursive.isChecked()
            args.make_seq = self.chk_make_seq.isChecked()
            args.target = self._target()
            args.skip_matching = self.chk_skip.isChecked()
            args.rescale = self.chk_rescale.isChecked()
            args.overwrite = overwrite
            args.output = output or None

            label = TARGETS[args.target][2]
            question = ("Convert %d %s to %s?\n\nOutput: %s"
                        % (len(files),
                           "frame(s) of '" + sequence + "'" if sequence
                           else "file(s)", label,
                           "the originals will be overwritten" if overwrite
                           else output))
            if args.target == TARGET_16:
                question += ("\n\nThis is lossy - values outside the range are "
                             "clipped and cannot be recovered.")
            answer = QtWidgets.QMessageBox.question(self, "Convert", question)
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
            self.lbl_status.setText("Converting...")

            thread = threading.Thread(target=self._worker, args=(args,),
                                      daemon=True)
            thread.start()

        def _worker(self, args) -> None:
            """Runs off the GUI thread; every Siril call is issued from here."""
            try:
                self.run_finished.emit(
                    run_pipeline(self.siril, args, self.cancel), None)
            except ConvertError as exc:
                self.run_finished.emit(None, str(exc))
            except Exception as exc:
                self.run_finished.emit(
                    None, exc.__class__.__name__ + ": " + str(exc))

        def _on_progress(self, done, total) -> None:
            self.progress.setValue(int(100.0 * done / max(total, 1)))
            self.lbl_status.setText("Converting... %d/%d" % (done, total))

        def _finish(self, result, error) -> None:
            self.running = False
            self.btn_run.setEnabled(True)
            if error:
                self.lbl_status.setText("Conversion failed.")
                self.text.appendPlainText("ERROR: " + error)
                QtWidgets.QMessageBox.critical(self, "Conversion failed", error)
                return

            converted, skipped, failed = result
            message = ("%d converted, %d skipped, %d failed."
                       % (converted, skipped, failed))
            self.lbl_status.setText(message)
            self.progress.setValue(100)
            self._refresh_count()
            if failed:
                QtWidgets.QMessageBox.warning(
                    self, "Finished", message + "\nSee the log for details.")
            else:
                QtWidgets.QMessageBox.information(self, "Finished", message)

        def closeEvent(self, event) -> None:
            if self.running:
                answer = QtWidgets.QMessageBox.question(
                    self, "Conversion is running",
                    "A conversion is still running. Close the window anyway?")
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
        raise ConvertError(
            "PyQt6 is not available in this Python environment.")

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])
    apply_siril_theme(app, siril)

    window = BitDepthWindow(siril, defaults)
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
        description="Convert FITS files between 16-bit unsigned integer and "
                    "32-bit float (Siril 1.4+).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version",
                   version="%(prog)s " + __version__)
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
