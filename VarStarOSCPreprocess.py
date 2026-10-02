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
OSC preprocessing up to registration, without stacking (Siril 1.4+).

Runs the classic Siril one-shot colour path - master bias, master flat, master
dark, calibration of the lights - keeps the calibrated frames, registers them,
and stops. The lights are NOT stacked: variable star photometry needs one
measurement per exposure, and stacking would throw that time resolution away.

Input
-----
  <work_dir>/lights/   light frames               (required)
  <work_dir>/biases/   bias / offset frames       (optional)
  <work_dir>/flats/    flat frames                (optional)
  <work_dir>/darks/    dark frames                (optional)

RAW (CR2/CR3/NEF/ARW/...) and FITS are both accepted - Siril's "convert" takes
whatever it can read.

Output
------
  <work_dir>/process/bias_stacked        master bias
  <work_dir>/process/pp_flat_stacked     master flat (bias-calibrated)
  <work_dir>/process/dark_stacked        master dark
  <work_dir>/process/pp_light_           calibrated lights, kept
  <work_dir>/process/r_pp_light_         calibrated + registered lights
  <work_dir>/process/G_r_pp_light_       one extracted channel (optional)

Dark optimization
-----------------
--dark-opt auto | exp scales the master dark before subtracting it, for lights
whose exposure does not match the dark's. Siril requires BOTH a master bias and
a master dark for this: it has to take the bias pedestal off the dark before the
remaining dark current can be scaled, so the bias master is passed to the lights
as well, which it is not otherwise. "auto" fits the factor from the data, "exp"
derives it from the exposure keyword.

Ready-made masters
------------------
--master-bias / --master-flat / --master-dark (or the Masters tab) take a file
you have already built and use it as it is, skipping both the conversion and the
stacking of that calibration type. The matching input directory is then ignored.
Anything left blank is built the usual way, so the three can be mixed freely.

Workflow
--------
  1. biases -> convert -> stack bias rej w 3 3 -nonorm -out=bias_stacked
                       (skipped when --master-bias is given)
  2. flats  -> convert -> calibrate flat -bias=bias_stacked
                       -> stack pp_flat rej w 3 3 -norm=mul -out=pp_flat_stacked
  3. darks  -> convert -> stack dark rej w 3 3 -nonorm -out=dark_stacked
  4. lights -> convert -> calibrate light -dark=... -flat=... -cc=dark -cfa
                                          -equalize_cfa -debayer [-opt] -prefix=pp_
  5. register pp_light -interp=none
  6. optionally extract one channel (R, G or B) into its own sequence
  7. stop - no stacking

Debayering
----------
The lights are debayered during calibration because Siril refuses to register a
sequence whose Bayer pattern is still intact ("you must debayer it prior to
registration"). That is why -debayer is required here, unlike a workflow that
ends in seqextract_Green.

Channel extraction
------------------
The optional last step splits one colour channel off the registered frames into
a new sequence, with a .seq of its own. On a debayered source that is a plain
channel pick at full resolution; on a CFA source green goes through
seqextract_Green and red/blue take their quarter of the Bayer cell, which halves
both dimensions. Only the requested channel is ever written.

Interpolation
-------------
Registration defaults to -interp=none, which forces a whole-pixel shift and
applies no interpolation at all. Photometry measures photon counts, and every
interpolating method redistributes them between neighbouring pixels; a plain
shift leaves the values untouched. Pick another method only if the frames really
need rotation or scaling, and accept the cost.

Toolkit
-------
The dialog is built with PyQt6, the Qt binding that ships inside Siril's own
Python environment (Siril 1.4 bundles PyQt6 6.11 / Qt 6.11), so it needs nothing
installed on top. It follows Siril's own light/dark preference. With --no-gui the
script never touches Qt at all.

Usage
-----
  A) From the Siril GUI: copy this script into the scripts directory
     (Windows: %LOCALAPPDATA%\\siril\\scripts, Linux/macOS: ~/.siril/scripts)
     and run it from the Scripts menu. A settings dialog opens.

  B) From a command line, with Siril running and its Python environment active:
     python VarStarOSCPreprocess.py --work-dir "D:/astro/RR_Lyr" --no-gui

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
# (Siril 1.4 bundles PyQt6 6.11 / Qt 6.11). Import it lazily enough that
# --no-gui still works on an installation without it.
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError:
    try:
        s.ensure_installed("PyQt6")
        from PyQt6 import QtCore, QtGui, QtWidgets
    except Exception:
        QtCore = QtGui = QtWidgets = None


# Directory names looked for inside the working directory.
BIAS_DIR_CANDIDATES = ("biases", "bias", "offsets", "offset")
FLAT_DIR_CANDIDATES = ("flats", "flat")
DARK_DIR_CANDIDATES = ("darks", "dark")
LIGHT_DIR_CANDIDATES = ("lights", "light")

# Sequence and master names, kept identical to the stock Siril OSC scripts so
# the process directory looks familiar.
BIAS_SEQ = "bias"
FLAT_SEQ = "flat"
DARK_SEQ = "dark"
LIGHT_SEQ = "light"
MASTER_BIAS = "bias_stacked"
MASTER_FLAT = "pp_flat_stacked"
MASTER_DARK = "dark_stacked"
CALIBRATED_PREFIX = "pp_"
REGISTERED_PREFIX = "r_"

# label -> rejection code for the stack command
REJECTIONS = (
    ("Winsorized sigma clipping", "w"),
    ("Sigma clipping", "s"),
    ("Median sigma clipping", "m"),
    ("Linear fit clipping", "l"),
    ("Percentile clipping", "p"),
    ("Generalized Extreme Studentized Deviate", "g"),
    ("MAD clipping", "a"),
    ("No rejection (mean)", "n"),
)

# label -> -interp= value. "none" forces a whole-pixel shift, which is the only
# option that leaves the measured counts untouched.
INTERPOLATIONS = (
    ("None - whole-pixel shift, no interpolation", "none"),
    ("Nearest neighbour", "nearest"),
    ("Bilinear", "linear"),
    ("Bicubic", "cubic"),
    ("Lanczos-4", "lanczos4"),
    ("Area", "area"),
)

# label -> -transf= value (ignored when the interpolation is "none", which
# forces a shift)
TRANSFORMATIONS = (
    ("Shift", "shift"),
    ("Similarity", "similarity"),
    ("Affine", "affine"),
    ("Homography", "homography"),
)

# Siril's log colours, in a light and a dark variant so the embedded log stays
# readable whichever theme Siril is set to.
LOG_COLOURS = {
    "light": {"green": "#1b6e2b", "salmon": "#b34a20", "blue": "#14539a",
              "red": "#b3261e"},
    "dark": {"green": "#7fd18c", "salmon": "#ffb08f", "blue": "#7cb6f2",
             "red": "#ff9b94"},
}

# label -> -opt value for calibrate. Siril's own help: "-opt ... requires the
# supply of bias and dark masters, and automatically calculates the coefficient
# to be applied to dark, or calculates the coefficient thanks to the exposure
# keyword with -opt=exp".
DARK_OPTIMIZATIONS = (
    ("None - subtract the dark as it is", "none"),
    ("Automatic - fit the scaling factor", "auto"),
    ("From the exposure keyword", "exp"),
)

CHANNELS = ("R", "G", "B")
CHANNEL_LABELS = {"R": "Red", "G": "Green", "B": "Blue"}

# Bayer pattern -> which quarter of the 2x2 cell carries which colour, used only
# when the lights were left undebayered. split_cfa numbers the cell in reading
# order: 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right.
BAYER_PLANES = {
    "RGGB": {"R": 0, "G": 1, "B": 3},
    "BGGR": {"R": 3, "G": 1, "B": 0},
    "GRBG": {"R": 1, "G": 0, "B": 2},
    "GBRG": {"R": 2, "G": 0, "B": 1},
}

# Anything Siril's "convert" is likely to be handed.
INPUT_EXTENSIONS = {
    "fit", "fits", "fts", "fz", "xisf", "tif", "tiff", "png", "jpg", "jpeg",
    "bmp", "ppm", "pgm", "pnm", "ser", "avi",
    "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "orf", "raf",
    "rw2", "pef", "ptx", "dng", "raw", "x3f", "mrw", "kdc", "dcr", "mef",
    "mos", "erf", "iiq", "3fr", "bay", "cap", "dcs", "drf", "fff", "rwl",
    "rwz", "srw",
}


class PreprocessError(RuntimeError):
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
    """Run a Siril command, raising PreprocessError with context on failure."""
    argv = [str(a) for a in args]
    printable = " ".join(argv)
    if not quiet:
        log(siril, "  > " + printable)
    try:
        siril.cmd(*argv)
    except Exception as exc:
        raise PreprocessError(
            "Command failed: " + printable + "\n    -> " + str(exc)
        ) from exc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt_num(value: float) -> str:
    """3.0 -> "3" (Siril expects plain numbers, integers without a decimal)."""
    return str(int(value)) if float(value).is_integer() else str(value)


def quote(text: str) -> str:
    """Quote for Siril's parser if the path contains a space."""
    return '"' + text + '"' if " " in text else text


def siril_path(path: Path) -> str:
    """A path in the form Siril's parser understands (forward slashes, quotes)."""
    return quote(Path(path).as_posix())


def rel_posix(src: Path, dst: Path) -> str:
    """Relative path from src to dst; absolute when they are on different drives."""
    try:
        return Path(os.path.relpath(dst, src)).as_posix()
    except ValueError:
        return Path(dst).as_posix()


def resolve_dir(base: Path, explicit, candidates: tuple, label: str,
                required: bool):
    """Find an input directory - the given one, or the first known name."""
    if explicit:
        path = Path(explicit)
        path = path if path.is_absolute() else base / path
        if not path.is_dir():
            raise PreprocessError(
                "The " + label + " directory does not exist: " + str(path))
        return path.resolve()

    for name in candidates:
        path = base / name
        if path.is_dir():
            return path.resolve()

    wanted = {c.lower() for c in candidates}
    if base.is_dir():
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and entry.name.lower() in wanted:
                return entry.resolve()

    if required:
        raise PreprocessError(
            "No " + label + " subdirectory found in " + str(base)
            + " (looked for: " + ", ".join(candidates[:2])
            + "). Set the path manually.")
    return None


def resolve_master(base: Path, explicit, label: str):
    """A master file the user supplied, ready to hand to calibrate.

    Returns the name Siril should use: a bare stem when the file already sits
    in the process directory, otherwise an absolute path. None when the user
    gave nothing, which is what makes the script fall back to building it.
    """
    if not explicit:
        return None
    path = Path(explicit)
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        # a name without the extension is a perfectly normal way to give it
        found = sorted(path.parent.glob(path.name + ".fit*"))
        if not found:
            raise PreprocessError(
                "The master " + label + " does not exist: " + str(path))
        path = found[0]
    return path.resolve()


def master_argument(master: Path, process_dir: Path) -> str:
    """How a master is written into the calibrate command."""
    if master.parent == process_dir:
        return master.stem          # Siril finds it in the working directory
    return siril_path(master)


def count_inputs(directory) -> int:
    """How many files in the directory Siril could plausibly convert."""
    if directory is None:
        return 0
    try:
        return sum(1 for p in directory.iterdir()
                   if p.is_file()
                   and p.suffix.lower().lstrip(".") in INPUT_EXTENSIONS)
    except OSError:
        return 0


def normalise_pattern(pattern: str) -> str:
    return "".join(c for c in (pattern or "") if c.isalpha()).upper()


def count_frames(process_dir: Path, sequence: str, ext: str) -> int:
    return len(sorted(process_dir.glob(sequence + "_*" + ext)))


# ---------------------------------------------------------------------------
# individual steps
# ---------------------------------------------------------------------------

def convert_frames(siril, src_dir: Path, basename: str,
                   process_dir: Path) -> None:
    """Convert a directory of frames into the FITS sequence <basename>_.

    Without -debayer: the calibration frames must keep their CFA mosaic so the
    masters subtract pixel for pixel, and the lights are debayered later, in the
    calibrate step.
    """
    run(siril, "cd", siril_path(src_dir))
    run(siril, "convert", basename,
        "-out=" + quote(rel_posix(src_dir, process_dir)), "-start=1")


def stack_master(siril, process_dir: Path, sequence: str, out_name: str,
                 rejection: str, sigma_low: float, sigma_high: float,
                 normalise, count: int, ext: str) -> str:
    """Stack a calibration master. Returns the master's name."""
    run(siril, "cd", siril_path(process_dir))

    if count == 1:
        # A single frame is already its own master; stacking one frame fails.
        log(siril, "  only one frame - using it directly, without stacking.",
            "salmon")
        found = sorted(process_dir.glob(sequence + "_*" + ext))
        if not found:
            found = sorted(process_dir.glob(sequence + "*" + ext))
        if not found:
            raise PreprocessError(
                "Converting " + sequence + " produced no FITS file.")
        run(siril, "load", quote(found[0].stem))
        run(siril, "save", quote(out_name))
    else:
        args = ["stack", sequence, "rej", rejection,
                fmt_num(sigma_low), fmt_num(sigma_high)]
        args.append("-norm=" + normalise if normalise else "-nonorm")
        args.append("-out=" + out_name)
        run(siril, *args)

    if not sorted(process_dir.glob(out_name + ".fit*")):
        raise PreprocessError("The master was not created: " + out_name)
    return out_name


def inspect_lights(siril, process_dir: Path, sequence: str, ext: str) -> tuple:
    """Look at the first converted light. Returns (channels, bayer pattern)."""
    frames = sorted(process_dir.glob(sequence + "_*" + ext))
    if not frames:
        frames = sorted(process_dir.glob(sequence + "*" + ext))
    if not frames:
        raise PreprocessError(
            "Converting the lights produced no FITS file in "
            + str(process_dir) + ".")

    try:
        ffit = siril.load_image_from_file(str(frames[0]), with_pixels=False)
    except Exception as exc:
        log(siril, "  could not inspect " + frames[0].name + " (" + str(exc)
            + ") - continuing with the settings as given.", "salmon")
        return 0, ""
    if ffit is None:
        return 0, ""

    try:
        channels = int(ffit.naxes[2])
    except Exception:
        return 0, ""
    pattern = ""
    keywords = getattr(ffit, "keywords", None)
    if keywords is not None:
        pattern = (getattr(keywords, "bayer_pattern", "") or "").strip()
    return channels, pattern


def frame_exposure(siril, path) -> float:
    """EXPTIME of a frame, or 0.0 when it cannot be read."""
    try:
        ffit = siril.load_image_from_file(str(path), with_pixels=False)
        return float(getattr(ffit.keywords, "exposure", 0.0) or 0.0)
    except Exception:
        return 0.0


def report_exposures(siril, light: float, dark: float) -> None:
    """Say how the two exposures compare - the reason to optimize at all."""
    if not light or not dark:
        log(siril, "      exposures could not be compared (no EXPTIME).",
            "salmon")
        return
    log(siril, "      light %gs vs master dark %gs (ratio %.3f)"
        % (light, dark, light / dark))
    if abs(light - dark) < 0.01:
        log(siril, "      they match, so optimization has little to do.")
    elif light < dark:
        log(siril, "      the lights are shorter, so the dark is scaled down.")
    else:
        log(siril, "      the lights are LONGER than the dark; scaling a dark "
                   "up extrapolates its noise, which is worse than matching "
                   "the exposures.", "salmon")


def calibrate_lights(siril, process_dir: Path, sequence: str,
                     master_dark, master_flat, master_bias, cosmetic: bool,
                     cc_low: float, cc_high: float, cfa: bool,
                     equalize_cfa: bool, debayer: bool, dark_opt: str,
                     prefix: str) -> None:
    """Calibrate the lights and, when asked, debayer them in the same pass."""
    run(siril, "cd", siril_path(process_dir))

    optimising = dark_opt and dark_opt != "none"

    args = ["calibrate", sequence]
    if master_bias and (optimising or not master_dark):
        # Normally the dark already contains the bias, so subtracting it as
        # well would be wrong. Dark optimization is the exception: to scale the
        # dark current Siril first has to take the bias pedestal off the dark,
        # so it needs the bias master too.
        args.append("-bias=" + master_bias)
    if master_dark:
        args.append("-dark=" + master_dark)
    if master_flat:
        args.append("-flat=" + master_flat)
    if cosmetic and master_dark:
        args += ["-cc=dark", fmt_num(cc_low), fmt_num(cc_high)]
    if cfa:
        args.append("-cfa")
    if equalize_cfa and master_flat:
        args.append("-equalize_cfa")
    if debayer:
        args.append("-debayer")
    if optimising:
        args.append("-opt" if dark_opt == "auto" else "-opt=" + dark_opt)
    args.append("-prefix=" + prefix)

    run(siril, *args)


def register_lights(siril, process_dir: Path, sequence: str, interp: str,
                    transf: str, layer, prefix: str, ext: str) -> tuple:
    """Register the calibrated lights. Returns (sequence name, frame count)."""
    run(siril, "cd", siril_path(process_dir))

    args = ["register", sequence, "-interp=" + interp]
    if interp != "none":
        # "none" already forces a shift, so -transf= would only conflict.
        args.append("-transf=" + transf)
    if layer is not None:
        args.append("-layer=" + str(layer))
    if prefix != REGISTERED_PREFIX:
        args.append("-prefix=" + prefix)

    run(siril, *args)

    out_root = prefix + sequence
    count = count_frames(process_dir, out_root, ext)
    if not count:
        raise PreprocessError(
            "Registration produced no frames named " + out_root + "_*" + ext
            + ". Check the log - too few stars is the usual reason.")
    return out_root, count


def sequence_frame_paths(process_dir: Path, sequence: str, ext: str) -> list:
    """The frames of a sequence, in frame-number order."""
    found = []
    for path in process_dir.glob(sequence + "_*" + ext):
        digits = path.stem[len(sequence) + 1:]
        if digits.isdigit():
            found.append((int(digits), path))
    return [path for _n, path in sorted(found, key=lambda item: item[0])]


def frame_suffix(path: Path, sequence: str) -> str:
    """'r_pp_light_00007.fit' -> '_00007' (the digits are kept as written)."""
    return path.stem[len(sequence):]


def strip_bayer_cards(header: str) -> str:
    """Drop the CFA keywords - an extracted channel is no longer mosaiced."""
    if not header:
        return ""
    dropped = ("BAYERPAT", "XBAYROFF", "YBAYROFF")
    cards = [header[i:i + 80] for i in range(0, len(header), 80)]
    return "".join(card for card in cards
                   if card[:8].strip().upper() not in dropped)


def as_2d(data):
    """Siril hands over planar data; a one-channel image may arrive as (1,h,w)."""
    if data is None:
        raise PreprocessError("the frame carries no pixel data")
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    return data


def save_plane(siril, plane, header: str, path: Path) -> None:
    """Write a single 2-D plane as its own FITS file.

    save_image_file() writes straight to disk without touching the image loaded
    in Siril, which is what lets us produce one channel and nothing else.
    """
    array = np.ascontiguousarray(plane)
    if array.dtype not in (np.float32, np.uint16):
        array = array.astype(np.float32)
    if not siril.save_image_file(array, header, str(path)):
        raise PreprocessError("Siril refused to save " + path.name)


def register_sequence(siril, root: str, folder: Path) -> None:
    """Write the .seq for frames we produced ourselves."""
    try:
        if siril.create_new_seq(root):
            log(siril, "      sequence file written: " + root + ".seq", "green")
        else:
            log(siril, "      Siril did not write the .seq for " + root
                + " - use 'Search sequence' in the Sequence tab.", "salmon")
    except Exception as exc:
        log(siril, "      could not write the .seq for " + root + ": "
            + str(exc), "salmon")


def extract_channel(siril, process_dir: Path, sequence: str, channel: str,
                    prefix: str, ext: str, make_seq: bool) -> tuple:
    """Split one colour channel off a sequence into a new one.

    Works on both shapes this pipeline can produce:

      three channels (the normal case, after -debayer)  the channel is taken
          straight out of the planar array, at full resolution;
      one channel with a Bayer pattern (--no-debayer)   green goes through
          seqextract_Green, which averages both green pixels; red and blue are
          the matching quarter of the 2x2 cell. Half width and half height.

    Only the requested channel is ever written - Siril's own "split" and
    "split_cfa" always write every channel, so they are not used.
    """
    run(siril, "cd", siril_path(process_dir))

    frames = sequence_frame_paths(process_dir, sequence, ext)
    if not frames:
        raise PreprocessError(
            "No frames of " + sequence + " were found in " + str(process_dir)
            + " to extract from.")

    # what does the source look like?
    try:
        first = siril.load_image_from_file(str(frames[0]), with_pixels=False)
    except Exception as exc:
        raise PreprocessError(
            "Could not inspect " + frames[0].name + ": " + str(exc))
    channels = int(first.naxes[2]) if first is not None else 0
    pattern = ""
    keywords = getattr(first, "keywords", None) if first is not None else None
    if keywords is not None:
        pattern = normalise_pattern(getattr(keywords, "bayer_pattern", ""))

    out_root = prefix + sequence

    # --- undebayered CFA: green has its own Siril command -------------------
    if channels < 3 and pattern:
        if channel == "G":
            log(siril, "      CFA source: seqextract_Green averages both green "
                       "pixels of every cell.")
            run(siril, "seqextract_Green", sequence, "-prefix=" + prefix)
            count = count_frames(process_dir, out_root, ext)
            if not count:
                raise PreprocessError(
                    "seqextract_Green produced no frames named " + out_root
                    + "_*" + ext + ".")
            log(siril, "      result is half width and half height.", "salmon")
            return out_root, count

        planes = BAYER_PLANES.get(pattern)
        if not planes:
            raise PreprocessError(
                "The Bayer pattern '" + (pattern or "?") + "' is not one of "
                "RGGB / BGGR / GRBG / GBRG, so the plane carrying "
                + CHANNEL_LABELS[channel] + " is unknown.")
        plane = planes[channel]
        log(siril, "      CFA source, BAYERPAT %s: %s is Bayer plane %d."
            % (pattern, CHANNEL_LABELS[channel].lower(), plane))
        log(siril, "      result is half width and half height.", "salmon")
    elif channels >= 3:
        plane = None
        log(siril, "      %d-channel source: the %s channel is taken at full "
                   "resolution." % (channels, CHANNEL_LABELS[channel].lower()))
    else:
        raise PreprocessError(
            "The frames of " + sequence + " have one channel and no BAYERPAT, "
            "so they are monochrome - there is no colour channel to extract.")

    written = failed = 0
    for index, frame in enumerate(frames, start=1):
        target = process_dir / (out_root + frame_suffix(frame, sequence) + ext)
        try:
            ffit = siril.load_image_from_file(str(frame), with_pixels=True)
            if ffit is None:
                raise PreprocessError("the frame could not be read")
            header = ffit.header or ""
            if plane is None:
                data = ffit.data
                if data is None or data.ndim != 3:
                    raise PreprocessError("the frame is not a 3-channel image")
                pixels = data[CHANNELS.index(channel)]
            else:
                pixels = as_2d(ffit.data)[plane // 2::2, plane % 2::2]
                header = strip_bayer_cards(header)
            save_plane(siril, pixels, header, target)
            written += 1
        except Exception as exc:
            failed += 1
            log(siril, "      %s: FAILED - %s" % (frame.name, exc), "salmon")
        emit_progress(index, len(frames))

    if not written:
        raise PreprocessError(
            "No frame could be processed - nothing was extracted.")
    if failed:
        log(siril, "      %d frame(s) failed and were skipped." % failed,
            "salmon")
    if make_seq:
        # create_new_seq matches <root><5 digits><ext>, and the frames are
        # named <out_root>_NNNNN, so the root it needs carries the underscore
        register_sequence(siril, out_root + "_", process_dir)
    return out_root, written


# ---------------------------------------------------------------------------
# the whole processing run
# ---------------------------------------------------------------------------

def run_pipeline(siril, args) -> dict:
    """Run the complete preprocessing. Raises PreprocessError on failure."""
    original_wd = None
    try:
        original_wd = siril.get_siril_wd()
    except Exception:
        original_wd = None

    try:
        run(siril, "requires", "1.4.0")

        base = Path(args.work_dir).resolve() if args.work_dir \
            else Path(original_wd or ".").resolve()
        if not base.is_dir():
            raise PreprocessError(
                "Working directory does not exist: " + str(base))

        dark_opt = getattr(args, "dark_opt", "none") or "none"

        if args.register and not args.debayer:
            raise PreprocessError(
                "Siril cannot register a sequence whose Bayer pattern is still "
                "intact. Either allow debayering, or turn registration off.")

        lights_dir = resolve_dir(base, args.lights, LIGHT_DIR_CANDIDATES,
                                 "lights", required=True)

        # A master given on the form or the command line is used as it is; only
        # the ones left blank are built from their directory.
        given_bias = resolve_master(base, args.master_bias, "bias")
        given_flat = resolve_master(base, args.master_flat, "flat")
        given_dark = resolve_master(base, args.master_dark, "dark")

        bias_dir = None if given_bias else resolve_dir(
            base, args.biases, BIAS_DIR_CANDIDATES, "bias", required=False)
        flat_dir = None if given_flat else resolve_dir(
            base, args.flats, FLAT_DIR_CANDIDATES, "flats", required=False)
        dark_dir = None if given_dark else resolve_dir(
            base, args.darks, DARK_DIR_CANDIDATES, "darks", required=False)

        n_lights = count_inputs(lights_dir)
        n_bias = count_inputs(bias_dir)
        n_flats = count_inputs(flat_dir)
        n_darks = count_inputs(dark_dir)
        if not n_lights:
            raise PreprocessError("No frames found in " + str(lights_dir) + ".")

        process_dir = Path(args.process)
        process_dir = process_dir if process_dir.is_absolute() \
            else base / args.process
        process_dir.mkdir(parents=True, exist_ok=True)
        process_dir = process_dir.resolve()

        ext = ".fit"

        log(siril, "=" * 66, "green")
        log(siril, "OSC preprocessing up to registration - no stacking", "green")
        log(siril, "=" * 66, "green")
        log(siril, "  working directory : " + str(base))
        log(siril, "  lights            : " + str(lights_dir)
            + "  (" + str(n_lights) + ")")
        for label, directory, count, given in (
                ("bias", bias_dir, n_bias, given_bias),
                ("flats", flat_dir, n_flats, given_flat),
                ("darks", dark_dir, n_darks, given_dark)):
            if given is not None:
                log(siril, "  %-18s: master given, not rebuilt - %s"
                    % (label, given), "green")
            elif directory is None or not count:
                log(siril, "  %-18s: none found - that master is skipped"
                    % label, "salmon")
            else:
                log(siril, "  %-18s: %s  (%d)" % (label, directory, count))
        log(siril, "  intermediates     : " + str(process_dir))

        run(siril, "setext", "fit")
        run(siril, "set32bits" if args.use32bits else "set16bits")

        total_steps = (4 + (1 if args.register else 0)
                       + (1 if args.extract else 0))
        step = 0
        master_bias = master_flat = master_dark = None

        # --- 1) master bias ------------------------------------------------
        step += 1
        if given_bias is not None:
            master_bias = master_argument(given_bias, process_dir)
            log(siril, "[%d/%d] Master bias: using the one given."
                % (step, total_steps), "blue")
        elif bias_dir is not None and n_bias:
            log(siril, "[%d/%d] Master bias..." % (step, total_steps), "blue")
            convert_frames(siril, bias_dir, BIAS_SEQ, process_dir)
            master_bias = stack_master(
                siril, process_dir, BIAS_SEQ, MASTER_BIAS, args.rejection,
                args.sigma_low, args.sigma_high, None, n_bias, ext)
            log(siril, "      " + master_bias, "green")
        else:
            log(siril, "[%d/%d] Master bias skipped." % (step, total_steps),
                "salmon")
        emit_progress(step, total_steps)

        # --- 2) master flat ------------------------------------------------
        step += 1
        if given_flat is not None:
            master_flat = master_argument(given_flat, process_dir)
            log(siril, "[%d/%d] Master flat: using the one given."
                % (step, total_steps), "blue")
        elif flat_dir is not None and n_flats:
            log(siril, "[%d/%d] Master flat..." % (step, total_steps), "blue")
            convert_frames(siril, flat_dir, FLAT_SEQ, process_dir)
            flat_sequence = FLAT_SEQ
            if master_bias:
                run(siril, "cd", siril_path(process_dir))
                run(siril, "calibrate", FLAT_SEQ, "-bias=" + master_bias,
                    "-prefix=" + CALIBRATED_PREFIX)
                flat_sequence = CALIBRATED_PREFIX + FLAT_SEQ
            else:
                log(siril, "  no master bias - the flats are stacked raw.",
                    "salmon")
            master_flat = stack_master(
                siril, process_dir, flat_sequence, MASTER_FLAT, args.rejection,
                args.sigma_low, args.sigma_high, "mul", n_flats, ext)
            log(siril, "      " + master_flat, "green")
        else:
            log(siril, "[%d/%d] Master flat skipped." % (step, total_steps),
                "salmon")
        emit_progress(step, total_steps)

        # --- 3) master dark ------------------------------------------------
        step += 1
        if given_dark is not None:
            master_dark = master_argument(given_dark, process_dir)
            log(siril, "[%d/%d] Master dark: using the one given."
                % (step, total_steps), "blue")
        elif dark_dir is not None and n_darks:
            log(siril, "[%d/%d] Master dark..." % (step, total_steps), "blue")
            convert_frames(siril, dark_dir, DARK_SEQ, process_dir)
            master_dark = stack_master(
                siril, process_dir, DARK_SEQ, MASTER_DARK, args.rejection,
                args.sigma_low, args.sigma_high, None, n_darks, ext)
            log(siril, "      " + master_dark, "green")
        else:
            log(siril, "[%d/%d] Master dark skipped." % (step, total_steps),
                "salmon")

        dark_exposure = 0.0
        if dark_opt != "none" and master_dark:
            source = (given_dark if given_dark is not None
                      else process_dir / (MASTER_DARK + ext))
            dark_exposure = frame_exposure(siril, source)
        emit_progress(step, total_steps)

        # Siril needs both masters to separate the bias pedestal from the
        # dark current before it can scale the latter.
        if dark_opt != "none" and not (master_dark and master_bias):
            missing = " and ".join(
                name for name, value in (("dark", master_dark),
                                         ("bias", master_bias)) if not value)
            raise PreprocessError(
                "Dark optimization needs both a master bias and a master dark, "
                "but the master " + missing + " is missing. Supply it, or turn "
                "the optimization off.")

        # --- 4) calibrate the lights ---------------------------------------
        step += 1
        log(siril, "[%d/%d] Converting and calibrating the lights..."
            % (step, total_steps), "blue")
        convert_frames(siril, lights_dir, LIGHT_SEQ, process_dir)

        channels, pattern = inspect_lights(siril, process_dir, LIGHT_SEQ, ext)
        cfa = args.cfa
        debayer = args.debayer
        if channels >= 3:
            log(siril, "  the converted lights already have %d channels, so "
                       "Siril debayered them on import - -debayer and -cfa are "
                       "dropped." % channels, "salmon")
            debayer = False
            cfa = False
        elif pattern:
            log(siril, "  lights are CFA, Bayer pattern " + pattern + ".",
                "green")
        elif channels:
            log(siril, "  the lights carry no BAYERPAT, so this is not "
                       "one-shot colour data - -debayer and -cfa are dropped. "
                       "Calibration and registration still apply.", "salmon")
            debayer = False
            cfa = False
            if args.register:
                log(siril, "  mono data registers without debayering, so this "
                           "is fine.")

        if not (master_bias or master_flat or master_dark):
            log(siril, "  no masters at all - the lights are only converted, "
                       "not calibrated.", "salmon")
            light_sequence = LIGHT_SEQ
            if debayer:
                log(siril, "  debayering still has to happen before "
                           "registration, so the lights go through calibrate "
                           "with no master.", "salmon")
                calibrate_lights(siril, process_dir, LIGHT_SEQ, None, None,
                                 None, False, 0, 0, cfa, False, debayer,
                                 CALIBRATED_PREFIX)
                light_sequence = CALIBRATED_PREFIX + LIGHT_SEQ
        else:
            if dark_opt != "none":
                log(siril, "  dark optimization: %s" % dark_opt, "blue")
                report_exposures(
                    siril,
                    frame_exposure(siril, sorted(process_dir.glob(
                        LIGHT_SEQ + "_*" + ext))[0]),
                    dark_exposure)
            calibrate_lights(
                siril, process_dir, LIGHT_SEQ, master_dark, master_flat,
                master_bias, args.cosmetic, args.cc_sigma_low,
                args.cc_sigma_high, cfa, args.equalize_cfa, debayer,
                dark_opt, CALIBRATED_PREFIX)
            light_sequence = CALIBRATED_PREFIX + LIGHT_SEQ

        n_calibrated = count_frames(process_dir, light_sequence, ext)
        log(siril, "      calibrated lights saved: %s_  (%d frames)"
            % (light_sequence, n_calibrated), "green")
        emit_progress(step, total_steps)

        # --- 5) registration -----------------------------------------------
        registered = None
        n_registered = 0
        if args.register:
            step += 1
            log(siril, "[%d/%d] Registering the calibrated lights..."
                % (step, total_steps), "blue")
            if args.interp == "none":
                log(siril, "  -interp=none: whole-pixel shift only, pixel "
                           "values are left untouched.")
            else:
                log(siril, "  -interp=" + args.interp + ": the pixels are "
                           "resampled, which redistributes the counts between "
                           "neighbours. For photometry prefer 'none'.",
                    "salmon")
            registered, n_registered = register_lights(
                siril, process_dir, light_sequence, args.interp, args.transf,
                args.layer, REGISTERED_PREFIX, ext)
            log(siril, "      registered: %s_  (%d of %d frames)"
                % (registered, n_registered, n_calibrated), "green")
            if n_registered < n_calibrated:
                log(siril, "  %d frame(s) were dropped by registration - too "
                           "few stars, most likely."
                    % (n_calibrated - n_registered), "salmon")
            emit_progress(step, total_steps)

        # --- 6) single channel ---------------------------------------------
        extracted = None
        n_extracted = 0
        if args.extract:
            step += 1
            channel = args.extract.upper()
            # take it from the registered frames when there are any, so the
            # extracted sequence is aligned like everything else
            source = registered or light_sequence
            prefix = args.extract_prefix or (channel + "_")
            log(siril, "[%d/%d] Extracting the %s channel from %s_..."
                % (step, total_steps, CHANNEL_LABELS[channel].lower(), source),
                "blue")
            extracted, n_extracted = extract_channel(
                siril, process_dir, source, channel, prefix, ext,
                args.make_seq)
            log(siril, "      extracted: %s_  (%d frames)"
                % (extracted, n_extracted), "green")
            emit_progress(step, total_steps)

        log(siril, "-" * 66, "green")
        log(siril, "DONE.", "green")
        log(siril, "  calibrated lights : " + light_sequence + "_  ("
            + str(n_calibrated) + " frames)", "green")
        if registered:
            log(siril, "  registered lights : " + registered + "_  ("
                + str(n_registered) + " frames)", "green")
        if extracted:
            log(siril, "  %s channel        : %s_  (%d frames)"
                % (args.extract.upper(), extracted, n_extracted), "green")
        log(siril, "  in                : " + str(process_dir), "green")
        log(siril, "  Not stacked, as intended: one frame per measurement.")
        log(siril, "-" * 66, "green")

        return {"calibrated": light_sequence, "n_calibrated": n_calibrated,
                "registered": registered, "n_registered": n_registered,
                "extracted": extracted, "n_extracted": n_extracted,
                "channel": args.extract.upper() if args.extract else None,
                "process": str(process_dir), "pattern": pattern,
                "bias": master_bias, "flat": master_flat, "dark": master_dark}

    finally:
        if original_wd:
            try:
                siril.cmd("cd", siril_path(Path(original_wd)))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GUI (PyQt6 - the Qt binding that ships in Siril's Python environment)
# ---------------------------------------------------------------------------

if QtWidgets is not None:

    class PreprocessWindow(QtWidgets.QWidget):
        """Settings dialog; the processing runs on its own thread.

        Everything the worker thread has to say travels as a Qt signal. Qt
        delivers those on the GUI thread, so the worker never touches a widget
        itself and no polling timer is needed.
        """

        log_line = QtCore.pyqtSignal(str, object)
        progress_changed = QtCore.pyqtSignal(int, int)
        run_finished = QtCore.pyqtSignal(object, object)

        def __init__(self, siril, defaults):
            super().__init__()
            self.siril = siril
            self.running = False
            self.log_theme = "dark" if siril_is_dark(siril) else "light"

            self.setWindowTitle(
                "OSC preprocessing up to registration v" + __version__)

            self._build_widgets(defaults)
            self._autofill_dirs()

            self.log_line.connect(self._append_log)
            self.progress_changed.connect(self._on_progress)
            self.run_finished.connect(self._finish)

            self._sync_cosmetic()
            self._sync_dark_opt()
            self._sync_register()
            self._sync_extract()
            self._fit_to_screen()

        # -- layout ---------------------------------------------------------

        def _build_widgets(self, d) -> None:
            """Settings live in tabs so the window stays short enough for a
            laptop; only the progress area and the buttons are always on."""
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(8, 8, 8, 8)

            self.tabs = QtWidgets.QTabWidget()
            outer.addWidget(self.tabs)

            try:
                wd = self.siril.get_siril_wd() or ""
            except Exception:
                wd = ""

            # ========================== Input ==============================
            tab = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(tab)
            self.ed_work = self._dir_row(
                grid, 0, "Working directory:", d.work_dir or wd,
                "Project directory; the subdirectories are looked up inside it.")
            self.ed_lights = self._dir_row(
                grid, 1, "Lights:", d.lights or "",
                "Required. Empty = the 'lights' subdirectory is found "
                "automatically.")
            self.ed_biases = self._dir_row(
                grid, 2, "Bias / offset:", d.biases or "",
                "Optional. Empty = 'biases' is looked for; without it the "
                "master bias is skipped.")
            self.ed_flats = self._dir_row(
                grid, 3, "Flats:", d.flats or "",
                "Optional. Empty = 'flats' is looked for; without it the "
                "master flat is skipped.")
            self.ed_darks = self._dir_row(
                grid, 4, "Darks:", d.darks or "",
                "Optional. Empty = 'darks' is looked for; without it the "
                "master dark is skipped.")
            self.ed_process = self._dir_row(
                grid, 5, "Intermediates:", d.process,
                "Directory for the sequences and the masters.", browse=False)

            self.lbl_found = QtWidgets.QLabel("")
            self.lbl_found.setWordWrap(True)
            grid.addWidget(self.lbl_found, 6, 0, 1, 3)
            grid.setRowStretch(7, 1)
            self.ed_work.editingFinished.connect(self._autofill_dirs)
            self.tabs.addTab(tab, "Input")

            # ========================== Masters ============================
            tab = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(tab)
            note = QtWidgets.QLabel(
                "Point these at masters you have already built and they are "
                "used as they are. Leave one blank and it is built from its "
                "directory on the Input tab, the usual way.")
            note.setWordWrap(True)
            grid.addWidget(note, 0, 0, 1, 3)
            self.ed_master_bias = self._file_row(
                grid, 1, "Master bias:", d.master_bias or "",
                "Skips building a master bias from the bias directory.")
            self.ed_master_flat = self._file_row(
                grid, 2, "Master flat:", d.master_flat or "",
                "Skips building a master flat, and with it the bias "
                "calibration of the flats.")
            self.ed_master_dark = self._file_row(
                grid, 3, "Master dark:", d.master_dark or "",
                "Skips building a master dark from the dark directory.")
            self.lbl_masters = QtWidgets.QLabel("")
            self.lbl_masters.setWordWrap(True)
            grid.addWidget(self.lbl_masters, 4, 0, 1, 3)
            grid.setRowStretch(5, 1)
            self.tabs.addTab(tab, "Masters")

            # ======================= Calibration ===========================
            tab = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(tab)

            grid.addWidget(QtWidgets.QLabel("Master rejection:"), 0, 0)
            self.cmb_rejection = QtWidgets.QComboBox()
            self.cmb_rejection.addItems([name for name, _c in REJECTIONS])
            self.cmb_rejection.setToolTip(
                "Bias and darks are stacked without normalisation, flats with "
                "-norm=mul.")
            grid.addWidget(self.cmb_rejection, 0, 1)

            grid.addWidget(QtWidgets.QLabel("sigma low / high:"), 0, 2)
            self.ed_sigma_low = self._small_entry(fmt_num(d.sigma_low))
            self.ed_sigma_high = self._small_entry(fmt_num(d.sigma_high))
            grid.addLayout(self._pair(self.ed_sigma_low, self.ed_sigma_high),
                           0, 3)

            self.chk_cosmetic = QtWidgets.QCheckBox(
                "Cosmetic correction from master dark")
            self.chk_cosmetic.setChecked(d.cosmetic)
            self.chk_cosmetic.setToolTip(
                "calibrate -cc=dark. Needs a master dark; dropped "
                "automatically without one.")
            self.chk_cosmetic.toggled.connect(self._sync_cosmetic)
            grid.addWidget(self.chk_cosmetic, 1, 0, 1, 2)

            grid.addWidget(QtWidgets.QLabel("sigma low / high:"), 1, 2)
            self.ed_cc_low = self._small_entry(fmt_num(d.cc_sigma_low))
            self.ed_cc_high = self._small_entry(fmt_num(d.cc_sigma_high))
            grid.addLayout(self._pair(self.ed_cc_low, self.ed_cc_high), 1, 3)

            flags = QtWidgets.QHBoxLayout()
            self.chk_cfa = QtWidgets.QCheckBox("-cfa")
            self.chk_cfa.setChecked(d.cfa)
            self.chk_cfa.setToolTip(
                "Makes the cosmetic correction aware of the Bayer matrix.")
            self.chk_equalize = QtWidgets.QCheckBox("-equalize_cfa")
            self.chk_equalize.setChecked(d.equalize_cfa)
            self.chk_equalize.setToolTip(
                "Equalises the RGB means of the master flat; only applied when "
                "there is one.")
            self.chk_debayer = QtWidgets.QCheckBox("-debayer")
            self.chk_debayer.setChecked(d.debayer)
            self.chk_debayer.setToolTip(
                "Required for registration: Siril refuses to register a "
                "sequence whose Bayer pattern is still intact.")
            self.chk_32bits = QtWidgets.QCheckBox("32-bit float")
            self.chk_32bits.setChecked(d.use32bits)
            self.chk_32bits.setToolTip("Off means 16-bit processing.")
            for widget in (self.chk_cfa, self.chk_equalize, self.chk_debayer,
                           self.chk_32bits):
                flags.addWidget(widget)
            flags.addStretch(1)
            grid.addLayout(flags, 2, 0, 1, 4)

            grid.addWidget(QtWidgets.QLabel("Dark optimization:"), 3, 0)
            self.cmb_dark_opt = QtWidgets.QComboBox()
            self.cmb_dark_opt.addItems([label for label, _c
                                        in DARK_OPTIMIZATIONS])
            self.cmb_dark_opt.setCurrentIndex(
                self._index_of(DARK_OPTIMIZATIONS,
                               getattr(d, "dark_opt", "none")))
            self.cmb_dark_opt.setToolTip(
                "Scales the master dark before subtracting it, for lights "
                "whose exposure does not match the dark's. Needs a master bias "
                "as well as a master dark.")
            self.cmb_dark_opt.currentIndexChanged.connect(self._sync_dark_opt)
            grid.addWidget(self.cmb_dark_opt, 3, 1, 1, 3)

            self.lbl_dark_opt = QtWidgets.QLabel("")
            self.lbl_dark_opt.setWordWrap(True)
            grid.addWidget(self.lbl_dark_opt, 4, 0, 1, 4)
            grid.setRowStretch(5, 1)
            self.tabs.addTab(tab, "Calibration")

            # ======================= Registration ==========================
            tab = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(tab)

            self.chk_register = QtWidgets.QCheckBox(
                "Register the calibrated lights")
            self.chk_register.setChecked(d.register)
            self.chk_register.toggled.connect(self._sync_register)
            grid.addWidget(self.chk_register, 0, 0, 1, 2)

            grid.addWidget(QtWidgets.QLabel("Interpolation:"), 1, 0)
            self.cmb_interp = QtWidgets.QComboBox()
            self.cmb_interp.addItems([label for label, _c in INTERPOLATIONS])
            self.cmb_interp.setCurrentIndex(
                self._index_of(INTERPOLATIONS, d.interp))
            self.cmb_interp.setToolTip(
                "'None' shifts by whole pixels and does not interpolate, so the "
                "measured counts survive unchanged - the right choice for "
                "photometry.")
            self.cmb_interp.currentIndexChanged.connect(self._sync_register)
            grid.addWidget(self.cmb_interp, 1, 1)

            grid.addWidget(QtWidgets.QLabel("Transformation:"), 2, 0)
            self.cmb_transf = QtWidgets.QComboBox()
            self.cmb_transf.addItems([label for label, _c in TRANSFORMATIONS])
            self.cmb_transf.setCurrentIndex(
                self._index_of(TRANSFORMATIONS, d.transf))
            grid.addWidget(self.cmb_transf, 2, 1)

            self.lbl_reg = QtWidgets.QLabel("")
            self.lbl_reg.setWordWrap(True)
            grid.addWidget(self.lbl_reg, 3, 0, 1, 2)
            grid.setRowStretch(4, 1)
            self.tabs.addTab(tab, "Registration")

            # ======================== Extraction ===========================
            tab = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(tab)

            self.chk_extract = QtWidgets.QCheckBox(
                "Extract a single colour channel")
            self.chk_extract.setChecked(bool(d.extract))
            self.chk_extract.setToolTip(
                "Runs after registration, on the registered frames when there "
                "are any.")
            self.chk_extract.toggled.connect(self._sync_extract)
            grid.addWidget(self.chk_extract, 0, 0, 1, 3)

            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Channel:"))
            self.channel_group = QtWidgets.QButtonGroup(self)
            self.radios = []
            wanted = (d.extract or "G").upper()
            for channel in CHANNELS:
                radio = QtWidgets.QRadioButton(
                    "%s (%s)" % (CHANNEL_LABELS[channel], channel))
                radio.setProperty("channel", channel)
                radio.setChecked(channel == wanted)
                radio.toggled.connect(self._sync_extract)
                self.channel_group.addButton(radio)
                row.addWidget(radio)
                self.radios.append(radio)
            row.addStretch(1)
            grid.addLayout(row, 1, 0, 1, 3)

            grid.addWidget(QtWidgets.QLabel("Prefix:"), 2, 0)
            self.ed_prefix = QtWidgets.QLineEdit(
                d.extract_prefix or (wanted + "_"))
            self.ed_prefix.setToolTip(
                "The new sequence is <prefix><source>, e.g. G_r_pp_light_.")
            grid.addWidget(self.ed_prefix, 2, 1)
            self.chk_auto_prefix = QtWidgets.QCheckBox("from the channel")
            self.chk_auto_prefix.setChecked(not d.extract_prefix)
            self.chk_auto_prefix.toggled.connect(self._sync_extract)
            grid.addWidget(self.chk_auto_prefix, 2, 2)

            self.chk_make_seq = QtWidgets.QCheckBox(
                "Write a .seq for the extracted sequence")
            self.chk_make_seq.setChecked(d.make_seq)
            grid.addWidget(self.chk_make_seq, 3, 0, 1, 3)

            self.lbl_extract = QtWidgets.QLabel("")
            self.lbl_extract.setWordWrap(True)
            grid.addWidget(self.lbl_extract, 4, 0, 1, 3)
            grid.setRowStretch(5, 1)
            self.tabs.addTab(tab, "Extraction")

            # ================= progress (always visible) ===================
            box = QtWidgets.QGroupBox("Progress")
            inner = QtWidgets.QVBoxLayout(box)
            self.lbl_status = QtWidgets.QLabel("Ready.")
            inner.addWidget(self.lbl_status)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setRange(0, 100)
            inner.addWidget(self.progress)
            self.text = QtWidgets.QPlainTextEdit()
            self.text.setReadOnly(True)
            self.text.setLineWrapMode(
                QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.text.setMinimumHeight(130)
            inner.addWidget(self.text, 1)
            outer.addWidget(box, 1)

            # --- buttons ---
            buttons = QtWidgets.QHBoxLayout()
            buttons.addWidget(QtWidgets.QLabel("The lights are never stacked."))
            buttons.addStretch(1)
            self.btn_close = QtWidgets.QPushButton("Close")
            self.btn_close.clicked.connect(self.close)
            buttons.addWidget(self.btn_close)
            self.btn_run = QtWidgets.QPushButton("Run preprocessing")
            self.btn_run.setDefault(True)
            self.btn_run.clicked.connect(self._start)
            buttons.addWidget(self.btn_run)
            outer.addLayout(buttons)

        # -- small layout helpers -------------------------------------------

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

        def _file_row(self, grid, row, label, value, tip):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            edit = QtWidgets.QLineEdit(value)
            edit.setToolTip(tip)
            edit.textChanged.connect(self._autofill_dirs)
            grid.addWidget(edit, row, 1)
            button = QtWidgets.QPushButton("...")
            button.setFixedWidth(32)
            button.clicked.connect(lambda _=False, e=edit: self._browse_file(e))
            grid.addWidget(button, row, 2)
            grid.setColumnStretch(1, 1)
            return edit

        def _browse_file(self, edit) -> None:
            start = (edit.text().strip() or self.ed_work.text().strip()
                     or os.getcwd())
            chosen, _filter = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select a master frame", start,
                "FITS images (*.fit *.fits *.fts);;All files (*)")
            if chosen:
                edit.setText(os.path.normpath(chosen))

        def _small_entry(self, value):
            edit = QtWidgets.QLineEdit(value)
            edit.setFixedWidth(52)
            return edit

        def _pair(self, first, second):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(first)
            row.addWidget(second)
            row.addStretch(1)
            return row

        @staticmethod
        def _index_of(table, code) -> int:
            for index, (_label, value) in enumerate(table):
                if value == code:
                    return index
            return 0

        def _fit_to_screen(self) -> None:
            """Size the window to its content, never larger than the screen."""
            available = QtGui.QGuiApplication.primaryScreen().availableGeometry()
            hint = self.sizeHint()
            width = min(max(hint.width(), 620), int(available.width() * 0.92))
            height = min(max(hint.height(), 430), int(available.height() * 0.85))
            self.setMinimumSize(min(600, width), min(400, height))
            self.resize(width, height)
            self.move(available.x() + (available.width() - width) // 2,
                      available.y() + (available.height() - height) // 3)

        # -- widget callbacks -----------------------------------------------

        def _browse(self, edit) -> None:
            start = edit.text().strip() or self.ed_work.text().strip() \
                or os.getcwd()
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select a directory", start)
            if chosen:
                edit.setText(os.path.normpath(chosen))
                if edit is self.ed_work:
                    for other in (self.ed_lights, self.ed_biases,
                                  self.ed_flats, self.ed_darks):
                        other.clear()
                self._autofill_dirs()

        def _autofill_dirs(self) -> None:
            """Fill in the subdirectories that can be found, and report counts."""
            text = self.ed_work.text().strip()
            base = Path(text) if text else None
            if base is None or not base.is_dir():
                self.lbl_found.setText("The working directory does not exist.")
                return

            masters = {"bias": self.ed_master_bias,
                       "flats": self.ed_master_flat,
                       "darks": self.ed_master_dark}
            found = []
            for edit, candidates, label in (
                    (self.ed_lights, LIGHT_DIR_CANDIDATES, "lights"),
                    (self.ed_biases, BIAS_DIR_CANDIDATES, "bias"),
                    (self.ed_flats, FLAT_DIR_CANDIDATES, "flats"),
                    (self.ed_darks, DARK_DIR_CANDIDATES, "darks")):
                if label in masters and masters[label].text().strip():
                    # a supplied master wins, so its directory is not consulted
                    edit.setEnabled(False)
                    found.append(label + ": master")
                    continue
                edit.setEnabled(True)
                directory = None
                if edit.text().strip():
                    candidate = Path(edit.text().strip())
                    candidate = candidate if candidate.is_absolute() \
                        else base / candidate
                    directory = candidate if candidate.is_dir() else None
                else:
                    try:
                        directory = resolve_dir(base, None, candidates, label,
                                                required=False)
                    except PreprocessError:
                        directory = None
                    if directory is not None:
                        edit.setText(str(directory))

                count = count_inputs(directory)
                found.append("%s: %s" % (label, count if count else "-"))

            self.lbl_found.setText("Found   " + "      ".join(found))
            self._describe_masters()
            if hasattr(self, "lbl_dark_opt"):
                self._sync_dark_opt()

        def _describe_masters(self) -> None:
            if not hasattr(self, "lbl_masters"):
                return
            supplied = [name for name, edit in (("bias", self.ed_master_bias),
                                                ("flat", self.ed_master_flat),
                                                ("dark", self.ed_master_dark))
                        if edit.text().strip()]
            if not supplied:
                self.lbl_masters.setText(
                    "Nothing given - every master is built from its directory.")
                return
            one = len(supplied) == 1
            self.lbl_masters.setText(
                "Given: %s - the matching director%s on the Input tab %s "
                "ignored." % (", ".join(supplied), "y" if one else "ies",
                              "is" if one else "are"))

        def _dark_opt(self) -> str:
            return DARK_OPTIMIZATIONS[self.cmb_dark_opt.currentIndex()][1]

        def _has_master(self, kind: str) -> bool:
            """Either a master was given, or its directory will produce one."""
            edits = {"bias": (self.ed_master_bias, self.ed_biases),
                     "dark": (self.ed_master_dark, self.ed_darks)}
            given, directory = edits[kind]
            if given.text().strip():
                return True
            text = directory.text().strip()
            return bool(text) and count_inputs(Path(text)) > 0

        def _sync_dark_opt(self) -> None:
            """Optimization is only possible with both a bias and a dark."""
            if self._dark_opt() == "none":
                self.lbl_dark_opt.setText(
                    "The master dark is subtracted as it is. Use one of the "
                    "other modes when the lights are shorter than the dark.")
                return

            missing = [kind for kind in ("bias", "dark")
                       if not self._has_master(kind)]
            if missing:
                self.lbl_dark_opt.setText(
                    "Not possible yet: Siril needs a master %s too, and there "
                    "is none. Supply it on the Masters tab, or point the Input "
                    "tab at its directory."
                    % " and a master ".join(missing))
                return

            how = ("the scaling factor is fitted from the data"
                   if self._dark_opt() == "auto"
                   else "the scaling factor comes from the exposure keyword")
            self.lbl_dark_opt.setText(
                "Siril takes the bias pedestal off the dark, scales the "
                "remaining dark current and subtracts that - " + how
                + ". The bias master is passed to the lights as well, which it "
                  "is not otherwise.")

        def _sync_cosmetic(self) -> None:
            on = self.chk_cosmetic.isChecked()
            self.ed_cc_low.setEnabled(on)
            self.ed_cc_high.setEnabled(on)

        def _sync_register(self) -> None:
            """Registration needs debayered data, and 'none' forces a shift."""
            registering = self.chk_register.isChecked()
            self.cmb_interp.setEnabled(registering)

            interp = INTERPOLATIONS[self.cmb_interp.currentIndex()][1]
            shift_only = interp == "none"
            self.cmb_transf.setEnabled(registering and not shift_only)

            if registering:
                # Siril will not register a CFA sequence, so debayering is not
                # optional here - make that visible instead of failing later.
                self.chk_debayer.setChecked(True)
                self.chk_debayer.setEnabled(False)
            else:
                self.chk_debayer.setEnabled(True)

            if not registering:
                self.lbl_reg.setText(
                    "Registration is off - only the calibrated lights are "
                    "produced.")
            elif shift_only:
                self.lbl_reg.setText(
                    "Whole-pixel shift, no interpolation: the pixel values "
                    "survive unchanged, which is what photometry needs. The "
                    "transformation is forced to shift, so that choice is "
                    "greyed out. Debayering is forced on - Siril will not "
                    "register a CFA sequence.")
            else:
                self.lbl_reg.setText(
                    "This method resamples the pixels and redistributes the "
                    "counts between neighbours, which biases photometry. Use it "
                    "only if the frames genuinely rotate or scale. Debayering "
                    "is forced on - Siril will not register a CFA sequence.")

            if hasattr(self, "lbl_extract"):
                self._sync_extract()

        def _channel(self) -> str:
            for radio in self.radios:
                if radio.isChecked():
                    return radio.property("channel")
            return "G"

        def _sync_extract(self) -> None:
            """Enable the channel controls, and say what the result will be."""
            on = self.chk_extract.isChecked()
            for radio in self.radios:
                radio.setEnabled(on)
            self.chk_auto_prefix.setEnabled(on)
            self.chk_make_seq.setEnabled(on)

            channel = self._channel()
            if on and self.chk_auto_prefix.isChecked():
                self.ed_prefix.setText(channel + "_")
            self.ed_prefix.setEnabled(on and not self.chk_auto_prefix.isChecked())

            if not on:
                self.lbl_extract.setText(
                    "No channel extraction - the run ends with the calibrated "
                    "(and registered) lights.")
                return

            source = (REGISTERED_PREFIX + CALIBRATED_PREFIX + LIGHT_SEQ
                      if self.chk_register.isChecked()
                      else CALIBRATED_PREFIX + LIGHT_SEQ)
            text = ("Result: %s%s_  - one mono file per frame, plus its .seq."
                    % (self.ed_prefix.text().strip() or (channel + "_"), source))
            if self.chk_debayer.isChecked():
                text += (" The source is debayered, so the %s channel comes out "
                         "at full resolution."
                         % CHANNEL_LABELS[channel].lower())
            else:
                text += (" The source is still CFA, so the result is half width "
                         "and half height - the real sensor pixels, not "
                         "interpolated.")
            self.lbl_extract.setText(text)

        # -- starting the processing ----------------------------------------

        def _error(self, message: str) -> None:
            QtWidgets.QMessageBox.critical(self, "Error", message)

        def _collect(self):
            """Build the same settings object argparse produces, from the form."""
            args = parse_args([])
            work = self.ed_work.text().strip()
            if not work or not Path(work).is_dir():
                self._error("The working directory does not exist.")
                return None

            def number(edit, label, minimum=0.0):
                try:
                    value = float(edit.text().replace(",", "."))
                except ValueError:
                    raise ValueError(label + ": enter a number.")
                if value < minimum:
                    raise ValueError(label + ": the value must be >= "
                                     + fmt_num(minimum))
                return value

            try:
                args.sigma_low = number(self.ed_sigma_low,
                                        "Sigma low (rejection)")
                args.sigma_high = number(self.ed_sigma_high,
                                         "Sigma high (rejection)")
                args.cc_sigma_low = number(self.ed_cc_low,
                                           "Sigma low (cosmetic correction)")
                args.cc_sigma_high = number(self.ed_cc_high,
                                            "Sigma high (cosmetic correction)")
            except ValueError as exc:
                self._error(str(exc))
                return None

            args.work_dir = work
            args.lights = self.ed_lights.text().strip() or None
            args.biases = self.ed_biases.text().strip() or None
            args.flats = self.ed_flats.text().strip() or None
            args.darks = self.ed_darks.text().strip() or None
            args.process = self.ed_process.text().strip() or "process"
            args.master_bias = self.ed_master_bias.text().strip() or None
            args.master_flat = self.ed_master_flat.text().strip() or None
            args.master_dark = self.ed_master_dark.text().strip() or None
            for value, label in ((args.master_bias, "bias"),
                                 (args.master_flat, "flat"),
                                 (args.master_dark, "dark")):
                try:
                    resolve_master(Path(work), value, label)
                except PreprocessError as exc:
                    self._error(str(exc))
                    return None

            args.rejection = REJECTIONS[self.cmb_rejection.currentIndex()][1]
            args.use32bits = self.chk_32bits.isChecked()
            args.cosmetic = self.chk_cosmetic.isChecked()
            args.cfa = self.chk_cfa.isChecked()
            args.equalize_cfa = self.chk_equalize.isChecked()
            args.debayer = self.chk_debayer.isChecked()
            args.dark_opt = self._dark_opt()
            if args.dark_opt != "none":
                missing = [k for k in ("bias", "dark")
                           if not self._has_master(k)]
                if missing:
                    self._error(
                        "Dark optimization needs both a master bias and a "
                        "master dark, but the master "
                        + " and the master ".join(missing) + " is missing.")
                    return None
            args.register = self.chk_register.isChecked()
            args.interp = INTERPOLATIONS[self.cmb_interp.currentIndex()][1]
            args.transf = TRANSFORMATIONS[self.cmb_transf.currentIndex()][1]

            args.extract = self._channel() if self.chk_extract.isChecked() \
                else None
            prefix = self.ed_prefix.text().strip()
            if args.extract:
                if not prefix:
                    self._error("The extraction prefix must not be empty.")
                    return None
                if any(ch in prefix for ch in r'\/:*?"<>|'):
                    self._error(
                        "The prefix must not contain path characters.")
                    return None
            args.extract_prefix = prefix or None
            args.make_seq = self.chk_make_seq.isChecked()
            return args

        def _start(self) -> None:
            if self.running:
                return
            args = self._collect()
            if args is None:
                return

            self.running = True
            self.btn_run.setEnabled(False)
            self.progress.setValue(0)
            self.text.clear()
            self.lbl_status.setText("Processing...")

            thread = threading.Thread(target=self._worker, args=(args,),
                                      daemon=True)
            thread.start()

        def _worker(self, args) -> None:
            """Runs off the GUI thread; every Siril command is issued here."""
            try:
                result = run_pipeline(self.siril, args)
                self.run_finished.emit(result, None)
            except PreprocessError as exc:
                self.run_finished.emit(None, str(exc))
            except Exception as exc:
                self.run_finished.emit(
                    None, exc.__class__.__name__ + ": " + str(exc))

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
            if message.startswith("["):
                self.lbl_status.setText(message)

        def _on_progress(self, done, total) -> None:
            self.progress.setValue(int(100.0 * done / max(total, 1)))

        def _finish(self, result, error) -> None:
            self.running = False
            self.btn_run.setEnabled(True)
            if error:
                self.lbl_status.setText("Preprocessing failed.")
                self.text.appendPlainText("ERROR: " + error)
                QtWidgets.QMessageBox.critical(self, "Preprocessing failed",
                                               error)
                return

            summary = ("Calibrated lights:\n  %s_  (%d frames)"
                       % (result["calibrated"], result["n_calibrated"]))
            if result["registered"]:
                summary += ("\n\nRegistered lights:\n  %s_  (%d frames)"
                            % (result["registered"], result["n_registered"]))
            if result.get("extracted"):
                summary += ("\n\n%s channel:\n  %s_  (%d frames)"
                            % (result["channel"], result["extracted"],
                               result["n_extracted"]))
            summary += "\n\nIn:\n  " + result["process"]
            summary += "\n\nThe lights were not stacked."

            self.lbl_status.setText("Done - " + (result.get("extracted")
                                                 or result["registered"]
                                                 or result["calibrated"]))
            self.progress.setValue(100)
            QtWidgets.QMessageBox.information(self, "Done", summary)

        # -- closing ---------------------------------------------------------

        def closeEvent(self, event) -> None:
            if self.running:
                answer = QtWidgets.QMessageBox.question(
                    self, "Processing is running",
                    "Processing is still running. Close the window anyway?")
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
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
        raise PreprocessError(
            "PyQt6 is not available in this Python environment.")

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])
    apply_siril_theme(app, siril)

    window = PreprocessWindow(siril, defaults)
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

def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OSC preprocessing: build the masters, calibrate the "
                    "lights, keep them, register them, and stop before "
                    "stacking (Siril 1.4+).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version",
                   version="%(prog)s " + __version__)
    p.add_argument("--no-gui", dest="gui", action="store_false",
                   help="Skip the dialog and start processing right away.")
    p.add_argument("--work-dir", default=None,
                   help="Working directory; defaults to Siril's working "
                        "directory.")
    p.add_argument("--lights", default=None,
                   help="Directory with the light frames (default 'lights').")
    p.add_argument("--biases", default=None,
                   help="Directory with the bias frames (default 'biases').")
    p.add_argument("--flats", default=None,
                   help="Directory with the flat frames (default 'flats').")
    p.add_argument("--darks", default=None,
                   help="Directory with the dark frames (default 'darks').")
    p.add_argument("--process", default="process",
                   help="Directory for the sequences and the masters.")

    p.add_argument("--master-bias", default=None,
                   help="Use this ready-made master bias instead of building "
                        "one from the bias directory.")
    p.add_argument("--master-flat", default=None,
                   help="Use this ready-made master flat instead of building "
                        "one from the flat directory.")
    p.add_argument("--master-dark", default=None,
                   help="Use this ready-made master dark instead of building "
                        "one from the dark directory.")

    p.add_argument("--rejection", default="w",
                   choices=["p", "s", "m", "w", "l", "g", "a", "n"],
                   help="Rejection type for stacking the masters "
                        "(w = Winsorized sigma clipping, n = none).")
    p.add_argument("--sigma-low", type=float, default=3.0,
                   help="Low sigma for rejection.")
    p.add_argument("--sigma-high", type=float, default=3.0,
                   help="High sigma for rejection.")
    p.add_argument("--16bit", dest="use32bits", action="store_false",
                   help="Work in 16-bit mode (32-bit float is the default).")

    p.add_argument("--no-cosmetic", dest="cosmetic", action="store_false",
                   help="Disable hot/cold pixel cosmetic correction "
                        "(calibrate -cc=dark).")
    p.add_argument("--cc-sigma-low", type=float, default=3.0,
                   help="Sigma for cold pixels (cosmetic correction).")
    p.add_argument("--cc-sigma-high", type=float, default=3.0,
                   help="Sigma for hot pixels (cosmetic correction).")
    p.add_argument("--no-cfa", dest="cfa", action="store_false",
                   help="Do not pass -cfa to calibrate.")
    p.add_argument("--no-equalize-cfa", dest="equalize_cfa",
                   action="store_false",
                   help="Do not pass -equalize_cfa to calibrate.")
    p.add_argument("--dark-opt", default="none",
                   choices=["none", "auto", "exp"],
                   help="Optimize the master dark before subtracting it. "
                        "'auto' fits the scaling factor, 'exp' derives it from "
                        "the exposure keyword. Both need a master bias as well "
                        "as a master dark.")
    p.add_argument("--no-debayer", dest="debayer", action="store_false",
                   help="Do not debayer during calibration. Incompatible with "
                        "registration, which Siril refuses to run on CFA data.")

    p.add_argument("--no-register", dest="register", action="store_false",
                   help="Stop after calibration, without registering.")
    p.add_argument("--interp", default="none",
                   choices=["none", "nearest", "linear", "cubic", "lanczos4",
                            "area"],
                   help="Registration interpolation. 'none' shifts by whole "
                        "pixels and does not touch the values - the right "
                        "choice for photometry.")
    p.add_argument("--transf", default="shift",
                   choices=["shift", "similarity", "affine", "homography"],
                   help="Registration transformation. Ignored with "
                        "--interp=none, which forces a shift.")
    p.add_argument("--layer", type=int, default=None,
                   help="Layer the registration is computed on "
                        "(1 = green, Siril's default for colour).")

    p.add_argument("--extract", default=None, choices=["R", "G", "B",
                                                       "r", "g", "b"],
                   help="Extract this colour channel into a new sequence as a "
                        "last step. Naming a channel is what enables it.")
    p.add_argument("--extract-prefix", default=None,
                   help="Prefix of the extracted sequence "
                        "(default: the channel letter, e.g. 'G_').")
    p.add_argument("--no-seq", dest="make_seq", action="store_false",
                   help="Do not write a .seq for the extracted sequence.")

    p.set_defaults(gui=True, cosmetic=True, cfa=True, equalize_cfa=True,
                   debayer=True, register=True, use32bits=True, make_seq=True)
    return p.parse_args(argv)


def main(argv: list) -> int:
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
        if args.gui:
            try:
                return launch_gui(siril, args)
            except Exception as exc:
                log(siril, "Could not start the GUI (" + str(exc)
                    + "), continuing in text mode.", "salmon")

        run_pipeline(siril, args)
        return 0

    except PreprocessError as exc:
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
