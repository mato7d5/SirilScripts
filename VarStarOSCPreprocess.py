#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

Workflow
--------
  1. biases -> convert -> stack bias rej w 3 3 -nonorm -out=bias_stacked
  2. flats  -> convert -> calibrate flat -bias=bias_stacked
                       -> stack pp_flat rej w 3 3 -norm=mul -out=pp_flat_stacked
  3. darks  -> convert -> stack dark rej w 3 3 -nonorm -out=dark_stacked
  4. lights -> convert -> calibrate light -dark=... -flat=... -cc=dark -cfa
                                          -equalize_cfa -debayer -prefix=pp_
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


def calibrate_lights(siril, process_dir: Path, sequence: str,
                     master_dark, master_flat, master_bias, cosmetic: bool,
                     cc_low: float, cc_high: float, cfa: bool,
                     equalize_cfa: bool, debayer: bool, prefix: str) -> None:
    """Calibrate the lights and, when asked, debayer them in the same pass."""
    run(siril, "cd", siril_path(process_dir))

    args = ["calibrate", sequence]
    if master_bias and not master_dark:
        # With a dark there is no point subtracting the bias as well: the dark
        # already contains it.
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

        if args.register and not args.debayer:
            raise PreprocessError(
                "Siril cannot register a sequence whose Bayer pattern is still "
                "intact. Either allow debayering, or turn registration off.")

        lights_dir = resolve_dir(base, args.lights, LIGHT_DIR_CANDIDATES,
                                 "lights", required=True)
        bias_dir = resolve_dir(base, args.biases, BIAS_DIR_CANDIDATES,
                               "bias", required=False)
        flat_dir = resolve_dir(base, args.flats, FLAT_DIR_CANDIDATES,
                               "flats", required=False)
        dark_dir = resolve_dir(base, args.darks, DARK_DIR_CANDIDATES,
                               "darks", required=False)

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
        for label, directory, count in (("bias", bias_dir, n_bias),
                                        ("flats", flat_dir, n_flats),
                                        ("darks", dark_dir, n_darks)):
            if directory is None or not count:
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
        if bias_dir is not None and n_bias:
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
        if flat_dir is not None and n_flats:
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
        if dark_dir is not None and n_darks:
            log(siril, "[%d/%d] Master dark..." % (step, total_steps), "blue")
            convert_frames(siril, dark_dir, DARK_SEQ, process_dir)
            master_dark = stack_master(
                siril, process_dir, DARK_SEQ, MASTER_DARK, args.rejection,
                args.sigma_low, args.sigma_high, None, n_darks, ext)
            log(siril, "      " + master_dark, "green")
        else:
            log(siril, "[%d/%d] Master dark skipped." % (step, total_steps),
                "salmon")
        emit_progress(step, total_steps)

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
            calibrate_lights(
                siril, process_dir, LIGHT_SEQ, master_dark, master_flat,
                master_bias, args.cosmetic, args.cc_sigma_low,
                args.cc_sigma_high, cfa, args.equalize_cfa, debayer,
                CALIBRATED_PREFIX)
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


class PreprocessGUI:
    """Settings dialog; the processing runs on its own thread."""

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

        root.title("OSC preprocessing up to registration")

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
        self.var_lights = tk.StringVar(value=d.lights or "")
        self.var_biases = tk.StringVar(value=d.biases or "")
        self.var_flats = tk.StringVar(value=d.flats or "")
        self.var_darks = tk.StringVar(value=d.darks or "")
        self.var_process = tk.StringVar(value=d.process)

        self.var_rejection = tk.StringVar(value=REJECTIONS[0][0])
        self.var_sigma_low = tk.StringVar(value=fmt_num(d.sigma_low))
        self.var_sigma_high = tk.StringVar(value=fmt_num(d.sigma_high))
        self.var_32bits = tk.BooleanVar(value=d.use32bits)

        self.var_cosmetic = tk.BooleanVar(value=d.cosmetic)
        self.var_cc_low = tk.StringVar(value=fmt_num(d.cc_sigma_low))
        self.var_cc_high = tk.StringVar(value=fmt_num(d.cc_sigma_high))
        self.var_cfa = tk.BooleanVar(value=d.cfa)
        self.var_equalize = tk.BooleanVar(value=d.equalize_cfa)
        self.var_debayer = tk.BooleanVar(value=d.debayer)

        self.var_register = tk.BooleanVar(value=d.register)
        interp_labels = dict((code, label) for label, code in INTERPOLATIONS)
        transf_labels = dict((code, label) for label, code in TRANSFORMATIONS)
        self.var_interp = tk.StringVar(
            value=interp_labels.get(d.interp, INTERPOLATIONS[0][0]))
        self.var_transf = tk.StringVar(
            value=transf_labels.get(d.transf, TRANSFORMATIONS[0][0]))

        self.var_extract = tk.BooleanVar(value=bool(d.extract))
        self.var_channel = tk.StringVar(value=(d.extract or "G").upper())
        self.var_ext_prefix = tk.StringVar(
            value=d.extract_prefix or ((d.extract or "G").upper() + "_"))
        self.var_auto_prefix = tk.BooleanVar(value=not d.extract_prefix)
        self.var_make_seq = tk.BooleanVar(value=d.make_seq)

        self.var_found = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="Ready.")

        self._build_widgets()
        self._autofill_dirs()
        self._fit_to_screen()
        self.var_work.trace_add("write", lambda *_a: self._autofill_dirs())
        self.pump_id = self.root.after(100, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout -------------------------------------------------------------

    def _build_widgets(self) -> None:
        """Settings live in tabs so the window stays short enough for a laptop.

        Only the progress area and the buttons are always on screen; everything
        else is one notebook page deep.
        """
        ttk, tk = self.ttk, self.tk
        from tkinter import scrolledtext

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)          # the log takes the spare height

        notebook = ttk.Notebook(main)
        notebook.grid(row=0, column=0, sticky="ew")

        wrap = 520          # fits the narrower tab width

        # ============================ Input ================================
        tab = ttk.Frame(notebook, padding=8)
        tab.columnconfigure(1, weight=1)
        notebook.add(tab, text="Input")

        self._dir_row(tab, 0, "Working directory:", self.var_work,
                      "Project directory; the subdirectories are looked up "
                      "inside it.")
        self._dir_row(tab, 1, "Lights:", self.var_lights,
                      "Required. Empty = the 'lights' subdirectory is found "
                      "automatically.")
        self._dir_row(tab, 2, "Bias / offset:", self.var_biases,
                      "Optional. Empty = 'biases' is looked for; without it "
                      "the master bias is skipped.")
        self._dir_row(tab, 3, "Flats:", self.var_flats,
                      "Optional. Empty = 'flats' is looked for; without it the "
                      "master flat is skipped.")
        self._dir_row(tab, 4, "Darks:", self.var_darks,
                      "Optional. Empty = 'darks' is looked for; without it the "
                      "master dark is skipped.")
        self._entry_row(tab, 5, "Intermediates:", self.var_process,
                        "Directory for the sequences and the masters.")
        ttk.Label(tab, textvariable=self.var_found, wraplength=wrap,
                  justify="left").grid(row=6, column=0, columnspan=3,
                                       sticky="w", pady=(6, 0))

        # ========================= Calibration =============================
        tab = ttk.Frame(notebook, padding=8)
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=1)
        notebook.add(tab, text="Calibration")

        ttk.Label(tab, text="Master rejection:").grid(row=0, column=0,
                                                      sticky="w", pady=2)
        combo = ttk.Combobox(tab, textvariable=self.var_rejection,
                             state="readonly", width=22,
                             values=[name for name, _c in REJECTIONS])
        combo.grid(row=0, column=1, sticky="ew", padx=(6, 12), pady=2)
        self._tip(combo, "Bias and darks are stacked without normalisation, "
                         "flats with -norm=mul.")

        ttk.Label(tab, text="sigma low / high:").grid(row=0, column=2,
                                                      sticky="e", pady=2)
        frame = ttk.Frame(tab)
        frame.grid(row=0, column=3, sticky="w", padx=(6, 0), pady=2)
        ttk.Entry(frame, textvariable=self.var_sigma_low, width=5).pack(
            side="left")
        ttk.Entry(frame, textvariable=self.var_sigma_high, width=5).pack(
            side="left", padx=(4, 0))

        chk = ttk.Checkbutton(tab, text="Cosmetic correction from master dark",
                              variable=self.var_cosmetic,
                              command=self._sync_cosmetic)
        chk.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        self._tip(chk, "calibrate -cc=dark. Needs a master dark; dropped "
                       "automatically without one.")

        ttk.Label(tab, text="sigma low / high:").grid(row=1, column=2,
                                                      sticky="e", pady=2)
        frame = ttk.Frame(tab)
        frame.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=2)
        self.ent_cc_low = ttk.Entry(frame, textvariable=self.var_cc_low, width=5)
        self.ent_cc_low.pack(side="left")
        self.ent_cc_high = ttk.Entry(frame, textvariable=self.var_cc_high,
                                     width=5)
        self.ent_cc_high.pack(side="left", padx=(4, 0))

        # the three flags sit side by side instead of on three rows
        flags = ttk.Frame(tab)
        flags.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        chk = ttk.Checkbutton(flags, text="-cfa", variable=self.var_cfa)
        chk.pack(side="left", padx=(0, 14))
        self._tip(chk, "Makes the cosmetic correction aware of the Bayer "
                       "matrix.")
        chk = ttk.Checkbutton(flags, text="-equalize_cfa",
                              variable=self.var_equalize)
        chk.pack(side="left", padx=(0, 14))
        self._tip(chk, "Equalises the RGB means of the master flat; only "
                       "applied when there is one.")
        self.chk_debayer = ttk.Checkbutton(flags, text="-debayer",
                                           variable=self.var_debayer)
        self.chk_debayer.pack(side="left", padx=(0, 14))
        self._tip(self.chk_debayer,
                  "Required for registration: Siril refuses to register a "
                  "sequence whose Bayer pattern is still intact.")
        chk = ttk.Checkbutton(flags, text="32-bit float",
                              variable=self.var_32bits)
        chk.pack(side="left")
        self._tip(chk, "Off means 16-bit processing.")

        # ======================== Registration =============================
        tab = ttk.Frame(notebook, padding=8)
        tab.columnconfigure(1, weight=1)
        notebook.add(tab, text="Registration")

        chk = ttk.Checkbutton(tab, text="Register the calibrated lights",
                              variable=self.var_register,
                              command=self._sync_register)
        chk.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(tab, text="Interpolation:").grid(row=1, column=0, sticky="w",
                                                   pady=2)
        self.cmb_interp = ttk.Combobox(
            tab, textvariable=self.var_interp, state="readonly",
            values=[label for label, _c in INTERPOLATIONS])
        self.cmb_interp.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)
        self.cmb_interp.bind("<<ComboboxSelected>>",
                             lambda _e: self._sync_register())
        self._tip(self.cmb_interp,
                  "'None' shifts by whole pixels and does not interpolate, so "
                  "the measured counts survive unchanged - the right choice for "
                  "photometry.")

        ttk.Label(tab, text="Transformation:").grid(row=2, column=0, sticky="w",
                                                    pady=2)
        self.cmb_transf = ttk.Combobox(
            tab, textvariable=self.var_transf, state="readonly",
            values=[label for label, _c in TRANSFORMATIONS])
        self.cmb_transf.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=2)

        self.lbl_reg = ttk.Label(tab, text="", wraplength=wrap, justify="left")
        self.lbl_reg.grid(row=3, column=0, columnspan=2, sticky="w",
                          pady=(6, 0))

        # ========================== Extraction =============================
        tab = ttk.Frame(notebook, padding=8)
        tab.columnconfigure(1, weight=1)
        notebook.add(tab, text="Extraction")

        chk = ttk.Checkbutton(tab, text="Extract a single colour channel",
                              variable=self.var_extract,
                              command=self._sync_extract)
        chk.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        self._tip(chk, "Runs after registration, on the registered frames when "
                       "there are any.")

        row = ttk.Frame(tab)
        row.grid(row=1, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(row, text="Channel:").pack(side="left", padx=(0, 10))
        self.radios = []
        for channel in CHANNELS:
            radio = ttk.Radiobutton(
                row, text=CHANNEL_LABELS[channel] + " (" + channel + ")",
                variable=self.var_channel, value=channel,
                command=self._sync_extract)
            radio.pack(side="left", padx=(0, 12))
            self.radios.append(radio)

        ttk.Label(tab, text="Prefix:").grid(row=2, column=0, sticky="w", pady=2)
        self.ent_prefix = ttk.Entry(tab, textvariable=self.var_ext_prefix)
        self.ent_prefix.grid(row=2, column=1, sticky="ew", padx=(6, 6), pady=2)
        self._tip(self.ent_prefix,
                  "The new sequence is <prefix><source>, e.g. G_r_pp_light_.")
        self.chk_auto_prefix = ttk.Checkbutton(
            tab, text="from the channel", variable=self.var_auto_prefix,
            command=self._sync_extract)
        self.chk_auto_prefix.grid(row=2, column=2, sticky="w", pady=2)

        self.chk_make_seq = ttk.Checkbutton(
            tab, text="Write a .seq for the extracted sequence",
            variable=self.var_make_seq)
        self.chk_make_seq.grid(row=3, column=0, columnspan=3, sticky="w",
                               pady=2)

        self.lbl_extract = ttk.Label(tab, text="", wraplength=wrap,
                                     justify="left")
        self.lbl_extract.grid(row=4, column=0, columnspan=3, sticky="w",
                              pady=(6, 0))

        # ===================== progress (always visible) ====================
        box = ttk.LabelFrame(main, text="Progress", padding=6)
        box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(2, weight=1)

        ttk.Label(box, textvariable=self.var_status).grid(row=0, column=0,
                                                          sticky="w")
        self.progress = ttk.Progressbar(box, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(4, 6))
        self.text = scrolledtext.ScrolledText(box, height=7, wrap="none")
        self.text.grid(row=2, column=0, sticky="nsew")
        self.text.configure(state="disabled")

        # --- buttons ---
        buttons = ttk.Frame(main, padding=(0, 8, 0, 0))
        buttons.grid(row=2, column=0, sticky="ew")
        ttk.Label(buttons, text="The lights are never stacked.").pack(
            side="left")
        self.btn_run = ttk.Button(buttons, text="Run preprocessing",
                                  command=self._start)
        self.btn_run.pack(side="right")
        ttk.Button(buttons, text="Close", command=self._on_close).pack(
            side="right", padx=(0, 6))

        self._sync_cosmetic()
        self._sync_register()
        self._sync_extract()

    def _fit_to_screen(self) -> None:
        """Size the window to its content, but never larger than the screen."""
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        # leave room for the taskbar and the window decorations
        max_w, max_h = int(screen_w * 0.92), int(screen_h * 0.85)

        width = min(max(self.root.winfo_reqwidth(), 620), max_w)
        height = min(max(self.root.winfo_reqheight(), 430), max_h)
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 3, 0)
        self.root.geometry("%dx%d+%d+%d" % (width, height, x, y))
        # the floor has to fit too, or the window is unusable on a small screen
        self.root.minsize(min(600, max_w), min(400, max_h))

    def _dir_row(self, parent, row, label, var, tip) -> None:
        ttk = self.ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        self._tip(entry, tip)
        ttk.Button(parent, text="...", width=3,
                   command=lambda v=var: self._browse(v)).grid(row=row,
                                                               column=2)

    def _entry_row(self, parent, row, label, var, tip) -> None:
        ttk = self.ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        self._tip(entry, tip)

    def _tip(self, widget, text) -> None:
        if tksiril is not None:
            try:
                tksiril.create_tooltip(widget, text)
            except Exception:
                pass

    # -- widget callbacks ---------------------------------------------------

    def _browse(self, var) -> None:
        from tkinter import filedialog
        initial = var.get() or self.var_work.get() or os.getcwd()
        chosen = filedialog.askdirectory(initialdir=initial, parent=self.root)
        if chosen:
            var.set(os.path.normpath(chosen))
            if var is self.var_work:
                for other in (self.var_lights, self.var_biases,
                              self.var_flats, self.var_darks):
                    other.set("")
                self._autofill_dirs()

    def _autofill_dirs(self) -> None:
        """Fill in the subdirectories that can be found, and report the counts."""
        text = self.var_work.get().strip()
        base = Path(text) if text else None
        if base is None or not base.is_dir():
            self.var_found.set("The working directory does not exist.")
            return

        found = []
        for var, candidates, label in (
                (self.var_lights, LIGHT_DIR_CANDIDATES, "lights"),
                (self.var_biases, BIAS_DIR_CANDIDATES, "bias"),
                (self.var_flats, FLAT_DIR_CANDIDATES, "flats"),
                (self.var_darks, DARK_DIR_CANDIDATES, "darks")):
            directory = None
            if var.get().strip():
                candidate = Path(var.get().strip())
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
                    var.set(str(directory))

            count = count_inputs(directory)
            found.append("%s: %s" % (label, count if count else "-"))

        self.var_found.set("Found   " + "      ".join(found))

    def _sync_extract(self) -> None:
        """Enable the channel controls, and say what the result will look like."""
        on = self.var_extract.get()
        state = "!disabled" if on else "disabled"
        for radio in self.radios:
            radio.state([state])
        self.chk_auto_prefix.state([state])
        self.chk_make_seq.state([state])

        channel = self.var_channel.get()
        if on and self.var_auto_prefix.get():
            self.var_ext_prefix.set(channel + "_")
        self.ent_prefix.configure(
            state="normal" if (on and not self.var_auto_prefix.get())
            else "disabled")

        if not on:
            self.lbl_extract.configure(
                text="No channel extraction - the run ends with the "
                     "calibrated (and registered) lights.")
            return

        source = (REGISTERED_PREFIX + CALIBRATED_PREFIX + LIGHT_SEQ
                  if self.var_register.get()
                  else CALIBRATED_PREFIX + LIGHT_SEQ)
        text = ("Result: %s%s_  - one mono file per frame, plus its .seq."
                % (self.var_ext_prefix.get() or (channel + "_"), source))
        if self.var_debayer.get():
            text += (" The source is debayered, so the %s channel comes out at "
                     "full resolution."
                     % CHANNEL_LABELS[channel].lower())
        else:
            text += (" The source is still CFA, so the result is half width and "
                     "half height - the real sensor pixels, not interpolated.")
        self.lbl_extract.configure(text=text)

    def _sync_cosmetic(self) -> None:
        state = "normal" if self.var_cosmetic.get() else "disabled"
        self.ent_cc_low.configure(state=state)
        self.ent_cc_high.configure(state=state)

    def _sync_register(self) -> None:
        """Registration needs debayered data, and 'none' forces a plain shift."""
        registering = self.var_register.get()
        self.cmb_interp.configure(state="readonly" if registering else "disabled")

        interp = dict((label, code)
                      for label, code in INTERPOLATIONS)[self.var_interp.get()]
        shift_only = interp == "none"
        self.cmb_transf.configure(
            state="disabled" if (not registering or shift_only) else "readonly")

        if registering:
            # Siril will not register a CFA sequence, so debayering is not
            # optional here - make that visible instead of failing later.
            self.var_debayer.set(True)
            self.chk_debayer.state(["disabled"])
        else:
            self.chk_debayer.state(["!disabled"])

        if hasattr(self, "lbl_extract"):
            self._sync_extract()

        if not registering:
            self.lbl_reg.configure(
                text="Registration is off - only the calibrated lights are "
                     "produced.")
        elif shift_only:
            self.lbl_reg.configure(
                text="Whole-pixel shift, no interpolation: the pixel values "
                     "survive unchanged, which is what photometry needs. The "
                     "transformation is forced to shift, so that choice is "
                     "greyed out. Debayering is forced on - Siril will not "
                     "register a CFA sequence.")
        else:
            self.lbl_reg.configure(
                text="This method resamples the pixels and redistributes the "
                     "counts between neighbours, which biases photometry. Use "
                     "it only if the frames genuinely rotate or scale. "
                     "Debayering is forced on - Siril will not register a CFA "
                     "sequence.")

    # -- starting the processing --------------------------------------------

    def _collect(self):
        """Build the same settings object argparse produces, from the form."""
        from tkinter import messagebox

        args = parse_args([])
        work = self.var_work.get().strip()
        if not work or not Path(work).is_dir():
            messagebox.showerror("Error",
                                 "The working directory does not exist.",
                                 parent=self.root)
            return None

        def number(var, label, minimum=0.0):
            try:
                value = float(var.get().replace(",", "."))
            except ValueError:
                raise ValueError(label + ": enter a number.")
            if value < minimum:
                raise ValueError(label + ": the value must be >= "
                                 + fmt_num(minimum))
            return value

        try:
            args.sigma_low = number(self.var_sigma_low, "Sigma low (rejection)")
            args.sigma_high = number(self.var_sigma_high,
                                     "Sigma high (rejection)")
            args.cc_sigma_low = number(self.var_cc_low,
                                       "Sigma low (cosmetic correction)")
            args.cc_sigma_high = number(self.var_cc_high,
                                        "Sigma high (cosmetic correction)")
        except ValueError as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)
            return None

        args.work_dir = work
        args.lights = self.var_lights.get().strip() or None
        args.biases = self.var_biases.get().strip() or None
        args.flats = self.var_flats.get().strip() or None
        args.darks = self.var_darks.get().strip() or None
        args.process = self.var_process.get().strip() or "process"

        args.rejection = dict((n, c) for n, c in REJECTIONS)[
            self.var_rejection.get()]
        args.use32bits = self.var_32bits.get()
        args.cosmetic = self.var_cosmetic.get()
        args.cfa = self.var_cfa.get()
        args.equalize_cfa = self.var_equalize.get()
        args.debayer = self.var_debayer.get()
        args.register = self.var_register.get()
        args.interp = dict((l, c) for l, c in INTERPOLATIONS)[
            self.var_interp.get()]
        args.transf = dict((l, c) for l, c in TRANSFORMATIONS)[
            self.var_transf.get()]

        args.extract = self.var_channel.get() if self.var_extract.get() else None
        prefix = self.var_ext_prefix.get().strip()
        if args.extract:
            if not prefix:
                messagebox.showerror("Error",
                                     "The extraction prefix must not be empty.",
                                     parent=self.root)
                return None
            if any(ch in prefix for ch in r'\/:*?"<>|'):
                messagebox.showerror(
                    "Error", "The prefix must not contain path characters.",
                    parent=self.root)
                return None
        args.extract_prefix = prefix or None
        args.make_seq = self.var_make_seq.get()
        return args

    def _start(self) -> None:
        if self.running:
            return
        args = self._collect()
        if args is None:
            return

        self.running = True
        self.btn_run.state(["disabled"])
        self.progress.configure(value=0)
        self._clear_log()
        self.var_status.set("Processing...")

        thread = threading.Thread(target=self._worker, args=(args,), daemon=True)
        thread.start()

    def _worker(self, args) -> None:
        """Runs off the GUI thread; every Siril command is issued from here."""
        try:
            result = run_pipeline(self.siril, args)
            self.queue.put(("done", result, None))
        except PreprocessError as exc:
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

    def _finish(self, result, error) -> None:
        from tkinter import messagebox
        self.running = False
        self.btn_run.state(["!disabled"])
        if error:
            self.var_status.set("Preprocessing failed.")
            self._append_log("ERROR: " + error)
            messagebox.showerror("Preprocessing failed", error,
                                 parent=self.root)
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

        self.var_status.set("Done - " + (result.get("extracted")
                                         or result["registered"]
                                         or result["calibrated"]))
        self.progress.configure(value=100)
        messagebox.showinfo("Done", summary, parent=self.root)

    def _on_close(self) -> None:
        from tkinter import messagebox
        if self.running:
            if not messagebox.askyesno(
                    "Processing is running",
                    "Processing is still running. Close the window anyway?",
                    parent=self.root):
                return
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
    gui = PreprocessGUI(root, siril, defaults)
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

def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OSC preprocessing: build the masters, calibrate the "
                    "lights, keep them, register them, and stop before "
                    "stacking (Siril 1.4+).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
