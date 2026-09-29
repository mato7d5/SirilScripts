#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate light frames with a master dark (Siril 1.4.4, "sirilpy" Python API).

Input
-----
  <work_dir>/lights/   ... light frames in RAW format (CR2/CR3/NEF/ARW/...)
  <work_dir>/darks/    ... dark frames in RAW format

Output
------
  <work_dir>/calibrated/light_00001.tif ...  calibrated light frames as TIFF
  <work_dir>/process/                        intermediates (FITS sequences, master dark)

The calibrated frames are LINEAR: in an ordinary image viewer they look almost
black, which is correct - subtracting the dark also removes the offset pedestal
that lifted the RAW background. Stretching belongs after stacking, so keep them
linear for registration and stacking. For a viewable copy use --stretch
(non-linear, preview only). The script prints the statistics of the first frame;
a background pinned at 0 means the dark is over-subtracting - check that darks
match the lights in exposure, ISO/gain and temperature, or use --pedestal.

Workflow
--------
  1. darks  -> convertraw -> sequence "dark_"       (no debayer, the CFA must be kept)
  2. stack dark rej w 3 3 -nonorm -out=master_dark  ... master dark
  3. lights -> convertraw -> sequence "light_"      (no debayer)
  4. calibrate light -dark=master_dark [-cc=dark] [-cfa] [-debayer] -prefix=pp_
  5. per frame: load pp_light_NNNNN + savetif ../calibrated/light_NNNNN
     (Siril has no command to export a whole sequence to TIFF, so it is done
     one frame at a time)

Usage
-----
  A) From the Siril GUI: copy this script into the scripts directory
     (Windows: %LOCALAPPDATA%\\siril\\scripts, Linux/macOS: ~/.siril/scripts)
     and run it from the Scripts menu. A settings dialog opens.

  B) From a command line, as long as Siril is running with its Python
     environment active:
     python siril_dark_calibration.py --work-dir "D:/astro/M31" --no-gui

Without arguments the GUI opens. Command line arguments pre-fill the form;
with --no-gui the script runs straight away, without a window.
"""

from __future__ import annotations

import argparse
import html
import os
import sys
import threading
from pathlib import Path

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


# Extensions of DSLR/mirrorless RAW files that Siril (libraw) can read.
RAW_EXTENSIONS = {
    "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "orf", "raf",
    "rw2", "pef", "ptx", "dng", "raw", "x3f", "mrw", "kdc", "dcr", "mef",
    "mos", "erf", "iiq", "3fr", "bay", "cap", "dcs", "drf", "fff", "rwl",
    "rwz", "srw",
}

LIGHT_DIR_CANDIDATES = ("lights", "light", "Lights", "Light")
DARK_DIR_CANDIDATES = ("darks", "dark", "Darks", "Dark")

# TIFF bit depth -> Siril command
TIFF_COMMANDS = {8: "savetif8", 16: "savetif", 32: "savetif32"}

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


# stretch mode -> label shown in the GUI
STRETCH_KEYS = ("none", "autostretch", "asinh")
STRETCH_LABELS = {
    "none": "None (linear, for stacking)",
    "autostretch": "Autostretch (viewing only)",
    "asinh": "Asinh (viewing only)",
}


# Siril's log colours, in a light and a dark variant so the embedded log stays
# readable whichever theme Siril is set to.
LOG_COLOURS = {
    "light": {"green": "#1b6e2b", "salmon": "#b34a20", "blue": "#14539a",
              "red": "#b3261e"},
    "dark": {"green": "#7fd18c", "salmon": "#ffb08f", "blue": "#7cb6f2",
             "red": "#ff9b94"},
}


class CalibrationError(RuntimeError):
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
    """Run a Siril command, raising CalibrationError with context on failure."""
    argv = [str(a) for a in args]
    printable = " ".join(argv)
    if not quiet:
        log(siril, "  > " + printable)
    try:
        siril.cmd(*argv)
    except Exception as exc:
        raise CalibrationError(
            "Command failed: " + printable + "\n    -> " + str(exc)
        ) from exc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt_num(value: float) -> str:
    """3.0 -> "3" (Siril expects plain numbers, integers without a decimal part)."""
    return str(int(value)) if float(value).is_integer() else str(value)


def quote(text: str) -> str:
    """Quote for Siril's parser if the path contains a space."""
    return '"' + text + '"' if " " in text else text


def siril_path(path: Path) -> str:
    """A path in the form Siril's parser understands (forward slashes, quotes)."""
    return quote(Path(path).as_posix())


def rel_posix(src: Path, dst: Path) -> str:
    """Unquoted relative path from src to dst; absolute if they are on different drives."""
    try:
        return Path(os.path.relpath(dst, src)).as_posix()
    except ValueError:
        return Path(dst).as_posix()


def resolve_dir(base: Path, explicit: str | None,
                candidates: tuple[str, ...], label: str) -> Path:
    """Find an input directory - either the given one, or the first known name."""
    if explicit:
        path = Path(explicit)
        path = path if path.is_absolute() else base / path
        if not path.is_dir():
            raise CalibrationError(
                "The " + label + " directory does not exist: " + str(path))
        return path.resolve()

    for name in candidates:
        path = base / name
        if path.is_dir():
            return path.resolve()

    # last attempt - case-insensitive scan of the working directory
    wanted = {c.lower() for c in candidates}
    if base.is_dir():
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and entry.name.lower() in wanted:
                return entry.resolve()

    raise CalibrationError(
        "No " + label + " subdirectory found in " + str(base)
        + " (looked for: " + ", ".join(candidates[:2])
        + "). Set the path manually."
    )


def list_raws(directory: Path) -> list[Path]:
    """RAW files in a directory, sorted the way Siril takes them (by name)."""
    return sorted(
        (p for p in directory.iterdir()
         if p.is_file() and p.suffix.lower().lstrip(".") in RAW_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


# ---------------------------------------------------------------------------
# individual steps
# ---------------------------------------------------------------------------

def convert_raws(siril, src_dir: Path, basename: str, process_dir: Path,
                 fitseq: bool = False) -> None:
    """Convert the RAW files in src_dir into the FITS sequence <basename>_ in process_dir.

    Deliberately without -debayer: both darks and lights must stay in their CFA
    form so the mosaic can be subtracted pixel by pixel. Debayering happens later,
    in the calibrate step.
    """
    run(siril, "cd", siril_path(src_dir))
    args = ["convertraw", basename,
            "-out=" + quote(rel_posix(src_dir, process_dir)), "-start=1"]
    if fitseq:
        args.append("-fitseq")
    try:
        run(siril, *args)
    except CalibrationError:
        # convertraw may not exist in every build - fall back to convert
        log(siril, "  convertraw failed, trying the generic convert command", "salmon")
        args[0] = "convert"
        run(siril, *args)


def build_master_dark(siril, process_dir: Path, dark_seq: str, master_name: str,
                      rejection: str, sigma_low: float, sigma_high: float,
                      n_darks: int, ext: str) -> Path:
    """Stack the master dark (without normalisation - darks are never normalised)."""
    run(siril, "cd", siril_path(process_dir))

    if n_darks == 1:
        log(siril, "  Only one dark frame - skipping stacking, using it directly.",
            "salmon")
        candidates = sorted(process_dir.glob(dark_seq + "_*" + ext))
        if not candidates:
            raise CalibrationError("Converting the dark frame produced no FITS file.")
        run(siril, "load", candidates[0].stem)
        run(siril, "save", master_name)
    else:
        run(siril, "stack", dark_seq, "rej", rejection,
            fmt_num(sigma_low), fmt_num(sigma_high), "-nonorm", "-out=" + master_name)

    master = process_dir / (master_name + ext)
    if not master.is_file():
        # fallback - Siril may have used a different extension (e.g. .fits)
        found = sorted(process_dir.glob(master_name + ".fit*"))
        if not found:
            raise CalibrationError("The master dark was not created: " + str(master))
        master = found[0]
    return master


def calibrate_lights(siril, process_dir: Path, light_seq: str, master_name: str,
                     prefix: str, cfa: bool, debayer: bool, cosmetic: bool,
                     cc_low: float, cc_high: float) -> None:
    """Subtract the master dark from every light frame."""
    run(siril, "cd", siril_path(process_dir))

    args = ["calibrate", light_seq, "-dark=" + master_name]
    if cosmetic:
        # detect hot/cold pixels from the master dark itself
        args += ["-cc=dark", fmt_num(cc_low), fmt_num(cc_high)]
    if cfa:
        args.append("-cfa")
    if debayer:
        args.append("-debayer")
    args.append("-prefix=" + prefix)

    run(siril, *args)


def tiff_names(frames: list[Path], raws: list[Path], basename: str,
               from_raw: bool, siril) -> list[str]:
    """Names of the output TIFF files (without extension)."""
    if from_raw:
        if len(raws) == len(frames):
            return [p.stem for p in raws]
        log(siril, "  RAW file count does not match the sequence frame count, "
                   "falling back to sequence numbering.", "salmon")
    return [basename + "_" + str(i).zfill(5) for i in range(1, len(frames) + 1)]


def report_stats(siril, label: str, stretched: bool) -> None:
    """Log statistics of the loaded frame and flag an over-subtracting dark.

    Calibrated frames are linear, so they look almost black in an ordinary image
    viewer - that is expected. A background pinned at zero is not: it means the
    dark removed more signal than the light frames contain.
    """
    try:
        stats = siril.get_image_stats(0)
    except Exception:
        return
    if stats is None:
        return

    # 32-bit float data is in [0, 1], 16-bit data in [0, 65535]
    try:
        max_value = float(getattr(stats, "max", 0) or 0)
    except (TypeError, ValueError):
        max_value = 0.0
    scale = 65535.0 if max_value > 1.5 else 1.0
    try:
        median = float(stats.median) / scale
        mean = float(stats.mean) / scale
        sigma = float(stats.sigma) / scale
    except Exception:
        return

    log(siril, "  " + label + ": median=" + ("%.5f" % median)
        + " (" + str(int(round(median * 65535))) + " ADU of 65535)"
        + ", mean=" + ("%.5f" % mean) + ", sigma=" + ("%.5f" % sigma))

    if median < 0.0005:
        log(siril, "  WARNING: the background sits at ~0, so the dark is probably "
                   "over-subtracting. Check that the darks match the lights in "
                   "exposure time, ISO/gain and sensor temperature; --pedestal "
                   "can keep the signal off the clipping point.", "salmon")
    elif not stretched:
        log(siril, "  NOTE: calibrated frames are LINEAR - they look dark in an "
                   "ordinary viewer, which is correct for registering and stacking. "
                   "Use --stretch for a viewable (non-linear) copy.", "salmon")


def export_tiff(siril, process_dir: Path, calibrated_dir: Path, seq_glob: str,
                names: list[str], bits: int, astro: bool, deflate: bool,
                stretch: str = "none", pedestal: float = 0.0,
                asinh_stretch: float = 100.0) -> int:
    """Load each calibrated frame and save it as TIFF into calibrated_dir.

    Siril has no command to export a whole sequence to TIFF, so this is done one
    frame at a time: load <frame> + savetif <path>.
    """
    frames = sorted(process_dir.glob(seq_glob))
    if not frames:
        raise CalibrationError(
            "Calibration produced no frames (" + seq_glob + " in "
            + str(process_dir) + ").")

    save_cmd = TIFF_COMMANDS[bits]
    options = []
    if astro:
        options.append("-astro")
    if deflate:
        options.append("-deflate")

    rel_out = rel_posix(process_dir, calibrated_dir)
    rel_back = rel_posix(calibrated_dir, process_dir)
    use_relative = True  # savetif ../calibrated/name; fallback = switching with cd

    run(siril, "cd", siril_path(process_dir))
    log(siril, "  exporting " + str(len(frames)) + " frames to TIFF ("
        + str(bits) + "-bit" + (", Astro-TIFF" if astro else "")
        + (", deflate" if deflate else "") + ")")

    if stretch != "none":
        log(siril, "  stretch: " + stretch + " (non-linear - for viewing only, "
                   "do not register or stack these files)", "salmon")
    if pedestal:
        log(siril, "  pedestal: +" + fmt_num(pedestal) + " ADU")

    for index, frame in enumerate(frames):
        name = names[index]
        run(siril, "load", frame.stem, quiet=True)

        if index == 0:
            report_stats(siril, "statistics of " + frame.stem, stretch != "none")

        # keep the signal off the clipping point, then optionally stretch
        if pedestal:
            run(siril, "offset", fmt_num(pedestal), quiet=True)
        if stretch == "autostretch":
            run(siril, "autostretch", "-linked", quiet=True)
        elif stretch == "asinh":
            run(siril, "asinh", fmt_num(asinh_stretch), quiet=True)

        if use_relative:
            try:
                run(siril, save_cmd, quote(rel_out + "/" + name), *options, quiet=True)
            except CalibrationError:
                # some builds reject a path inside the file name - use cd instead
                log(siril, "  Saving with a relative path failed, "
                           "switching into the target directory instead.", "salmon")
                use_relative = False

        if not use_relative:
            run(siril, "cd", siril_path(calibrated_dir), quiet=True)
            run(siril, save_cmd, quote(name), *options, quiet=True)
            run(siril, "cd", quote(rel_back), quiet=True)

        emit_progress(index + 1, len(frames))
        if (index + 1) % 10 == 0 or index + 1 == len(frames):
            log(siril, "    [" + str(index + 1) + "/" + str(len(frames)) + "] "
                + name + ".tif")

    return len(frames)


# ---------------------------------------------------------------------------
# the whole processing run
# ---------------------------------------------------------------------------

def run_pipeline(siril, args) -> int:
    """Run the complete processing. Raises CalibrationError on failure."""
    original_wd = None
    try:
        original_wd = siril.get_siril_wd()
    except Exception:
        original_wd = None

    try:
        # minimum Siril version
        run(siril, "requires", "1.4.0")

        base = Path(args.work_dir).resolve() if args.work_dir \
            else Path(original_wd or ".").resolve()
        if not base.is_dir():
            raise CalibrationError("Working directory does not exist: " + str(base))

        lights_dir = resolve_dir(base, args.lights, LIGHT_DIR_CANDIDATES, "lights")
        darks_dir = resolve_dir(base, args.darks, DARK_DIR_CANDIDATES, "darks")

        light_raws = list_raws(lights_dir)
        dark_raws = list_raws(darks_dir)
        if not light_raws:
            raise CalibrationError("No RAW files in " + str(lights_dir) + ".")
        if not dark_raws:
            raise CalibrationError("No RAW files in " + str(darks_dir) + ".")

        process_dir = Path(args.process)
        process_dir = process_dir if process_dir.is_absolute() else base / args.process
        process_dir.mkdir(parents=True, exist_ok=True)
        process_dir = process_dir.resolve()

        calibrated_dir = Path(args.calibrated)
        calibrated_dir = calibrated_dir if calibrated_dir.is_absolute() \
            else base / args.calibrated
        calibrated_dir.mkdir(parents=True, exist_ok=True)
        calibrated_dir = calibrated_dir.resolve()

        cfa = not args.mono
        debayer = args.debayer and not args.mono

        log(siril, "=" * 62, "green")
        log(siril, "Calibrating light frames with a master dark -> TIFF", "green")
        log(siril, "=" * 62, "green")
        log(siril, "  working directory : " + str(base))
        log(siril, "  lights            : " + str(lights_dir)
            + "  (" + str(len(light_raws)) + " RAW)")
        log(siril, "  darks             : " + str(darks_dir)
            + "  (" + str(len(dark_raws)) + " RAW)")
        log(siril, "  intermediates     : " + str(process_dir))
        log(siril, "  output (TIFF)     : " + str(calibrated_dir))
        log(siril, "  sensor            : " + ("mono" if args.mono else "OSC/CFA")
            + (", debayer during calibration" if debayer else ""))

        # shared settings
        run(siril, "setext", "fit")
        run(siril, "set32bits" if args.use32bits else "set16bits")

        # 1) master dark
        log(siril, "[1/4] Converting dark frames into a FITS sequence...", "blue")
        convert_raws(siril, darks_dir, args.dark_name, process_dir, args.dark_fitseq)

        log(siril, "[2/4] Stacking the master dark...", "blue")
        master = build_master_dark(
            siril, process_dir, args.dark_name, args.master_name,
            args.rejection, args.sigma_low, args.sigma_high, len(dark_raws), ".fit",
        )
        log(siril, "      master dark: " + str(master), "green")

        # 2) calibrate the lights
        log(siril, "[3/4] Converting and calibrating light frames...", "blue")
        convert_raws(siril, lights_dir, args.light_name, process_dir)
        calibrate_lights(
            siril, process_dir, args.light_name, master.stem, args.prefix,
            cfa, debayer, args.cosmetic, args.cc_sigma_low, args.cc_sigma_high,
        )

        # 3) export to TIFF
        log(siril, "[4/4] Saving the calibrated frames as TIFF...", "blue")
        seq_glob = args.prefix + args.light_name + "_*.fit*"
        frames = sorted(process_dir.glob(seq_glob))
        names = tiff_names(frames, light_raws, args.tiff_basename,
                           args.name_from_raw, siril)
        exported = export_tiff(
            siril, process_dir, calibrated_dir, seq_glob, names,
            args.tiff_bits, args.astro, args.deflate,
            args.stretch, args.pedestal, args.asinh_stretch,
        )

        log(siril, "-" * 62, "green")
        log(siril, "DONE. Frames calibrated and saved: " + str(exported), "green")
        log(siril, "  TIFF output : " + str(calibrated_dir), "green")
        log(siril, "  master dark : " + str(master), "green")
        log(siril, "-" * 62, "green")
        return exported

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

    class CalibrationWindow(QtWidgets.QWidget):
        """Settings dialog; the processing runs on its own thread.

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
            self.log_theme = "dark" if siril_is_dark(siril) else "light"

            self.setWindowTitle("Calibrate Lights with Master Dark")
            self._build_widgets(defaults)

            self.log_line.connect(self._append_log)
            self.progress_changed.connect(self._on_progress)
            self.run_finished.connect(self._finish)

            self._autofill_dirs()
            self._sync_sensor()
            self._sync_cosmetic()
            self._sync_naming()
            self._fit_to_screen()

        # -- layout ---------------------------------------------------------

        def _build_widgets(self, d) -> None:
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(8, 8, 8, 8)

            try:
                wd = self.siril.get_siril_wd() or ""
            except Exception:
                wd = ""

            # --- directories ---
            box = QtWidgets.QGroupBox("Directories")
            grid = QtWidgets.QGridLayout(box)
            self.ed_work = self._dir_row(
                grid, 0, "Working directory:", d.work_dir or wd,
                "Project directory; the lights/darks subdirectories are looked "
                "up inside it.")
            self.ed_work.editingFinished.connect(self._autofill_dirs)
            self.ed_lights = self._dir_row(
                grid, 1, "Light frames (RAW):", d.lights or "",
                "Empty = the 'lights' subdirectory is detected automatically.")
            self.ed_darks = self._dir_row(
                grid, 2, "Dark frames (RAW):", d.darks or "",
                "Empty = the 'darks' subdirectory is detected automatically.")
            self.ed_calibrated = self._dir_row(
                grid, 3, "Output (TIFF):", d.calibrated,
                "Name (or path) of the directory for the calibrated TIFF "
                "frames.", browse=False)
            self.ed_process = self._dir_row(
                grid, 4, "Intermediates:", d.process,
                "Directory for the FITS sequences and the master dark.",
                browse=False)
            outer.addWidget(box)

            # --- calibration ---
            box = QtWidgets.QGroupBox("Calibration")
            grid = QtWidgets.QGridLayout(box)

            grid.addWidget(QtWidgets.QLabel("Sensor:"), 0, 0)
            self.cmb_sensor = QtWidgets.QComboBox()
            self.cmb_sensor.addItems(["OSC / colour (CFA)", "Mono"])
            self.cmb_sensor.setCurrentIndex(1 if d.mono else 0)
            self.cmb_sensor.setToolTip(
                "Mono disables both CFA cosmetic correction and debayering.")
            self.cmb_sensor.currentIndexChanged.connect(self._sync_sensor)
            grid.addWidget(self.cmb_sensor, 0, 1)

            self.chk_debayer = QtWidgets.QCheckBox("Debayer during calibration")
            self.chk_debayer.setChecked(d.debayer)
            self.chk_debayer.setToolTip(
                "Debayer only after the dark is subtracted - darks and lights "
                "must stay in their CFA form.")
            grid.addWidget(self.chk_debayer, 0, 2, 1, 2)

            self.chk_cosmetic = QtWidgets.QCheckBox(
                "Cosmetic correction from master dark")
            self.chk_cosmetic.setChecked(d.cosmetic)
            self.chk_cosmetic.setToolTip(
                "calibrate -cc=dark: hot/cold pixel map taken from the dark.")
            self.chk_cosmetic.toggled.connect(self._sync_cosmetic)
            grid.addWidget(self.chk_cosmetic, 1, 0, 1, 2)

            grid.addWidget(QtWidgets.QLabel("sigma low / high:"), 1, 2)
            self.ed_cc_low = self._small(fmt_num(d.cc_sigma_low))
            self.ed_cc_high = self._small(fmt_num(d.cc_sigma_high))
            grid.addLayout(self._pair(self.ed_cc_low, self.ed_cc_high), 1, 3)

            grid.addWidget(QtWidgets.QLabel("Dark stacking rejection:"), 2, 0)
            self.cmb_rejection = QtWidgets.QComboBox()
            self.cmb_rejection.addItems([name for name, _c in REJECTIONS])
            self.cmb_rejection.setToolTip(
                "Winsorized is a good choice for a typical number of darks.")
            grid.addWidget(self.cmb_rejection, 2, 1)

            grid.addWidget(QtWidgets.QLabel("sigma low / high:"), 2, 2)
            self.ed_sigma_low = self._small(fmt_num(d.sigma_low))
            self.ed_sigma_high = self._small(fmt_num(d.sigma_high))
            grid.addLayout(self._pair(self.ed_sigma_low, self.ed_sigma_high),
                           2, 3)
            grid.setColumnStretch(1, 1)
            outer.addWidget(box)

            # --- TIFF output ---
            box = QtWidgets.QGroupBox("TIFF output")
            grid = QtWidgets.QGridLayout(box)

            grid.addWidget(QtWidgets.QLabel("Bit depth:"), 0, 0)
            self.cmb_bits = QtWidgets.QComboBox()
            self.cmb_bits.addItems(["8", "16", "32"])
            self.cmb_bits.setCurrentText(str(d.tiff_bits))
            self.cmb_bits.setToolTip(
                "16-bit is the usual compromise, 32-bit keeps full precision.")
            grid.addWidget(self.cmb_bits, 0, 1)

            self.chk_astro = QtWidgets.QCheckBox(
                "Astro-TIFF (FITS header inside the TIFF)")
            self.chk_astro.setChecked(d.astro)
            grid.addWidget(self.chk_astro, 0, 2)
            self.chk_deflate = QtWidgets.QCheckBox("Compression (deflate)")
            self.chk_deflate.setChecked(d.deflate)
            grid.addWidget(self.chk_deflate, 0, 3)

            grid.addWidget(QtWidgets.QLabel("Naming:"), 1, 0)
            self.cmb_naming = QtWidgets.QComboBox()
            self.cmb_naming.addItems(["Sequence numbering",
                                      "From original RAW names"])
            self.cmb_naming.setCurrentIndex(1 if d.name_from_raw else 0)
            self.cmb_naming.setToolTip(
                "Naming from RAW files assumes conversion in alphabetical "
                "order.")
            self.cmb_naming.currentIndexChanged.connect(self._sync_naming)
            grid.addWidget(self.cmb_naming, 1, 1)

            grid.addWidget(QtWidgets.QLabel("Base name:"), 1, 2)
            self.ed_basename = QtWidgets.QLineEdit(d.tiff_basename)
            grid.addWidget(self.ed_basename, 1, 3)

            grid.addWidget(QtWidgets.QLabel("Stretch:"), 2, 0)
            self.cmb_stretch = QtWidgets.QComboBox()
            self.cmb_stretch.addItems([STRETCH_LABELS[k] for k in STRETCH_KEYS])
            self.cmb_stretch.setCurrentText(STRETCH_LABELS[d.stretch])
            self.cmb_stretch.setToolTip(
                "Calibrated data is linear and looks dark in an ordinary "
                "viewer. A stretch makes it viewable but non-linear - never "
                "register or stack stretched files.")
            grid.addWidget(self.cmb_stretch, 2, 1)

            grid.addWidget(QtWidgets.QLabel("Pedestal (ADU):"), 2, 2)
            self.ed_pedestal = QtWidgets.QLineEdit(fmt_num(d.pedestal))
            grid.addWidget(self.ed_pedestal, 2, 3)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            outer.addWidget(box)

            outer.addWidget(self._log_box(), 1)

            buttons = QtWidgets.QHBoxLayout()
            buttons.addStretch(1)
            btn = QtWidgets.QPushButton("Close")
            btn.clicked.connect(self.close)
            buttons.addWidget(btn)
            self.btn_run = QtWidgets.QPushButton("Run calibration")
            self.btn_run.setDefault(True)
            self.btn_run.clicked.connect(self._start)
            buttons.addWidget(self.btn_run)
            outer.addLayout(buttons)

        def _small(self, value):
            edit = QtWidgets.QLineEdit(value)
            edit.setFixedWidth(52)
            return edit

        def _pair(self, first, second):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(first)
            row.addWidget(second)
            row.addStretch(1)
            return row

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
            start = edit.text().strip() or self.ed_work.text().strip() \
                or os.getcwd()
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select a directory", start)
            if chosen:
                edit.setText(os.path.normpath(chosen))
                if edit is self.ed_work:
                    self.ed_lights.clear()
                    self.ed_darks.clear()
                    self._autofill_dirs()

        def _autofill_dirs(self) -> None:
            """Pre-fill lights/darks if they can be found in the working dir."""
            text = self.ed_work.text().strip()
            base = Path(text) if text else None
            if base is None or not base.is_dir():
                return
            for edit, candidates, label in (
                    (self.ed_lights, LIGHT_DIR_CANDIDATES, "lights"),
                    (self.ed_darks, DARK_DIR_CANDIDATES, "darks")):
                if edit.text().strip():
                    continue
                try:
                    edit.setText(str(resolve_dir(base, None, candidates, label)))
                except CalibrationError:
                    pass

        def _sync_sensor(self) -> None:
            self.chk_debayer.setEnabled(self.cmb_sensor.currentIndex() == 0)

        def _sync_cosmetic(self) -> None:
            on = self.chk_cosmetic.isChecked()
            self.ed_cc_low.setEnabled(on)
            self.ed_cc_high.setEnabled(on)

        def _sync_naming(self) -> None:
            self.ed_basename.setEnabled(self.cmb_naming.currentIndex() == 0)

        # -- starting the processing ----------------------------------------

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
                args.pedestal = number(self.ed_pedestal, "Pedestal")
            except ValueError as exc:
                self._error(str(exc))
                return None

            args.work_dir = work
            args.lights = self.ed_lights.text().strip() or None
            args.darks = self.ed_darks.text().strip() or None
            args.process = self.ed_process.text().strip() or "process"
            args.calibrated = self.ed_calibrated.text().strip() or "calibrated"

            args.mono = self.cmb_sensor.currentIndex() == 1
            args.debayer = self.chk_debayer.isChecked() and not args.mono
            args.cosmetic = self.chk_cosmetic.isChecked()
            args.rejection = REJECTIONS[self.cmb_rejection.currentIndex()][1]

            args.tiff_bits = int(self.cmb_bits.currentText())
            args.astro = self.chk_astro.isChecked()
            args.deflate = self.chk_deflate.isChecked()
            args.name_from_raw = self.cmb_naming.currentIndex() == 1
            args.tiff_basename = self.ed_basename.text().strip() or "light"
            args.stretch = STRETCH_KEYS[self.cmb_stretch.currentIndex()]
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
                self.run_finished.emit(run_pipeline(self.siril, args), None)
            except CalibrationError as exc:
                self.run_finished.emit(0, str(exc))
            except Exception as exc:
                self.run_finished.emit(
                    0, exc.__class__.__name__ + ": " + str(exc))

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

        def _finish(self, count, error) -> None:
            self.running = False
            self.btn_run.setEnabled(True)
            if error:
                self.lbl_status.setText("Processing failed.")
                self.text.appendPlainText("ERROR: " + error)
                QtWidgets.QMessageBox.critical(self, "Calibration failed",
                                               error)
                return
            self.lbl_status.setText("Done - frames saved: " + str(count))
            self.progress.setValue(100)
            QtWidgets.QMessageBox.information(
                self, "Done", "Frames calibrated and saved: " + str(count))

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
        raise CalibrationError(
            "PyQt6 is not available in this Python environment.")

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])
    apply_siril_theme(app, siril)

    window = CalibrationWindow(siril, defaults)
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
        description="Build a master dark, calibrate the light frames with it and "
                    "save them as TIFF into the 'calibrated' directory (Siril 1.4.4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-gui", dest="gui", action="store_false",
                   help="Skip the dialog and start processing right away.")
    p.add_argument("--work-dir", default=None,
                   help="Working directory; defaults to Siril's working directory.")
    p.add_argument("--lights", default=None,
                   help="Directory with the light RAW frames (default 'lights').")
    p.add_argument("--darks", default=None,
                   help="Directory with the dark RAW frames (default 'darks').")
    p.add_argument("--process", default="process",
                   help="Directory for intermediates (FITS sequences, master dark).")
    p.add_argument("--calibrated", default="calibrated",
                   help="Output directory for the calibrated TIFF frames.")
    p.add_argument("--light-name", default="light",
                   help="Base name of the light sequence.")
    p.add_argument("--dark-name", default="dark",
                   help="Base name of the dark sequence.")
    p.add_argument("--master-name", default="master_dark",
                   help="File name of the master dark.")
    p.add_argument("--prefix", default="pp_",
                   help="Prefix of the calibrated FITS sequence in the process directory.")

    p.add_argument("--tiff-bits", type=int, default=16, choices=[8, 16, 32],
                   help="Bit depth of the output TIFF (16 = savetif, 32 = float).")
    p.add_argument("--tiff-basename", default="light",
                   help="Base name of the output TIFF files (light_00001.tif ...).")
    p.add_argument("--name-from-raw", action="store_true",
                   help="Name the TIFFs after the original RAW files instead of numbering.")
    p.add_argument("--no-astro-tiff", dest="astro", action="store_false",
                   help="Do not use the Astro-TIFF format (no FITS header in the TIFF).")
    p.add_argument("--deflate", action="store_true",
                   help="Enable lossless TIFF compression.")
    p.add_argument("--stretch", default="none",
                   choices=["none", "autostretch", "asinh"],
                   help="Stretch the frames before saving. Calibrated data is "
                        "linear and looks dark; a stretch makes it viewable but "
                        "NON-LINEAR, so do not register or stack such files.")
    p.add_argument("--asinh-stretch", type=float, default=100.0,
                   help="Stretch factor for --stretch asinh.")
    p.add_argument("--pedestal", type=float, default=0.0,
                   help="Constant added to every frame after calibration (ADU), "
                        "so a slightly over-subtracting dark does not clip at 0.")

    p.add_argument("--rejection", default="w",
                   choices=["p", "s", "m", "w", "l", "g", "a", "n"],
                   help="Rejection type for dark stacking "
                        "(w = Winsorized sigma clipping, n = none).")
    p.add_argument("--sigma-low", type=float, default=3.0, help="Low sigma for rejection.")
    p.add_argument("--sigma-high", type=float, default=3.0, help="High sigma for rejection.")

    p.add_argument("--mono", action="store_true",
                   help="Monochrome sensor - disables both -cfa and debayering.")
    p.add_argument("--no-debayer", dest="debayer", action="store_false",
                   help="Do not debayer during calibration (output stays CFA).")
    p.add_argument("--no-cosmetic", dest="cosmetic", action="store_false",
                   help="Disable hot/cold pixel cosmetic correction.")
    p.add_argument("--cc-sigma-low", type=float, default=3.0,
                   help="Sigma for cold pixels (cosmetic correction).")
    p.add_argument("--cc-sigma-high", type=float, default=3.0,
                   help="Sigma for hot pixels (cosmetic correction).")

    p.add_argument("--dark-fitseq", action="store_true",
                   help="Store the dark sequence as a single FITS file "
                        "(the light sequence must stay per-frame for the TIFF export).")
    p.add_argument("--16bit", dest="use32bits", action="store_false",
                   help="Work in 16-bit mode (32-bit float is the default).")
    p.set_defaults(gui=True, debayer=True, cosmetic=True, use32bits=True, astro=True)
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
        if args.gui:
            try:
                return launch_gui(siril, args)
            except Exception as exc:
                log(siril, "Could not start the GUI (" + str(exc)
                    + "), continuing in text mode.", "salmon")

        run_pipeline(siril, args)
        return 0

    except CalibrationError as exc:
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
