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

try:
    from sirilpy import tksiril  # matches the GUI to Siril's theme (1.4.x)
except ImportError:
    tksiril = None


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


class CalibrationGUI:
    """Settings dialog; the processing runs on its own thread."""

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

        root.title("Calibrate Lights with Master Dark")
        root.minsize(700, 640)

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
        self.var_darks = tk.StringVar(value=d.darks or "")
        self.var_process = tk.StringVar(value=d.process)
        self.var_calibrated = tk.StringVar(value=d.calibrated)

        self.var_sensor = tk.StringVar(value="Mono" if d.mono else "OSC / colour (CFA)")
        self.var_debayer = tk.BooleanVar(value=d.debayer)
        self.var_cosmetic = tk.BooleanVar(value=d.cosmetic)
        self.var_cc_low = tk.StringVar(value=fmt_num(d.cc_sigma_low))
        self.var_cc_high = tk.StringVar(value=fmt_num(d.cc_sigma_high))

        self.var_rejection = tk.StringVar(value=REJECTIONS[0][0])
        self.var_sigma_low = tk.StringVar(value=fmt_num(d.sigma_low))
        self.var_sigma_high = tk.StringVar(value=fmt_num(d.sigma_high))

        self.var_bits = tk.StringVar(value=str(d.tiff_bits))
        self.var_astro = tk.BooleanVar(value=d.astro)
        self.var_deflate = tk.BooleanVar(value=d.deflate)
        self.var_naming = tk.StringVar(
            value="From original RAW names" if d.name_from_raw
            else "Sequence numbering")
        self.var_basename = tk.StringVar(value=d.tiff_basename)
        self.var_stretch = tk.StringVar(value=STRETCH_LABELS[d.stretch])
        self.var_pedestal = tk.StringVar(value=fmt_num(d.pedestal))

        self.var_status = tk.StringVar(value="Ready.")

        self._build_widgets()
        self._autofill_dirs()
        self.pump_id = self.root.after(100, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout -------------------------------------------------------------

    def _build_widgets(self) -> None:
        ttk, tk = self.ttk, self.tk
        from tkinter import scrolledtext

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # --- directories ---
        box = ttk.LabelFrame(main, text="Directories", padding=8)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)

        self._dir_row(box, 0, "Working directory:", self.var_work, True,
                      "Project directory; the lights/darks subdirectories are "
                      "looked up inside it.")
        self._dir_row(box, 1, "Light frames (RAW):", self.var_lights, True,
                      "Empty = the 'lights' subdirectory is detected automatically.")
        self._dir_row(box, 2, "Dark frames (RAW):", self.var_darks, True,
                      "Empty = the 'darks' subdirectory is detected automatically.")
        self._entry_row(box, 3, "Output (TIFF):", self.var_calibrated,
                        "Name (or path) of the directory for the calibrated TIFF frames.")
        self._entry_row(box, 4, "Intermediates:", self.var_process,
                        "Directory for the FITS sequences and the master dark.")

        # --- calibration ---
        box = ttk.LabelFrame(main, text="Calibration", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)
        box.columnconfigure(3, weight=1)

        ttk.Label(box, text="Sensor:").grid(row=0, column=0, sticky="w", pady=2)
        sensor = ttk.Combobox(box, textvariable=self.var_sensor, state="readonly",
                              values=["OSC / colour (CFA)", "Mono"])
        sensor.grid(row=0, column=1, sticky="ew", padx=(6, 12), pady=2)
        sensor.bind("<<ComboboxSelected>>", lambda _e: self._sync_sensor())
        self._tip(sensor, "Mono disables both CFA cosmetic correction and debayering.")

        self.chk_debayer = ttk.Checkbutton(
            box, text="Debayer during calibration", variable=self.var_debayer)
        self.chk_debayer.grid(row=0, column=2, columnspan=2, sticky="w", pady=2)
        self._tip(self.chk_debayer,
                  "Debayer only after the dark is subtracted - darks and lights "
                  "must stay in their CFA form.")

        chk = ttk.Checkbutton(box, text="Cosmetic correction from master dark",
                              variable=self.var_cosmetic, command=self._sync_cosmetic)
        chk.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        self._tip(chk, "calibrate -cc=dark: hot/cold pixel map taken from the dark.")

        ttk.Label(box, text="sigma low / high:").grid(row=1, column=2, sticky="e", pady=2)
        frame = ttk.Frame(box)
        frame.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=2)
        self.ent_cc_low = ttk.Entry(frame, textvariable=self.var_cc_low, width=6)
        self.ent_cc_low.pack(side="left")
        self.ent_cc_high = ttk.Entry(frame, textvariable=self.var_cc_high, width=6)
        self.ent_cc_high.pack(side="left", padx=(4, 0))

        ttk.Label(box, text="Dark stacking rejection:").grid(
            row=2, column=0, sticky="w", pady=2)
        combo = ttk.Combobox(box, textvariable=self.var_rejection, state="readonly",
                             values=[name for name, _code in REJECTIONS])
        combo.grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=2)
        self._tip(combo, "Winsorized is a good choice for a typical number of darks.")

        ttk.Label(box, text="sigma low / high:").grid(row=2, column=2, sticky="e", pady=2)
        frame = ttk.Frame(box)
        frame.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=2)
        ttk.Entry(frame, textvariable=self.var_sigma_low, width=6).pack(side="left")
        ttk.Entry(frame, textvariable=self.var_sigma_high, width=6).pack(
            side="left", padx=(4, 0))

        # --- output ---
        box = ttk.LabelFrame(main, text="TIFF output", padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)
        box.columnconfigure(3, weight=1)

        ttk.Label(box, text="Bit depth:").grid(row=0, column=0, sticky="w", pady=2)
        bits = ttk.Combobox(box, textvariable=self.var_bits, state="readonly",
                            width=8, values=["8", "16", "32"])
        bits.grid(row=0, column=1, sticky="w", padx=(6, 12), pady=2)
        self._tip(bits, "16-bit is the usual compromise, 32-bit keeps full precision.")

        chk = ttk.Checkbutton(box, text="Astro-TIFF (FITS header inside the TIFF)",
                              variable=self.var_astro)
        chk.grid(row=0, column=2, sticky="w", pady=2)
        chk = ttk.Checkbutton(box, text="Compression (deflate)", variable=self.var_deflate)
        chk.grid(row=0, column=3, sticky="w", pady=2)

        ttk.Label(box, text="Naming:").grid(row=1, column=0, sticky="w", pady=2)
        naming = ttk.Combobox(box, textvariable=self.var_naming, state="readonly",
                              values=["Sequence numbering", "From original RAW names"])
        naming.grid(row=1, column=1, sticky="ew", padx=(6, 12), pady=2)
        naming.bind("<<ComboboxSelected>>", lambda _e: self._sync_naming())
        self._tip(naming, "Naming from RAW files assumes conversion in alphabetical order.")

        ttk.Label(box, text="Base name:").grid(row=1, column=2, sticky="e", pady=2)
        self.ent_basename = ttk.Entry(box, textvariable=self.var_basename, width=16)
        self.ent_basename.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=2)

        ttk.Label(box, text="Stretch:").grid(row=2, column=0, sticky="w", pady=2)
        stretch = ttk.Combobox(box, textvariable=self.var_stretch, state="readonly",
                               values=[STRETCH_LABELS[key] for key in STRETCH_KEYS])
        stretch.grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=2)
        self._tip(stretch,
                  "Calibrated data is linear and looks dark in an ordinary viewer. "
                  "A stretch makes it viewable but non-linear - never register or "
                  "stack stretched files.")

        ttk.Label(box, text="Pedestal (ADU):").grid(row=2, column=2, sticky="e", pady=2)
        ttk.Entry(box, textvariable=self.var_pedestal, width=16).grid(
            row=2, column=3, sticky="w", padx=(6, 0), pady=2)

        # --- progress ---
        box = ttk.LabelFrame(main, text="Progress", padding=8)
        box.grid(row=3, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(2, weight=1)
        main.rowconfigure(3, weight=1)

        ttk.Label(box, textvariable=self.var_status).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(box, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(4, 6))
        self.text = scrolledtext.ScrolledText(box, height=12, wrap="none")
        self.text.grid(row=2, column=0, sticky="nsew")
        self.text.configure(state="disabled")

        # --- buttons ---
        buttons = ttk.Frame(main, padding=(0, 8, 0, 0))
        buttons.grid(row=4, column=0, sticky="ew")
        self.btn_run = ttk.Button(buttons, text="Run calibration", command=self._start)
        self.btn_run.pack(side="right")
        self.btn_close = ttk.Button(buttons, text="Close", command=self._on_close)
        self.btn_close.pack(side="right", padx=(0, 6))

        self._sync_sensor()
        self._sync_cosmetic()
        self._sync_naming()

    def _dir_row(self, parent, row, label, var, browse, tip) -> None:
        ttk = self.ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        self._tip(entry, tip)
        if browse:
            ttk.Button(parent, text="...", width=3,
                       command=lambda v=var: self._browse(v)).grid(row=row, column=2)

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
            var.set(chosen)
            if var is self.var_work:
                self.var_lights.set("")
                self.var_darks.set("")
                self._autofill_dirs()

    def _autofill_dirs(self) -> None:
        """Pre-fill lights/darks if they can be found in the working directory."""
        base = Path(self.var_work.get()) if self.var_work.get() else None
        if not base or not base.is_dir():
            return
        for var, candidates, label in ((self.var_lights, LIGHT_DIR_CANDIDATES, "lights"),
                                       (self.var_darks, DARK_DIR_CANDIDATES, "darks")):
            if var.get():
                continue
            try:
                found = resolve_dir(base, None, candidates, label)
                var.set(str(found))
            except CalibrationError:
                pass

    def _sync_sensor(self) -> None:
        mono = self.var_sensor.get() == "Mono"
        self.chk_debayer.state(["disabled"] if mono else ["!disabled"])

    def _sync_cosmetic(self) -> None:
        state = "normal" if self.var_cosmetic.get() else "disabled"
        self.ent_cc_low.configure(state=state)
        self.ent_cc_high.configure(state=state)

    def _sync_naming(self) -> None:
        from_raw = self.var_naming.get().startswith("From")
        self.ent_basename.configure(state="disabled" if from_raw else "normal")

    # -- starting the processing --------------------------------------------

    def _collect(self):
        """Build the same settings object argparse produces, from the form."""
        from tkinter import messagebox

        args = parse_args([])
        work = self.var_work.get().strip()
        if not work or not Path(work).is_dir():
            messagebox.showerror("Error", "The working directory does not exist.",
                                 parent=self.root)
            return None

        def number(var, label, minimum=0.0):
            try:
                value = float(var.get().replace(",", "."))
            except ValueError:
                raise ValueError(label + ": enter a number.")
            if value < minimum:
                raise ValueError(label + ": the value must be >= " + fmt_num(minimum))
            return value

        try:
            args.sigma_low = number(self.var_sigma_low, "Sigma low (rejection)")
            args.sigma_high = number(self.var_sigma_high, "Sigma high (rejection)")
            args.cc_sigma_low = number(self.var_cc_low, "Sigma low (cosmetic correction)")
            args.cc_sigma_high = number(self.var_cc_high,
                                        "Sigma high (cosmetic correction)")
            args.pedestal = number(self.var_pedestal, "Pedestal")
        except ValueError as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)
            return None

        args.work_dir = work
        args.lights = self.var_lights.get().strip() or None
        args.darks = self.var_darks.get().strip() or None
        args.process = self.var_process.get().strip() or "process"
        args.calibrated = self.var_calibrated.get().strip() or "calibrated"

        args.mono = self.var_sensor.get() == "Mono"
        args.debayer = self.var_debayer.get() and not args.mono
        args.cosmetic = self.var_cosmetic.get()
        args.rejection = dict((name, code) for name, code in REJECTIONS)[
            self.var_rejection.get()]

        args.tiff_bits = int(self.var_bits.get())
        args.astro = self.var_astro.get()
        args.deflate = self.var_deflate.get()
        args.name_from_raw = self.var_naming.get().startswith("From")
        args.tiff_basename = self.var_basename.get().strip() or "light"
        args.stretch = dict((label, key)
                            for key, label in STRETCH_LABELS.items())[
                                self.var_stretch.get()]
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
            count = run_pipeline(self.siril, args)
            self.queue.put(("done", count, None))
        except CalibrationError as exc:
            self.queue.put(("done", 0, str(exc)))
        except Exception as exc:
            self.queue.put(("done", 0, exc.__class__.__name__ + ": " + str(exc)))

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

    def _finish(self, count, error) -> None:
        from tkinter import messagebox
        self.running = False
        self.btn_run.state(["!disabled"])
        if error:
            self.var_status.set("Processing failed.")
            self._append_log("ERROR: " + error)
            messagebox.showerror("Calibration failed", error, parent=self.root)
        else:
            self.var_status.set("Done - frames saved: " + str(count))
            self.progress.configure(value=100)
            messagebox.showinfo(
                "Done",
                "Frames calibrated and saved: " + str(count),
                parent=self.root)

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
    gui = CalibrationGUI(root, siril, defaults)
    add_log_sink(gui.sink_log)
    add_progress_sink(gui.sink_progress)
    root.mainloop()
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
