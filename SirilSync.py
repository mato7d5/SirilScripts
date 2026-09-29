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

import html
import os
import re
import sys
import threading

import sirilpy as s

# PyQt6 is the Qt binding that ships in Siril's own Python environment
# (Siril 1.4 bundles PyQt6 6.11 / Qt 6.11).
try:
    from PyQt6 import QtCore, QtGui, QtWidgets   # noqa: E402
except ImportError:
    s.ensure_installed("PyQt6")
    from PyQt6 import QtCore, QtGui, QtWidgets   # noqa: E402


# Siril's log colours, in a light and a dark variant so the embedded log stays
# readable whichever theme Siril is set to.
LOG_COLOURS = {
    "light": {"green": "#1b6e2b", "salmon": "#b34a20", "blue": "#14539a",
              "red": "#b3261e"},
    "dark": {"green": "#7fd18c", "salmon": "#ffb08f", "blue": "#7cb6f2",
             "red": "#ff9b94"},
}


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

# ---------------------------------------------------------------------------
# GUI (PyQt6 - the Qt binding that ships in Siril's Python environment)
# ---------------------------------------------------------------------------

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


class ChangeRow(QtWidgets.QWidget):
    """One history entry: a checkbox, the entry text, and its command."""

    def __init__(self, number, text, command, level, note, pre_existing):
        super().__init__()
        self.level = level
        self.note = note
        self.pre_existing = pre_existing

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setVerticalSpacing(2)

        self.check = QtWidgets.QCheckBox()
        self.check.setChecked(bool(command) and not pre_existing)
        layout.addWidget(self.check, 0, 0, 2, 1,
                         QtCore.Qt.AlignmentFlag.AlignTop)

        if pre_existing:
            tag = "   [already stored in the file]"
        elif level == PARTIAL:
            tag = "   [incomplete]"
        elif level == NONE:
            tag = "   [no command]"
        else:
            tag = ""
        label = QtWidgets.QLabel("%d. %s%s" % (number, text, tag))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("command:"))
        self.entry = QtWidgets.QLineEdit(command)
        self.entry.textChanged.connect(self._on_command_change)
        row.addWidget(self.entry, 1)
        layout.addLayout(row, 1, 1)

        if note:
            hint = QtWidgets.QLabel(note)
            hint.setWordWrap(True)
            layout.addWidget(hint, 2, 1)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(line, 3, 0, 1, 2)
        layout.setColumnStretch(1, 1)

        self._on_command_change()

    def _on_command_change(self, *_args) -> None:
        if self.entry.text().strip():
            self.check.setEnabled(True)
        else:
            self.check.setChecked(False)
            self.check.setEnabled(False)

    def command(self) -> str:
        return self.entry.text().strip()

    def set_enabled(self, value: bool) -> None:
        if value and not self.command():
            return
        self.check.setChecked(value)

    def is_active(self) -> bool:
        return self.check.isChecked() and bool(self.command())


class SirilSyncWindow(QtWidgets.QWidget):
    """Main window; the sync runs on its own thread.

    The worker reports back through Qt signals, which Qt delivers on the GUI
    thread, so no widget is touched from the wrong thread.
    """

    log_line = QtCore.pyqtSignal(str, object)
    status_changed = QtCore.pyqtSignal(str)
    progress_max = QtCore.pyqtSignal(int)
    progress_changed = QtCore.pyqtSignal(int)
    run_finished = QtCore.pyqtSignal(int, int)

    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.rows = []
        self.source_path = None
        self.worker = None
        self.cancel = threading.Event()
        self.log_theme = "dark" if siril_is_dark(siril) else "light"

        self.setWindowTitle("SirilSync")
        self._build_widgets()

        self.log_line.connect(self._append_log)
        self.status_changed.connect(self.status.setText)
        self.progress_max.connect(self._set_maximum)
        self.progress_changed.connect(self.progress.setValue)
        self.run_finished.connect(self.on_worker_finished)

        self.load_history()
        self.update_output_state()
        self._fit_to_screen()

    # -- layout -------------------------------------------------------------

    def _build_widgets(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        self.source_label = QtWidgets.QLabel("")
        self.source_label.setWordWrap(True)
        outer.addWidget(self.source_label)

        # --- target ---
        box = QtWidgets.QGroupBox("Images to process")
        grid = QtWidgets.QGridLayout(box)
        grid.addWidget(QtWidgets.QLabel("Folder:"), 0, 0)
        self.ed_folder = QtWidgets.QLineEdit()
        self.ed_folder.textChanged.connect(self.refresh_count)
        grid.addWidget(self.ed_folder, 0, 1)
        btn = QtWidgets.QPushButton("Browse...")
        btn.clicked.connect(self.pick_folder)
        grid.addWidget(btn, 0, 2)

        self.chk_recursive = QtWidgets.QCheckBox("Include subfolders")
        self.chk_recursive.toggled.connect(self.refresh_count)
        grid.addWidget(self.chk_recursive, 1, 1)

        self.count_label = QtWidgets.QLabel("No folder selected.")
        grid.addWidget(self.count_label, 2, 1)
        grid.setColumnStretch(1, 1)
        outer.addWidget(box)

        # --- changes ---
        box = QtWidgets.QGroupBox("Changes to apply")
        inner = QtWidgets.QVBoxLayout(box)
        bar = QtWidgets.QHBoxLayout()
        for text, slot in (("Select all", lambda: self.select_all(True)),
                           ("Select none", lambda: self.select_all(False)),
                           ("Reload from image", self.load_history)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            bar.addWidget(button)
        bar.addStretch(1)
        inner.addLayout(bar)

        self.scroller = QtWidgets.QScrollArea()
        self.scroller.setWidgetResizable(True)
        self.list_frame = QtWidgets.QWidget()
        self.list_layout = QtWidgets.QVBoxLayout(self.list_frame)
        self.list_layout.setContentsMargins(4, 4, 4, 4)
        self.list_layout.addStretch(1)
        self.scroller.setWidget(self.list_frame)
        self.scroller.setMinimumHeight(180)
        inner.addWidget(self.scroller, 1)
        outer.addWidget(box, 1)

        # --- output ---
        box = QtWidgets.QGroupBox("Output")
        grid = QtWidgets.QGridLayout(box)
        self.chk_overwrite = QtWidgets.QCheckBox("Overwrite the original images")
        self.chk_overwrite.toggled.connect(self.update_output_state)
        grid.addWidget(self.chk_overwrite, 0, 0, 1, 3)
        self.out_caption = QtWidgets.QLabel("Save to:")
        grid.addWidget(self.out_caption, 1, 0)
        self.ed_output = QtWidgets.QLineEdit()
        grid.addWidget(self.ed_output, 1, 1)
        self.out_button = QtWidgets.QPushButton("Browse...")
        self.out_button.clicked.connect(self.pick_output)
        grid.addWidget(self.out_button, 1, 2)
        grid.setColumnStretch(1, 1)
        outer.addWidget(box)

        # --- progress ---
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        outer.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Ready.")
        outer.addWidget(self.status)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setMinimumHeight(110)
        outer.addWidget(self.text)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.close)
        buttons.addWidget(btn)
        self.sync_button = QtWidgets.QPushButton("Sync")
        self.sync_button.setDefault(True)
        self.sync_button.clicked.connect(self.on_sync)
        buttons.addWidget(self.sync_button)
        outer.addLayout(buttons)

    def _fit_to_screen(self) -> None:
        """Size the window to its content, never larger than the screen."""
        available = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        hint = self.sizeHint()
        width = min(max(hint.width(), 640), int(available.width() * 0.92))
        height = min(max(hint.height(), 520), int(available.height() * 0.85))
        self.setMinimumSize(min(620, width), min(460, height))
        self.resize(width, height)
        self.move(available.x() + (available.width() - width) // 2,
                  available.y() + (available.height() - height) // 3)

    # -- history ------------------------------------------------------------

    def _clear_rows(self) -> None:
        for row in self.rows:
            row.setParent(None)
            row.deleteLater()
        self.rows = []
        while self.list_layout.count() > 1:      # keep the trailing stretch
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def load_history(self) -> None:
        self._clear_rows()

        try:
            if not self.siril.is_image_loaded():
                self.source_label.setText(
                    "No image is loaded in Siril. Load the image you have "
                    "processed, then press \"Reload from image\".")
                return
            filename = self.siril.get_image_filename()
            history = self.siril.get_image_history() or []
        except s.SirilError as exc:
            self.source_label.setText(
                "Could not read the loaded image: %s" % exc)
            return

        self.source_path = os.path.abspath(filename) if filename else None
        self.source_label.setText(
            "Source image: %s" % (filename or "(unsaved image)"))

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
            label = QtWidgets.QLabel("This image has no HISTORY entries.")
            self.list_layout.insertWidget(self.list_layout.count() - 1, label)
            return

        for index, entry in enumerate(history):
            command, level, note = translate(entry)
            row = ChangeRow(index + 1, entry, command, level, note,
                            pre_existing=index < baseline_len)
            self.list_layout.insertWidget(self.list_layout.count() - 1, row)
            self.rows.append(row)

        if baseline is None:
            self.write_log("The history stored on disk could not be read, so "
                           "every entry is listed - check the selection.")
        else:
            self.write_log("%d of %d history entries were added after the "
                           "image was loaded."
                           % (len(history) - baseline_len, len(history)))

    def select_all(self, value) -> None:
        for row in self.rows:
            row.set_enabled(value)

    # -- folders ------------------------------------------------------------

    def pick_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the folder with the images",
            self.ed_folder.text().strip() or os.getcwd())
        if path:
            self.ed_folder.setText(os.path.normpath(path))

    def pick_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the output folder",
            self.ed_output.text().strip() or os.getcwd())
        if path:
            self.ed_output.setText(os.path.normpath(path))

    def update_output_state(self) -> None:
        on = not self.chk_overwrite.isChecked()
        self.ed_output.setEnabled(on)
        self.out_button.setEnabled(on)
        self.out_caption.setEnabled(on)

    def refresh_count(self) -> None:
        folder = self.ed_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            self.count_label.setText("No folder selected.")
            return
        files = list_images(folder, self.chk_recursive.isChecked())
        self.count_label.setText("%d image(s) found." % len(files))

    # -- logging ------------------------------------------------------------

    def write_log(self, text) -> None:
        self._append_log(text, None)

    def _append_log(self, message, color=None) -> None:
        colour = LOG_COLOURS[self.log_theme].get(
            (color or "").lower() if color else "")
        if colour:
            self.text.appendHtml(
                '<span style="color:%s; white-space:pre">%s</span>'
                % (colour, html.escape(message)))
        else:
            self.text.appendPlainText(message)

    def _set_maximum(self, value) -> None:
        self.progress.setRange(0, max(value, 1))
        self.progress.setValue(0)

    # -- sync ---------------------------------------------------------------

    def on_sync(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        commands = [row.command() for row in self.rows if row.is_active()]
        if not commands:
            QtWidgets.QMessageBox.warning(self, "SirilSync",
                                          "No change is selected.")
            return

        folder = self.ed_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(
                self, "SirilSync",
                "Select a valid folder with the images to process.")
            return

        files = list_images(folder, self.chk_recursive.isChecked())
        if not files:
            QtWidgets.QMessageBox.warning(
                self, "SirilSync",
                "No supported image was found in that folder.")
            return

        # The loaded image is processed like any other file in the folder: the
        # copy on disk still is the unprocessed one, so replaying the commands
        # on it gives the same result as the one currently on screen.
        includes_source = bool(self.source_path) and any(
            os.path.abspath(f) == self.source_path for f in files)

        overwrite = self.chk_overwrite.isChecked()
        out_folder = self.ed_output.text().strip()
        if not overwrite:
            if not out_folder:
                QtWidgets.QMessageBox.warning(
                    self, "SirilSync",
                    "Choose an output folder, or enable \"Overwrite the "
                    "original images\".")
                return
            if not os.path.isdir(out_folder):
                try:
                    os.makedirs(out_folder)
                except OSError as exc:
                    QtWidgets.QMessageBox.critical(
                        self, "SirilSync",
                        "Cannot create the output folder:\n%s" % exc)
                    return
            if os.path.abspath(out_folder) == os.path.abspath(folder):
                QtWidgets.QMessageBox.critical(
                    self, "SirilSync",
                    "The output folder is the same as the source folder. "
                    "Enable overwriting, or pick a different folder.")
                return

        flagged = [row for row in self.rows
                   if row.is_active() and row.level != EXACT]
        warning = ""
        if flagged:
            warning = ("\n\nThese selected changes could not be fully "
                       "recovered from the history:\n"
                       + "\n".join("    %s" % row.command() for row in flagged)
                       + "\nCheck their commands before continuing.")

        source_note = ""
        if includes_source:
            source_note = ("\n\nThe image currently loaded in Siril is part of "
                           "this batch and will be processed too. Make sure "
                           "its file on disk does not already contain these "
                           "changes, otherwise they are applied twice.")

        target = ("the original files will be overwritten" if overwrite
                  else out_folder)
        answer = QtWidgets.QMessageBox.question(
            self, "SirilSync",
            "Apply %d change(s) to %d image(s)?\n\nOutput: %s%s%s"
            % (len(commands), len(files), target, source_note, warning))
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self.cancel.clear()
        self.sync_button.setEnabled(False)
        self.progress_max.emit(len(files))
        self.worker = threading.Thread(
            target=self._run, args=(files, commands, overwrite, out_folder),
            daemon=True)
        self.worker.start()

    # -- the sinks the worker writes into -----------------------------------

    def post(self, kind, payload=None) -> None:
        """Kept so the worker body reads the same as it always did."""
        if kind == "log":
            self.log_line.emit(payload, None)
        elif kind == "status":
            self.status_changed.emit(payload)
        elif kind == "progress":
            self.progress_changed.emit(payload)
        elif kind == "maximum":
            self.progress_max.emit(payload)
        elif kind == "done":
            self.run_finished.emit(payload[0], payload[1])

    def on_worker_finished(self, done, failed) -> None:
        self.sync_button.setEnabled(True)
        message = "Finished: %d processed, %d failed." % (done, failed)
        self.status.setText(message)
        self.write_log(message)
        try:
            self.siril.log("SirilSync: " + message)
        except Exception:
            pass
        if failed:
            QtWidgets.QMessageBox.warning(
                self, "SirilSync", message + "\nSee the log for details.")
        else:
            QtWidgets.QMessageBox.information(
                self, "SirilSync", "%d image(s) processed." % done)

    # -- closing ------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.is_alive():
            answer = QtWidgets.QMessageBox.question(
                self, "SirilSync", "A sync is running. Stop it and close?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel.set()
            self.worker.join(timeout=15)
        try:
            self.siril.disconnect()
        except Exception:
            pass
        event.accept()


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


def main():
    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilError as exc:
        print("SirilSync: could not connect to Siril: %s" % exc)
        return

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])
    apply_siril_theme(app, siril)

    window = SirilSyncWindow(siril)
    window.show()
    window.raise_()
    window.activateWindow()

    if owns_app:
        app.exec()


if __name__ == "__main__":
    main()
