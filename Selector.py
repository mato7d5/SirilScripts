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
Selector - frame selection for calibrated light frames (Siril 1.4+).

Point the script at a folder holding calibrated light frames - either a Siril
sequence (.seq) or just the individual image files - and it measures every
frame:

  FWHM          how sharp the stars are (seeing, focus)
  eccentricity  how elongated the stars are (tracking, wind, tilt)
  stars         how many stars were found (clouds, haze, dew)
  background    the sky level (moon, dawn, light pollution, clouds)

The results are shown in a table and in charts. Each frame gets a suggestion -
keep or reject - with the reason, computed from limits you can tune while you
look at the charts. The frames you finally mark are then deleted, moved to a
"rejected" subfolder, or only unselected in the sequence.

How a frame is measured
-----------------------
  Sequence          Siril's own registration data is used: FWHM, roundness,
                    star count and background per frame. If the sequence has
                    none yet, "register <seq> -2pass" computes it - this only
                    measures, it does not write any registered images.
                    An undebayered (CFA) sequence cannot be registered, so its
                    frames are measured one by one instead, as below.
  Individual files  every file is loaded and "findstar" is run on it; the
                    median FWHM and roundness of the detected stars are used.

Requires Siril >= 1.4 with the Python (sirilpy) interface.
Copy this file into your Siril scripts directory to get it in the script menu.
"""

from __future__ import annotations

__version__ = "1.0.0"

import csv
import html
import math
import os
import shutil
import sys
import threading
from dataclasses import dataclass, field

import numpy as np

import sirilpy as s

try:
    from sirilpy import LogColor
except ImportError:  # builds of sirilpy without LogColor
    LogColor = None

# PyQt6 is the Qt binding that ships in Siril's own Python environment
# (Siril 1.4 bundles PyQt6 6.11 / Qt 6.11).
try:
    from PyQt6 import QtCore, QtGui, QtWidgets   # noqa: E402
except ImportError:
    s.ensure_installed("PyQt6")
    from PyQt6 import QtCore, QtGui, QtWidgets   # noqa: E402


TITLE = "Selector"

# Siril's log colours, in a light and a dark variant so the embedded log stays
# readable whichever theme Siril is set to.
LOG_COLOURS = {
    "light": {"green": "#1b6e2b", "salmon": "#b34a20", "blue": "#14539a",
              "red": "#b3261e"},
    "dark": {"green": "#7fd18c", "salmon": "#ffb08f", "blue": "#7cb6f2",
             "red": "#ff9b94"},
}

# Chart / table colours, per theme.
PLOT_COLOURS = {
    "light": {"keep": "#2a6fdb", "reject": "#d0342c", "override": "#e08a00",
              "line": "#9aa7b8", "median": "#6b7785", "limit": "#d0342c",
              "grid": "#e3e6ea", "row_reject": "#fbe3e1",
              "row_override": "#fdf0d8"},
    "dark": {"keep": "#6ea8ff", "reject": "#ff6b61", "override": "#ffb347",
             "line": "#5b6675", "median": "#9aa5b1", "limit": "#ff6b61",
             "grid": "#3a3f45", "row_reject": "#5a2a27",
             "row_override": "#5a4520"},
}

FITS_EXTS = (".fit", ".fits", ".fts")
IMAGE_EXTS = FITS_EXTS + (".fit.fz", ".fits.fz", ".fts.fz",
                          ".tif", ".tiff", ".xisf")

REJECTED_DIR = "rejected"
ACCEPTED_DIR = "accepted"


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def quote(path: str) -> str:
    """A path or name in the form Siril's parser understands."""
    return '"%s"' % str(path).replace("\\", "/")


def split_ext(path: str):
    """Split a path, keeping '.fits.fz' style double extensions together."""
    base, ext = os.path.splitext(path)
    if ext.lower() == ".fz":
        base2, ext2 = os.path.splitext(base)
        if ext2.lower() in FITS_EXTS:
            return base2, ext2.lower() + ext.lower()
    return base, ext.lower()


def list_images(folder: str) -> list:
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    found = [os.path.join(folder, name) for name in names
             if split_ext(name)[1] in IMAGE_EXTS
             and os.path.isfile(os.path.join(folder, name))]
    return sorted(found, key=lambda p: os.path.basename(p).lower())


def _sequences_in(folder: str) -> list:
    """Sequence names in one folder: 'name' for a .seq, 'name.ser' for SER."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    found = []
    for name in names:
        lower = name.lower()
        if lower.endswith(".seq"):
            found.append(name[:-4])
        elif lower.endswith(".ser"):
            found.append(name)
    return sorted(found, key=str.lower)


def transfer(path: str, target_dir: str, copy: bool) -> None:
    """Copy or move a file into target_dir, replacing a file of that name."""
    target = os.path.join(target_dir, os.path.basename(path))
    if os.path.exists(target):
        os.remove(target)
    if copy:
        shutil.copy2(path, target)
    else:
        shutil.move(path, target)


def list_sequences(folder: str) -> list:
    """(label, sequence folder, name) of every sequence in the folder and in
    its direct subfolders - Siril keeps its sequences in e.g. 'process'."""
    result = [(name, folder, name) for name in _sequences_in(folder)]
    try:
        subfolders = sorted((d for d in os.listdir(folder)
                             if os.path.isdir(os.path.join(folder, d))
                             and d.lower() != REJECTED_DIR), key=str.lower)
    except OSError:
        subfolders = []
    for sub in subfolders:
        path = os.path.join(folder, sub)
        result += [("%s/%s" % (sub, name), path, name)
                   for name in _sequences_in(path)]
    return result


def robust_sigma(values: np.ndarray, median: float) -> float:
    """1.4826 x MAD, falling back to the plain standard deviation."""
    if values.size < 2:
        return 0.0
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    if sigma <= 0:
        sigma = float(np.std(values))
    return sigma


def fmt(value, digits=2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, int):
        return str(value)
    return "%.*f" % (digits, value)


def fmt_bg(value) -> str:
    """Backgrounds are 0..1 for float data and ADU for integer data."""
    if value is None:
        return "-"
    return "%.1f" % value if abs(value) >= 10 else "%.5f" % value


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

@dataclass
class FrameStat:
    """What was measured on one frame."""

    index: int                    # frame index in the sequence / file order
    name: str
    path: str | None = None       # None for frames inside a SER / FITSEQ
    fwhm: float | None = None     # px
    wfwhm: float | None = None    # Siril's weighted FWHM (sequences only)
    roundness: float | None = None
    stars: int = 0
    background: float | None = None
    noise: float | None = None
    date: str = ""
    error: str = ""
    # filled in by evaluate()
    suggested: bool = False
    reasons: list = field(default_factory=list)
    # the user's decision (starts as the suggestion)
    marked: bool = False

    @property
    def valid(self) -> bool:
        return self.fwhm is not None and self.fwhm > 0 and self.stars > 0

    @property
    def eccentricity(self) -> float | None:
        if self.roundness is None or self.roundness <= 0:
            return None
        r = min(self.roundness, 1.0)
        return math.sqrt(1.0 - r * r)

    def value(self, key):
        if key == "fwhm":
            return self.fwhm
        if key == "ecc":
            return self.eccentricity
        if key == "stars":
            return float(self.stars) if self.valid else None
        if key == "bg":
            return self.background
        return None


# key, label, unit, higher values are worse, enabled by default,
# limit caption, limit range, decimals
METRICS = [
    ("fwhm", "FWHM", "px", True, True, "max:", (0.0, 100.0), 2),
    ("ecc", "Eccentricity", "", True, True, "max:", (0.0, 1.0), 2),
    ("stars", "Stars", "", False, True, "min:", (0.0, 1000000.0), 0),
    ("bg", "Background", "", True, False, "max:", (0.0, 65535.0), 5),
]
METRIC = {meta[0]: meta for meta in METRICS}

ECC_DEFAULT = 0.60      # stars start to look oval above this


@dataclass
class Rule:
    key: str
    enabled: bool
    limit: float    # in the metric's own units: px, eccentricity, stars, ADU


@dataclass
class MetricSummary:
    median: float | None = None
    low: float | None = None
    high: float | None = None
    threshold: float | None = None
    failing: int = 0


def metric_values(frames: list, key: str) -> np.ndarray:
    return np.array([f.value(key) for f in frames
                     if f.value(key) is not None], dtype=float)


def propose_limit(frames: list, key: str) -> float | None:
    """A sensible starting limit derived from the data itself.

    FWHM: median + 2 robust sigma; background: median + 3 robust sigma;
    stars: half the median; eccentricity: 0.60, or a little above the typical
    value when every frame is more elongated than that.
    """
    values = metric_values(frames, key)
    if not values.size:
        return ECC_DEFAULT if key == "ecc" else None
    median = float(np.median(values))
    sigma = robust_sigma(values, median)
    if key == "fwhm":
        return round(median + 2.0 * sigma, 2)
    if key == "ecc":
        return max(ECC_DEFAULT, round(median + 2.0 * sigma, 2))
    if key == "stars":
        return float(int(median * 0.5))
    if key == "bg":
        return median + 3.0 * sigma
    return None


def evaluate(frames: list, rules: list) -> dict:
    """Mark every frame keep / reject against the limits; return summaries."""
    summaries = {}
    for frame in frames:
        frame.suggested = False
        frame.reasons = []

    active = [rule for rule in rules if rule.enabled]
    for rule in rules:
        label, higher_is_worse = METRIC[rule.key][1], METRIC[rule.key][3]
        values = metric_values(frames, rule.key)
        summary = MetricSummary(threshold=rule.limit)
        if values.size:
            summary.median = float(np.median(values))
            summary.low, summary.high = float(values.min()), float(values.max())
        summaries[rule.key] = summary
        if not rule.enabled:
            continue

        for frame in frames:
            value = frame.value(rule.key)
            if value is None:
                continue
            if higher_is_worse and value > rule.limit:
                sign = ">"
            elif not higher_is_worse and value < rule.limit:
                sign = "<"
            else:
                continue
            summary.failing += 1
            frame.reasons.append("%s %s %s %s" % (
                label, fmt_value(rule.key, value), sign,
                fmt_value(rule.key, rule.limit)))

    for frame in frames:
        if not frame.valid and active:
            frame.reasons.insert(0, frame.error or "no stars detected")
        frame.suggested = bool(frame.reasons)
    return summaries


def fmt_value(key, value) -> str:
    if value is None:
        return "-"
    if key == "stars":
        return "%d" % round(value)
    if key == "bg":
        return fmt_bg(value)
    return "%.2f" % value


# --------------------------------------------------------------------------- #
#  Theme
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  Chart
# --------------------------------------------------------------------------- #

class MetricChart(QtWidgets.QWidget):
    """One metric plotted against the frame number, drawn with QPainter.

    Points marked for rejection are red; a frame whose mark differs from the
    suggestion is orange. The dashed lines are the median and the limit.
    Clicking a point selects that frame in the table.
    """

    frame_clicked = QtCore.pyqtSignal(int)

    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 54, 12, 24, 26

    def __init__(self, key, title, theme):
        super().__init__()
        self.key = key
        self.title = title
        self.colours = PLOT_COLOURS[theme]
        self.frames = []
        self.summary = MetricSummary()
        self.rule_on = True
        self.current = -1
        self._points = []           # (x, y, position) in widget coordinates
        self.setMouseTracking(True)
        self.setMinimumSize(240, 140)

    def set_data(self, frames, summary, rule_on) -> None:
        self.frames = frames
        self.summary = summary
        self.rule_on = rule_on
        self.update()

    def set_current(self, position) -> None:
        self.current = position
        self.update()

    # -- geometry -----------------------------------------------------------

    def _ranges(self):
        values = [f.value(self.key) for f in self.frames]
        valid = [v for v in values if v is not None]
        if not valid:
            return values, None
        low, high = min(valid), max(valid)
        for extra in (self.summary.median, self.summary.threshold):
            if extra is not None:
                low, high = min(low, extra), max(high, extra)
        if high - low < 1e-9:
            pad = abs(high) * 0.1 or 1.0
        else:
            pad = (high - low) * 0.08
        low, high = low - pad, high + pad
        if low < 0 <= min(valid):       # no negative axis for >= 0 metrics
            low = 0.0
        return values, (low, high)

    # -- painting -----------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        text_colour = pal.color(QtGui.QPalette.ColorRole.WindowText)
        painter.fillRect(self.rect(), pal.color(QtGui.QPalette.ColorRole.Base))

        font = painter.font()
        bold = QtGui.QFont(font)
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(text_colour)
        caption = self.title if self.rule_on else self.title + "  (not used)"
        painter.drawText(8, 16, caption)
        painter.setFont(font)

        left, top = self.MARGIN_L, self.MARGIN_T
        width = self.width() - self.MARGIN_L - self.MARGIN_R
        height = self.height() - self.MARGIN_T - self.MARGIN_B
        self._points = []
        if width < 20 or height < 20:
            return

        values, yr = self._ranges()
        if yr is None:
            painter.setPen(pal.color(QtGui.QPalette.ColorRole.PlaceholderText))
            painter.drawText(QtCore.QRectF(left, top, width, height),
                             QtCore.Qt.AlignmentFlag.AlignCenter, "no data")
            return
        y0, y1 = yr
        n = len(values)

        def px(i):
            return left + (width * (i + 0.5) / n if n else 0)

        def py(v):
            return top + height - (v - y0) / (y1 - y0) * height

        # grid and y labels
        metrics = QtGui.QFontMetrics(font)
        grid_pen = QtGui.QPen(QtGui.QColor(self.colours["grid"]))
        for step in range(5):
            v = y0 + (y1 - y0) * step / 4
            y = py(v)
            painter.setPen(grid_pen)
            painter.drawLine(QtCore.QPointF(left, y),
                             QtCore.QPointF(left + width, y))
            painter.setPen(text_colour)
            label = fmt_value(self.key, v) if self.key != "fwhm" else "%.2f" % v
            painter.drawText(QtCore.QRectF(0, y - 8, left - 6, 16),
                             QtCore.Qt.AlignmentFlag.AlignRight
                             | QtCore.Qt.AlignmentFlag.AlignVCenter, label)

        # x labels: frame numbers, spaced so they do not overlap
        painter.setPen(text_colour)
        label_w = metrics.horizontalAdvance("0000") + 8
        every = max(1, math.ceil(n / max(1, width // label_w)))
        for i in range(0, n, every):
            painter.drawText(QtCore.QRectF(px(i) - 20, top + height + 4, 40, 16),
                             QtCore.Qt.AlignmentFlag.AlignHCenter,
                             str(self.frames[i].index + 1))

        # the connecting line
        painter.setPen(QtGui.QPen(QtGui.QColor(self.colours["line"]), 1))
        previous = None
        for i, v in enumerate(values):
            if v is None:
                previous = None
                continue
            point = QtCore.QPointF(px(i), py(v))
            if previous is not None:
                painter.drawLine(previous, point)
            previous = point

        # median (labelled on the left) and limit (labelled on the right)
        for value, colour, text, at_left in (
                (self.summary.median, self.colours["median"], "median", True),
                (self.summary.threshold if self.rule_on else None,
                 self.colours["limit"], "limit", False)):
            if value is None:
                continue
            pen = QtGui.QPen(QtGui.QColor(colour), 1,
                             QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen)
            y = py(value)
            painter.drawLine(QtCore.QPointF(left, y),
                             QtCore.QPointF(left + width, y))
            x = (left + 4 if at_left
                 else left + width - metrics.horizontalAdvance(text) - 4)
            painter.drawText(QtCore.QPointF(x, y - 3), text)

        # the points
        radius = 3.5 if n <= 150 else 2.5
        for i, (frame, v) in enumerate(zip(self.frames, values)):
            if v is None:
                continue
            if frame.marked != frame.suggested:
                colour = self.colours["override"]
            elif frame.marked:
                colour = self.colours["reject"]
            else:
                colour = self.colours["keep"]
            centre = QtCore.QPointF(px(i), py(v))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(colour))
            painter.drawEllipse(centre, radius, radius)
            if i == self.current:
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.setPen(QtGui.QPen(text_colour, 1.5))
                painter.drawEllipse(centre, radius + 3, radius + 3)
            self._points.append((centre.x(), centre.y(), i))

        # frame
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(self.colours["line"])))
        painter.drawRect(QtCore.QRectF(left, top, width, height))

    # -- interaction --------------------------------------------------------

    def _nearest(self, pos):
        best, best_d = None, 12.0 ** 2
        for x, y, i in self._points:
            d = (x - pos.x()) ** 2 + (y - pos.y()) ** 2
            if d < best_d:
                best, best_d = i, d
        return best

    def mouseMoveEvent(self, event) -> None:
        i = self._nearest(event.position())
        if i is None:
            QtWidgets.QToolTip.hideText()
            return
        frame = self.frames[i]
        text = "#%d  %s\n%s: %s" % (frame.index + 1, frame.name, self.title,
                                    fmt_value(self.key, frame.value(self.key)))
        if frame.reasons:
            text += "\n" + "; ".join(frame.reasons)
        QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), text, self)

    def mousePressEvent(self, event) -> None:
        i = self._nearest(event.position())
        if i is not None:
            self.frame_clicked.emit(i)


# --------------------------------------------------------------------------- #
#  Table
# --------------------------------------------------------------------------- #

class NumericItem(QtWidgets.QTableWidgetItem):
    """A cell that shows formatted text but sorts by its number."""

    def __init__(self, text, number):
        super().__init__(text)
        self.number = number
        self.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                              | QtCore.Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            a = self.number if self.number is not None else float("inf")
            b = other.number if other.number is not None else float("inf")
            return a < b
        return super().__lt__(other)


COLUMNS = ["#", "File", "Reject", "Suggestion", "FWHM", "wFWHM",
           "Eccentricity", "Roundness", "Stars", "Background", "Noise",
           "Date", "Reason"]
COL_NUM, COL_FILE, COL_REJECT, COL_SUGGEST = 0, 1, 2, 3
COL_REASON = len(COLUMNS) - 1


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #

class SelectorWindow(QtWidgets.QWidget):
    """Main window; measuring runs on its own thread.

    The worker reports back through Qt signals, which Qt delivers on the GUI
    thread, so no widget is touched from the wrong thread.
    """

    log_line = QtCore.pyqtSignal(str, object)
    status_changed = QtCore.pyqtSignal(str)
    progress_max = QtCore.pyqtSignal(int)
    progress_changed = QtCore.pyqtSignal(int)
    measure_finished = QtCore.pyqtSignal(object)

    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.frames = []
        self.summaries = {}
        self.source = None          # ("seq", folder, name) or ("files", folder)
        self.seq_stale = False      # the sequence was rebuilt after analysis
        self.worker = None
        self.cancel = threading.Event()
        self.theme = "dark" if siril_is_dark(siril) else "light"
        self.colours = PLOT_COLOURS[self.theme]
        self._filling_table = False

        self.setWindowTitle(TITLE + " - frame selection v" + __version__)
        self._build_widgets()

        self.log_line.connect(self._append_log)
        self.status_changed.connect(self.status.setText)
        self.progress_max.connect(self._set_maximum)
        self.progress_changed.connect(self.progress.setValue)
        self.measure_finished.connect(self.on_measure_finished)

        try:
            folder = siril.get_siril_wd()
        except Exception:
            folder = ""
        self.ed_folder.setText(folder or os.getcwd())
        self._fit_to_screen()

    # -- layout -------------------------------------------------------------

    def _build_widgets(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self._build_source_box(), 2)
        top.addWidget(self._build_criteria_box(), 3)
        outer.addLayout(top)

        # --- results: table above, charts below ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        table_box = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_box)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QtWidgets.QLabel("Nothing analysed yet.")
        table_layout.addWidget(self.summary_label)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSortIndicator(
            COL_NUM, QtCore.Qt.SortOrder.AscendingOrder)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.setToolTip("Tick \"Reject\" to change the decision for a "
                              "frame. Double-click a row to open that frame "
                              "in Siril.")
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        self.table.cellDoubleClicked.connect(self.on_row_double_clicked)
        table_layout.addWidget(self.table, 1)
        splitter.addWidget(table_box)

        charts = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(charts)
        grid.setContentsMargins(0, 0, 0, 0)
        self.charts = {}
        for n, meta in enumerate(METRICS):
            key, label, unit = meta[0], meta[1], meta[2]
            chart = MetricChart(key, label + (" [%s]" % unit if unit else ""),
                                self.theme)
            chart.frame_clicked.connect(self.select_frame)
            grid.addWidget(chart, n // 2, n % 2)
            self.charts[key] = chart
        splitter.addWidget(charts)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 400])
        outer.addWidget(splitter, 1)

        # --- decision area: rejected frames above, accepted frames below ---
        actions = QtWidgets.QGridLayout()
        bar = QtWidgets.QHBoxLayout()
        for text, slot in (("Reset to suggestions", self.reset_marks),
                           ("Keep all", lambda: self.set_all_marks(False)),
                           ("Export CSV...", self.export_csv)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            bar.addWidget(button)
        actions.addLayout(bar, 0, 0)
        actions.setColumnStretch(1, 1)

        actions.addWidget(QtWidgets.QLabel("Rejected frames:"), 0, 2)
        self.cmb_action = QtWidgets.QComboBox()
        self.cmb_action.addItem("Delete the files", "delete")
        self.cmb_action.addItem("Move to the \"%s\" subfolder" % REJECTED_DIR,
                                "move")
        self.cmb_action.addItem("Only unselect them in the sequence",
                                "unselect")
        self.cmb_action.setToolTip(
            "Delete: the files are removed for good.\n"
            "Move: the files go to a \"%s\" subfolder, so you can bring them "
            "back.\nUnselect: the files stay, the frames are only excluded "
            "in the .seq (sequences only)." % REJECTED_DIR)
        actions.addWidget(self.cmb_action, 0, 3)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.on_apply)
        actions.addWidget(self.apply_button, 0, 4)

        actions.addWidget(QtWidgets.QLabel("Accepted frames:"), 1, 2)
        self.cmb_accepted = QtWidgets.QComboBox()
        self.cmb_accepted.addItem("Copy to the \"%s\" subfolder"
                                  % ACCEPTED_DIR, "copy")
        self.cmb_accepted.addItem("Move to the \"%s\" subfolder"
                                  % ACCEPTED_DIR, "move")
        self.cmb_accepted.setToolTip(
            "The frames not marked for rejection go to a \"%s\" subfolder "
            "next to them.\nCopy: the originals stay where they are.\n"
            "Move: the originals are taken out of the folder.\n"
            "For a sequence a new .seq is created in the subfolder."
            % ACCEPTED_DIR)
        actions.addWidget(self.cmb_accepted, 1, 3)
        self.accept_button = QtWidgets.QPushButton("Apply")
        self.accept_button.setEnabled(False)
        self.accept_button.clicked.connect(self.on_accept)
        actions.addWidget(self.accept_button, 1, 4)
        outer.addLayout(actions)

        # --- progress and log ---
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        outer.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Ready.")
        outer.addWidget(self.status)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setMaximumHeight(80)
        outer.addWidget(self.text)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.close)
        buttons.addWidget(btn)
        outer.addLayout(buttons)

    def _build_source_box(self):
        box = QtWidgets.QGroupBox("Calibrated light frames")
        grid = QtWidgets.QGridLayout(box)

        grid.addWidget(QtWidgets.QLabel("Folder:"), 0, 0)
        self.ed_folder = QtWidgets.QLineEdit()
        self.ed_folder.textChanged.connect(self.refresh_source)
        grid.addWidget(self.ed_folder, 0, 1, 1, 2)
        btn = QtWidgets.QPushButton("Browse...")
        btn.clicked.connect(self.pick_folder)
        grid.addWidget(btn, 0, 3)

        self.rb_seq = QtWidgets.QRadioButton("Sequence:")
        self.rb_files = QtWidgets.QRadioButton("Individual files")
        self.rb_seq.setChecked(True)
        self.rb_seq.toggled.connect(self._update_mode)
        grid.addWidget(self.rb_seq, 1, 0)
        self.cmb_seq = QtWidgets.QComboBox()
        self.cmb_seq.setToolTip("A sequence (.seq or .ser) in the folder above "
                                "or in one of its subfolders.")
        self.cmb_seq.setPlaceholderText("no sequence found in this folder")
        self.cmb_seq.activated.connect(
            lambda _i: self.rb_seq.setChecked(True))
        grid.addWidget(self.cmb_seq, 1, 1, 1, 3)

        grid.addWidget(self.rb_files, 2, 0)
        self.files_label = QtWidgets.QLabel("")
        grid.addWidget(self.files_label, 2, 1, 1, 3)

        self.chk_remeasure = QtWidgets.QCheckBox(
            "Re-measure even if the sequence already has star data")
        self.chk_remeasure.setToolTip(
            "Runs \"register <seq> -2pass\" again. This replaces the "
            "registration data stored in the .seq (no images are written).")
        grid.addWidget(self.chk_remeasure, 3, 0, 1, 4)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("Stop")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel.set)
        row.addWidget(self.cancel_button)
        self.analyse_button = QtWidgets.QPushButton("Analyse")
        self.analyse_button.setDefault(True)
        self.analyse_button.clicked.connect(self.on_analyse)
        row.addWidget(self.analyse_button)
        grid.addLayout(row, 4, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(5, 1)
        return box

    def _build_criteria_box(self):
        box = QtWidgets.QGroupBox("Rejection criteria")
        grid = QtWidgets.QGridLayout(box)
        self.rule_widgets = {}
        for row, meta in enumerate(METRICS):
            (key, label, unit, _higher_is_worse, enabled, caption,
             limit_range, decimals) = meta

            check = QtWidgets.QCheckBox(label)
            check.setChecked(enabled)
            grid.addWidget(check, row, 0)
            grid.addWidget(QtWidgets.QLabel(caption), row, 1)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(*limit_range)
            spin.setDecimals(decimals)
            spin.setSingleStep({"fwhm": 0.05, "ecc": 0.01, "stars": 10.0}
                               .get(key, 0.001))
            spin.setSuffix(" " + unit if unit else "")
            spin.setMinimumWidth(110)
            if key == "ecc":
                spin.setValue(ECC_DEFAULT)
            grid.addWidget(spin, row, 2)

            info = QtWidgets.QLabel("")
            info.setMinimumWidth(170)
            grid.addWidget(info, row, 3)

            check.toggled.connect(self.reevaluate)
            spin.valueChanged.connect(self.reevaluate)
            self.rule_widgets[key] = (check, spin, info)

        row = QtWidgets.QHBoxLayout()
        hint = QtWidgets.QLabel(
            "A frame is suggested for rejection when it fails any ticked "
            "criterion, or when no stars were found. The limits are filled in "
            "from the data after each analysis - type your own values and the "
            "suggestions follow at once (your own ticks are reset).")
        hint.setWordWrap(True)
        row.addWidget(hint, 1)
        button = QtWidgets.QPushButton("Propose from data")
        button.setToolTip("FWHM: median + 2σ, stars: half the median, "
                          "eccentricity: 0.60, background: median + 3σ.")
        button.clicked.connect(self.propose_limits)
        row.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        grid.addLayout(row, len(METRICS), 0, 1, 4)
        grid.setColumnStretch(3, 1)
        return box

    def _fit_to_screen(self) -> None:
        """Size the window generously, never larger than the screen."""
        available = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        width = min(1250, int(available.width() * 0.94))
        height = min(900, int(available.height() * 0.9))
        self.setMinimumSize(min(760, width), min(560, height))
        self.resize(width, height)
        self.move(available.x() + (available.width() - width) // 2,
                  available.y() + (available.height() - height) // 3)

    # -- source -------------------------------------------------------------

    def pick_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the folder with the calibrated lights",
            self.ed_folder.text().strip() or os.getcwd())
        if path:
            self.ed_folder.setText(os.path.normpath(path))

    def refresh_source(self) -> None:
        folder = self.ed_folder.text().strip()
        sequences = list_sequences(folder) if os.path.isdir(folder) else []
        files = list_images(folder) if os.path.isdir(folder) else []

        current = self.cmb_seq.currentText()
        self.cmb_seq.blockSignals(True)
        self.cmb_seq.clear()
        for label, seq_folder, name in sequences:
            self.cmb_seq.addItem(label, (seq_folder, name))
        index = self.cmb_seq.findText(current)
        self.cmb_seq.setCurrentIndex(index if index >= 0
                                     else (0 if sequences else -1))
        self.cmb_seq.blockSignals(False)

        self.files_label.setText("%d image file(s) in the folder" % len(files)
                                 if os.path.isdir(folder)
                                 else "the folder does not exist")
        # A new folder picks the likely mode; the radio buttons stay free to
        # change afterwards.
        if sequences:
            self.rb_seq.setChecked(True)
        elif files:
            self.rb_files.setChecked(True)
        self._update_mode()

    def _update_mode(self, *_args) -> None:
        self.chk_remeasure.setEnabled(self.rb_seq.isChecked())

    # -- measuring ----------------------------------------------------------

    def on_analyse(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        folder = self.ed_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(self, TITLE,
                                          "Select a valid folder.")
            return
        folder = os.path.abspath(folder)

        if self.rb_seq.isChecked():
            data = self.cmb_seq.currentData()
            if not data:
                QtWidgets.QMessageBox.warning(
                    self, TITLE, "No sequence (.seq / .ser) was found in this "
                                 "folder or its subfolders. Pick another "
                                 "folder, or choose \"Individual files\".")
                return
            seq_folder, name = data
            seq_file = (name if name.lower().endswith(".ser")
                        else name + ".seq")
            if not os.path.isfile(os.path.join(seq_folder, seq_file)):
                QtWidgets.QMessageBox.warning(
                    self, TITLE, "The sequence %s no longer exists." % seq_file)
                self.refresh_source()
                return
            seq_folder = os.path.abspath(seq_folder)
            source = ("seq", seq_folder, name)
            args = (seq_folder, name, self.chk_remeasure.isChecked())
            target = self._measure_sequence
        else:
            files = list_images(folder)
            if not files:
                QtWidgets.QMessageBox.warning(
                    self, TITLE, "No FITS, TIFF or XISF image was found in "
                                 "that folder.")
                return
            source = ("files", folder)
            args = (folder, files)
            target = self._measure_folder

        self.source = source
        self.seq_stale = False
        self.frames = []
        self.summaries = {}
        self._fill_table()
        self._refresh_charts()
        self.cancel.clear()
        self.analyse_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.accept_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = threading.Thread(target=self._worker_main,
                                       args=(target, args), daemon=True)
        self.worker.start()

    def _worker_main(self, target, args) -> None:
        frames = []
        try:
            frames = target(*args) or []
        except Exception as exc:     # report, never kill the window
            self.post_log("Analysis failed: %s" % exc, "red")
        self.measure_finished.emit(frames)

    def _measure_sequence(self, folder, name, remeasure):
        siril = self.siril
        self.post_log("Sequence %s in %s" % (name, folder), "blue")
        siril.cmd("cd", quote(folder))

        frames = None
        loaded = False
        try:
            siril.cmd("load_seq", quote(name))
            loaded = True
        except s.SirilError as exc:
            self.post_log("Could not load the sequence: %s" % exc, "salmon")

        if loaded and not remeasure:
            frames = self._read_sequence(folder)
            if frames:
                self.post_log("Using the star data already stored in the "
                              "sequence.")

        if not frames and not self.cancel.is_set():
            self.status_changed.emit("Measuring the stars with "
                                     "register -2pass ...")
            self.progress_max.emit(0)          # busy indicator
            self.post_log("register %s -2pass  (measures only, writes no "
                          "images)" % name)
            try:
                siril.cmd("register", quote(name), "-2pass")
                # reload, so the data register just wrote to the .seq is read
                siril.cmd("load_seq", quote(name))
                frames = self._read_sequence(folder)
            except s.SirilError as exc:
                self.post_log("register -2pass failed: %s" % exc, "salmon")
                frames = None

        if not frames and not self.cancel.is_set():
            # Typically an undebayered CFA sequence, which Siril refuses to
            # register: fall back to measuring every frame on its own.
            paths = self._sequence_paths(folder, name)
            if not paths:
                raise RuntimeError(
                    "No star data could be computed for this sequence and its "
                    "frames are not separate FITS files that could be "
                    "measured one by one.")
            self.post_log("Measuring every frame on its own with findstar.",
                          "salmon")
            frames = self._measure_files(paths)
        return frames

    def _sequence_paths(self, folder, name):
        """(index, path) of every frame of the sequence, if it is one file each."""
        siril = self.siril
        try:
            if not siril.is_sequence_loaded():
                siril.cmd("load_seq", quote(name))
            seq = siril.get_seq()
            if seq.type not in (None, s.SequenceType.SEQ_REGULAR):
                return []
            paths = []
            for i in range(seq.number):
                path = siril.get_seq_frame_filename(i)
                if not path:
                    return []
                if not os.path.isabs(path):
                    path = os.path.join(folder, path)
                paths.append((i, os.path.normpath(path)))
            return paths
        except Exception as exc:
            self.post_log("Could not list the frames: %s" % exc, "salmon")
            return []

    def _read_sequence(self, folder):
        """FrameStats from the registration data of the loaded sequence.

        Returns None when the sequence carries no star data.
        """
        siril = self.siril
        seq = siril.get_seq()
        if seq is None or seq.number <= 0:
            return None
        n = seq.number
        layers = max(1, seq.nb_layers)
        container = seq.type not in (None, s.SequenceType.SEQ_REGULAR)

        # Registration data lives on one layer only (green for colour data).
        best_layer, best_regs, best_count = None, None, 0
        for layer in range(layers):
            regs = []
            for i in range(n):
                try:
                    regs.append(siril.get_seq_regdata(i, layer))
                except s.SirilError:
                    regs.append(None)
            count = sum(1 for r in regs if r is not None and r.fwhm > 0)
            if count > best_count:
                best_layer, best_regs, best_count = layer, regs, count
        if not best_count:
            return None
        if layers > 1:
            self.post_log("Star data taken from layer %d." % best_layer)

        frames = []
        self.progress_max.emit(n)
        for i in range(n):
            reg = best_regs[i]
            frame = FrameStat(index=i, name="frame %d" % (i + 1))
            try:
                path = siril.get_seq_frame_filename(i)
            except s.SirilError:
                path = None
            if path:
                if not os.path.isabs(path):
                    path = os.path.join(folder, path)
                frame.name = os.path.basename(path)
                frame.path = None if container else os.path.normpath(path)
            try:
                img = siril.get_seq_imgdata(i)
                if img is not None and img.date_obs is not None:
                    frame.date = img.date_obs.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            if reg is not None and reg.fwhm > 0:
                frame.fwhm = float(reg.fwhm)
                frame.wfwhm = float(reg.weighted_fwhm) or None
                frame.roundness = float(reg.roundness) or None
                frame.stars = int(reg.number_of_stars)
                frame.background = float(reg.background_lvl)
            try:
                stats = siril.get_seq_stats(i, best_layer)
                if stats is not None and stats.bgnoise > 0:
                    frame.noise = float(stats.bgnoise)
            except Exception:
                pass
            frames.append(frame)
            self.progress_changed.emit(i + 1)
        return frames

    def _measure_folder(self, folder, files):
        self.post_log("%d file(s) in %s" % (len(files), folder), "blue")
        return self._measure_files(list(enumerate(files)))

    def _measure_files(self, indexed_paths):
        """Load every file, detect its stars and summarise them."""
        siril = self.siril
        frames = []
        total = len(indexed_paths)
        self.progress_max.emit(total)
        for done, (index, path) in enumerate(indexed_paths, start=1):
            if self.cancel.is_set():
                self.post_log("Stopped - %d of %d frame(s) measured."
                              % (len(frames), total), "salmon")
                break
            name = os.path.basename(path)
            self.status_changed.emit("[%d/%d] %s" % (done, total, name))
            frame = FrameStat(index=index, name=name, path=path)
            try:
                siril.cmd("load", quote(path))
                try:
                    siril.cmd("findstar")
                except s.SirilError:
                    pass      # no star found - reported as such below
                stars = siril.get_image_stars() or []
                good = [st for st in stars if st.fwhmx > 0 and st.fwhmy > 0]
                frame.stars = len(good)
                if good:
                    frame.fwhm = float(np.median([st.fwhmx for st in good]))
                    frame.roundness = float(np.median(
                        [min(st.fwhmy, st.fwhmx) / max(st.fwhmy, st.fwhmx)
                         for st in good]))
                shape = siril.get_image_shape()
                channel = 1 if shape and shape[0] >= 3 else 0
                stats = siril.get_image_stats(channel)
                if stats is not None:
                    frame.background = float(stats.median)
                    frame.noise = float(stats.bgnoise) or None
                try:
                    keywords = siril.get_image_keywords()
                    if keywords is not None and keywords.date_obs is not None:
                        frame.date = keywords.date_obs.strftime(
                            "%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            except s.SirilError as exc:
                frame.error = "could not be measured: %s" % exc
                self.post_log("%s: %s" % (name, frame.error), "red")
            frames.append(frame)
            self.progress_changed.emit(done)
        return frames

    def on_measure_finished(self, frames) -> None:
        self.analyse_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._set_maximum(1)
        self.progress.setValue(1 if frames else 0)
        self.frames = frames
        if not frames:
            self.status.setText("Nothing was measured.")
            self._fill_table()
            return
        valid = sum(1 for f in frames if f.valid)
        self.status.setText("%d frame(s) measured, %d with stars."
                            % (len(frames), valid))
        self.write_log("%d frame(s) measured, %d with stars."
                       % (len(frames), valid), "green")
        self.propose_limits()
        self.apply_button.setEnabled(True)
        self.accept_button.setEnabled(True)

    # -- evaluation ---------------------------------------------------------

    def rules(self):
        return [Rule(key, check.isChecked(), spin.value())
                for key, (check, spin, _info) in self.rule_widgets.items()]

    def propose_limits(self) -> None:
        """Fill the limits in from the measured data, then re-evaluate."""
        for key, (_check, spin, _info) in self.rule_widgets.items():
            value = propose_limit(self.frames, key)
            if value is None:
                continue
            if key == "bg":
                # 0..1 for float data, ADU for integer data
                magnitude = 10 ** math.floor(math.log10(max(value, 1e-6)))
                spin.setSingleStep(max(magnitude / 100.0, 1e-5))
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.reevaluate(reset_marks=True)

    def reevaluate(self, *_args, reset_marks=True) -> None:
        rules = self.rules()
        self.summaries = evaluate(self.frames, rules)
        if reset_marks:
            for frame in self.frames:
                frame.marked = frame.suggested
        for rule in rules:
            summary = self.summaries.get(rule.key, MetricSummary())
            info = self.rule_widgets[rule.key][2]
            if summary.median is None:
                info.setText("")
                continue
            text = "median %s   (%s – %s)" % (
                fmt_value(rule.key, summary.median),
                fmt_value(rule.key, summary.low),
                fmt_value(rule.key, summary.high))
            if rule.enabled:
                text += "   →  %d frame(s) %s" % (
                    summary.failing,
                    "above" if METRIC[rule.key][3] else "below")
            info.setText(text)
            info.setEnabled(rule.enabled)
        self._fill_table()
        self._refresh_charts()

    # -- table --------------------------------------------------------------

    def _fill_table(self) -> None:
        self._filling_table = True
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(self.frames))
        for row, frame in enumerate(self.frames):
            items = [
                NumericItem(str(frame.index + 1), frame.index),
                QtWidgets.QTableWidgetItem(frame.name),
                QtWidgets.QTableWidgetItem(""),
                QtWidgets.QTableWidgetItem(""),
                NumericItem(fmt(frame.fwhm), frame.fwhm),
                NumericItem(fmt(frame.wfwhm), frame.wfwhm),
                NumericItem(fmt(frame.eccentricity), frame.eccentricity),
                NumericItem(fmt(frame.roundness), frame.roundness),
                NumericItem(str(frame.stars), frame.stars),
                NumericItem(fmt_bg(frame.background), frame.background),
                NumericItem(fmt_bg(frame.noise), frame.noise),
                QtWidgets.QTableWidgetItem(frame.date),
                QtWidgets.QTableWidgetItem("; ".join(frame.reasons)),
            ]
            for col, item in enumerate(items):
                item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
                if col != COL_REJECT:
                    item.setFlags(item.flags()
                                  & ~QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                self.table.setItem(row, col, item)
            check = items[COL_REJECT]
            check.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                           | QtCore.Qt.ItemFlag.ItemIsSelectable
                           | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            self._style_row(row, frame)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(COL_FILE,
                                  min(self.table.columnWidth(COL_FILE), 260))
        self._filling_table = False
        self._update_summary()

    def _style_row(self, row, frame) -> None:
        """Colour one table row after the frame's mark and suggestion."""
        was = self._filling_table
        self._filling_table = True
        check = self.table.item(row, COL_REJECT)
        check.setCheckState(QtCore.Qt.CheckState.Checked if frame.marked
                            else QtCore.Qt.CheckState.Unchecked)
        suggest = self.table.item(row, COL_SUGGEST)
        suggest.setText("REJECT" if frame.suggested else "keep")
        suggest.setForeground(QtGui.QColor(
            self.colours["reject"] if frame.suggested else self.colours["keep"]))
        font = suggest.font()
        font.setBold(frame.suggested)
        suggest.setFont(font)
        tooltip = ("; ".join(frame.reasons) if frame.reasons
                   else "all criteria passed")
        if frame.marked != frame.suggested:
            background = QtGui.QColor(self.colours["row_override"])
            tooltip += "\n(your decision differs from the suggestion)"
        elif frame.marked:
            background = QtGui.QColor(self.colours["row_reject"])
        else:
            background = None
        for col in range(len(COLUMNS)):
            item = self.table.item(row, col)
            item.setBackground(background if background is not None
                               else QtGui.QBrush())
            item.setToolTip(tooltip)
        self._filling_table = was

    def _row_of(self, position):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_NUM)
            if item and item.data(QtCore.Qt.ItemDataRole.UserRole) == position:
                return row
        return -1

    def on_item_changed(self, item) -> None:
        if self._filling_table or item.column() != COL_REJECT:
            return
        position = item.data(QtCore.Qt.ItemDataRole.UserRole)
        frame = self.frames[position]
        frame.marked = item.checkState() == QtCore.Qt.CheckState.Checked
        self._style_row(item.row(), frame)
        self._update_summary()
        self._refresh_charts()

    def on_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        position = -1
        if rows:
            item = self.table.item(rows[0].row(), COL_NUM)
            if item is not None:
                position = item.data(QtCore.Qt.ItemDataRole.UserRole)
        for chart in self.charts.values():
            chart.set_current(position)

    def select_frame(self, position) -> None:
        row = self._row_of(position)
        if row >= 0:
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, COL_NUM))

    def on_row_double_clicked(self, row, _col) -> None:
        position = self.table.item(row, COL_NUM).data(
            QtCore.Qt.ItemDataRole.UserRole)
        frame = self.frames[position]
        if not frame.path or not os.path.isfile(frame.path):
            return
        if self.worker and self.worker.is_alive():
            return
        try:
            self.siril.cmd("load", quote(frame.path))
            self.write_log("Opened %s in Siril." % frame.name)
        except s.SirilError as exc:
            self.write_log("Could not open %s: %s" % (frame.name, exc), "red")

    def reset_marks(self) -> None:
        for frame in self.frames:
            frame.marked = frame.suggested
        self._restyle_all()

    def set_all_marks(self, value) -> None:
        for frame in self.frames:
            frame.marked = value
        self._restyle_all()

    def _restyle_all(self) -> None:
        for row in range(self.table.rowCount()):
            position = self.table.item(row, COL_NUM).data(
                QtCore.Qt.ItemDataRole.UserRole)
            self._style_row(row, self.frames[position])
        self._update_summary()
        self._refresh_charts()

    def _update_summary(self) -> None:
        if not self.frames:
            self.summary_label.setText("Nothing analysed yet.")
            return
        suggested = sum(1 for f in self.frames if f.suggested)
        marked = sum(1 for f in self.frames if f.marked)
        self.summary_label.setText(
            "<b>%d</b> frame(s)  ·  <b>%d</b> suggested for rejection  ·  "
            "<b>%d</b> marked to reject  ·  <b>%d</b> kept"
            % (len(self.frames), suggested, marked, len(self.frames) - marked))

    def _refresh_charts(self) -> None:
        rules = {rule.key: rule for rule in self.rules()}
        for key, chart in self.charts.items():
            chart.set_data(self.frames,
                           self.summaries.get(key, MetricSummary()),
                           rules[key].enabled)

    # -- export -------------------------------------------------------------

    def export_csv(self) -> None:
        if not self.frames:
            return
        folder = self.source[1] if self.source else os.getcwd()
        stem = self.source[2] if self.source and self.source[0] == "seq" \
            else "frames"
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export the statistics",
            os.path.join(folder, "%s_selection.csv" % stem),
            "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["frame", "file", "fwhm_px", "wfwhm_px",
                                 "eccentricity", "roundness", "stars",
                                 "background", "noise", "date",
                                 "suggestion", "marked", "reason"])
                def num(value, spec):
                    return "" if value is None else spec % value

                for f in self.frames:
                    writer.writerow([
                        f.index + 1, f.name, num(f.fwhm, "%.3f"),
                        num(f.wfwhm, "%.3f"), num(f.eccentricity, "%.3f"),
                        num(f.roundness, "%.3f"), f.stars,
                        num(f.background, "%.6g"), num(f.noise, "%.6g"),
                        f.date, "reject" if f.suggested else "keep",
                        "reject" if f.marked else "keep",
                        "; ".join(f.reasons)])
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, TITLE,
                                           "Could not write the file:\n%s" % exc)
            return
        self.write_log("Statistics exported to %s" % path, "green")

    # -- applying the decision ----------------------------------------------

    def on_apply(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        marked = [f for f in self.frames if f.marked]
        if not marked:
            QtWidgets.QMessageBox.information(
                self, TITLE, "No frame is marked for rejection.")
            return
        action = self.cmb_action.currentData()
        seq_mode = self.source and self.source[0] == "seq"

        if action == "unselect":
            if not seq_mode:
                QtWidgets.QMessageBox.warning(
                    self, TITLE, "Unselecting only works for a sequence. "
                                 "Choose to delete or move the files instead.")
                return
            if self.seq_stale:
                QtWidgets.QMessageBox.warning(
                    self, TITLE, "The sequence was rebuilt since it was "
                                 "analysed. Press Analyse again first.")
                return
        elif any(not f.path for f in marked):
            QtWidgets.QMessageBox.warning(
                self, TITLE, "These frames are stored inside one sequence file "
                             "(SER / FITSEQ), so they cannot be deleted one by "
                             "one. Choose \"Only unselect them in the "
                             "sequence\" instead.")
            return

        names = "\n".join("    " + f.name for f in marked[:15])
        if len(marked) > 15:
            names += "\n    ... and %d more" % (len(marked) - 15)
        if action == "delete":
            question = ("Permanently DELETE %d file(s)?\n\n%s\n\nThis cannot "
                        "be undone." % (len(marked), names))
        elif action == "move":
            question = ("Move %d file(s) to the \"%s\" subfolder?\n\n%s"
                        % (len(marked), REJECTED_DIR, names))
        else:
            question = ("Unselect %d frame(s) in the sequence %s?\n\n%s"
                        % (len(marked), self.source[2], names))
        if seq_mode and action != "unselect":
            question += ("\n\nThe sequence %s.seq is rebuilt from the remaining "
                         "files. Its registration data is dropped, so register "
                         "the sequence again before stacking."
                         % self.source[2])
        icon = (QtWidgets.QMessageBox.Icon.Warning if action == "delete"
                else QtWidgets.QMessageBox.Icon.Question)
        box = QtWidgets.QMessageBox(icon, TITLE, question,
                                    QtWidgets.QMessageBox.StandardButton.Yes
                                    | QtWidgets.QMessageBox.StandardButton.No,
                                    self)
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        if box.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        if action == "unselect":
            self._unselect(marked)
        else:
            self._remove(marked, action)

    def _unselect(self, marked) -> None:
        name = self.source[2]
        indexes = sorted(f.index for f in marked)
        # contiguous runs, so one command covers a whole block of frames
        runs, start, prev = [], indexes[0], indexes[0]
        for i in indexes[1:]:
            if i != prev + 1:
                runs.append((start, prev))
                start = i
            prev = i
        runs.append((start, prev))
        try:
            self.siril.cmd("cd", quote(self.source[1]))
            for first, last in runs:
                self.siril.cmd("unselect", quote(name), str(first), str(last))
        except s.SirilError as exc:
            self.write_log("unselect failed: %s" % exc, "red")
            QtWidgets.QMessageBox.critical(self, TITLE,
                                           "Unselecting failed:\n%s" % exc)
            return
        message = "%d frame(s) unselected in %s." % (len(marked), name)
        self.write_log(message, "green")
        self.siril_log(message)
        QtWidgets.QMessageBox.information(self, TITLE, message)

    def _remove(self, marked, action) -> None:
        seq_mode = self.source[0] == "seq"
        folder = self.source[1]
        if seq_mode:
            try:
                self.siril.cmd("close")   # release the sequence's files
            except s.SirilError:
                pass

        target_dir = os.path.normpath(os.path.join(folder, REJECTED_DIR))
        if action == "move":
            try:
                os.makedirs(target_dir, exist_ok=True)
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self, TITLE, "Cannot create the folder:\n%s" % exc)
                return

        removed, failed = [], 0
        for frame in marked:
            try:
                if action == "delete":
                    os.remove(frame.path)
                else:
                    transfer(frame.path, target_dir, copy=False)
                removed.append(frame)
            except OSError as exc:
                failed += 1
                self.write_log("%s: %s" % (frame.name, exc), "red")

        verb = "deleted" if action == "delete" else "moved to " + target_dir
        message = "%d file(s) %s." % (len(removed), verb)
        if failed:
            message += " %d could not be removed." % failed
        self.write_log(message, "green" if not failed else "salmon")
        self.siril_log(message)

        if seq_mode and removed:
            self._rebuild_sequence()

        gone = {id(f) for f in removed}
        self.frames = [f for f in self.frames if id(f) not in gone]
        self.reevaluate(reset_marks=False)
        if failed:
            QtWidgets.QMessageBox.warning(self, TITLE,
                                          message + "\nSee the log for details.")
        else:
            QtWidgets.QMessageBox.information(self, TITLE, message)

    def on_accept(self) -> None:
        """Copy or move the frames that are kept into the accepted folder."""
        if self.worker and self.worker.is_alive():
            return
        accepted = [f for f in self.frames if not f.marked]
        if not accepted:
            QtWidgets.QMessageBox.information(
                self, TITLE, "Every frame is marked for rejection.")
            return
        if any(not f.path for f in accepted):
            QtWidgets.QMessageBox.warning(
                self, TITLE, "These frames are stored inside one sequence file "
                             "(SER / FITSEQ), so they cannot be copied or moved "
                             "one by one.")
            return

        copy = self.cmb_accepted.currentData() == "copy"
        seq_mode = self.source[0] == "seq"
        folder = self.source[1]
        target_dir = os.path.normpath(os.path.join(folder, ACCEPTED_DIR))

        names = "\n".join("    " + f.name for f in accepted[:15])
        if len(accepted) > 15:
            names += "\n    ... and %d more" % (len(accepted) - 15)
        question = ("%s %d accepted file(s) to\n%s?\n\n%s"
                    % ("Copy" if copy else "Move", len(accepted), target_dir,
                       names))
        existing = list_images(target_dir) if os.path.isdir(target_dir) else []
        if existing:
            question += ("\n\nThe folder already holds %d image file(s). They "
                         "are kept; files with the same name are replaced."
                         % len(existing))
        if seq_mode:
            question += ("\n\nA sequence %s.seq is created in the subfolder."
                         % self.source[2])
            if not copy:
                question += (" The original %s.seq is rebuilt from the files "
                             "that stay behind." % self.source[2])
        answer = QtWidgets.QMessageBox.question(self, TITLE, question)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, TITLE, "Cannot create the folder:\n%s" % exc)
            return
        if seq_mode:
            try:
                # release the files, and let a sequence of the same name be
                # created in the subfolder
                self.siril.cmd("close")
            except s.SirilError:
                pass

        done, failed = [], 0
        self.progress_max.emit(len(accepted))
        for n, frame in enumerate(accepted, start=1):
            try:
                transfer(frame.path, target_dir, copy)
                done.append(frame)
            except OSError as exc:
                failed += 1
                self.write_log("%s: %s" % (frame.name, exc), "red")
            self.progress.setValue(n)
            QtWidgets.QApplication.processEvents()

        message = "%d accepted file(s) %s to %s." % (
            len(done), "copied" if copy else "moved", target_dir)
        if failed:
            message += " %d failed." % failed
        self.write_log(message, "green" if not failed else "salmon")
        self.siril_log(message)

        if seq_mode and done:
            self._create_sequence(target_dir, self.source[2])
            if not copy:
                self._rebuild_sequence()
        if not copy:
            gone = {id(f) for f in done}
            self.frames = [f for f in self.frames if id(f) not in gone]
            self.reevaluate(reset_marks=False)
        if failed:
            QtWidgets.QMessageBox.warning(self, TITLE,
                                          message + "\nSee the log for details.")
        else:
            QtWidgets.QMessageBox.information(self, TITLE, message)

    def _create_sequence(self, folder, name) -> None:
        """Write a fresh .seq for the files of sequence 'name' in folder."""
        seq_file = os.path.join(folder, name + ".seq")
        try:
            if os.path.isfile(seq_file):
                os.remove(seq_file)
            self.siril.cmd("cd", quote(folder))
            if self.siril.create_new_seq(name):
                self.write_log("Sequence %s.seq created in %s."
                               % (name, folder), "green")
            else:
                self.write_log("Siril did not create %s.seq in %s - use "
                               "\"Search sequence\" in the Sequence tab."
                               % (name, folder), "salmon")
        except Exception as exc:
            self.write_log("Could not create %s.seq: %s" % (name, exc), "red")
        finally:
            try:
                self.siril.cmd("cd", quote(self.source[1]))
            except Exception:
                pass

    def _rebuild_sequence(self) -> None:
        """Recreate the analysed sequence from the files left in its folder."""
        self.seq_stale = True
        self._create_sequence(self.source[1], self.source[2])
        self.write_log("Register %s again before stacking - the rebuilt "
                       "sequence has no registration data." % self.source[2],
                       "salmon")

    # -- logging ------------------------------------------------------------

    def post_log(self, text, color=None) -> None:
        """Thread-safe: log into the window and into Siril's log."""
        self.log_line.emit(text, color)
        self.siril_log(text, color)

    def siril_log(self, text, color=None) -> None:
        try:
            if LogColor is not None and color:
                self.siril.log(TITLE + ": " + text,
                               getattr(LogColor, color.upper(),
                                       LogColor.DEFAULT))
            else:
                self.siril.log(TITLE + ": " + text)
        except Exception:
            pass

    def write_log(self, text, color=None) -> None:
        self._append_log(text, color)

    def _append_log(self, message, color=None) -> None:
        colour = LOG_COLOURS[self.theme].get((color or "").lower())
        if colour:
            self.text.appendHtml(
                '<span style="color:%s; white-space:pre">%s</span>'
                % (colour, html.escape(message)))
        else:
            self.text.appendPlainText(message)

    def _set_maximum(self, value) -> None:
        self.progress.setRange(0, max(value, 0))   # 0 = busy indicator
        self.progress.setValue(0)

    # -- closing ------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.is_alive():
            answer = QtWidgets.QMessageBox.question(
                self, TITLE, "The analysis is running. Stop it and close?")
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


def main():
    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilError as exc:
        print("Selector: could not connect to Siril: %s" % exc)
        return

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])
    apply_siril_theme(app, siril)

    window = SelectorWindow(siril)
    window.refresh_source()
    window.show()
    window.raise_()
    window.activateWindow()

    if owns_app:
        app.exec()


if __name__ == "__main__":
    main()
