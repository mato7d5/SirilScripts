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
StarsStatistics - a table of the stars in an image or a sequence (Siril 1.4+).

Runs Siril's star detection ("findstar") either on the image loaded in Siril
or on every frame of a sequence, and lists every star with

  RA, Dec       from the image's plate solution (empty when it has none)
  magnitude     calibrated with a catalogue (APASS, Gaia, NOMAD - read with
                Siril's conesearch): every frame's zero point is fitted to
                the catalogue stars and added to the instrumental magnitude,
                25 - 2.5 log10(flux), which is listed too, as is the
                catalogue's own magnitude
  flux          the integral of the fitted PSF above the background, in
                16-bit ADU (0..65535) whatever the bit depth of the image
  max flux      the star's brightest pixel, 16-bit ADU
  exposure      EXPTIME / EXPOSURE from the FITS header, when present

For a sequence the stars are matched between the frames - by RA/Dec when the
frames are plate solved, by pixel position otherwise (registered frames) - and
either averaged (arithmetic mean or median) or listed frame by frame.

The result is shown as a table in its own window, from where it can be copied
or saved as a CSV file.

Requires Siril >= 1.4 with the Python (sirilpy) interface.
Copy this file into your Siril scripts directory to get it in the script menu.
"""

from __future__ import annotations

__version__ = "1.0.0"

import csv
import html
import io
import math
import os
import re
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import timezone

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


TITLE = "StarsStatistics"

# Siril's log colours, in a light and a dark variant so the embedded log stays
# readable whichever theme Siril is set to.
LOG_COLOURS = {
    "light": {"green": "#1b6e2b", "salmon": "#b34a20", "blue": "#14539a",
              "red": "#b3261e"},
    "dark": {"green": "#7fd18c", "salmon": "#ffb08f", "blue": "#7cb6f2",
             "red": "#ff9b94"},
}

ADU16 = 65535.0                                # float data is 0..1 in Siril
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
EXPOSURE_KEYS = ("EXPTIME", "EXPOSURE", "EXP_TIME", "EXPOSURE_TIME")

# How a sequence is summarised.
RESULT_MEAN, RESULT_MEDIAN, RESULT_FRAMES = "mean", "median", "frames"
# How stars are matched between the frames of a sequence.
MATCH_AUTO, MATCH_SKY, MATCH_PIXEL = "auto", "sky", "pixel"

DELIMITERS = [(";", ";"), (",", ","), ("Tab", "\t")]

# Nominal zero point of the instrumental magnitude, 25 - 2.5 log10(flux).
INSTRUMENTAL_ZP = 25.0

# Circles drawn around the stars in Siril, packed 0xRRGGBBAA.
MARK_COLOUR = 0x40FF40FF
MARK_SATURATED = 0xFF4040FF
MARK_HIGHLIGHT = 0xFFD700FF        # the star double-clicked in the table


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def quote(path: str) -> str:
    """A path or name in the form Siril's parser understands."""
    return '"%s"' % str(path).replace("\\", "/")


FRAME_RE = re.compile(r"^(.*?)(\d+)\.(fit|fits|fts)$", re.IGNORECASE)


def _sequences_in(folder: str) -> list:
    """Sequence names in one folder: 'name' of every .seq file, plus every
    run of numbered FITS frames (name00001.fit, ...) that has no .seq yet -
    Siril writes that .seq only when it first opens the folder."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    found = {name[:-4] for name in names if name.lower().endswith(".seq")}
    counts = {}
    for name in names:
        match = FRAME_RE.match(name)
        if match and match.group(1):
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    lower = {name.lower() for name in found}
    found.update(base for base, count in counts.items()
                 if count >= 2 and base.lower() not in lower)
    return sorted(found, key=str.lower)


def list_sequences(folder: str) -> list:
    """(label, sequence folder, name) of every sequence in the folder and in
    its direct subfolders - Siril keeps its sequences in e.g. 'process'."""
    result = [(name, folder, name) for name in _sequences_in(folder)]
    try:
        subfolders = sorted((d for d in os.listdir(folder)
                             if os.path.isdir(os.path.join(folder, d))),
                            key=str.lower)
    except OSError:
        subfolders = []
    for sub in subfolders:
        path = os.path.join(folder, sub)
        result += [("%s/%s" % (sub, name), path, name)
                   for name in _sequences_in(path)]
    return result


def read_seq_file(folder: str, name: str):
    """(frames, reference) from a Siril .seq file.

    frames is a list of (index, selected, path); reference is the index of
    the reference frame or -1. The .seq is read directly rather than through
    load_seq, which Siril does not allow in headless (siril-cli) scripts.
    """
    with open(os.path.join(folder, name + ".seq"), encoding="utf-8",
              errors="replace") as handle:
        lines = [line.strip() for line in handle]

    header = next((line for line in lines if line.startswith("S ")), None)
    if header is None:
        raise RuntimeError("%s.seq is not a Siril sequence file." % name)
    match = re.match(r"S\s+'(.*)'\s+(.*)$", header)
    numbers = (match.group(2) if match else "").split()
    if len(numbers) < 5:
        raise RuntimeError("%s.seq has an unknown format." % name)
    fixed_len, reference = int(numbers[3]), int(numbers[4])
    if any(re.match(r"T\s*[SFA]", line) for line in lines):
        raise RuntimeError("%s is a single-file sequence (SER / FITSEQ). Its "
                           "frames cannot be loaded one by one - convert it "
                           "to separate FITS files first." % name)

    try:
        files = {f.lower(): f for f in os.listdir(folder)}
    except OSError:
        files = {}
    frames = []
    for line in lines:
        if not line.startswith("I "):
            continue
        parts = line.split()
        filenum, selected = int(parts[1]), parts[2] != "0"
        stem = "%s%0*d" % (name, fixed_len, filenum) if fixed_len \
            else "%s%d" % (name, filenum)
        path = None
        for ext in (".fit", ".fits", ".fts", ".fit.fz", ".fits.fz",
                    ".fts.fz"):
            real = files.get((stem + ext).lower())
            if real:
                path = os.path.join(folder, real)
                break
        frames.append((len(frames), selected, path))
    return frames, reference


def robust_sigma(values: np.ndarray, median: float) -> float:
    """1.4826 x MAD, falling back to the plain standard deviation."""
    if values.size < 2:
        return 0.0
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    if sigma <= 0:
        sigma = float(np.std(values, ddof=1))
    return sigma


def ra_to_hms(ra: float) -> str:
    total = (ra % 360.0) / 15.0 * 3600.0
    total = round(total, 2) % 86400.0
    hours = int(total // 3600)
    minutes = int((total - hours * 3600) // 60)
    seconds = total - hours * 3600 - minutes * 60
    return "%02d:%02d:%05.2f" % (hours, minutes, seconds)


def dec_to_dms(dec: float) -> str:
    sign = "-" if dec < 0 else "+"
    total = round(abs(dec) * 3600.0, 1)
    degrees = int(total // 3600)
    minutes = int((total - degrees * 3600) // 60)
    seconds = total - degrees * 3600 - minutes * 60
    return "%s%02d:%02d:%04.1f" % (sign, degrees, minutes, seconds)


def parse_angle(text: str, hours: bool) -> float | None:
    """Degrees from '21:41:21.5', '21 41 21.5', '21h41m21.5s', '+69°41'34"'
    or a plain number in degrees; RA (hours=True) in sexagesimal form is in
    hours. None for an empty text, ValueError for one that is no angle."""
    text = text.strip().lower()
    if not text:
        return None
    sign = -1.0 if text.startswith("-") else 1.0
    body = text.lstrip("+-").strip()
    sexagesimal = any(ch in body for ch in ":hdms°'\" ")
    for ch in ":hdms°'\"":
        body = body.replace(ch, " ")
    try:
        parts = [float(p.replace(",", ".")) for p in body.split()]
    except ValueError:
        parts = []
    what = "RA" if hours else "Dec"
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts) or \
            any(p >= 60 for p in parts[1:]):
        raise ValueError("'%s' is not a valid %s." % (text, what))
    value = parts[0] + sum(p / 60.0 ** n for n, p in enumerate(parts[1:], 1))
    if hours and sexagesimal:
        value *= 15.0
    value *= sign
    if hours and not 0.0 <= value < 360.0:
        raise ValueError("RA must be within 0..24 h (0..360°).")
    if not hours and not -90.0 <= value <= 90.0:
        raise ValueError("Dec must be within -90..+90°.")
    return value


def separation_arcsec(ra1, dec1, ra2, dec2) -> float:
    """Angular distance of two positions given in degrees."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = math.sin((dec2 - dec1) / 2.0) ** 2 + math.cos(dec1) * math.cos(dec2) \
        * math.sin((ra2 - ra1) / 2.0) ** 2
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a)))) * 3600.0


def normalise_name(name: str) -> str:
    """'EK  Cep' and 'ekcep' compare equal."""
    return "".join(name.split()).lower()


def psf_flux(star) -> float | None:
    """Integral of the fitted PSF above the background, in image units.

    Gaussian: 2 pi A sx sy with s = FWHM / 2.3548.
    Moffat:   pi A ax ay / (beta - 1) with a = FWHM / (2 sqrt(2^(1/beta) - 1)).
    """
    fx, fy, amplitude = star.fwhmx, star.fwhmy, star.A
    if fx <= 0 or fy <= 0 or amplitude <= 0:
        return None
    beta = star.beta
    if int(star.profile) == int(s.StarProfile.MOFFAT) and beta > 1.0:
        k = 2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0)
        return math.pi * amplitude * (fx / k) * (fy / k) / (beta - 1.0)
    return (2.0 * math.pi * amplitude
            * (fx / FWHM_PER_SIGMA) * (fy / FWHM_PER_SIGMA))


def valid_radec(ra, dec) -> bool:
    """Siril reports -1 / -1 (or 0 / 0) for a star without coordinates."""
    if ra is None or dec is None:
        return False
    if (ra == -1.0 and dec == -1.0) or (ra == 0.0 and dec == 0.0):
        return False
    return -90.0 <= dec <= 90.0 and math.isfinite(ra)


def header_number(header, keys):
    """The first of the keys that holds a number in a FITS header dict."""
    if not isinstance(header, dict):
        return None
    for key in keys:
        value = header.get(key)
        if isinstance(value, (list, tuple)):     # (value, comment) pairs
            value = value[0] if value else None
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def header_gain(header) -> float | None:
    """GAIN from a FITS header dict; 0 is a valid gain on most cameras."""
    if not isinstance(header, dict):
        return None
    value = header.get("GAIN")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        gain = float(value)
    except (TypeError, ValueError):
        return None
    return gain if math.isfinite(gain) and gain >= 0 else None


def header_text(header, key) -> str | None:
    """A text card of a FITS header dict, None when missing or blank."""
    if not isinstance(header, dict):
        return None
    value = header.get(key)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def header_date(header) -> str:
    """DATE-OBS exactly as the FITS header has it (UT)."""
    if not isinstance(header, dict):
        return ""
    value = header.get("DATE-OBS")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value).strip() if value else ""


def utc_text(date) -> str:
    """sirilpy hands DATE-OBS over as a naive local time; turn it back to UT."""
    if date is None:
        return ""
    try:
        date = date.astimezone(timezone.utc)
    except (ValueError, OSError):
        pass
    return date.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

@dataclass
class Star:
    """One detected star, with its fluxes already in 16-bit ADU."""

    x: float
    y: float
    ra: float | None = None
    dec: float | None = None
    flux: float | None = None          # PSF integral, 16-bit ADU
    max_flux: float | None = None      # brightest pixel, 16-bit ADU
    layer: int = 0                     # channel the star was detected on
    fwhm: float | None = None          # px, mean of the two axes
    saturated: bool = False
    # the matching star of the photometric catalogue
    cat_mag: float | None = None
    cat_bmag: float | None = None
    cat_mag_err: float | None = None
    cat_dist: float | None = None      # arcsec
    name: str | None = None            # from VSX / SIMBAD
    mag: float | None = None           # mag_inst + the frame's zero point
    match: int = -1                    # internal: same star in other frames
    star_id: int = 0                   # final number, 1 = brightest

    @property
    def mag_inst(self) -> float | None:
        """Instrumental magnitude from the 16-bit flux, on the usual scale
        with a nominal zero point of 25 (as IRAF's phot), so it is positive."""
        if self.flux is None or self.flux <= 0:
            return None
        return INSTRUMENTAL_ZP - 2.5 * math.log10(self.flux)


@dataclass
class FrameResult:
    """The stars measured on one image or one frame of a sequence."""

    index: int
    name: str
    path: str | None = None
    exposure: float | None = None      # s
    gain: float | None = None          # camera gain setting (GAIN)
    filter: str | None = None          # FILTER from the FITS header
    date: str = ""
    float_data: bool = False
    plate_solved: bool = False
    stars: list = field(default_factory=list)
    error: str = ""
    zero_point: float | None = None    # catalogue mag - instrumental mag
    sky_shift: tuple | None = None     # measured - catalogue, arcsec
    zp_sigma: float | None = None
    zp_stars: int = 0


@dataclass
class Measurement:
    """Everything one run of the measuring produced."""

    kind: str                          # "image" or "seq"
    label: str                         # file or sequence name
    folder: str
    frames: list = field(default_factory=list)
    reference: int = 0                 # position in frames, sequences only
    match_mode: str = ""               # what match_stars() actually used
    matched: bool = False
    catalogue: tuple | None = None     # CATALOGUES entry used, if any
    names: bool = False                # star names were looked up


# Where the star names come from, in order of preference: the variable
# star designation first, then SIMBAD's main identifier.
NAME_CATALOGUES = [("vsx", "VSX"), ("simbad", "SIMBAD")]
NAME_LIMIT = 18.0               # faintest magnitude asked for
NAME_RADIUS = 3.0               # arcsec

# key for conesearch -cat=, label, band of "mag", band of "bmag", default
# limiting magnitude
CATALOGUES = [
    ("apass", "APASS - V, B (online)", "V", "B", 17.0),
    ("localgaia", "Gaia DR3 - G (local catalogue, offline)", "G", None, 17.0),
    ("gaia", "Gaia DR3 - G, BP (online)", "G", "BP", 17.0),
    ("nomad", "NOMAD - V, B (online)", "V", "B", 17.0),
]


def read_catalogue_csv(path: str) -> list:
    """(ra, dec, mag, bmag, e_mag) of every star of a conesearch -out= file."""
    def number(row, key):
        try:
            value = float(row.get(key) or "")
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    stars = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            ra, dec = number(row, "ra"), number(row, "dec")
            mag = number(row, "mag")
            if ra is None or dec is None or mag is None:
                continue
            stars.append((ra, dec, mag, number(row, "bmag"),
                          number(row, "e_mag")))
    return stars


class SkyProjection:
    """Offsets in arcsec on the tangent plane around a field centre."""

    def __init__(self, entries):
        self.ra0 = float(np.median([e[0] for e in entries]))
        self.dec0 = float(np.median([e[1] for e in entries]))
        self.cos_dec = math.cos(math.radians(self.dec0))

    def __call__(self, ra, dec):
        dra = ((ra - self.ra0 + 180.0) % 360.0) - 180.0
        return dra * self.cos_dec * 3600.0, (dec - self.dec0) * 3600.0


def _pairs(stars, entries, radius, shift):
    """(distance, star index, entry index, dx, dy) of the unique nearest
    pairs within radius; shift (arcsec) is taken off the star positions."""
    if not entries:
        return []
    project = SkyProjection(entries)
    radius = max(radius, 1e-6)
    grid = {}
    points = []
    for n, entry in enumerate(entries):
        q = project(entry[0], entry[1])
        points.append(q)
        key = (int(math.floor(q[0] / radius)), int(math.floor(q[1] / radius)))
        grid.setdefault(key, []).append(n)

    pairs = []
    for i, st in enumerate(stars):
        if st.ra is None:
            continue
        p = project(st.ra, st.dec)
        p = (p[0] - shift[0], p[1] - shift[1])
        cx = int(math.floor(p[0] / radius))
        cy = int(math.floor(p[1] / radius))
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for n in grid.get((gx, gy), ()):
                    q = points[n]
                    dx, dy = p[0] - q[0], p[1] - q[1]
                    d = math.hypot(dx, dy)
                    if d <= radius:
                        pairs.append((d, i, n, dx, dy))
    pairs.sort()
    result, used_stars, used_entries = [], set(), set()
    for pair in pairs:
        if pair[1] in used_stars or pair[2] in used_entries:
            continue
        used_stars.add(pair[1])
        used_entries.add(pair[2])
        result.append(pair)
    return result


def sky_offset(stars, entries, radius) -> tuple:
    """Systematic offset (arcsec, east and north) of the measured positions
    from the catalogue - from the plate solution and from where Siril puts a
    pixel's centre; typically a fraction of a pixel, but enough to push
    stars out of a tight matching radius. Measured with a wide radius, as the
    median over all pairs; (0, 0) when there are too few."""
    pairs = _pairs(stars, entries, max(3.0 * radius, 10.0), (0.0, 0.0))
    if len(pairs) < 5:
        return 0.0, 0.0
    return (float(np.median([p[3] for p in pairs])),
            float(np.median([p[4] for p in pairs])))


def nearest_pairs(stars: list, entries: list, radius: float,
                  shift=(0.0, 0.0)) -> list:
    """(star index, entry index, distance in arcsec) for every star that has
    a catalogue entry within radius once shift is taken off; entries are
    (ra, dec, ...) tuples.

    Every entry goes to one star at most - the nearest one - so a faint
    detection next to a bright star does not inherit its data.
    """
    return [(i, n, d) for d, i, n, _dx, _dy in
            _pairs(stars, entries, radius, shift)]


def pixel_scale(stars) -> float | None:
    """Arcsec per pixel, from the measured stars' positions and RA / Dec."""
    sky = [st for st in stars if st.ra is not None]
    if len(sky) < 2:
        return None
    first = sky[0]
    scales = []
    for other in sky[1:300]:
        pixels = math.hypot(other.x - first.x, other.y - first.y)
        if pixels > 100:
            scales.append(separation_arcsec(first.ra, first.dec,
                                            other.ra, other.dec) / pixels)
    return float(np.median(scales)) if scales else None


def match_radius(stars, radius) -> float:
    """The matching radius, but never under 1.5 pixels: at a coarse scale
    (e.g. 3"/px) a few arcsec is less than the measuring precision."""
    scale = pixel_scale(stars)
    return max(radius, 1.5 * scale) if scale else radius


def match_catalogue(stars: list, catalogue: list, radius: float,
                    shift=(0.0, 0.0)) -> int:
    """Give each star the nearest catalogue star within radius (arcsec)
    once shift is taken off. Returns how many stars got a catalogue star."""
    for st in stars:
        st.cat_mag = st.cat_bmag = st.cat_mag_err = st.cat_dist = None
    pairs = nearest_pairs(stars, catalogue, radius, shift)
    for i, n, d in pairs:
        st = stars[i]
        _ra, _dec, st.cat_mag, st.cat_bmag, st.cat_mag_err = catalogue[n]
        st.cat_dist = d
    return len(pairs)


def clean_name(name: str) -> str:
    """'V* EK Cep' -> 'EK Cep', 'bet   Cep  ' -> 'bet Cep'."""
    name = " ".join(name.split())
    for prefix in ("V* ", "** ", "* ", "NAME "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def read_names_csv(path: str) -> list:
    """(ra, dec, name) of every named object of a conesearch -out= file."""
    entries = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                ra, dec = float(row.get("ra") or ""), float(row.get("dec") or "")
            except ValueError:
                continue
            name = clean_name(row.get("name") or "")
            if name:
                entries.append((ra, dec, name))
    return entries


def match_names(stars: list, sources: list, radius: float,
                shift=(0.0, 0.0)) -> int:
    """Name the stars from (ra, dec, name) lists; an earlier list wins.
    Returns how many stars got a name."""
    for st in stars:
        st.name = None
    for entries in sources:
        unnamed = [st for st in stars if st.name is None]
        for i, n, _d in nearest_pairs(unnamed, entries, radius, shift):
            unnamed[i].name = entries[n][2]
    return sum(1 for st in stars if st.name)


def calibrate(frame: FrameResult) -> None:
    """Zero point of one frame from its catalogue-matched stars, then
    mag = mag_inst + zero point for every star.

    The zero point is the median of (catalogue mag - instrumental mag) over
    the unsaturated matched stars, with outliers (variables, blends, wrong
    matches) clipped at 3 sigma.
    """
    frame.zero_point = frame.zp_sigma = None
    frame.zp_stars = 0
    for st in frame.stars:
        st.mag = None
    diffs = np.array([st.cat_mag - st.mag_inst for st in frame.stars
                      if st.cat_mag is not None and st.mag_inst is not None
                      and not st.saturated], dtype=float)
    if diffs.size < 3:
        return
    keep = diffs
    for _ in range(5):
        centre = float(np.median(keep))
        sigma = robust_sigma(keep, centre)
        clipped = diffs[np.abs(diffs - centre) <= 3.0 * sigma] \
            if sigma > 0 else keep
        if clipped.size < 3 or clipped.size == keep.size:
            keep = clipped if clipped.size >= 3 else keep
            break
        keep = clipped
    frame.zero_point = float(np.median(keep))
    frame.zp_sigma = robust_sigma(keep, frame.zero_point)
    frame.zp_stars = int(keep.size)
    for st in frame.stars:
        if st.mag_inst is not None:
            st.mag = st.mag_inst + frame.zero_point


# --------------------------------------------------------------------------- #
#  Matching and summarising
# --------------------------------------------------------------------------- #

def match_stars(measurement: Measurement, mode: str, tol_px: float,
                tol_arcsec: float) -> str:
    """Give the same Star.match to the detections of one star in every frame.

    Starts from the reference frame; a star that matches nothing already
    catalogued (within the tolerance) starts a new catalogue entry. Every
    catalogue entry takes at most one star per frame, the nearest one.
    Returns the mode used: MATCH_SKY or MATCH_PIXEL.
    """
    frames = [f for f in measurement.frames if f.stars]
    if mode == MATCH_AUTO:
        sky = bool(frames) and all(
            sum(1 for st in f.stars if st.ra is not None) * 2 >= len(f.stars)
            for f in frames)
        mode = MATCH_SKY if sky else MATCH_PIXEL
    elif mode == MATCH_SKY and not any(st.ra is not None
                                       for f in frames for st in f.stars):
        mode = MATCH_PIXEL              # no frame is plate solved

    reference = measurement.frames[measurement.reference] \
        if 0 <= measurement.reference < len(measurement.frames) else None
    if reference is None or not reference.stars:
        reference = max(frames, key=lambda f: len(f.stars), default=None)
    if reference is not None:
        measurement.reference = measurement.frames.index(reference)

    if mode == MATCH_SKY:
        with_sky = [st for st in (reference.stars if reference else [])
                    if st.ra is not None]
        if not with_sky:
            with_sky = [st for f in frames for st in f.stars
                        if st.ra is not None]
        if with_sky:
            ra0 = float(np.median([st.ra for st in with_sky]))
            dec0 = float(np.median([st.dec for st in with_sky]))
        else:
            ra0 = dec0 = 0.0
        cos_dec = math.cos(math.radians(dec0))
        tolerance = tol_arcsec

        def position(st):
            if st.ra is None:
                return None
            dra = ((st.ra - ra0 + 180.0) % 360.0) - 180.0
            return dra * cos_dec * 3600.0, (st.dec - dec0) * 3600.0
    else:
        tolerance = tol_px

        def position(st):
            return st.x, st.y

    tolerance = max(tolerance, 1e-6)
    catalogue = []                      # positions of the catalogue entries
    grid = {}                           # cell -> catalogue indexes

    def cell(p):
        return int(math.floor(p[0] / tolerance)), \
            int(math.floor(p[1] / tolerance))

    order = ([reference] if reference else []) + \
        [f for f in frames if f is not reference]
    for frame in order:
        pairs = []
        for n, st in enumerate(frame.stars):
            st.match = -1
            p = position(st)
            if p is None:
                continue
            cx, cy = cell(p)
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for entry in grid.get((gx, gy), ()):
                        q = catalogue[entry]
                        d = math.hypot(p[0] - q[0], p[1] - q[1])
                        if d <= tolerance:
                            pairs.append((d, n, entry))
        pairs.sort()
        used_stars, used_entries = set(), set()
        for _d, n, entry in pairs:
            if n in used_stars or entry in used_entries:
                continue
            frame.stars[n].match = entry
            used_stars.add(n)
            used_entries.add(entry)
        # new entries only after the frame, so that two stars of one frame
        # never end up as the same star
        for st in frame.stars:
            if st.match >= 0:
                continue
            p = position(st)
            if p is None:
                continue
            st.match = len(catalogue)
            catalogue.append(p)
            grid.setdefault(cell(p), []).append(st.match)

    number_stars(measurement)
    measurement.match_mode = mode
    measurement.matched = True
    return mode


def number_stars(measurement: Measurement) -> None:
    """Number the stars 1, 2, ... from the brightest (median flux) down."""
    groups = {}
    for frame in measurement.frames:
        for n, st in enumerate(frame.stars):
            key = st.match if st.match >= 0 else ("single", id(frame), n)
            groups.setdefault(key, []).append(st)
    if measurement.kind == "seq":
        # unmatched stars (no coordinates) cannot be followed between frames
        groups = {k: v for k, v in groups.items() if not isinstance(k, tuple)}
        for frame in measurement.frames:
            for st in frame.stars:
                st.star_id = 0

    def brightness(stars):
        fluxes = [st.flux for st in stars if st.flux is not None]
        return float(np.median(fluxes)) if fluxes else -1.0

    ranked = sorted(groups.values(), key=brightness, reverse=True)
    for number, stars in enumerate(ranked, start=1):
        for st in stars:
            st.star_id = number


def combine(values, method):
    """(value, spread) of a list with None entries skipped."""
    data = np.array([v for v in values if v is not None], dtype=float)
    if not data.size:
        return None, None
    if method == RESULT_MEDIAN:
        centre = float(np.median(data))
        return centre, (robust_sigma(data, centre) if data.size > 1 else None)
    centre = float(np.mean(data))
    return centre, (float(np.std(data, ddof=1)) if data.size > 1 else None)


def most_common(values):
    """The most frequent non-empty value, or None."""
    counts = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else None


def combine_ra(values, method):
    values = [v for v in values if v is not None]
    if not values:
        return None
    ref = values[0]
    unwrapped = [ref + ((v - ref + 180.0) % 360.0) - 180.0 for v in values]
    centre, _spread = combine(unwrapped, method)
    return centre % 360.0


# --------------------------------------------------------------------------- #
#  Table
# --------------------------------------------------------------------------- #

@dataclass
class Column:
    key: str
    header: str
    spec: str = "%s"            # printf format for numbers
    tooltip: str = ""


COLUMN_INFO = {
    "frame": ("frame", "%d", "Frame number in the sequence (1-based)."),
    "file": ("file", "%s", "File of the frame."),
    "name": ("name", "%s", "Name of the star: its variable star "
                           "designation from VSX, else SIMBAD's main "
                           "identifier. Empty for an uncatalogued star."),
    "star": ("star", "%d", "Star number - 1 is the brightest. In a sequence "
                           "the same star has the same number in every "
                           "frame."),
    "n_frames": ("n_frames", "%d", "In how many frames the star was found."),
    "x": ("x", "%.2f", "Position in the image, px."),
    "y": ("y", "%.2f", "Position in the image, px."),
    "x_ref": ("x_ref", "%.2f", "Position in the reference frame, px."),
    "y_ref": ("y_ref", "%.2f", "Position in the reference frame, px."),
    "ra_deg": ("ra_deg", "%.6f", "Right ascension, degrees (J2000). Empty "
                                 "when the image is not plate solved."),
    "dec_deg": ("dec_deg", "%.6f", "Declination, degrees (J2000)."),
    "ra_hms": ("ra_hms", "%s", "Right ascension, hh:mm:ss."),
    "dec_dms": ("dec_dms", "%s", "Declination, ±dd:mm:ss."),
    "mag": ("mag", "%.3f", "Calibrated magnitude: mag_inst + the zero "
                           "point of the frame, fitted to the catalogue "
                           "stars. "
                           "Given for every star, also those missing in the "
                           "catalogue."),
    "mag_sigma": ("mag_sigma", "%.4f", "Scatter of mag (of mag_inst "
                                       "without a catalogue) between the "
                                       "frames: standard deviation for the "
                                       "mean, 1.4826 x MAD for the median."),
    "mag_inst": ("mag_inst", "%.4f", "Instrumental magnitude, "
                                     "25 - 2.5 log10(flux_adu16). Not "
                                     "calibrated, but fine for differential "
                                     "photometry within a frame."),
    "cat_mag": ("cat_mag", "%.3f", "Magnitude of the matching catalogue "
                                   "star."),
    "cat_bmag": ("cat_bmag", "%.3f", "Second (blue) magnitude of the "
                                     "matching catalogue star."),
    "cat_mag_err": ("cat_err", "%.3f", "Error of the catalogue magnitude."),
    "cat_dist": ("cat_dist_arcsec", "%.2f", "Distance to the matching "
                                            "catalogue star, arcsec."),
    "zero_point": ("zero_point", "%.4f", "Zero point of the frame: median "
                                         "of cat_mag - mag_inst over the "
                                         "unsaturated catalogue stars; "
                                         "mag = mag_inst + zero_point."),
    "flux_adu16": ("flux_adu16", "%.1f", "Integral of the fitted PSF above "
                                         "the background, in 16-bit ADU "
                                         "(0..65535 scale)."),
    "max_flux": ("max_flux", "%.1f", "Value of the star's brightest pixel, "
                                     "background included, 16-bit ADU. At "
                                     "65535 the star is saturated."),
    "fwhm_px": ("fwhm_px", "%.3f", "FWHM, mean of both axes, px."),
    "saturated": ("saturated", "%d", "1 when the star has saturated "
                                     "pixels."),
    "saturated_frames": ("saturated_frames", "%d", "In how many frames the "
                                                   "star was saturated."),
    "exposure_s": ("exposure_s", "%.3f", "Exposure time from the FITS "
                                         "header, s."),
    "gain": ("gain", "%g", "Camera gain setting, GAIN in the FITS header "
                           "(e.g. 100 on a ZWO camera)."),
    "filter": ("filter", "%s", "FILTER from the FITS header; empty when the "
                               "header has none."),
    "date_obs": ("date_obs", "%s", "DATE-OBS of the frame, UT."),
}


def make_columns(keys, catalogue=None):
    """Columns for keys; the catalogue columns are named after its bands,
    e.g. cat_V, cat_B for APASS."""
    columns = [Column(key, *COLUMN_INFO[key]) for key in keys]
    if catalogue is not None:
        bands = {"cat_mag": catalogue[2], "cat_bmag": catalogue[3]}
        for column in columns:
            if bands.get(column.key):
                column.header = "cat_" + bands[column.key]
                column.tooltip += " %s %s." % (catalogue[1].split(" - ")[0],
                                               bands[column.key])
    return columns


def catalogue_keys(catalogue, averaged=False) -> list:
    """The magnitude columns: always the instrumental magnitude, and with a
    catalogue the calibrated one and the catalogue's."""
    sigma = ["mag_sigma"] if averaged else []
    if catalogue is None:
        return ["mag_inst"] + sigma
    keys = ["mag"] + sigma + ["mag_inst", "cat_mag"]
    if catalogue[3]:
        keys.append("cat_bmag")
    if catalogue[0] == "apass":
        keys.append("cat_mag_err")
    return keys + ([] if averaged else ["cat_dist"])


def star_row_values(st: Star, frame: FrameResult) -> dict:
    return {
        "star": st.star_id or None, "name": st.name, "x": st.x, "y": st.y,
        "ra_deg": st.ra, "dec_deg": st.dec,
        "ra_hms": ra_to_hms(st.ra) if st.ra is not None else None,
        "dec_dms": dec_to_dms(st.dec) if st.dec is not None else None,
        "mag": st.mag, "mag_inst": st.mag_inst, "cat_mag": st.cat_mag,
        "cat_bmag": st.cat_bmag, "cat_mag_err": st.cat_mag_err,
        "cat_dist": st.cat_dist, "zero_point": frame.zero_point,
        "flux_adu16": st.flux, "max_flux": st.max_flux, "fwhm_px": st.fwhm,
        "saturated": int(st.saturated),
        "exposure_s": frame.exposure, "gain": frame.gain,
        "filter": frame.filter,
        "date_obs": frame.date or None,
    }


def build_table(measurement: Measurement, result: str, hms: bool,
                skip_saturated: bool, min_percent: float):
    """(columns, rows, summary) for the result window."""
    sexa = ["ra_hms", "dec_dms"] if hms else []
    names = ["name"] if measurement.names else []
    catalogue = measurement.catalogue

    def keep(st):
        return not (skip_saturated and st.saturated)

    if measurement.kind == "image" or result == RESULT_FRAMES:
        lead = [] if measurement.kind == "image" else ["frame", "file"]
        zero = ["zero_point"] if catalogue and measurement.kind == "seq" \
            else []
        keys = (lead + ["star"] + names + ["x", "y", "ra_deg", "dec_deg"]
                + sexa
                + catalogue_keys(catalogue) + zero
                + ["flux_adu16", "max_flux", "fwhm_px", "saturated",
                   "exposure_s", "gain", "filter", "date_obs"])
        rows = []
        for frame in measurement.frames:
            stars = sorted((st for st in frame.stars if keep(st)),
                           key=lambda st: (st.star_id or 10 ** 9,
                                           -(st.flux or 0.0)))
            for st in stars:
                values = star_row_values(st, frame)
                values["frame"] = frame.index + 1
                values["file"] = frame.name
                rows.append([values[k] for k in keys])
        count = len(rows)
        summary = "%d star(s)" % count
        if measurement.kind == "seq":
            summary += " in %d frame(s)" % sum(
                1 for f in measurement.frames if f.stars)
        return make_columns(keys, catalogue), rows, summary

    # one row per star, combined over the frames
    keys = (["star"] + names + ["n_frames", "x_ref", "y_ref", "ra_deg",
                                "dec_deg"] + sexa
            + catalogue_keys(catalogue, averaged=True)
            + ["flux_adu16", "max_flux", "fwhm_px", "saturated_frames",
               "exposure_s", "gain", "filter"])
    used_frames = [f for f in measurement.frames if f.stars]
    min_count = max(1, int(math.ceil(len(used_frames) * min_percent / 100.0)))
    reference = (measurement.frames[measurement.reference]
                 if 0 <= measurement.reference < len(measurement.frames)
                 else None)

    groups = {}
    for frame in used_frames:
        for st in frame.stars:
            if st.star_id:
                groups.setdefault(st.star_id, []).append((frame, st))

    rows = []
    dropped = 0
    for star_id in sorted(groups):
        detections = groups[star_id]
        good = [(f, st) for f, st in detections if keep(st)]
        if len(good) < min_count:
            dropped += 1
            continue
        stars = [st for _f, st in good]
        mag, mag_sigma = combine([st.mag for st in stars], result)
        mag_inst, inst_sigma = combine([st.mag_inst for st in stars], result)
        if catalogue is None:
            mag_sigma = inst_sigma
        ref_star = next((st for f, st in detections if f is reference), None)
        values = {
            "star": star_id, "n_frames": len(good),
            "name": most_common([st.name for st in stars]),
            "x_ref": ref_star.x if ref_star else None,
            "y_ref": ref_star.y if ref_star else None,
            "ra_deg": combine_ra([st.ra for st in stars], result),
            "dec_deg": combine([st.dec for st in stars], result)[0],
            "mag": mag, "mag_sigma": mag_sigma, "mag_inst": mag_inst,
            # the same catalogue star in every frame; the median keeps a
            # single frame that matched a neighbour from changing it
            "cat_mag": combine([st.cat_mag for st in stars],
                               RESULT_MEDIAN)[0],
            "cat_bmag": combine([st.cat_bmag for st in stars],
                                RESULT_MEDIAN)[0],
            "cat_mag_err": combine([st.cat_mag_err for st in stars],
                                   RESULT_MEDIAN)[0],
            "flux_adu16": combine([st.flux for st in stars], result)[0],
            "max_flux": combine([st.max_flux for st in stars], result)[0],
            "fwhm_px": combine([st.fwhm for st in stars], result)[0],
            "saturated_frames": sum(1 for _f, st in detections
                                    if st.saturated),
            "exposure_s": combine([f.exposure for f, _st in good],
                                  result)[0],
            "gain": combine([f.gain for f, _st in good], RESULT_MEDIAN)[0],
            "filter": most_common([f.filter for f, _st in good]),
        }
        values["ra_hms"] = (ra_to_hms(values["ra_deg"])
                            if values["ra_deg"] is not None else None)
        values["dec_dms"] = (dec_to_dms(values["dec_deg"])
                             if values["dec_deg"] is not None else None)
        rows.append([values[k] for k in keys])

    summary = "%d star(s), %s of %d frame(s), found in at least %d frame(s)" \
        % (len(rows), "mean" if result == RESULT_MEAN else "median",
           len(used_frames), min_count)
    if dropped:
        summary += "; %d star(s) found in fewer frames left out" % dropped
    return make_columns(keys, catalogue), rows, summary


def format_value(value, spec, decimal_comma=False) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, str):
        return value
    text = spec % value
    return text.replace(".", ",") if decimal_comma else text


def to_csv(columns, rows, delimiter=";", decimal_comma=False) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow([c.header for c in columns])
    for row in rows:
        writer.writerow([format_value(v, c.spec, decimal_comma)
                         for v, c in zip(row, columns)])
    return buffer.getvalue()


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
#  Result window
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


class ResultWindow(QtWidgets.QWidget):
    """The statistics as a table and as CSV text, with copy and save."""

    def __init__(self, title, summary, columns, rows, folder, stem,
                 on_log=None, on_star=None):
        super().__init__()
        self.columns = columns
        self.rows = rows
        self.folder = folder
        self.stem = stem
        self.on_log = on_log
        self.on_star = on_star
        self.setWindowTitle("%s - %s" % (TITLE, title))
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        label = QtWidgets.QLabel("<b>%s</b>  ·  %s"
                                 % (html.escape(title), html.escape(summary)))
        label.setWordWrap(True)
        outer.addWidget(label)

        self.tabs = QtWidgets.QTabWidget()
        self.table = QtWidgets.QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels([c.header for c in columns])
        for n, column in enumerate(columns):
            header_item = self.table.horizontalHeaderItem(n)
            if header_item is not None:
                header_item.setToolTip(column.tooltip)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        if on_star is not None:
            self.table.setToolTip("Double-click a row to show the star in "
                                  "Siril.")
            self.table.cellDoubleClicked.connect(self._row_double_clicked)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive)
        self._fill_table()
        self.column_of = {c.key: n for n, c in enumerate(columns)}
        self.target = None          # (ra, dec, radius) of a position search
        # sorting moves the rows but not their hidden state - filter again
        # once the table has been sorted
        self.table.horizontalHeader().sortIndicatorChanged.connect(
            lambda *_a: QtCore.QTimer.singleShot(0, self.apply_filter))
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 4, 0, 0)
        page_layout.addLayout(self._build_search_bar())
        page_layout.addWidget(self.table, 1)
        self.tabs.addTab(page, "Table")

        self.csv_text = QtWidgets.QPlainTextEdit()
        self.csv_text.setReadOnly(True)
        self.csv_text.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.csv_text.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont))
        self.tabs.addTab(self.csv_text, "CSV")
        outer.addWidget(self.tabs, 1)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Delimiter:"))
        self.cmb_delimiter = QtWidgets.QComboBox()
        for text, value in DELIMITERS:
            self.cmb_delimiter.addItem(text, value)
        bar.addWidget(self.cmb_delimiter)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Decimal separator:"))
        self.cmb_decimal = QtWidgets.QComboBox()
        self.cmb_decimal.addItem(". (point)", False)
        self.cmb_decimal.addItem(", (comma)", True)
        self.cmb_decimal.setToolTip("A decimal comma suits a spreadsheet set "
                                    "to e.g. Slovak or German. It cannot be "
                                    "combined with the comma delimiter.")
        bar.addWidget(self.cmb_decimal)
        bar.addStretch(1)
        for text, slot in (("Copy CSV", self.copy_csv),
                           ("Save CSV...", self.save_csv),
                           ("Close", self.close)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            bar.addWidget(button)
        outer.addLayout(bar)

        self.cmb_delimiter.currentIndexChanged.connect(self._options_changed)
        self.cmb_decimal.currentIndexChanged.connect(self._options_changed)
        self._options_changed()
        self._fit_to_screen()

    def _fill_table(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            for c, (value, column) in enumerate(zip(row, self.columns)):
                text = format_value(value, column.spec)
                if isinstance(value, (int, float)) and not isinstance(
                        value, bool):
                    item = NumericItem(text, float(value)
                                       if math.isfinite(value) else None)
                else:
                    item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, r)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    # -- searching ----------------------------------------------------------

    def _build_search_bar(self):
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Name:"))
        self.ed_name = QtWidgets.QLineEdit()
        self.ed_name.setPlaceholderText("e.g. EK Cep")
        self.ed_name.setClearButtonEnabled(True)
        self.ed_name.setToolTip("Shows only the stars whose name contains "
                                "this text (case and spaces do not matter).")
        self.ed_name.textChanged.connect(self.apply_filter)
        bar.addWidget(self.ed_name, 2)
        if "name" not in self.column_of:
            self.ed_name.setEnabled(False)
            self.ed_name.setPlaceholderText("no star names in this table")

        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("RA:"))
        self.ed_ra = QtWidgets.QLineEdit()
        self.ed_ra.setPlaceholderText("21:41:21.5 or 325.34")
        self.ed_ra.setToolTip("Right ascension: hh:mm:ss (also '21 41 21.5' "
                              "or '21h41m21.5s'), or degrees.")
        bar.addWidget(self.ed_ra, 1)
        bar.addWidget(QtWidgets.QLabel("Dec:"))
        self.ed_dec = QtWidgets.QLineEdit()
        self.ed_dec.setPlaceholderText("+69:41:34 or 69.693")
        self.ed_dec.setToolTip("Declination: ±dd:mm:ss (also '69 41 34' or "
                               "'69d41m34s'), or degrees.")
        bar.addWidget(self.ed_dec, 1)
        bar.addWidget(QtWidgets.QLabel("within"))
        self.spin_radius = QtWidgets.QDoubleSpinBox()
        self.spin_radius.setRange(0.5, 36000.0)
        self.spin_radius.setDecimals(1)
        self.spin_radius.setValue(10.0)
        self.spin_radius.setSuffix(" arcsec")
        bar.addWidget(self.spin_radius)
        find = QtWidgets.QPushButton("Find")
        find.setToolTip("Shows the stars within the radius of RA / Dec, "
                        "nearest first, and selects the nearest one.")
        find.clicked.connect(self.find_position)
        self.ed_ra.returnPressed.connect(self.find_position)
        self.ed_dec.returnPressed.connect(self.find_position)
        bar.addWidget(find)
        clear = QtWidgets.QPushButton("Show all")
        clear.clicked.connect(self.clear_search)
        bar.addWidget(clear)
        if "ra_deg" not in self.column_of or not any(
                row[self.column_of["ra_deg"]] is not None
                for row in self.rows):
            for widget in (self.ed_ra, self.ed_dec, self.spin_radius, find):
                widget.setEnabled(False)
            self.ed_ra.setPlaceholderText("not plate solved")
            self.ed_dec.setPlaceholderText("")

        self.search_label = QtWidgets.QLabel("")
        bar.addWidget(self.search_label)
        return bar

    def find_position(self) -> None:
        try:
            ra = parse_angle(self.ed_ra.text(), hours=True)
            dec = parse_angle(self.ed_dec.text(), hours=False)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, TITLE, str(exc))
            return
        if ra is None or dec is None:
            QtWidgets.QMessageBox.warning(self, TITLE,
                                          "Enter both RA and Dec.")
            return
        self.target = (ra, dec, self.spin_radius.value())
        self.apply_filter()

    def clear_search(self) -> None:
        self.target = None
        self.ed_name.blockSignals(True)
        self.ed_name.clear()
        self.ed_name.blockSignals(False)
        self.apply_filter()

    def apply_filter(self, *_args) -> None:
        """Hide the rows that do not match the name and the position."""
        wanted = normalise_name(self.ed_name.text())
        name_col = self.column_of.get("name")
        ra_col, dec_col = self.column_of.get("ra_deg"), \
            self.column_of.get("dec_deg")
        nearest, nearest_d, shown = None, None, 0
        for r in range(self.table.rowCount()):
            row = self.rows[self.table.item(r, 0).data(
                QtCore.Qt.ItemDataRole.UserRole)]
            visible = True
            if wanted and name_col is not None:
                visible = wanted in normalise_name(row[name_col] or "")
            if visible and self.target is not None:
                ra, dec = row[ra_col], row[dec_col]
                if ra is None or dec is None:
                    visible = False
                else:
                    d = separation_arcsec(ra, dec, self.target[0],
                                          self.target[1])
                    visible = d <= self.target[2]
                    if visible and (nearest_d is None or d < nearest_d):
                        nearest, nearest_d = r, d
            self.table.setRowHidden(r, not visible)
            shown += visible
        if not wanted and self.target is None:
            self.search_label.setText("")
            return
        text = "%d of %d row(s)" % (shown, self.table.rowCount())
        if nearest is not None:
            text += ", nearest %.1f″" % nearest_d
            self.table.selectRow(nearest)
            self.table.scrollToItem(self.table.item(nearest, 0))
        elif shown:
            first = next(r for r in range(self.table.rowCount())
                         if not self.table.isRowHidden(r))
            self.table.scrollToItem(self.table.item(first, 0))
        self.search_label.setText(text)

    def _row_double_clicked(self, row, _col) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        index = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.on_star(dict(zip((c.key for c in self.columns),
                              self.rows[index])))

    def _fit_to_screen(self) -> None:
        available = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        width = min(1250, int(available.width() * 0.9))
        height = min(760, int(available.height() * 0.85))
        self.resize(width, height)
        self.move(available.x() + (available.width() - width) // 2 + 30,
                  available.y() + (available.height() - height) // 3 + 30)

    def _options_changed(self, *_args) -> None:
        # a decimal comma and a comma delimiter would make the file unreadable
        if self.cmb_decimal.currentData() and \
                self.cmb_delimiter.currentData() == ",":
            sender = self.sender()
            if sender is self.cmb_delimiter:
                self.cmb_decimal.blockSignals(True)
                self.cmb_decimal.setCurrentIndex(0)
                self.cmb_decimal.blockSignals(False)
            else:
                self.cmb_delimiter.blockSignals(True)
                self.cmb_delimiter.setCurrentIndex(0)
                self.cmb_delimiter.blockSignals(False)
        self.csv_text.setPlainText(self.csv())

    def csv(self) -> str:
        return to_csv(self.columns, self.rows,
                      self.cmb_delimiter.currentData(),
                      bool(self.cmb_decimal.currentData()))

    def copy_csv(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.csv())
        if self.on_log:
            self.on_log("CSV copied to the clipboard (%d row(s))."
                        % len(self.rows), "green")

    def save_csv(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the star statistics",
            os.path.join(self.folder or os.getcwd(),
                         "%s_stars.csv" % self.stem),
            "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                handle.write(self.csv())
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, TITLE,
                                           "Could not write the file:\n%s"
                                           % exc)
            return
        if self.on_log:
            self.on_log("Statistics saved to %s" % path, "green")


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #

class StarsStatisticsWindow(QtWidgets.QWidget):
    """Main window; measuring runs on its own thread.

    The worker reports back through Qt signals, which Qt delivers on the GUI
    thread, so no widget is touched from the wrong thread.
    """

    log_line = QtCore.pyqtSignal(str, object)
    status_changed = QtCore.pyqtSignal(str)
    progress_max = QtCore.pyqtSignal(int)
    progress_changed = QtCore.pyqtSignal(int)
    measure_finished = QtCore.pyqtSignal(object)
    marking_finished = QtCore.pyqtSignal()

    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.measurement = None
        self.worker = None
        self.cancel = threading.Event()
        self.theme = "dark" if siril_is_dark(siril) else "light"
        self.result_windows = []
        self.highlight_ids = []       # overlay polygons of the shown star

        self.setWindowTitle(TITLE + " - star statistics v" + __version__)
        self._build_widgets()

        self.log_line.connect(self._append_log)
        self.status_changed.connect(self.status.setText)
        self.progress_max.connect(self._set_maximum)
        self.progress_changed.connect(self.progress.setValue)
        self.measure_finished.connect(self.on_measure_finished)
        self.marking_finished.connect(self.on_marking_finished)

        try:
            folder = siril.get_siril_wd()
        except Exception:
            folder = ""
        self.ed_folder.setText(folder or os.getcwd())
        self._select_initial_source()
        self._fit_to_screen()

    # -- layout -------------------------------------------------------------

    def _build_widgets(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._build_source_box())
        outer.addWidget(self._build_sequence_box())
        outer.addWidget(self._build_catalogue_box())
        outer.addWidget(self._build_output_box())

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        outer.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Ready.")
        outer.addWidget(self.status)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setMinimumHeight(90)
        outer.addWidget(self.text, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.table_button = QtWidgets.QPushButton("Show table again")
        self.table_button.setToolTip(
            "Builds a new table from the last measurement with the options "
            "set now - no need to measure again.")
        self.table_button.setEnabled(False)
        self.table_button.clicked.connect(self.show_table)
        buttons.addWidget(self.table_button)
        buttons.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("Stop")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel.set)
        buttons.addWidget(self.cancel_button)
        self.measure_button = QtWidgets.QPushButton("Measure")
        self.measure_button.setDefault(True)
        self.measure_button.clicked.connect(self.on_measure)
        buttons.addWidget(self.measure_button)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.close)
        buttons.addWidget(btn)
        outer.addLayout(buttons)

    def _build_source_box(self):
        box = QtWidgets.QGroupBox("Source")
        grid = QtWidgets.QGridLayout(box)

        self.rb_image = QtWidgets.QRadioButton("Image loaded in Siril:")
        self.rb_image.toggled.connect(self._update_mode)
        grid.addWidget(self.rb_image, 0, 0)
        self.image_label = QtWidgets.QLabel("")
        grid.addWidget(self.image_label, 0, 1, 1, 2)
        btn = QtWidgets.QPushButton("Refresh")
        btn.setToolTip("Check again which image is loaded in Siril.")
        btn.clicked.connect(self.refresh_image)
        grid.addWidget(btn, 0, 3)

        self.chk_existing = QtWidgets.QCheckBox(
            "Use the stars already detected in Siril (do not run findstar)")
        self.chk_existing.setToolTip(
            "Keeps the star list you built in Siril's Dynamic PSF window, e.g. "
            "stars you picked by hand. Without it findstar runs again and "
            "replaces that list.")
        grid.addWidget(self.chk_existing, 1, 1, 1, 3)

        self.rb_seq = QtWidgets.QRadioButton("Sequence:")
        self.rb_seq.toggled.connect(self._update_mode)
        grid.addWidget(self.rb_seq, 2, 0)
        self.ed_folder = QtWidgets.QLineEdit()
        self.ed_folder.setToolTip("Folder that holds the sequence (.seq), or "
                                  "its parent folder.")
        self.ed_folder.textChanged.connect(self.refresh_sequences)
        grid.addWidget(self.ed_folder, 2, 1, 1, 2)
        btn = QtWidgets.QPushButton("Browse...")
        btn.clicked.connect(self.pick_folder)
        grid.addWidget(btn, 2, 3)

        self.cmb_seq = QtWidgets.QComboBox()
        self.cmb_seq.setPlaceholderText("no sequence found in this folder")
        self.cmb_seq.activated.connect(lambda _i: self.rb_seq.setChecked(True))
        grid.addWidget(self.cmb_seq, 3, 1, 1, 3)

        self.chk_selected = QtWidgets.QCheckBox(
            "Only the frames selected in the sequence")
        self.chk_selected.setChecked(True)
        grid.addWidget(self.chk_selected, 4, 1, 1, 3)
        grid.setColumnStretch(1, 1)
        return box

    def _build_sequence_box(self):
        self.seq_box = QtWidgets.QGroupBox("Sequence result")
        grid = QtWidgets.QGridLayout(self.seq_box)

        grid.addWidget(QtWidgets.QLabel("Result:"), 0, 0)
        self.cmb_result = QtWidgets.QComboBox()
        self.cmb_result.addItem("Arithmetic mean over the frames",
                                RESULT_MEAN)
        self.cmb_result.addItem("Median over the frames", RESULT_MEDIAN)
        self.cmb_result.addItem("Every frame separately (no averaging)",
                                RESULT_FRAMES)
        self.cmb_result.currentIndexChanged.connect(self._update_mode)
        grid.addWidget(self.cmb_result, 0, 1, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Match stars by:"), 1, 0)
        self.cmb_match = QtWidgets.QComboBox()
        self.cmb_match.addItem("Automatic", MATCH_AUTO)
        self.cmb_match.addItem("RA / Dec (plate-solved frames)", MATCH_SKY)
        self.cmb_match.addItem("Pixel position (registered frames)",
                               MATCH_PIXEL)
        self.cmb_match.setToolTip(
            "Automatic uses RA/Dec when every frame is plate solved, the pixel "
            "position otherwise.\nPixel matching only works on frames that "
            "are aligned (registered), e.g. an r_ sequence.\nPlate solve a "
            "whole sequence with seqplatesolve.")
        grid.addWidget(self.cmb_match, 1, 1, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Tolerance:"), 2, 0)
        self.spin_px = QtWidgets.QDoubleSpinBox()
        self.spin_px.setRange(0.1, 100.0)
        self.spin_px.setDecimals(1)
        self.spin_px.setValue(3.0)
        self.spin_px.setSuffix(" px")
        self.spin_px.setToolTip("Largest distance between two detections of "
                                "the same star when matching by pixel "
                                "position.")
        grid.addWidget(self.spin_px, 2, 1)
        self.spin_arcsec = QtWidgets.QDoubleSpinBox()
        self.spin_arcsec.setRange(0.1, 600.0)
        self.spin_arcsec.setDecimals(1)
        self.spin_arcsec.setValue(3.0)
        self.spin_arcsec.setSuffix(" arcsec")
        self.spin_arcsec.setToolTip("Largest distance between two detections "
                                    "of the same star when matching by "
                                    "RA / Dec.")
        grid.addWidget(self.spin_arcsec, 2, 2)

        grid.addWidget(QtWidgets.QLabel("Keep stars found in at least:"),
                       3, 0)
        self.spin_min = QtWidgets.QSpinBox()
        self.spin_min.setRange(1, 100)
        self.spin_min.setValue(50)
        self.spin_min.setSuffix(" % of the frames")
        self.spin_min.setToolTip("A star detected in fewer frames is left out "
                                 "of the averaged table.")
        grid.addWidget(self.spin_min, 3, 1, 1, 2)
        grid.setColumnStretch(3, 1)
        return self.seq_box

    def _build_catalogue_box(self):
        self.cat_box = QtWidgets.QGroupBox(
            "Calibrate the magnitudes with a catalogue")
        self.cat_box.setCheckable(True)
        self.cat_box.setChecked(True)
        self.cat_box.setToolTip(
            "Reads the catalogue stars of the field with Siril's conesearch "
            "(the image must be plate solved), pairs them with the measured "
            "stars and fits each frame's zero point.\nmag = mag_inst + zero "
            "point is then given for every star, also those missing in the "
            "catalogue.\nChanging these settings needs a new measurement.")
        grid = QtWidgets.QGridLayout(self.cat_box)

        grid.addWidget(QtWidgets.QLabel("Catalogue:"), 0, 0)
        self.cmb_catalogue = QtWidgets.QComboBox()
        for entry in CATALOGUES:
            self.cmb_catalogue.addItem(entry[1], entry[0])
        self.cmb_catalogue.setToolTip(
            "APASS V is the usual choice for variable stars measured in the "
            "green channel.\nThe local Gaia catalogue works offline but has "
            "only the G band.")
        grid.addWidget(self.cmb_catalogue, 0, 1, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Down to magnitude:"), 1, 0)
        self.spin_cat_limit = QtWidgets.QDoubleSpinBox()
        self.spin_cat_limit.setRange(5.0, 21.0)
        self.spin_cat_limit.setDecimals(1)
        self.spin_cat_limit.setValue(CATALOGUES[0][4])
        self.spin_cat_limit.setToolTip("Faintest catalogue stars to read.")
        grid.addWidget(self.spin_cat_limit, 1, 1)
        grid.addWidget(QtWidgets.QLabel("match within:"), 1, 2,
                       QtCore.Qt.AlignmentFlag.AlignRight)
        self.spin_cat_radius = QtWidgets.QDoubleSpinBox()
        self.spin_cat_radius.setRange(0.5, 30.0)
        self.spin_cat_radius.setDecimals(1)
        self.spin_cat_radius.setValue(3.0)
        self.spin_cat_radius.setSuffix(" arcsec")
        self.spin_cat_radius.setToolTip("Largest distance between a measured "
                                        "star and its catalogue star.")
        grid.addWidget(self.spin_cat_radius, 1, 3)
        grid.setColumnStretch(4, 1)
        return self.cat_box

    def _build_output_box(self):
        box = QtWidgets.QGroupBox("Output")
        layout = QtWidgets.QGridLayout(box)
        self.chk_hms = QtWidgets.QCheckBox(
            "Also RA / Dec as hh:mm:ss / ±dd:mm:ss")
        layout.addWidget(self.chk_hms, 0, 0)
        self.chk_skip_sat = QtWidgets.QCheckBox("Leave out saturated stars")
        self.chk_skip_sat.setToolTip("A saturated star has a wrong flux and "
                                     "magnitude.")
        layout.addWidget(self.chk_skip_sat, 0, 1)

        self.chk_marks = QtWidgets.QCheckBox(
            "Mark the stars in Siril with circles")
        self.chk_marks.setToolTip(
            "Draws a circle around every star of the table on the image "
            "shown in Siril - green, red for a saturated star.\nFor a sequence "
            "the frame Siril shows is marked. Siril's overlay button clears "
            "the circles.")
        layout.addWidget(self.chk_marks, 1, 0)
        self.chk_mark_numbers = QtWidgets.QCheckBox("with the star numbers")
        self.chk_mark_numbers.setToolTip("Writes the number from the table's "
                                         "\"star\" column next to each "
                                         "circle.")
        self.chk_mark_numbers.setEnabled(False)
        self.chk_marks.toggled.connect(self.chk_mark_numbers.setEnabled)
        layout.addWidget(self.chk_mark_numbers, 1, 1)

        self.chk_names = QtWidgets.QCheckBox(
            "Star names from VSX and SIMBAD (online)")
        self.chk_names.setChecked(True)
        self.chk_names.setToolTip(
            "Adds a \"name\" column: the variable star designation from VSX, "
            "else SIMBAD's main identifier.\nNeeds a plate-solved image and "
            "an internet connection. Changing it needs a new measurement.")
        layout.addWidget(self.chk_names, 2, 0, 1, 2)
        layout.setColumnStretch(2, 1)
        return box

    def _fit_to_screen(self) -> None:
        available = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        width = min(720, int(available.width() * 0.9))
        height = min(640, int(available.height() * 0.85))
        self.resize(width, height)
        self.move(available.x() + (available.width() - width) // 2,
                  available.y() + (available.height() - height) // 3)

    # -- source -------------------------------------------------------------

    def _select_initial_source(self) -> None:
        """Start on the sequence loaded in Siril, else on its image."""
        loaded_seq = None
        try:
            if self.siril.is_sequence_loaded():
                seq = self.siril.get_seq()
                if seq is not None and seq.seqname:
                    loaded_seq = os.path.basename(seq.seqname)
        except Exception:
            loaded_seq = None
        self.refresh_image()
        self.refresh_sequences()
        if loaded_seq:
            wd = os.path.normcase(os.path.abspath(
                self.ed_folder.text().strip() or os.getcwd()))
            for index in range(self.cmb_seq.count()):
                seq_folder, name = self.cmb_seq.itemData(index)
                if name == loaded_seq and os.path.normcase(
                        os.path.abspath(seq_folder)) == wd:
                    self.cmb_seq.setCurrentIndex(index)
                    break
            self.rb_seq.setChecked(True)
        elif self.rb_image.isEnabled():
            self.rb_image.setChecked(True)
        else:
            self.rb_seq.setChecked(True)
        self._update_mode()

    def refresh_image(self) -> None:
        try:
            loaded = self.siril.is_image_loaded()
            name = self.siril.get_image_filename() if loaded else None
        except Exception:
            loaded, name = False, None
        if loaded:
            self.image_label.setText(os.path.basename(name) if name
                                     else "(unsaved image)")
            self.image_label.setToolTip(name or "")
        else:
            self.image_label.setText("no image loaded")
            self.image_label.setToolTip("")
        self.rb_image.setEnabled(loaded)
        if not loaded and self.rb_image.isChecked():
            self.rb_seq.setChecked(True)

    def pick_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the folder with the sequence",
            self.ed_folder.text().strip() or os.getcwd())
        if path:
            self.ed_folder.setText(os.path.normpath(path))

    def refresh_sequences(self) -> None:
        folder = self.ed_folder.text().strip()
        sequences = list_sequences(folder) if os.path.isdir(folder) else []
        current = self.cmb_seq.currentText()
        self.cmb_seq.blockSignals(True)
        self.cmb_seq.clear()
        for label, seq_folder, name in sequences:
            self.cmb_seq.addItem(label, (seq_folder, name))
        index = self.cmb_seq.findText(current)
        self.cmb_seq.setCurrentIndex(index if index >= 0
                                     else (0 if sequences else -1))
        self.cmb_seq.blockSignals(False)

    def _update_mode(self, *_args) -> None:
        seq_mode = self.rb_seq.isChecked()
        self.chk_existing.setEnabled(not seq_mode)
        self.chk_selected.setEnabled(seq_mode)
        self.seq_box.setEnabled(seq_mode)
        averaged = self.cmb_result.currentData() != RESULT_FRAMES
        self.spin_min.setEnabled(averaged)

    # -- measuring ----------------------------------------------------------

    def on_measure(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if self.rb_image.isChecked():
            self.refresh_image()
        request = None
        if self.cat_box.isChecked():
            request = (CATALOGUES[self.cmb_catalogue.currentIndex()],
                       self.spin_cat_limit.value(),
                       self.spin_cat_radius.value())
        if self.rb_image.isChecked():
            target = self._measure_image
            args = (self.chk_existing.isChecked(), request,
                    self.chk_names.isChecked())
        else:
            data = self.cmb_seq.currentData()
            if not data:
                QtWidgets.QMessageBox.warning(
                    self, TITLE, "No sequence (.seq) was found in this folder "
                                 "or its subfolders, and no image is loaded "
                                 "in Siril.")
                return
            seq_folder, name = data
            if name not in _sequences_in(seq_folder):
                QtWidgets.QMessageBox.warning(
                    self, TITLE, "The sequence %s no longer exists." % name)
                self.refresh_sequences()
                return
            target = self._measure_sequence
            args = (os.path.abspath(seq_folder), name,
                    self.chk_selected.isChecked(), request,
                    self.chk_names.isChecked())

        self.cancel.clear()
        self.measure_button.setEnabled(False)
        self.table_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = threading.Thread(target=self._worker_main,
                                       args=(target, args), daemon=True)
        self.worker.start()

    def _worker_main(self, target, args) -> None:
        measurement = None
        try:
            measurement = target(*args)
        except Exception as exc:     # report, never kill the window
            self.post_log("Measuring failed: %s" % exc, "red")
        self.measure_finished.emit(measurement)

    def _measure_image(self, use_existing, request, names):
        siril = self.siril
        path = siril.get_image_filename()
        name = os.path.basename(path) if path else "image"
        self.post_log("Image %s" % name, "blue")
        self.progress_max.emit(0)
        if use_existing:
            self.status_changed.emit("Reading the stars detected in Siril ...")
        else:
            self.status_changed.emit("Detecting the stars (findstar) ...")
            try:
                siril.cmd("findstar")
            except s.SirilError as exc:
                self.post_log("findstar: %s" % exc, "salmon")
        frame = self._collect(0, name, path)
        folder = os.path.dirname(path) if path else ""
        if not folder:
            try:
                folder = siril.get_siril_wd()
            except Exception:
                folder = os.getcwd()
        measurement = Measurement("image", os.path.splitext(name)[0], folder,
                                  [frame])
        catalogue_read = bool(request or names) and self._has_sky(frame)
        if catalogue_read and request:
            self._apply_catalogue(measurement, self._fetch_catalogue(request),
                                  request)
        if catalogue_read and names:
            self._apply_names(measurement, self._fetch_names())
        # a star list the user built in Siril stays on screen
        self._clear_siril_marks(not use_existing, catalogue_read)
        number_stars(measurement)
        return measurement

    def _clear_siril_marks(self, stars, catalogue) -> None:
        """Take Siril's own marks off the image again: findstar circles every
        star it finds and conesearch every catalogue star. The script's
        circles are drawn only when asked for."""
        commands = ([("clearstar",)] if stars else []) + \
            ([("show", "-clear")] if catalogue else [])
        for args in commands:
            try:
                self.siril.cmd(*args)
            except s.SirilError:
                pass        # not allowed in headless Siril, nothing shown

    def _has_sky(self, frame) -> bool:
        if any(st.ra is not None for st in frame.stars):
            return True
        self.post_log("%s is not plate solved - catalogue magnitudes need a "
                      "plate solution (Tools > Astrometry, or the platesolve "
                      "/ seqplatesolve command)." % frame.name, "salmon")
        return False

    def _fetch_catalogue(self, request):
        """Catalogue stars in the field of the image loaded in Siril, read
        through conesearch; None when that fails."""
        catalogue, limit, _radius = request
        handle, path = tempfile.mkstemp(prefix="starsstatistics_",
                                        suffix=".csv")
        os.close(handle)
        name = catalogue[1].split(" (")[0]
        self.status_changed.emit("Reading the %s catalogue ..." % name)
        self.progress_max.emit(0)
        try:
            self.siril.cmd("conesearch", "%g" % limit, "-cat=" + catalogue[0],
                           '"-out=%s"' % path.replace("\\", "/"), "-log=off")
            stars = read_catalogue_csv(path)
        except (s.SirilError, OSError) as exc:
            self.post_log("The %s catalogue could not be read: %s"
                          % (name, exc), "red")
            return None
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if not stars:
            self.post_log("The %s catalogue has no star in this field "
                          "(down to mag %g)." % (name, limit), "salmon")
            return None
        self.post_log("%s: %d catalogue star(s) down to mag %g."
                      % (name, len(stars), limit))
        return stars

    def _fetch_names(self):
        """(ra, dec, name) lists from the name catalogues, best first."""
        self.status_changed.emit("Looking up the star names ...")
        self.progress_max.emit(0)
        sources = []
        for key, label in NAME_CATALOGUES:
            handle, path = tempfile.mkstemp(prefix="starsstatistics_",
                                            suffix=".csv")
            os.close(handle)
            try:
                self.siril.cmd("conesearch", "%g" % NAME_LIMIT, "-cat=" + key,
                               '"-out=%s"' % path.replace("\\", "/"),
                               "-log=off")
                entries = read_names_csv(path)
            except (s.SirilError, OSError) as exc:
                self.post_log("%s names could not be read: %s"
                              % (label, exc), "salmon")
                continue
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
            self.post_log("%s: %d named object(s) in the field."
                          % (label, len(entries)))
            sources.append(entries)
        return sources

    def _apply_names(self, measurement, sources) -> None:
        if not sources:
            return
        measurement.names = True
        for frame in measurement.frames:
            if frame.stars:
                radius = match_radius(frame.stars, NAME_RADIUS)
                if frame.sky_shift is None:
                    frame.sky_shift = sky_offset(
                        frame.stars, [e for entries in sources
                                      for e in entries], radius)
                named = match_names(frame.stars, sources, radius,
                                    frame.sky_shift)
                self.post_log("%s: %d star(s) named." % (frame.name, named))

    def _apply_catalogue(self, measurement, cat_stars, request) -> None:
        """Match every frame to the catalogue and calibrate its magnitudes."""
        if not cat_stars:
            return
        catalogue, _limit, radius = request
        matched_any = False
        for frame in measurement.frames:
            if not frame.stars:
                continue
            frame_radius = match_radius(frame.stars, radius)
            frame.sky_shift = sky_offset(frame.stars, cat_stars,
                                         frame_radius)
            matched = match_catalogue(frame.stars, cat_stars, frame_radius,
                                      frame.sky_shift)
            matched_any = matched_any or matched > 0
            calibrate(frame)
            self.post_log("%s: matched within %.1f″ after removing a "
                          "%.1f″ offset from the catalogue."
                          % (frame.name, frame_radius,
                             math.hypot(*frame.sky_shift)))
            if frame.zero_point is None:
                self.post_log("%s: %d star(s) in the catalogue - too few for "
                              "a zero point." % (frame.name, matched),
                              "salmon")
            else:
                self.post_log("%s: %d star(s) in the catalogue, zero point "
                              "%.3f ± %.3f from %d unsaturated star(s)."
                              % (frame.name, matched, frame.zero_point,
                                 frame.zp_sigma, frame.zp_stars))
        if matched_any:
            measurement.catalogue = catalogue

    def _measure_sequence(self, folder, name, selected_only, request,
                          names):
        siril = self.siril
        self.post_log("Sequence %s in %s" % (name, folder), "blue")
        siril.cmd("cd", quote(folder))
        if not os.path.isfile(os.path.join(folder, name + ".seq")):
            # numbered frames Siril has not opened yet
            if not siril.create_new_seq(name):
                raise RuntimeError("Siril could not create %s.seq." % name)
            self.post_log("Sequence file %s.seq created." % name)
        frames, reference = read_seq_file(folder, name)
        if not frames:
            raise RuntimeError("The sequence %s holds no frame." % name)

        entries = [(i, path) for i, selected, path in frames
                   if selected or not selected_only]
        if not entries:
            raise RuntimeError("No frame of %s is selected." % name)
        if len(entries) < len(frames):
            self.post_log("%d unselected frame(s) skipped."
                          % (len(frames) - len(entries)))
        missing = [i for i, path in entries if path is None]
        if missing:
            self.post_log("%d frame file(s) not found, e.g. frame %d."
                          % (len(missing), missing[0] + 1), "salmon")
            entries = [(i, path) for i, path in entries if path]
            if not entries:
                raise RuntimeError("No frame file of %s was found." % name)

        # without a reference frame match_stars() starts from the frame with
        # the most stars
        measurement = Measurement("seq", name, folder, reference=-1)
        cat_stars, name_sources, sky_seen = None, None, False
        total = len(entries)
        self.progress_max.emit(total)
        try:
            for done, (index, path) in enumerate(entries, start=1):
                if self.cancel.is_set():
                    self.post_log("Stopped - %d of %d frame(s) measured."
                                  % (done - 1, total), "salmon")
                    break
                frame_name = os.path.basename(path)
                self.status_changed.emit("[%d/%d] %s"
                                         % (done, total, frame_name))
                try:
                    siril.cmd("load", quote(path))
                    try:
                        siril.cmd("findstar")
                    except s.SirilError:
                        pass      # no star found - reported as such below
                    frame = self._collect(index, frame_name, path)
                    # the catalogue is read once, on the first plate-solved
                    # frame - the field hardly moves between the frames
                    if (request or names) and not sky_seen and \
                            any(st.ra is not None for st in frame.stars):
                        sky_seen = True
                        if request:
                            cat_stars = self._fetch_catalogue(request)
                        if names:
                            name_sources = self._fetch_names()
                        self.progress_max.emit(total)
                        self.progress_changed.emit(done - 1)
                except s.SirilError as exc:
                    frame = FrameResult(index, frame_name, path,
                                        error=str(exc))
                    self.post_log("%s: %s" % (frame_name, exc), "red")
                if frame.index == reference:
                    measurement.reference = len(measurement.frames)
                measurement.frames.append(frame)
                self.progress_changed.emit(done)
        finally:
            self._clear_siril_marks(True, sky_seen)
            # leave the sequence loaded again, as it was found
            try:
                siril.cmd("cd", quote(folder))
                siril.cmd("load_seq", quote(name))
            except s.SirilError:
                pass
        if sky_seen:
            if request:
                self._apply_catalogue(measurement, cat_stars, request)
            if names:
                self._apply_names(measurement, name_sources)
        elif (request or names) and measurement.frames:
            self._has_sky(measurement.frames[0])
        return measurement

    def _collect(self, index, name, path) -> FrameResult:
        """Read the stars and the metadata of the image loaded in Siril."""
        siril = self.siril
        frame = FrameResult(index, name, path)

        try:
            image = siril.get_image(with_pixels=False)
            frame.float_data = image is not None and \
                image.bitpix == s.BitpixType.FLOAT_IMG
        except Exception:
            frame.float_data = False
        scale = ADU16 if frame.float_data else 1.0

        keywords = header = None
        try:
            keywords = siril.get_image_keywords()
        except Exception:
            pass
        try:
            header = siril.get_image_fits_header(return_as="dict")
        except Exception:
            pass
        if keywords is not None:
            if keywords.exposure and keywords.exposure > 0:
                frame.exposure = float(keywords.exposure)
            frame.plate_solved = bool(getattr(keywords, "pltsolvd", False))
        if frame.exposure is None:
            frame.exposure = header_number(header, EXPOSURE_KEYS)
        frame.gain = header_gain(header)
        frame.filter = header_text(header, "FILTER")
        if frame.gain is None and keywords is not None and keywords.gain:
            frame.gain = float(keywords.gain)
        frame.date = header_date(header) or (
            utc_text(keywords.date_obs) if keywords is not None else "")

        try:
            psf_stars = siril.get_image_stars() or []
        except s.SirilError as exc:
            self.post_log("%s: no stars (%s)" % (name, exc), "salmon")
            psf_stars = []

        for st in psf_stars:
            flux = psf_flux(st)
            fwhms = [v for v in (st.fwhmx, st.fwhmy) if v and v > 0]
            star = Star(
                x=float(st.xpos), y=float(st.ypos),
                flux=flux * scale if flux is not None else None,
                fwhm=float(np.mean(fwhms)) if fwhms else None,
                saturated=bool(st.has_saturated),
                layer=int(st.layer))
            if valid_radec(st.ra, st.dec):
                star.ra, star.dec = float(st.ra), float(st.dec)
            frame.stars.append(star)
        if frame.stars:
            self._read_max_pixels(frame, scale)

        # Siril fills RA / Dec of every star of a plate-solved image; ask it
        # for the coordinates only when they did not come with the stars.
        if frame.stars and frame.plate_solved and \
                not any(st.ra is not None for st in frame.stars):
            try:
                for star in frame.stars:
                    radec = siril.pix2radec(star.x, star.y)
                    if radec and valid_radec(*radec):
                        star.ra, star.dec = float(radec[0]), float(radec[1])
            except (ValueError, s.SirilError) as exc:
                self.post_log("%s: no RA / Dec (%s)" % (name, exc), "salmon")

        with_sky = sum(1 for st in frame.stars if st.ra is not None)
        if not frame.stars:
            frame.error = "no stars detected"
        self.post_log("%s: %d star(s)%s, exposure %s, %s data" % (
            name, len(frame.stars),
            ", %d with RA/Dec" % with_sky if with_sky else ", no RA/Dec",
            "%g s" % frame.exposure if frame.exposure else "unknown",
            "32-bit float" if frame.float_data else "16-bit"))
        return frame

    def _read_max_pixels(self, frame, scale) -> None:
        """max_flux of every star: its brightest pixel, background included.

        Siril counts a star's y from the top of the image while the pixel
        array has the bottom row first, hence row = height - 1 - y. The 5 x 5
        box around the centre always holds the star's brightest pixel.
        """
        try:
            data = self.siril.get_image_pixeldata()
        except Exception as exc:
            self.post_log("%s: pixels not readable, no max_flux (%s)"
                          % (frame.name, exc), "salmon")
            return
        if data is None:
            return
        for star in frame.stars:
            plane = data[min(star.layer, data.shape[0] - 1)] \
                if data.ndim == 3 else data
            height, width = plane.shape
            col = int(round(star.x))
            row = int(round(height - 1 - star.y))
            if not (0 <= col < width and 0 <= row < height):
                continue
            box = plane[max(0, row - 2):row + 3, max(0, col - 2):col + 3]
            star.max_flux = float(box.max()) * scale

    def on_measure_finished(self, measurement) -> None:
        self.measure_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._set_maximum(1)
        self.progress.setValue(1 if measurement else 0)
        if measurement is None or not any(f.stars
                                          for f in measurement.frames):
            self.status.setText("No stars were measured.")
            if measurement is not None:
                self.write_log("No stars were found.", "salmon")
            return
        self.measurement = measurement
        self.table_button.setEnabled(True)
        stars = sum(len(f.stars) for f in measurement.frames)
        frames = sum(1 for f in measurement.frames if f.stars)
        message = "%d star(s) measured in %d frame(s)." % (stars, frames)
        self.status.setText(message)
        self.write_log(message, "green")
        self.show_table()

    # -- result -------------------------------------------------------------

    def show_table(self) -> None:
        measurement = self.measurement
        if measurement is None:
            return
        result = self.cmb_result.currentData()
        title = measurement.label
        if measurement.kind == "seq":
            mode = match_stars(measurement, self.cmb_match.currentData(),
                               self.spin_px.value(), self.spin_arcsec.value())
            self.write_log("Stars matched by %s." % (
                "RA / Dec" if mode == MATCH_SKY else "pixel position"))
            title += " (%s)" % {RESULT_MEAN: "mean", RESULT_MEDIAN: "median",
                                RESULT_FRAMES: "per frame"}[result]
        columns, rows, summary = build_table(
            measurement, result, self.chk_hms.isChecked(),
            self.chk_skip_sat.isChecked(), float(self.spin_min.value()))

        if measurement.kind == "seq":
            summary += "; matched by %s" % (
                "RA / Dec" if measurement.match_mode == MATCH_SKY
                else "pixel position")
        exposures = {f.exposure for f in measurement.frames if f.stars}
        if exposures == {None}:
            summary += "; exposure unknown"
        elif len(exposures) == 1:
            summary += "; exposure %g s" % next(iter(exposures))
        if any(f.float_data for f in measurement.frames):
            summary += "; 32-bit data scaled to 16-bit ADU"
        zero_points = [f.zero_point for f in measurement.frames
                       if f.zero_point is not None]
        if measurement.catalogue:
            summary += "; mag calibrated to %s %s" % (
                measurement.catalogue[1].split(" - ")[0],
                measurement.catalogue[2])
            if len(zero_points) == 1:
                frame = next(f for f in measurement.frames
                             if f.zero_point is not None)
                summary += ", zero point %.3f ± %.3f (%d stars)" % (
                    frame.zero_point, frame.zp_sigma, frame.zp_stars)
            elif zero_points:
                summary += ", zero point %.3f to %.3f" % (min(zero_points),
                                                          max(zero_points))

        stem = measurement.label
        if measurement.kind == "seq" and result != RESULT_FRAMES:
            stem += "_" + result
        window = ResultWindow(
            title, summary, columns, rows, measurement.folder, stem,
            self.write_log,
            lambda values, m=measurement: self.show_star(m, values))
        window.destroyed.connect(lambda _obj=None, w=window:
                                 self._forget(w))
        self.result_windows.append(window)
        window.show()
        window.raise_()
        window.activateWindow()
        self.write_log("Table: %s." % summary)
        if self.chk_marks.isChecked():
            self.start_marking(measurement, columns, rows)

    # -- marking the stars in Siril -----------------------------------------

    def start_marking(self, measurement, columns, rows) -> None:
        """Circle the stars of the table on the image Siril shows."""
        if self.worker and self.worker.is_alive():
            return
        star_col = next((n for n, c in enumerate(columns) if c.key == "star"),
                        None)
        if star_col is None:
            return
        ids = {row[star_col] for row in rows if row[star_col]}
        if not ids:
            return
        self.cancel.clear()
        self.measure_button.setEnabled(False)
        self.table_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = threading.Thread(
            target=self._mark_stars,
            args=(measurement, ids, self.chk_mark_numbers.isChecked()),
            daemon=True)
        self.worker.start()

    def _shown_frame(self, measurement):
        """The measured frame Siril shows now, or None."""
        siril = self.siril
        try:
            shown = siril.get_image_filename()
        except Exception:
            shown = None
        if shown:
            shown = os.path.normcase(os.path.abspath(shown))
            for frame in measurement.frames:
                if frame.path and os.path.normcase(
                        os.path.abspath(frame.path)) == shown:
                    return frame
        try:
            if measurement.kind == "seq" and siril.is_sequence_loaded():
                current = siril.get_seq().current
                for frame in measurement.frames:
                    if frame.index == current:
                        return frame
        except Exception:
            pass
        return None

    def _mark_stars(self, measurement, ids, numbers) -> None:
        siril = self.siril
        try:
            frame = self._shown_frame(measurement)
            if frame is None:
                self.post_log("The image Siril shows is not one of the "
                              "measured frames - no stars marked.", "salmon")
                return
            stars = [st for st in frame.stars if st.star_id in ids]
            self.status_changed.emit("Marking %d star(s) in Siril ..."
                                     % len(stars))
            self.progress_max.emit(len(stars))
            siril.overlay_clear_polygons()
            for done, st in enumerate(stars, start=1):
                if self.cancel.is_set():
                    self.post_log("Marking stopped after %d star(s)."
                                  % (done - 1), "salmon")
                    return
                radius = max(5.0, 2.0 * (st.fwhm or 2.5))
                colour = MARK_SATURATED if st.saturated else MARK_COLOUR
                points = [s.FPoint(st.x + radius * math.cos(a),
                                   st.y + radius * math.sin(a))
                          for a in (2.0 * math.pi * k / 20 for k in range(20))]
                siril.overlay_add_polygon(s.Polygon(points=points,
                                                    color=colour))
                if numbers:
                    # Siril centres a legend on its polygon, so the number
                    # goes on a tiny polygon beside the circle
                    x = st.x + radius * 0.8 + 6.0
                    y = st.y - radius * 0.8 - 6.0
                    tiny = [s.FPoint(x, y), s.FPoint(x + 0.01, y),
                            s.FPoint(x, y + 0.01)]
                    siril.overlay_add_polygon(s.Polygon(
                        points=tiny, color=colour, legend=str(st.star_id)))
                self.progress_changed.emit(done)
            self.post_log("%d star(s) marked on %s." % (len(stars),
                                                        frame.name), "green")
        except Exception as exc:     # report, never kill the window
            self.post_log("Marking the stars failed: %s" % exc, "red")
        finally:
            self.marking_finished.emit()

    # -- showing one star in Siril ------------------------------------------

    def show_star(self, measurement, values) -> None:
        """Circle the star of a table row in Siril and centre the view on it."""
        if self.worker and self.worker.is_alive():
            self.write_log("Siril is busy - try again when the work is done.",
                           "salmon")
            return
        siril = self.siril
        star_id = values.get("star")
        fwhm = values.get("fwhm_px")
        position = None

        # the star as measured on the image Siril shows
        frame = self._shown_frame(measurement)
        if frame is not None and star_id:
            star = next((st for st in frame.stars if st.star_id == star_id),
                        None)
            if star is not None:
                position, fwhm = (star.x, star.y), star.fwhm or fwhm
        # not found there: from its RA / Dec, else from the table
        if position is None and values.get("ra_deg") is not None:
            try:
                xy = siril.radec2pix(values["ra_deg"], values["dec_deg"])
                if xy:
                    position = (float(xy[0]), float(xy[1]))
            except (ValueError, s.SirilError):
                pass
        if position is None:
            x = values.get("x", values.get("x_ref"))
            y = values.get("y", values.get("y_ref"))
            if x is not None and y is not None:
                position = (x, y)
        if position is None:
            self.write_log("Star %s cannot be located on the image Siril "
                           "shows." % (star_id or "?"), "salmon")
            return

        try:
            self._highlight(position, fwhm, star_id)
        except Exception as exc:
            self.write_log("Could not mark the star: %s" % exc, "red")
        try:
            centred = self._centre_view(*position)
        except Exception as exc:
            centred = False
            self.write_log("Could not move the view: %s" % exc, "salmon")
        self.write_log("Star %s at x %.1f, y %.1f%s." % (
            star_id or "?", position[0], position[1],
            "" if centred else " (view not moved - look for the yellow "
                               "circle)"))

    def _highlight(self, position, fwhm, star_id) -> None:
        """A yellow circle with a cross-hair and the number; replaces the
        previous one."""
        siril = self.siril
        for polygon_id in self.highlight_ids:
            try:
                siril.overlay_delete_polygon(polygon_id)
            except Exception:
                pass
        self.highlight_ids = []
        x, y = position
        radius = max(12.0, 4.0 * (fwhm or 3.0))
        shapes = [[(x + radius * math.cos(a), y + radius * math.sin(a))
                   for a in (2.0 * math.pi * k / 32 for k in range(32))]]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shapes.append([(x + dx * radius * 1.3, y + dy * radius * 1.3),
                           (x + dx * radius * 2.5, y + dy * radius * 2.5)])
        for points in shapes:
            polygon = siril.overlay_add_polygon(s.Polygon(
                points=[s.FPoint(px, py) for px, py in points],
                color=MARK_HIGHLIGHT))
            self.highlight_ids.append(polygon.polygon_id)
        if star_id:
            lx, ly = x + radius * 1.6, y - radius * 1.6
            polygon = siril.overlay_add_polygon(s.Polygon(
                points=[s.FPoint(lx, ly), s.FPoint(lx + 0.01, ly),
                        s.FPoint(lx, ly + 0.01)],
                color=MARK_HIGHLIGHT, legend=str(star_id)))
            self.highlight_ids.append(polygon.polygon_id)

    def _centre_view(self, x, y) -> bool:
        """Centre Siril's view on (x, y), zoomed to at least 100 %.

        Siril's pan is the screen position of the image's corner, so the
        size of the view is needed: zoom-to-fit centres the whole image,
        which gives it away. False when that does not work out.
        """
        siril = self.siril
        shape = siril.get_image_shape()
        before = siril.get_siril_panzoom()
        if not shape or not before:
            return False
        _channels, height, width = shape
        siril.set_siril_zoom(-1)
        fit = siril.get_siril_panzoom()
        if not fit or fit[2] <= 0:
            siril.set_siril_zoom(before[2])
            siril.set_siril_pan(before[0], before[1])
            return False
        view_w = 2.0 * fit[0] + width * fit[2]
        view_h = 2.0 * fit[1] + height * fit[2]
        if not (50 < view_w < 20000 and 50 < view_h < 20000):
            siril.set_siril_zoom(before[2])
            siril.set_siril_pan(before[0], before[1])
            return False
        zoom = max(1.0, before[2])
        siril.set_siril_zoom(zoom)
        siril.set_siril_pan(view_w / 2.0 - x * zoom, view_h / 2.0 - y * zoom)
        return True

    def on_marking_finished(self) -> None:
        self.measure_button.setEnabled(True)
        self.table_button.setEnabled(self.measurement is not None)
        self.cancel_button.setEnabled(False)
        self.status.setText("Ready.")

    def _forget(self, window) -> None:
        if window in self.result_windows:
            self.result_windows.remove(window)

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
                self, TITLE, "The measuring is running. Stop it and close?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel.set()
            self.worker.join(timeout=15)
        for window in list(self.result_windows):
            window.close()
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
        print("StarsStatistics: could not connect to Siril: %s" % exc)
        return

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])
    apply_siril_theme(app, siril)

    window = StarsStatisticsWindow(siril)
    window.show()
    window.raise_()
    window.activateWindow()

    if owns_app:
        app.exec()


if __name__ == "__main__":
    main()
