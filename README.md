# Siril Scripts

Python scripts for [Siril](https://siril.org/) 1.4+ that use the `sirilpy`
interface. Each one opens a PyQt6 dialog, runs its work on a background thread
and reports progress into Siril's log, so the Siril window stays responsive.

| Script | What it does |
| --- | --- |
| [`SirilSync.py`](SirilSync.py) | Reads the FITS `HISTORY` of the image loaded in Siril, translates it back into Siril commands, and replays the ones you select onto a whole folder of images. |
| [`siril_dark_calibration.py`](siril_dark_calibration.py) | Builds a master dark from RAW darks, calibrates RAW light frames with it, and exports the calibrated frames as TIFF. |
| [`SirilChannelExtract.py`](SirilChannelExtract.py) | Detects whether a sequence holds OSC (one-shot colour) data and extracts the R, G or B channel into a new sequence. |
| [`VarStarOSCPreprocess.py`](VarStarOSCPreprocess.py) | Full OSC preprocessing: masters, light calibration, registration and optional single-channel extraction — deliberately stopping before stacking. |
| [`DepthFITSConversion.py`](DepthFITSConversion.py) | Converts a folder of FITS files, or the frames of one sequence, between 32-bit float and 16-bit unsigned integer, reporting any clipping. |
| [`Selector.py`](Selector.py) | Measures FWHM, eccentricity, star count and background of every calibrated light frame, shows them in a table and charts with a keep / reject suggestion, deletes, moves or unselects the frames you reject, and copies or moves the accepted ones to an `accepted` subfolder. |
| [`StarsStatistics.py`](StarsStatistics.py) | Detects the stars on the image loaded in Siril or on every frame of a sequence and lists their RA, Dec, magnitude calibrated with a catalogue (APASS, Gaia, NOMAD), flux (16-bit ADU) and exposure time — per frame or averaged (mean / median) over the sequence — in a table that can be copied or saved as CSV. |

## Requirements

- Siril **1.4** or newer, built with the Python (`sirilpy`) interface.
- **PyQt6** — already part of Siril's Python environment (Siril 1.4 bundles
  PyQt6 6.11 / Qt 6.11), so normally there is nothing to install. If it is
  somehow missing, the scripts ask `sirilpy.ensure_installed()` for it. The
  scripts that have a `--no-gui` mode do not need Qt at all in that mode;
  `SirilSync.py`, `Selector.py` and `StarsStatistics.py` are GUI-only and
  always need it.

## Installation

Copy the `.py` files into your Siril scripts directory and they appear in the
**Scripts** menu:

| OS | Path |
| --- | --- |
| Windows | `%LOCALAPPDATA%\siril\scripts` |
| Linux / macOS | `~/.siril/scripts` |

Every script except `SirilSync.py` can also be started from a shell, as long as
Siril is running and you use its Python environment:

```bash
python siril_dark_calibration.py --work-dir "D:/astro/M31" --no-gui
```

`SirilSync.py` has no command-line interface — it is GUI only, because it needs
an image loaded in the running Siril instance. `Selector.py` is GUI only as
well: rejecting frames is a decision you make while looking at the charts.
`StarsStatistics.py` is GUI only too — its result is a table in a window.

## The dialogs

Every dialog is a **PyQt6** window, built with the Qt binding that ships inside
Siril's own Python environment, so nothing has to be installed alongside.

- They follow Siril's own light/dark preference, read from `gui.theme`, and
  switch to a matching dark palette when Siril is dark.
- The embedded log is **colour-coded** the same way Siril's own log is — blue
  for step headings, green for results, salmon for warnings, red for errors —
  with a separate palette per theme so it stays readable either way.
- Work runs on a background thread and reports back through Qt signals, which Qt
  delivers on the GUI thread, so the window never freezes while Siril works.
- Each window sizes itself to its content and is clamped to the available screen
  area, so it fits on a laptop display.

---

# SirilSync.py

## What it is for

You process one image by hand in Siril — crop, rotate, stretch, denoise — and
then want the *same* sequence of operations applied to a folder of other images.
Siril records every editing step as a FITS `HISTORY` card. SirilSync parses those
cards, maps each one back to the Siril command that produced it, and replays the
selected commands over every image in a folder.

## Workflow

1. Load and process an image in Siril as usual.
2. Run **SirilSync** from the Scripts menu. It reads the loaded image's history
   and lists every entry with the reconstructed command.
3. Pick the target folder (optionally including subfolders).
4. Review, edit or untick individual commands.
5. Choose whether to overwrite the originals or write to a separate output folder.
6. Press **Sync**.

For each file the script runs `load` → your selected commands → a save command,
then moves on to the next one.

## The change list

Every history entry becomes one row with a checkbox and an editable command
field. Rows are tagged according to how well the command could be recovered:

| Tag | Meaning |
| --- | --- |
| *(none)* | **Exact** — every parameter was recoverable from the history. |
| `[incomplete]` | The operation was recognised but Siril does not record all of its parameters. A best-effort command is filled in; a note explains what is missing. |
| `[no command]` | No equivalent single-image command exists (e.g. deconvolution, manual colour calibration), or the entry is not a replayable single-image operation at all (stacking, registration, plate solving, pixel math…). |
| `[already stored in the file]` | The entry was already in the file on disk before you loaded it. |

The command field is free text, so you can correct an incomplete command or type
any Siril command yourself. Clearing a field disables its checkbox.

### Pre-existing history detection

The script reads the `HISTORY` cards directly from the FITS file on disk and
compares them, in order, against the in-memory history. The on-disk history is a
prefix of the in-memory one, so everything past the common prefix is what *you*
did in this session — only those entries are ticked by default. Entries already
baked into the file are listed but left unticked, so they are not applied twice.

If the on-disk history cannot be read (compressed FITS, non-FITS formats, an
unsaved image), the script says so in the log and lists every entry — check the
selection yourself in that case.

## Recognised operations

Translation is table-driven: each rule is a regular expression over the history
text plus a builder that assembles the command.

| Group | Operations |
| --- | --- |
| Geometry | crop, rotate, mirrorx, mirrory, resample *(partial)*, binxy |
| Stretching | mtf, asinh (incl. `-human`), autostretch, autoghs, ght, invght, modasinh, invmodasinh, linstretch (BP shift), ddp, neg |
| Colour | rmgreen (SCNR), satu, unpurple, pcc *(partial)*, manual colour calibration *(none)* |
| Filters / noise | gauss, unsharp, fmedian *(partial)*, epf (bilateral and guided), clahe, denoise *(partial)*, fixbanding, rgradient, find_cosme *(partial)* |
| Background | subsky *(partial)* |
| Stars | synthstar, starnet *(partial)*, deconvolution *(none)*, linear match *(none)* |
| Not replayable | stacking, registration, calibration, conversion, debayer, CFA split/merge, channel extraction, plate solving, pixel math, wavelets, ICC profile changes |

Anything the table does not match is listed as *unrecognised history entry* with
an empty command, so nothing is applied silently.

## Input formats

FITS (`.fit`, `.fits`, `.fts`, including the `.fz`-compressed variants), TIFF,
PNG, JPEG, BMP, PNM, XISF and the common camera RAW formats (CR2, CR3, NEF, ARW,
DNG, ORF, RAF, RW2, PEF).

## Output

- **Overwrite the original images** — each file is written back over itself.
- Otherwise pick an output folder. It is created if it does not exist, and it
  must differ from the source folder.

The save command follows the input extension: `savetif` for TIFF, `savepng` for
PNG, `savejpg … 95` for JPEG, and `save` (FITS) for everything else. Siril's FITS
extension setting is switched with `setext` when needed and restored afterwards.
Compressed `.fz` inputs are saved uncompressed, which is noted in the log.

## Notes and caveats

- **The loaded image is not skipped.** If it lives in the target folder it is
  processed like any other file. Normally that is what you want — the copy on
  disk is still unprocessed, so replaying the commands reproduces what is on
  screen. If you already saved it with those changes, they get applied twice.
  The confirmation dialog warns you when this situation is detected.
- Selected commands that were tagged incomplete are listed again in the
  confirmation dialog before anything is written.
- After the run, the source image is reloaded into Siril.
- Failures are per file: a failed image is counted and logged, and the run
  continues. A summary dialog reports how many succeeded and how many failed.
- Closing the window during a run asks for confirmation and cancels the worker.

---

# siril_dark_calibration.py

## What it is for

A dark-only calibration pipeline: stack a master dark, subtract it from the light
frames, and export the calibrated frames as TIFF for use in other software. No
flats or biases are involved.

## Expected layout

```
<work_dir>/
    lights/       RAW light frames  (CR2/CR3/NEF/ARW/DNG/...)
    darks/        RAW dark frames
    process/      created: FITS sequences and the master dark
    calibrated/   created: light_00001.tif ...
```

The `lights` and `darks` directories are found automatically (`lights`, `light`,
`darks`, `dark`, in any capitalisation), or you can point at them explicitly.
Roughly 30 camera RAW extensions are recognised; files are taken in
case-insensitive name order.

## Pipeline

1. `convertraw` the darks into the FITS sequence `dark_` — **without**
   `-debayer`, so the CFA mosaic is preserved.
2. `stack dark rej w 3 3 -nonorm -out=master_dark` — darks are never normalised.
   With a single dark frame, stacking is skipped and the frame is used directly.
3. `convertraw` the lights into the sequence `light_`, again without debayering.
4. `calibrate light -dark=master_dark [-cc=dark low high] [-cfa] [-debayer] -prefix=pp_`
5. Per frame: `load pp_light_NNNNN` + `savetif …/calibrated/light_NNNNN`.
   Siril has no command to export a whole sequence to TIFF, so the frames are
   exported one at a time.

Before step 1 the script sets `setext fit` and 32-bit (or 16-bit) mode, and it
restores Siril's original working directory when it finishes. If `convertraw` is
missing in your build, it falls back to the generic `convert`.

## Linear output — read this

The calibrated frames are **linear** and look almost black in an ordinary image
viewer. That is correct: subtracting the dark also removes the offset pedestal
that lifted the RAW background. Keep them linear for registration and stacking;
stretching belongs after stacking.

The script logs the statistics of the first exported frame. A background pinned
at ~0 (median below 0.0005) means the dark is over-subtracting — check that the
darks match the lights in exposure time, ISO/gain and sensor temperature, or add
a `--pedestal`.

`--stretch autostretch` / `--stretch asinh` produce a **non-linear** viewable
copy. Never register or stack stretched files.

## GUI

The dialog groups the settings into **Directories**, **Calibration**, **TIFF
output** and **Progress** (status line, progress bar and a live copy of the log).
Fields carry tooltips, and dependent controls grey themselves out — selecting a
mono sensor disables debayering, turning off cosmetic correction disables its
sigma fields, and naming from RAW filenames disables the base-name field.
Numeric fields accept a comma as the decimal separator and are validated before
the run starts. If the GUI cannot start at all, the script reports it and
continues in text mode.

## Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--no-gui` | GUI on | Skip the dialog and start processing immediately. |
| `--work-dir PATH` | Siril's working directory | Project directory. |
| `--lights PATH` | auto (`lights`) | Directory with the light RAW frames. |
| `--darks PATH` | auto (`darks`) | Directory with the dark RAW frames. |
| `--process NAME` | `process` | Directory for intermediates. |
| `--calibrated NAME` | `calibrated` | Output directory for the TIFF frames. |
| `--light-name NAME` | `light` | Base name of the light sequence. |
| `--dark-name NAME` | `dark` | Base name of the dark sequence. |
| `--master-name NAME` | `master_dark` | File name of the master dark. |
| `--prefix STR` | `pp_` | Prefix of the calibrated FITS sequence. |
| `--tiff-bits {8,16,32}` | `16` | TIFF bit depth (`savetif8` / `savetif` / `savetif32`). |
| `--tiff-basename NAME` | `light` | Base name of the output TIFFs. |
| `--name-from-raw` | off | Name the TIFFs after the original RAW files instead of numbering them. |
| `--no-astro-tiff` | Astro-TIFF on | Do not embed a FITS header in the TIFF. |
| `--deflate` | off | Lossless TIFF compression. |
| `--stretch {none,autostretch,asinh}` | `none` | Stretch before saving (viewing only — non-linear). |
| `--asinh-stretch F` | `100.0` | Stretch factor for `--stretch asinh`. |
| `--pedestal F` | `0.0` | Constant (ADU) added after calibration so a slightly over-subtracting dark does not clip at 0. |
| `--rejection {p,s,m,w,l,g,a,n}` | `w` | Rejection for dark stacking (`w` = Winsorized, `n` = none). |
| `--sigma-low F` | `3.0` | Low sigma for rejection. |
| `--sigma-high F` | `3.0` | High sigma for rejection. |
| `--mono` | off | Monochrome sensor — disables both `-cfa` and debayering. |
| `--no-debayer` | debayer on | Leave the output in CFA form. |
| `--no-cosmetic` | cosmetic on | Disable hot/cold pixel correction. |
| `--cc-sigma-low F` | `3.0` | Sigma for cold pixels. |
| `--cc-sigma-high F` | `3.0` | Sigma for hot pixels. |
| `--dark-fitseq` | off | Store the dark sequence as a single FITS file. The light sequence must stay per-frame for the TIFF export. |
| `--16bit` | 32-bit float | Work in 16-bit mode. |

`--name-from-raw` assumes the RAW files were converted in alphabetical order; if
the RAW count does not match the number of sequence frames, the script logs a
warning and falls back to sequence numbering.

## Examples

Defaults, no dialog:

```bash
python siril_dark_calibration.py --work-dir "D:/astro/M31" --no-gui
```

Mono camera, 32-bit TIFF, no cosmetic correction:

```bash
python siril_dark_calibration.py --work-dir "D:/astro/M31" --mono --tiff-bits 32 --no-cosmetic --no-gui
```

Stretched preview copies with a pedestal:

```bash
python siril_dark_calibration.py --work-dir "D:/astro/M31" --stretch autostretch --pedestal 100 --calibrated preview --no-gui
```

## Error handling

Every Siril command that fails raises a `CalibrationError` carrying the command
and Siril's own message. In the GUI the error is shown in a message box and
appended to the log; from the command line it is written to Siril's log and the
process exits with status 1. Two failure modes have automatic fallbacks: a
missing `convertraw` falls back to `convert`, and if a build rejects a relative
path inside `savetif`, the script switches into the target directory with `cd`
instead.

---

# SirilChannelExtract.py

## What it is for

You have a sequence from a one-shot colour (OSC) camera and want just one colour
channel as its own sequence — the red channel for a Ha-ish continuum, the green
channel for luminance or star detection, and so on. Give the script a sequence
name; it looks at the first frame, works out whether the data is OSC at all, and
offers R, G or B.

## Workflow

1. Run **SirilChannelExtract** from the Scripts menu.
2. Pick the working directory. The **Sequence** dropdown lists every `.seq`
   found there; you can also type a name such as `light_`.
3. The script analyses the first frame and reports what it found.
4. Choose the channel, press **Extract**.

The result is a new sequence `R_<sequence>` / `G_<sequence>` / `B_<sequence>`,
with its `.seq` written so it shows up in Siril without a manual *Search
sequence*.

## Detection

The first frame is read with `load_image_from_file()`, which does **not** disturb
the image currently loaded in Siril. Three outcomes:

| Detected | Condition | How the channel is extracted |
| --- | --- | --- |
| **CFA (undebayered)** | one channel per frame **and** a `BAYERPAT` header | `seqsplit_cfa` / `seqextract_Green` |
| **RGB (debayered)** | three channels per frame | `split`, frame by frame |
| **Monochrome** | one channel, no `BAYERPAT` | nothing — extraction is refused |

For a mono sequence the channel buttons and the **Extract** button stay greyed
out, and the dialog says why. Nothing meaningless gets produced.

## How each case is extracted

### CFA (undebayered)

`split_cfa` cuts the 2×2 Bayer cell into four quarter-size planes, numbered in
reading order — 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right.
`BAYERPAT` names the same four positions in the same order, so the plane holding
each colour is a direct lookup:

| `BAYERPAT` | R | G | B |
| --- | --- | --- | --- |
| `RGGB` | 0 | 1, 2 | 3 |
| `BGGR` | 3 | 1, 2 | 0 |
| `GRBG` | 1 | 0, 3 | 2 |
| `GBRG` | 2 | 0, 3 | 1 |

- **R / B** → the frame's pixel data is read and the matching plane is taken
  straight out of it (`data[y::2, x::2]`), then written as its own file.
- **G** → `seqextract_Green <seq> -prefix=G_`. Green sits at two of the four
  positions, and `seqextract_Green` combines both of them rather than throwing
  half the green signal away — and it writes only one sequence, so nothing extra
  is produced. Setting an explicit CFA plane overrides this and takes a single
  green plane through the pixel-data path instead.

The output is **half the width and half the height** of the input. These are the
real sensor pixels — nothing is interpolated, which is exactly why this is
preferable to debayering and then splitting.

Because the plane is now sliced out directly, the script also cross-checks the
mapping against the pixels themselves on the first frame: both green positions
carry the same filter, so their medians are nearly identical while red and blue
differ. Whichever diagonal of the 2×2 cell holds that matching pair is the green
one. If that disagrees with `BAYERPAT`, the run logs a warning that red and blue
are probably swapped. The check stays quiet when the two pairs are too alike to
call.

> **If red and blue come out swapped**, use the **CFA plane** dropdown to pick
> the plane by hand. `BAYERPAT` is interpreted relative to how the data is stored
> and can be shifted by `BAYERPAT` X/Y offsets or a different `ROWORDER`; the
> dialog shows both so you can see what it is working from. The script warns when
> the Bayer offset is non-zero, and refuses to guess for an X-Trans matrix or an
> unrecognised pattern — pick the plane manually in that case.

### RGB (already debayered)

Each frame is read, the requested channel is taken out of the planar pixel array
and written on its own. Full resolution, and per-frame progress in the dialog. A
frame that fails is logged and counted, and the run continues.

### Only the requested channel is ever written

Siril's own `split` and `split_cfa` commands always write **every** channel, so
they are not used here. Instead each frame is read with `load_image_from_file()`,
the wanted plane is sliced out of the numpy array, and `save_image_file()` writes
that one plane — neither call disturbs the image currently loaded in Siril. The
other channels are never created, so there is nothing to clean up afterwards and
no wasted disk I/O.

The `BAYERPAT`, `XBAYROFF` and `YBAYROFF` header cards are stripped from CFA
output, since an extracted plane is no longer mosaiced and nothing downstream
should try to debayer it. The rest of the header is carried over.

## Options

| Control | Default | Meaning |
| --- | --- | --- |
| **Prefix** | the channel letter (`R_`, `G_`, `B_`) | Prefix of the new sequence. Untick *from the channel* to type your own. |
| **CFA plane** | Automatic (from `BAYERPAT`) | Force a `split_cfa` plane. Only enabled for CFA sequences. |
| **Create a `.seq` for the result** | on | Writes the `.seq` via `create_new_seq()`. |

Your source frames are never touched — only the new `<prefix><seq>NNNNN` files
are written. If output files with those names already exist, the dialog asks
before overwriting.

## Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--no-gui` | GUI on | Skip the dialog and start straight away. Requires `--sequence`. |
| `--work-dir PATH` | Siril's working directory | Directory holding the sequence. |
| `--sequence NAME` | — | Sequence name, e.g. `light_`. A trailing `.seq` is stripped. |
| `--channel {R,G,B}` | `R` | Channel to extract. |
| `--prefix STR` | the channel letter | Prefix of the new sequence. |
| `--plane {0,1,2,3}` | auto | Force a `split_cfa` plane instead of deriving it from `BAYERPAT`. CFA sequences only. |
| `--no-seq` | write it | Do not write a `.seq` for the result. |
| `--force-cfa` | off | Treat the sequence as undebayered CFA even when its type could not be detected (SER, FITSEQ). |

## Examples

Red channel of an OSC sequence, no dialog:

```bash
python SirilChannelExtract.py --sequence light_ --channel R --no-gui
```

Green, skipping the `.seq`:

```bash
python SirilChannelExtract.py --sequence pp_light_ --channel G --no-seq --no-gui
```

Blue with the Bayer plane forced and a custom prefix:

```bash
python SirilChannelExtract.py --sequence light_ --channel B --plane 3 --prefix blue_ --no-gui
```

## Limitations

- Detection needs one FITS file per frame. For **SER** and **FITSEQ** sequences
  the script cannot read the first frame this way; it says so and you can pass
  `--force-cfa` to run the CFA path anyway, since Siril's own sequence commands
  handle those containers.
- `create_new_seq()` matches files named `<root>NNNNN<ext>` with exactly five
  digits, which is Siril's default. A sequence numbered differently still gets
  its frames written — only the automatic `.seq` is skipped, with a note in the
  log telling you to use *Search sequence*.

---

# DepthFITSConversion.py

## What it is for

Converting a folder of FITS files from one pixel format to the other:

- **32-bit float → 16-bit unsigned integer** — halves the file size, lossy.
- **16-bit unsigned integer → 32-bit float** — lossless, doubles the file size.

Siril works internally with exactly those two formats — unsigned 16-bit in
`[0, 65535]` and 32-bit float in `[0, 1]` — so they are the only two targets.

## What it converts

The input is either **a folder of FITS files** or **one named sequence**:

| Mode | What is converted |
| --- | --- |
| **Every FITS file in the folder** | Every `.fit` / `.fits` / `.fts` (and `.fz`) file found, optionally including subfolders. |
| **One sequence** | Only the frames of that sequence, e.g. `light_00001.fit` … — other files in the same folder are left alone. |

Sequence mode picks the frames by name, so a folder holding several sequences
side by side converts only the one you asked for. Only sequences stored as one
file per frame work; SER and FITSEQ keep every frame inside a single container,
and the script says so rather than doing something surprising.

## Workflow

1. Run **DepthFITSConversion** from the Scripts menu.
2. Pick the folder. The **Sequence** dropdown lists the `.seq` files found there.
3. Choose the input mode — the whole folder, or one sequence. The dialog shows
   how many files or frames that comes to.
4. Choose the target depth. A line under the choice states what it costs.
5. Choose overwrite, or an output folder.
6. Press **Convert**, and confirm the summary.

In sequence mode the `.seq` is rewritten next to the converted frames, so Siril
picks them up without a manual *Search sequence*. That also refreshes the depth
that an in-place conversion has just made stale. Switch it off with `--no-seq`.

## How the conversion is done

There is no Siril command that converts an existing file's bit depth —
`set16bits` / `set32bits` set a *global processing preference* rather than
converting a given file, so the script does not touch them. Instead each file is
read with `load_image_from_file()`, its pixel array is converted, and
`save_image_file()` writes the result. Neither call disturbs the image currently
loaded in Siril, and your Siril 16/32-bit preference is left exactly as it was.

Because the output array's dtype determines the written format, the resulting
`BITPIX` is guaranteed rather than inferred — and the script reads the `BITPIX`
back off the finished file to confirm it, failing the file loudly if it does not
match.

## Scaling

The two formats use different value ranges, so values are rescaled to keep the
picture identical:

| Direction | Operation |
| --- | --- |
| float → integer | `round(clip(value, 0, 1) × 65535)` |
| integer → float | `value / 65535` |

Float files already stored in ADU (maximum well above 1.0, which happens with
FITS from outside Siril) are detected and rounded **without** rescaling —
otherwise everything above 1.0 would clip to pure white. The decision is written
to the log for each run.

`--no-rescale` casts the raw numbers instead, for the rare case where you want
the stored values left alone. This changes how the image looks.

## Clipping

Float → 16-bit is the lossy direction: values outside the range are clipped and
the gradation between two integer steps is gone. The script **counts the clipped
pixels of every file** and reports the count and percentage:

```
  light_00001.fit: 32-bit float -> 16-bit unsigned integer, 1843 pixel(s) clipped (0.021%)
```

A summary line at the end says how many files were affected. Clipping usually
means the source was not in the range the scaling assumed — for example linear
calibrated data with a negative background, which is exactly the case where
16-bit conversion destroys information. Converting back to 32-bit does not
recover it.

## Options

| Control | Default | Meaning |
| --- | --- | --- |
| **Include subfolders** | off | Folder mode only; the folder structure is reproduced in the output folder. |
| **Write a .seq for the converted sequence** | on | Sequence mode only. |
| **Skip files that already have the target depth** | on | Read from each file's `BITPIX`. |
| **Rescale the values between the two conventions** | on | Turn off to cast raw values. |
| **Overwrite the original files** | off | Otherwise an output folder is required; it must differ from the source. |

Compressed `.fz` inputs are written uncompressed. A file that fails is logged and
counted, and the run continues; the final line reports converted / skipped /
failed.

## Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--no-gui` | GUI on | Skip the dialog and start straight away. |
| `--folder PATH` | Siril's working directory | Folder with the FITS files, or the folder the sequence lives in. |
| `--sequence NAME` | — | Convert only the frames of this sequence, e.g. `light_`. A trailing `.seq` is stripped. Naming a sequence is what selects sequence mode. |
| `--recursive` | off | Also convert files in subfolders. Ignored when a sequence is named. |
| `--no-seq` | write it | Do not write a `.seq` for the converted sequence. |
| `--target {16,32}` | `16` | Target depth: 16 = unsigned integer, 32 = float. |
| `--output PATH` | — | Folder for the converted files. Required unless `--overwrite`. |
| `--overwrite` | off | Write the converted files over the originals. |
| `--no-skip` | skip them | Convert every file, even one already at the target depth. |
| `--no-rescale` | rescale | Cast raw values instead of rescaling. |

## Examples

Float to 16-bit into a separate folder:

```bash
python DepthFITSConversion.py --folder "D:/astro/M31/process" --target 16 --output "D:/astro/M31/16bit" --no-gui
```

Whole tree back to 32-bit float, in place:

```bash
python DepthFITSConversion.py --folder "D:/astro/M31" --recursive --target 32 --overwrite --no-gui
```

Only the frames of one sequence, leaving the rest of the folder alone:

```bash
python DepthFITSConversion.py --folder "D:/astro/M31/process" --sequence pp_light_ --target 16 --output "D:/astro/M31/16bit" --no-gui
```

---

# VarStarOSCPreprocess.py

## What it is for

The complete Siril one-shot colour path — master bias, master flat, master dark,
calibration of the lights, registration — stopping deliberately before the stack.
Variable star photometry needs **one measurement per exposure**, so stacking
would throw away exactly the time resolution the light curve is made of.

Both the calibrated and the registered sequences are kept, so you can measure
either one.

## Expected layout

```
<work_dir>/
    lights/       light frames                (required)
    biases/       bias / offset frames        (optional)
    flats/        flat frames                 (optional)
    darks/        dark frames                 (optional)
    process/      created: sequences and masters
```

RAW (CR2/CR3/NEF/ARW/…) and FITS are both fine. Directory names are matched
case-insensitively — `bias`/`offset`, `flat`, `dark`, `light` and their plurals
all work — or you can point at each one explicitly.

Any calibration folder may be missing: that master is skipped and the
`calibrate` call is built from what is actually present. The dialog reports the
frame counts it found before you start.

## Dark optimization

When the lights were shot at a shorter exposure than the master dark, the dark
can be **scaled** before it is subtracted instead of being used as it is:

| `--dark-opt` | What Siril does |
| --- | --- |
| `none` *(default)* | The master dark is subtracted unchanged. |
| `auto` | `calibrate -opt` — the scaling factor is fitted from the data. |
| `exp` | `calibrate -opt=exp` — the factor comes from the exposure keyword. |

Both modes need **a master bias as well as a master dark**. A dark frame is bias
pedestal plus dark current, and only the dark current scales with exposure, so
Siril has to remove the pedestal first — which is why the bias master is passed
to the lights in this mode, and only in this mode. The script checks for both up
front and refuses with a message naming the missing one, rather than letting
`calibrate` fail partway through.

With optimization on, the log compares the two exposures before calibrating:

```
  dark optimization: exp
      light 120s vs master dark 300s (ratio 0.400)
      the lights are shorter, so the dark is scaled down.
```

If the lights turn out to be *longer* than the dark it says so too — scaling a
dark up extrapolates its noise, and matching the exposures is the better fix.

## Ready-made masters

If you already have a master, point the script at it and that whole branch is
skipped — no conversion, no stacking, and the matching input directory is
ignored:

| Given | Effect |
| --- | --- |
| `--master-bias PATH` | The bias directory is not touched; the flats are calibrated with this master. |
| `--master-flat PATH` | The flat directory is not touched, and neither is the bias calibration of the flats. |
| `--master-dark PATH` | The dark directory is not touched. |

They mix freely: give only a master dark and the bias and flat masters are still
built from their directories as usual. The extension may be left off
(`--master-dark process/dark_stacked` finds `dark_stacked.fit`), a relative path
is taken from the working directory, and a master that already sits in the
process directory is passed to `calibrate` by its bare name rather than as a
path. A file that does not exist stops the run before any work is done.

In the dialog this is the **Masters** tab. Filling one in greys out the matching
directory on the Input tab, and the Input summary shows `darks: master` in place
of a frame count, so it is always clear which branch will run.

## Pipeline

| Step | Commands |
| --- | --- |
| 1. Master bias | `convert bias` → `stack bias rej w 3 3 -nonorm -out=bias_stacked` |
| 2. Master flat | `convert flat` → `calibrate flat -bias=bias_stacked` → `stack pp_flat rej w 3 3 -norm=mul -out=pp_flat_stacked` |
| 3. Master dark | `convert dark` → `stack dark rej w 3 3 -nonorm -out=dark_stacked` |
| 4. Lights | `convert light` → `calibrate light -dark=… -flat=… -cc=dark 3 3 -cfa -equalize_cfa -debayer -prefix=pp_` |
| 5. Registration | `register pp_light -interp=none` |
| 6. Extraction *(optional)* | one colour channel → `G_r_pp_light_` |
| 7. | **stop** — no stacking |

Bias and darks are stacked without normalisation, flats with `-norm=mul`. A
calibration folder holding a single frame is used directly instead of being
"stacked", which would fail. When there is no master dark the master bias is
passed to the lights instead; with a dark it is not, because the dark already
contains it.

## Output

```
process/bias_stacked        master bias
process/pp_flat_stacked     master flat (bias-calibrated)
process/dark_stacked        master dark
process/pp_light_           calibrated lights          <- kept
process/r_pp_light_         calibrated + registered    <- kept
process/G_r_pp_light_       one extracted channel      <- optional
```

If registration drops frames — too few stars is the usual reason — the log says
how many, comparing the registered count against the calibrated one.

## Debayering is required here

Siril refuses to register a sequence whose Bayer pattern is still intact
(*"you must debayer it prior to registration"*), so `-debayer` is part of the
`calibrate` call. In the dialog the debayer checkbox is therefore **forced on and
greyed out** whenever registration is ticked, with the reason spelled out next to
it; from the command line, `--no-debayer` together with registration is refused
with the same explanation rather than failing halfway through.

Untick registration and the debayer checkbox becomes editable again, which is how
you get a calibrated CFA sequence — the input a green-channel extraction needs.

The script also looks at the first converted light and adapts:

- already three channels (Siril debayered them on import) → `-debayer` and `-cfa`
  are dropped;
- one channel with a `BAYERPAT` → the normal OSC path;
- one channel without one → not OSC data, so `-debayer` and `-cfa` are dropped;
  calibration and registration still apply perfectly well to mono frames.

## Channel extraction

The optional last step splits **one** colour channel off into a sequence of its
own, with a `.seq` written next to it so Siril picks it straight up. It runs on
the registered frames when there are any, otherwise on the calibrated ones.

| Source | How | Result |
| --- | --- | --- |
| Debayered (the normal case) | the channel is taken straight out of the planar array | **full resolution** |
| Still CFA (`--no-debayer --no-register`) | green via `seqextract_Green`, red/blue take their quarter of the Bayer cell | half width and half height |

Only the requested channel is ever written — Siril's `split` and `split_cfa`
always write all of them, so neither is used. On a CFA source the `BAYERPAT`,
`XBAYROFF` and `YBAYROFF` cards are stripped, since an extracted channel is no
longer mosaiced.

The output is `<prefix><source>_NNNNN`, e.g. `G_r_pp_light_00001.fit`, with the
prefix defaulting to the channel letter. Extracting from monochrome frames is
refused — there is no colour channel to take.

## Interpolation — the setting that matters for photometry

Registration defaults to **`-interp=none`**, which makes Siril apply a whole-pixel
shift and no interpolation at all. Photometry measures photon counts, and every
interpolating method redistributes those counts between neighbouring pixels,
biasing the measurement and correlating the noise. A plain shift leaves the
values untouched.

Passing `none` also forces the transformation to a shift, so the transformation
chooser is greyed out in that mode and `-transf=` is not sent. Choose another
method only when the frames genuinely rotate or scale between exposures — the
dialog warns about the cost when you do.

## Options

The settings sit on four tabs — **Input**, **Calibration**,
**Registration** and **Extraction** — with the progress area and the buttons always visible below
them, so the window stays short enough for a laptop screen. On first open it
sizes itself to its content and is clamped to the available screen area.

| Control | Tab | Default | Meaning |
| --- | --- | --- | --- |
| **Rejection** + sigma low/high | Calibration | Winsorized, 3 / 3 | Used for every master stack. |
| **32-bit float** | Calibration | on | Off means 16-bit. |
| **Cosmetic correction from master dark** | Calibration | on | `-cc=dark`; dropped automatically without a master dark. |
| **-cfa** | Calibration | on | Makes the cosmetic correction aware of the Bayer matrix. |
| **-equalize_cfa** | Calibration | on | Equalises the RGB means of the master flat; only applied when there is one. |
| **-debayer** | Calibration | on | Forced on while registration is enabled. |
| **Dark optimization** | Calibration | None | Greyed-out explanation when a master bias or dark is missing. |
| **Register the calibrated lights** | Registration | on | Untick to stop after calibration. |
| **Interpolation** | Registration | None (whole-pixel shift) | See above. |
| **Transformation** | Registration | Shift | Ignored when the interpolation is *None*. |
| **Extract a single colour channel** | Extraction | off | Adds the extraction step. |
| **Channel** | Extraction | Green | R, G or B. |
| **Prefix** | Extraction | the channel letter | Untick *from the channel* to type your own. |
| **Write a .seq for the extracted sequence** | Extraction | on | |

## Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--no-gui` | GUI on | Skip the dialog and start straight away. |
| `--work-dir PATH` | Siril's working directory | Project directory. |
| `--lights` / `--biases` / `--flats` / `--darks PATH` | auto | Input directories. |
| `--process NAME` | `process` | Directory for the sequences and masters. |
| `--master-bias PATH` | — | Use this master bias instead of building one. |
| `--master-flat PATH` | — | Use this master flat instead of building one. |
| `--master-dark PATH` | — | Use this master dark instead of building one. |
| `--rejection {p,s,m,w,l,g,a,n}` | `w` | Rejection for the master stacks. |
| `--sigma-low F` / `--sigma-high F` | `3.0` | Rejection sigmas. |
| `--16bit` | 32-bit float | Work in 16-bit mode. |
| `--no-cosmetic` | on | Disable `-cc=dark`. |
| `--cc-sigma-low F` / `--cc-sigma-high F` | `3.0` | Cosmetic correction sigmas. |
| `--dark-opt {none,auto,exp}` | `none` | Scale the master dark before subtracting it. Needs a master bias too. |
| `--no-cfa` | on | Do not pass `-cfa`. |
| `--no-equalize-cfa` | on | Do not pass `-equalize_cfa`. |
| `--no-debayer` | on | Keep the CFA mosaic. Incompatible with registration. |
| `--no-register` | on | Stop after calibration. |
| `--interp {none,nearest,linear,cubic,lanczos4,area}` | `none` | Registration interpolation. |
| `--transf {shift,similarity,affine,homography}` | `shift` | Ignored with `--interp=none`. |
| `--layer N` | Siril's default (green) | Layer the registration is computed on. |
| `--extract {R,G,B}` | — | Extract this channel as a last step. Naming a channel is what enables it. |
| `--extract-prefix STR` | the channel letter | Prefix of the extracted sequence. |
| `--no-seq` | write it | Do not write a `.seq` for the extracted sequence. |

## Examples

Whole path plus the green channel, no dialog:

```bash
python VarStarOSCPreprocess.py --work-dir "D:/astro/RR_Lyr" --extract G --no-gui
```

Short lights against a longer master dark, scaled by exposure:

```bash
python VarStarOSCPreprocess.py --work-dir "D:/astro/RR_Lyr" --dark-opt exp --no-gui
```

Reusing masters built on an earlier night, so only the lights are processed:

```bash
python VarStarOSCPreprocess.py --work-dir "D:/astro/RR_Lyr" --master-dark "D:/astro/masters/dark_600s.fit" --master-flat "D:/astro/masters/flat.fit" --extract G --no-gui
```

Whole path without extraction:

```bash
python VarStarOSCPreprocess.py --work-dir "D:/astro/RR_Lyr" --no-gui
```

Calibrate only, keeping the CFA mosaic for a later green extraction:

```bash
python VarStarOSCPreprocess.py --work-dir "D:/astro/RR_Lyr" --no-register --no-debayer --no-gui
```

Frames that rotate between exposures, so interpolation is unavoidable:

```bash
python VarStarOSCPreprocess.py --work-dir "D:/astro/RR_Lyr" --interp lanczos4 --transf homography --no-gui
```

---

# Selector.py

## What it is for

Before stacking you want to throw out the bad subs: frames with soft stars
(seeing, focus), elongated stars (tracking, wind), few stars (clouds, haze,
dew) or a bright sky (moon, dawn). Selector measures every calibrated light
frame, shows the numbers in a table and in charts, suggests which frames to
reject and why, and then removes the ones you finally mark.

## Workflow

1. Run **Selector** from the Scripts menu. The folder starts at Siril's working
   directory.
2. Choose the input: a **sequence** (`.seq` or `.ser`) from the folder or one
   of its direct subfolders (e.g. `process/pp_light_`), or the **individual
   files** in the folder itself (FITS, `.fz`-compressed FITS, TIFF, XISF).
   Picking a sequence from the list switches to sequence mode.
3. Press **Analyse**.
4. Look at the table and the charts, adjust the limits — the suggestions update
   at once — and tick / untick **Reject** for individual frames if you disagree.
5. Optionally **Export CSV...** the statistics.
6. Choose what happens to the rejected frames and press **Apply** — and/or
   copy or move the accepted frames to an `accepted` subfolder with the
   second **Apply**.

## How the frames are measured

| Input | Method |
| --- | --- |
| Sequence with star data | The registration data already stored in the `.seq` is used: FWHM, weighted FWHM, roundness, star count, background. |
| Sequence without star data | `register <seq> -2pass` is run first. With `-2pass` Siril only measures and computes the transforms — no registered images are written. Tick *Re-measure* to force this even when data exists (it replaces the registration data in the `.seq`). |
| Undebayered (CFA) sequence | Siril refuses to register a CFA sequence, so each frame is measured on its own, as below. Star measurements on a CFA mosaic are less precise than on debayered data. |
| Individual files | Each file is loaded and `findstar` is run; the **median** FWHM and roundness of the detected stars are used, the background is the median of the image (green channel for colour data). |

Eccentricity is computed from Siril's roundness *r* (minor / major FWHM) as
*e* = √(1 − *r*²): 0 is a perfectly round star, 0.5 is barely visible
elongation, above ~0.6 stars look clearly oval.

## Rejection criteria

Every criterion is a plain value in the metric's own units — a frame fails it
when its value is past that limit. Tick the criteria you want to use.

| Metric | Default | Rejected when | Proposed limit |
| --- | --- | --- | --- |
| FWHM | on | FWHM > max (px) | median + 2σ |
| Eccentricity | on | eccentricity > max | 0.60 (higher only when every frame is more elongated) |
| Stars | on | stars < min | half the median |
| Background | off | background > max | median + 3σ |

After each analysis the limits are filled in with the proposed values
(σ = 1.4826 · MAD, a robust spread, so a few bad frames do not widen it).
Overwrite them with your own values — e.g. FWHM max `6.27` px — and the
suggestions update at once; **Propose from data** brings the proposals back.
Next to each limit the median and the range (min – max) of the data are shown,
together with how many frames fail it.

A frame in which no stars were found is always suggested for rejection.
Changing a limit recomputes the suggestions and resets your own ticks to them.

## The table

| Column | Meaning |
| --- | --- |
| `#` | Frame number in the sequence / file order |
| Reject | Your decision — starts as the suggestion, click to change |
| Suggestion | `keep` or `REJECT` |
| FWHM, wFWHM | Star FWHM in pixels; wFWHM is Siril's weighted FWHM (sequences only), which also penalises frames with fewer stars |
| Eccentricity, Roundness | Star elongation |
| Stars | Number of detected stars |
| Background, Noise | Sky level and background noise (0–1 for float data, ADU for integer data) |
| Date | `DATE-OBS` of the frame |
| Reason | Why the frame is suggested for rejection, e.g. `FWHM 4.19 > 3.12` |

Rows marked for rejection are tinted red, rows where your decision differs from
the suggestion orange. Columns sort by value. **Double-click** a row to open
that frame in Siril for a visual check.

## The charts

FWHM, eccentricity, stars and background are plotted against the frame number,
with the median and the limit as dashed lines. Blue points are kept, red ones
rejected, orange ones differ from the suggestion. Hover a point for its values,
click it to select the frame in the table.

## Applying the decision

Rejected and accepted frames each have their own action and **Apply** button;
you can use either one or both.

**Rejected frames** (marked *Reject*):

| Action | What happens |
| --- | --- |
| Delete the files | The rejected files are deleted permanently (after a confirmation). |
| Move to the `rejected` subfolder | The files are moved to `<folder>/rejected`, so you can bring them back. |
| Only unselect them in the sequence | The files stay; the frames are excluded in the `.seq` with `unselect` (sequences only). |

**Accepted frames** (not marked):

| Action | What happens |
| --- | --- |
| Copy to the `accepted` subfolder | The accepted files are copied to `<folder>/accepted`; the originals stay. |
| Move to the `accepted` subfolder | The accepted files are moved to `<folder>/accepted`; only the rejected ones are left behind. |

For a **sequence**, `<folder>` is the folder of the `.seq`, and a new sequence
of the same name (e.g. `accepted/pp_light_.seq`) is created from the accepted
files, ready to be registered and stacked; it also shows up in the sequence
list. Files already in `accepted/` are kept, files with the same name are
replaced — empty the folder first when you run the selection again with
stricter limits.

When files of a **sequence** are deleted or moved out, the sequence is closed in
Siril, its `.seq` is removed and rebuilt from the remaining files. The rebuilt
sequence has no registration data, so **register it again** before stacking.
Frames stored inside a single SER / FITSEQ file cannot be removed, copied or
moved one by one — use *unselect* for those.

---

# StarsStatistics.py

## What it is for

A star list with coordinates and photometry: for every star Siril detects, the
script reports where it is (pixel position and RA / Dec), how bright it is
(flux, and a magnitude calibrated against a star catalogue) and under which
exposure it was taken. Run it on a single
image — a stack, or one sub — or on a whole sequence, where the stars are
followed from frame to frame and averaged.

## Workflow

1. Run **StarsStatistics** from the Scripts menu. It starts on the sequence
   loaded in Siril, otherwise on the loaded image.
2. Choose the source:
   - **Image loaded in Siril** — `findstar` is run on it. Tick *Use the stars
     already detected in Siril* to keep the list from the Dynamic PSF window
     instead (for example stars you picked by hand).
   - **Sequence** — a sequence from the folder or one of its direct
     subfolders. Numbered FITS frames (`r_pp_light_00001.fit`, ...) are listed
     even when Siril has not written their `.seq` yet; the script creates it.
     By default only the frames selected in the sequence are measured.
3. For a sequence, choose the result: **arithmetic mean**, **median**, or
   **every frame separately**.
4. Leave **Calibrate the magnitudes with a catalogue** ticked and pick the
   catalogue (see below) — the image must be plate solved for it.
5. Press **Measure**. The table opens in its own window when it is done.
6. In the table window, pick the delimiter and the decimal separator, then
   **Copy CSV** or **Save CSV...**. The *CSV* tab shows the exact text.
7. **Search** the table from the bar above it:
   - **Name** — shows only the stars whose name contains the text, as you
     type; case and spaces do not matter (`ekcep` finds `EK Cep`).
   - **RA / Dec** — `21:41:21.5`, `21 41 21.5`, `21h41m21.5s` or degrees
     (`325.34`); Dec likewise (`+69:41:34`, `69d41m34s`, `69.693`). **Find**
     (or Enter) shows the stars within the radius (10″ by default) and selects
     the nearest one; the distance is shown next to the bar.
   - Both can be combined; **Show all** clears the search. The search only
     hides rows — **Copy CSV** and **Save CSV...** still take the whole table.
8. **Double-click a row** to find that star in Siril: it gets a yellow circle
   with a cross-hair and its number, and the view is centred on it at 100 %
   zoom (or more, if you were zoomed in further). The star is looked up on the
   image Siril shows — for a sequence on the frame currently shown, else from
   its RA / Dec. The next double-click moves the circle.

**Show table again** builds a new table from the last measurement with the
options set now — switching between mean, median and per frame, or changing the
matching, does not need a new measurement. Changing the catalogue does.

## The columns

| Column | Meaning |
| --- | --- |
| `star` | Star number, 1 = the brightest. In a sequence the same star keeps its number in every frame, so the per-frame table can be pivoted into light curves. |
| `name` | Name of the star: its variable star designation from VSX (e.g. `EK Cep`, `ASASSN-V J...`), else SIMBAD's main identifier (`HD 207636`, `TYC 4465-965-1`, `Gaia DR3 ...`); prefixes such as `V*` are dropped. Empty for an uncatalogued star. Only with **Star names from VSX and SIMBAD** ticked (on by default; online, plate-solved image). |
| `x`, `y` | Position in the image, px (`x_ref`, `y_ref` in the averaged table: the position in the reference frame). |
| `ra_deg`, `dec_deg` | RA / Dec in degrees, from the plate solution. Empty when the image is not plate solved. `ra_hms` / `dec_dms` are added on request. |
| `mag` | Calibrated magnitude, `mag_inst` plus the frame's zero point (see below) — for every star, also those missing in the catalogue. Only with a catalogue. |
| `mag_inst` | Instrumental magnitude `25 − 2.5 log10(flux_adu16)` — the usual scale with a nominal zero point of 25 (as IRAF's `phot`), so it is positive. Always in the table and the CSV, with or without a catalogue; not calibrated, but differences between stars of one frame are real magnitude differences. |
| `cat_V`, `cat_B` / `cat_G`, `cat_BP` | The magnitudes of the matching catalogue star, named after the catalogue's bands; empty for a star not in the catalogue. |
| `cat_err`, `cat_dist_arcsec` | APASS's error of `cat_V`, and the distance to the catalogue star. |
| `zero_point` | Per-frame table: the zero point of that frame, `mag − mag_inst`. |
| `flux_adu16` | Integral of the fitted PSF above the local background (Gaussian or Moffat, whichever Siril fitted), in **16-bit ADU**: a 32-bit float image (0..1) is scaled by 65535, a 16-bit image is used as is. It is the sum over all the star's pixels, so it can be far above 65535. |
| `max_flux` | Value of the star's brightest pixel, background included, in 16-bit ADU. This one is limited to 65535: a star near it is saturated. |
| `fwhm_px`, `saturated` | FWHM (mean of both axes), and whether the star has saturated pixels. |
| `exposure_s` | `EXPTIME` / `EXPOSURE` from the FITS header; empty when the file has none (e.g. a TIFF without it). |
| `gain` | Camera gain setting, `GAIN` from the FITS header (e.g. 100 on a ZWO camera; 0 is a valid value). Empty when the header has none, e.g. for a DSLR. In the averaged table the median over the frames. |
| `filter` | `FILTER` from the FITS header (e.g. `L`, `V`, `Ha`); empty when the header has none. In the averaged table the most common filter of the frames. |
| `focal_ratio` | Focal ratio (f-number), `FOCRATIO` from the FITS header (e.g. `3.45`); empty when the header has none. In the averaged table the median over the frames. |
| `date_obs` | `DATE-OBS` of the frame (single image and per-frame table). |
| `n_frames`, `mag_sigma`, `saturated_frames` | Averaged table only: in how many frames the star was found, the scatter of `mag` (of `mag_inst` without a catalogue) between the frames (standard deviation for the mean, 1.4826 × MAD for the median), and in how many frames it was saturated. |

Tick **Leave out saturated stars** to drop them from the table (and from the
averages) — their flux and magnitude are wrong.

Tick **Mark the stars in Siril with circles** to see which stars the table
holds: every star of the table gets a circle on the image Siril shows — green,
red for a saturated star — optionally **with the star numbers** from the `star`
column, so a row of the table is easy to find in the image. For a sequence the
frame Siril currently shows is marked, with the positions measured on that
frame. The circles live on Siril's overlay; its overlay button clears them, and
a new table replaces them. Marking a few thousand stars takes a few seconds and
can be stopped with **Stop**.

## Star names

With **Star names from VSX and SIMBAD** ticked, the script also runs
`conesearch -cat=vsx` and `conesearch -cat=simbad` (down to mag 18) on the
plate-solved image — for a sequence on its first plate-solved frame — and names
every star that has a catalogue object within 3″ (at least 1.5 pixels, after
the same offset removal as below). VSX comes first, so a known
variable star is always listed under its variable star designation; SIMBAD
names the rest. On the EK Cep test field that is about 1250 named objects,
mostly Gaia, Tycho and UCAC4 numbers besides the variable stars.

## Catalogue magnitudes

The catalogue stars of the field are read with Siril's `conesearch` (once — for
a sequence on its first plate-solved frame), down to the chosen magnitude:

| Catalogue | Bands | Note |
| --- | --- | --- |
| **APASS** | V, B | Online. The usual choice for variable stars measured in the green channel. |
| **Gaia DR3, local** | G | Offline, from Siril's local catalogue. G is a broad band, so expect a colour-dependent offset. |
| **Gaia DR3, online** | G, BP | Online, through VizieR. |
| **NOMAD** | V, B | Online. |

Every measured star is paired with the nearest catalogue star within the
radius, each catalogue star with one measured star at most. Two things keep
that pairing reliable whatever the image scale:

- **Offset removal.** The measured positions are usually shifted against the
  catalogue as a whole — by the plate solution and by where Siril puts a
  pixel's centre, typically half a pixel. A first pass with a wide radius
  measures that shift (median over all pairs) and it is taken off before the
  real pairing. The log reports it per frame.
- **Radius floor.** The radius (3″ by default) is never smaller than
  1.5 pixels. At a coarse scale such as 3.2″/px, 3″ is less than one pixel.

On the RW Lac field (3.2″/px, 1.2″ offset) this raised the APASS matches from
3300 to 6075 and found RW Lac itself, which was measured 3.5″ from its
catalogue position.
Each frame then gets its **zero point**: the median of `cat − mag_inst` over the
unsaturated matched stars, with outliers (variable stars, blends, wrong matches)
clipped at 3σ. `mag = mag_inst + zero point` — a per-frame zero point also
removes changes of transparency and airmass between the frames. The log lists
each frame's zero point and its scatter.

There is no colour term: a star much redder or bluer than average comes out a
little off, and the catalogue's own errors grow at its faint end. On the test
field (EK Cep, APASS V) the calibrated `mag` agreed with `cat_V` to 0.04 mag
(1σ) between V 10 and 14. Saturated stars keep a wrong `mag` — tick **Leave out
saturated stars** to drop them.

## Matching stars in a sequence

Every frame is loaded and measured on its own, so the script has to work out
which detection in one frame is the same star as in another:

| Mode | Use it for |
| --- | --- |
| **RA / Dec** | Plate-solved frames (e.g. after `seqplatesolve`). Works on frames that are not registered. Default tolerance 3″. |
| **Pixel position** | Registered (aligned) frames, e.g. an `r_` sequence. Default tolerance 3 px. |
| **Automatic** | RA / Dec when every frame is plate solved, pixel position otherwise. |

Matching starts from the sequence's reference frame. Each star is paired with
the nearest detection within the tolerance, at most one per frame; a star that
matches nothing starts a new entry. The averaged table keeps only stars found in
at least the given share of the frames (50 % by default).

Only sequences of separate files can be measured — the frames of a SER or
FITSEQ file cannot be loaded one by one. The frame list, the selection and the
reference frame are read from the `.seq` file itself, so the script also works
in headless Siril (`siril-cli`), where `load_seq` is not allowed. In the GUI the
sequence is loaded in Siril again after the run.

`date_obs` is taken from the `DATE-OBS` card exactly as written (UT). sirilpy
hands the date over converted to the computer's local time, so it is not used
when the header has the card.

## License

Copyright (C) 2026 Martin Mancuska <martin@martin-in.space>

These scripts are free software; you can redistribute them and/or modify them
under the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option) any
later version.

They are distributed in the hope that they will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the [LICENSE](LICENSE) file for the full text of the
GNU General Public License version 2.
