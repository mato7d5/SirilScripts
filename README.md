# Siril Scripts

Python scripts for [Siril](https://siril.org/) 1.4+ that use the `sirilpy`
interface. Each one opens a Tkinter dialog, runs its work on a background thread
and reports progress into Siril's log, so the Siril window stays responsive.

| Script | What it does |
| --- | --- |
| [`SirilSync.py`](SirilSync.py) | Reads the FITS `HISTORY` of the image loaded in Siril, translates it back into Siril commands, and replays the ones you select onto a whole folder of images. |
| [`siril_dark_calibration.py`](siril_dark_calibration.py) | Builds a master dark from RAW darks, calibrates RAW light frames with it, and exports the calibrated frames as TIFF. |
| [`SirilChannelExtract.py`](SirilChannelExtract.py) | Detects whether a sequence holds OSC (one-shot colour) data and extracts the R, G or B channel into a new sequence. |
| [`VarStarOSCPreprocess.py`](VarStarOSCPreprocess.py) | Full OSC preprocessing: masters, light calibration, registration and optional single-channel extraction — deliberately stopping before stacking. |
| [`DepthFITSConversion.py`](DepthFITSConversion.py) | Converts a folder of FITS files, or the frames of one sequence, between 32-bit float and 16-bit unsigned integer, reporting any clipping. |

## Requirements

- Siril **1.4** or newer, built with the Python (`sirilpy`) interface.
- `ttkthemes` — installed on demand through `sirilpy.ensure_installed()`.
  Every script except `SirilSync.py` falls back to a plain `tkinter.Tk` window
  if the themed toolkit is unavailable; `SirilSync.py` requires it.

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
an image loaded in the running Siril instance.

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

The settings sit on four tabs — **Input**, **Calibration**, **Registration**
and **Extraction** — with the progress area and the buttons always visible below
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
| `--rejection {p,s,m,w,l,g,a,n}` | `w` | Rejection for the master stacks. |
| `--sigma-low F` / `--sigma-high F` | `3.0` | Rejection sigmas. |
| `--16bit` | 32-bit float | Work in 16-bit mode. |
| `--no-cosmetic` | on | Disable `-cc=dark`. |
| `--cc-sigma-low F` / `--cc-sigma-high F` | `3.0` | Cosmetic correction sigmas. |
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
