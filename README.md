# Modern DeepDream Generator

Just a vibe-coded toy I wanted to run on MacOS. Based on Google's original **Inception5h
(GoogLeNet / Inception v1)** frozen TensorFlow graph. It does **not** train or
fine-tune the model. A companion script creates source/DeepDream image pairs in
parallel for downstream experiments.

Project code is **Apache-2.0**. The downloaded model has its **own upstream
Apache-2.0 license**, copyright The TensorFlow Authors. Weights are not included
or relicensed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Features

- Process a single image or every supported image immediately inside a folder.
- Choose multiple layers and specific, zero-based channels independently per layer.
- Three activation objectives: `mean`, `square`, and `relu`.
- Multiscale octaves, tiled gradients, random spatial jitter, and optional resizing.
- Skip octaves below a minimum size; merge tile edges shorter than 64 pixels.
- JPEG quality 95 by default, no chroma subsampling, optimized encoding. Quality
  is Pillow's 1–100 setting, not a percentage of retained visual information.
- Save a JSON sidecar with processing settings beside each generated image.
- Preserve existing image/sidecar names by choosing a numeric suffix.
- Generate aligned JPEG pairs recursively with three concurrent child processes
  by default; resumable completed pairs and nonzero exit codes on failure.

Supported input extensions: JPEG, PNG, WebP, BMP, TIFF. Images become 8-bit RGB;
Codec availability depends on the installed Pillow build.

## Requirements and operating systems

Use **64-bit CPython 3.11 or 3.12**, a virtual environment, and the dependencies
in `requirements.txt`: TensorFlow 2.16.2, NumPy 1.26.4, and Pillow 11.3–12.x.
Python 3.13/3.14 are outside this dependency baseline. TensorFlow is deliberately
pinned because this script imports an old GraphDef through `tf.compat.v1`; newer
TensorFlow versions have not been validated for this release.

| Platform | Execution path | Verification for this package |
| --- | --- | --- |
| macOS, Apple Silicon | CPU; native arm64 Python, macOS 12+ for the TF wheel | Real graph, gradients, JPEG and parallel-pair smoke tests passed |
| macOS, Intel | CPU; x86-64 TensorFlow 2.16.2 wheel | Expected compatible; not run here |
| Linux x86-64 | CPU; NVIDIA GPU optional with a compatible TensorFlow/CUDA installation | Expected compatible; not run here |
| Windows x86-64 | CPU using native Python; NVIDIA GPU through WSL2 | Expected compatible; not run here |
| Windows ARM, Linux ARM, other Python runtimes | Depends on available binary wheels and graph support | Not validated |

The table describes compatibility expectations, not a completed cross-platform
certification. [TensorFlow installation instructions](https://www.tensorflow.org/install/pip)
macOS CPU is the baseline; `tensorflow-metal`
is not installed or validated, and legacy graph operations may not run on it.
For Apple Silicon, TensorFlow's [2.16 release notes](https://blog.tensorflow.org/2024/03/whats-new-in-tensorflow-216.html)
use the `tensorflow` package directly.

Memory/time depend heavily on image size, tiles, steps, and
workers. Start with `--workers 1` and modest images on a constrained machine.
Each parallel child loads a separate model.

## Example results

The same original image processed with each of the five separately tested layers (`mixed4a`–`mixed4e`). All were generated with 30 steps, 4 octaves, step size 1.5, tile size 512, and JPEG quality 95. Photos are sample assets and are not covered by the project code license.

| Original | `mixed4a` — scales / feathers | `mixed4b` — insects / reptiles |
| --- | --- | --- |
| ![Original input](examples/original.jpg) | ![mixed4a result](examples/mixed4a.jpg) | ![mixed4b result](examples/mixed4b.jpg) |

| `mixed4c` — dogs / slugs | `mixed4d` — large animals | `mixed4e` — more animals |
| --- | --- | --- |
| ![mixed4c result](examples/mixed4c.jpg) | ![mixed4d result](examples/mixed4d.jpg) | ![mixed4e result](examples/mixed4e.jpg) |


## Install

Run commands from this repository's root. On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_model.py
```

On Windows PowerShell (no activation required):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
```

Use `.\.venv\Scripts\python.exe` instead of `python` in subsequent Windows
commands. Multiline examples below use Bash continuation; in PowerShell run them
on one line, or use PowerShell's backtick continuation.

The bootstrap stores the graph, labels, and original license under
`models/inception5h/`, relative to the scripts, not the shell's current directory.

## Generate images

Single image:

```bash
python dream.py --input images/photo.jpg --output results/dream.jpg \
  --layers mixed4c --steps 30 --octaves 4 --step-size 0.75 \
  --max-size 0 --tile-size 512 --jpeg-quality 95
```

Choose different channels in different layers:

```bash
python dream.py --input images/photo.jpg --output results/channels.jpg \
  --layers mixed4c:123,124 mixed4d:88,91 --loss-mode mean
```

There is **no separate `--channels` argument**. A bare layer name selects all its
channels. `mixed4c` has 512 channels and `mixed4d` has 528 in this artifact.

Folder mode (nonrecursive):

```bash
python dream.py --input images --output results --layers mixed4c mixed4d
```

Without `--output`, results go into `result/` beside the input file or inside the
input directory. A single-file `--output` is a filename; a folder-mode `--output`
is a directory. Existing results receive `_1`, `_2`, etc.

| Argument | `dream.py` default | Meaning |
| --- | --- | --- |
| `--layers` | `mixed4c` | One or more layer/channel specifications |
| `--steps` | `50` | Gradient updates per retained octave; at least 1 |
| `--octaves` | `5` | Smaller-scale levels; processes exponents `-N..0`, up to **N+1** scales; range 0–100 |
| `--step-size` | `1.5` | Positive finite gradient-ascent step in 0–255 pixel space |
| `--max-size` | `0` | Long-side cap; zero preserves size, positive values only downscale |
| `--tile-size` | `512` | Tile edge, at least 64; short trailing strips merge into the prior tile |
| `--min-octave-size` | `64` | Minimum height AND width, at least 64; smaller octaves skip |
| `--jpeg-quality` | `95` | JPEG quality 1–100; does not affect PNG/TIFF/etc. |
| `--loss-mode` | `mean` | Sum of layer means, mean squares, or means after ReLU |
| `--model-dir` | `models/inception5h` beside scripts | Verified model directory |

Octave scale is 1.4. `--octaves 0` processes the original size once. Tiling limits activation memory, but the full image and gradient remain in memory. Random jitter makes results nondeterministic.

## Generate training pairs in parallel

```bash
python make_deepdream_pairs_parallel.py \
  --source-root images --output-root pairs/mixed4c_animals \
  --preset-name mixed4c_animals --layers mixed4c \
  --steps 30 --octaves 4 --step-size 0.75 \
  --input-long-side 1536 --dream-max-size 0 --tile-size 512 \
  --jpeg-quality 95 --workers 3 --limit 6
```

Remove `--limit 6` for all images; use `--skip-existing` to resume completed pairs
with the same settings. Folder scanning here **is recursive**. Relative folders
are preserved. For `photos/a.png`, output paths are:

```text
pairs/mixed4c_animals/
  inputs/photos/a.png.jpg
  targets/photos/a.png.jpg
  metadata/photos/a.png.json
```

The original extension remains in the basename to prevent collisions with
`a.jpg`. Input and target both use JPEG quality 95. Inputs are oriented, converted
to RGB and downscaled to a maximum long side of 1536 by default; small images are
not enlarged. Targets are checked for matching dimensions before publication.

Wrapper-only settings:

- `--workers 3`: concurrent image processes; use 1 if memory is scarce.
- `--input-long-side 1536`: input resize cap; 0 preserves dimensions.
- `--dream-max-size 0`: passed to `dream.py` as `--max-size`; keep zero for aligned pairs.
- `--tile-size 0`: omit the child argument, using its default of 512. This does
  **not** disable tiling. Direct `dream.py --tile-size 0` is invalid.
- `--limit 0`: process all source files; a positive value selects the first N sorted paths.
- `--preset-name`: metadata label only, not a built-in style lookup.
- `--dream-script`: defaults to the bundled `dream.py`; replacements must implement
  its CLI, JPEG and JSON-sidecar contract.


## Verify

```bash
python test_smoke.py
python test_smoke.py --model-dir models/inception5h
```

Validated on macOS arm64, Python 3.11.15, TensorFlow 2.16.2, NumPy 1.26.4 and
Pillow 12.3.0. TensorFlow can print a deprecation warning for `extract_sub_graph`;
the pinned version's real inference passed. Batch image failures continue to the
next image but make the process exit nonzero.

## Model provenance and repository layout

The inspected uploaded script explicitly downloaded:

[TensorFlow's original inception5h.zip](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip)

Its graph is `tensorflow_inception_graph.pb`, imported with `input:0`; pixel values
remain in `[0,255]` and the script subtracts 117 before inference. TensorFlow's
[historical Android example](https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/android/README.md)
links this exact archive. This is not the Keras InceptionV3 DeepDream model, and
this project does not claim to have trained Google's weights.

```text
dream.py                         Main generator, based on uploaded source
make_deepdream_pairs_parallel.py Reconstructed pair-generation wrapper
download_model.py                Verified model bootstrap, standard library only
test_smoke.py                    Runnable checks
requirements.txt                 Dependency baseline
LICENSE                         Project Apache-2.0 text
NOTICE                          Project and upstream attribution
THIRD_PARTY_NOTICES.md           Model identity, hashes and license scope
SOURCE_NOTES.md                  Source history and publication changes
third_party/inception5h-LICENSE.txt  Verbatim license from the model archive
models/README.md                 Local model layout (weights ignored)
examples/README.md               Gallery notes and sample-image rights
.gitignore                      Excludes weights, environments and output data
```

Made with Google Antigravity free tier and ChatGPT 5.5. 
[https://netsvetaev.com](https://netsvetaev.com)
