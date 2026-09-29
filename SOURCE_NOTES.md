# Source notes

Prepared 2026-09-29 from the actual uploaded `dream_channels_batch(1).py` in the
referenced conversation. Its SHA-256 before modification was:

`d3fa567f18a3cb939bdfaef82d1259322aed6e27eba24011849a51517f77a662`

A locally available copy of that attachment was byte-identical. Other available
`dream.py` and `dream_channels(1).py` attachments were older variants; this release
uses the batch/channel variant as its base. The uploaded file contained no
copyright/license header or named third-party code attribution to preserve.
The project's own code is released under Apache-2.0 at the user's request; this
source inspection is not proof of every historical contribution's authorship.

The conversation referenced generated `dream_channels_batch_jpeg95.py` and
`make_deepdream_pairs_parallel.py` download links, but their file contents were
not available from the reference or the relevant local attachment directories.
This package does not claim those generated files were recovered verbatim.
Instead, JPEG and minimum-octave settings described there were applied to the
actual uploaded generator, and the parallel wrapper was reconstructed from the
recorded interface. The wrapper's output layout and resume contract are documented
in README; it is not guaranteed to resume datasets from the missing older wrapper.

## Changes from the uploaded generator

- Renamed the canonical script to `dream.py`; kept model preprocessing, layer/channel
  selection, activation losses, gradient ascent and sequential folder processing.
- Added Apache-2.0 identification and this explicit modification record.
- Removed the global TLS-verification bypass and automatic ZIP extraction.
- Added explicit standard-library bootstrap with HTTPS, archive/file hash checks,
  exact-member extraction and the original model license. Model paths are based
  on the script directory and configurable with `--model-dir`.
- Added JPEG quality 95, 4:4:4 subsampling and optimized encoding; other image
  formats keep their existing encoding path.
- Added minimum-octave skipping and an error when every scale is too small.
- Merged tile remainders smaller than 64 pixels into their neighboring tile.
- Added numeric CLI validation, tuple/list activation support, EXIF orientation,
  image/JSON collision avoidance and a nonzero status when any image fails.
- Reconstructed the parallel wrapper with matching layer syntax, settings passed
  through to targets, recursive discovery, collision-safe names, pair dimension
  checks, completion metadata and conservative resume behavior.

No training, weight modifications, conversion to another model, dataset copying,
cloud publishing or GitHub repository creation was performed. Weights were used
locally for verification and excluded from the deliverable.

## Checks performed

On macOS arm64, Python 3.11.15, TensorFlow 2.16.2, NumPy 1.26.4, Pillow 12.3.0:

- Python compilation and runnable smoke checks passed.
- Upstream HTTPS archive fetched and all retained members hashed; model install
  from the downloaded archive verified. The bootstrap's own HTTPS download and
  installation were also run successfully.
- Real multi-layer/channel inference and gradient ascent passed at several scales,
  including short tile boundaries; output was finite and changed the pixels.
- Parallel pair generation with two same-stem, different-extension images passed;
  paired sizes matched, JPEG subsampling was disabled and resume preserved files.
- Tiny inputs produced a nonzero CLI status; corrupt archives and failed child
  processes were rejected without publishing a completed pair.

Other operating systems, GPU execution and long production batches were not run.
