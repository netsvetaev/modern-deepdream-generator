# Third-party notices

## Inception5h model artifact

**Publisher/source:** Google / TensorFlow. **Artifact:** `inception5h.zip`, a
GoogLeNet/Inception-v1-family frozen GraphDef, not Keras InceptionV3.

- [Original artifact](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip)
- [TensorFlow r0.8 example linking the exact artifact](https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/android/README.md)
- [Verbatim artifact LICENSE](third_party/inception5h-LICENSE.txt)

The exact downloaded ZIP was inspected during packaging on 2026-09-29. It contains
only `tensorflow_inception_graph.pb`, `imagenet_comp_graph_label_strings.txt`, and
`LICENSE`. The LICENSE begins `Copyright 2015 The TensorFlow Authors. All rights
reserved.` and supplies Apache License 2.0. No separate NOTICE or different
weights-specific license was present in this archive. The archive's own license
is the evidence for the model's upstream Apache-2.0 terms; this conclusion is not
inferred from this project's license, TensorFlow's library license, or a different
DeepDream repository.

This project's license grant does not replace that upstream grant. Google /
TensorFlow retain attribution to their model. We do not include the graph or
labels in the GitHub source package. `download_model.py` downloads them directly
and retains the original LICENSE beside them. If distributing those files in a
separate package, preserve their original attribution and license, indicate any
modifications, and comply with upstream Apache-2.0 conditions. This repository
does not grant rights to Google's training data, user photographs or trademarks,
or imply Google endorsement.

### Artifact identity

These SHA-256 values were computed from the fetched archive; they are local
integrity pins, not an independently signed checksum published by the vendor.

| File | SHA-256 |
| --- | --- |
| `inception5h.zip` | `d13569f6a98159de37e92e9c8ec4dae8f674fbf475f69fe6199b514f756d4364` |
| `tensorflow_inception_graph.pb` | `a39b08b826c9d5a5532ff424c03a3a11a202967544e389aca4b06c2bd8aef63f` |
| `imagenet_comp_graph_label_strings.txt` | `da2a31ecfe9f212ae8dd07379b11a74cb2d7a110eba12c5fc8c862a65b8e6606` |
| `LICENSE` | `f086f362c12f3a0295ba186c8caa1d2778beb6b9a7651c499791f202c2429c0d` |

If the source archive changes, the bootstrap fails instead of silently accepting
new weights or terms. Review the changed artifact and license before updating
these pins. The bundled license is an exact copy, without a project header.

## Runtime dependencies (installed separately)

| Component | License/reference | Scope |
| --- | --- | --- |
| TensorFlow 2.16.2 | [Apache-2.0](https://github.com/tensorflow/tensorflow/blob/v2.16.2/LICENSE) | TensorFlow code, not proof of a model license |
| NumPy 1.26.4 | [BSD-3-Clause and bundled notices](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt) | Installed NumPy distribution |
| Pillow 11.3–12.x | [MIT-CMU / historical PIL terms](https://github.com/python-pillow/Pillow/blob/main/LICENSE) | Installed Pillow distribution |
| Python | [PSF license and bundled notices](https://docs.python.org/3/license.html) | Interpreter and standard library |

Those libraries and their transitive dependencies are not vendored. Their wheel
and source distributions may carry additional bundled-library notices; retain
those when redistributing an environment or executable. Project Apache-2.0 does
not relicense dependencies. No code or images were copied from `google/deepdream`
during this packaging work. Source derivation is recorded in SOURCE_NOTES.md.
