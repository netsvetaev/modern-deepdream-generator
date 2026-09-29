# Local models

Run `python download_model.py` from the repository root. The bootstrap creates:

```text
inception5h/
  tensorflow_inception_graph.pb
  imagenet_comp_graph_label_strings.txt
  LICENSE
```

These downloaded files are ignored by Git. The original upstream license is also
preserved in `../third_party/inception5h-LICENSE.txt`. Do not replace it with the
project LICENSE. Model provenance and integrity hashes are in
`../THIRD_PARTY_NOTICES.md`. No weights ship in the repository archive.
