# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepDream Generator contributors.
"""Fetch the original TensorFlow Inception5h archive, retaining its own license."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

URL = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
ARCHIVE_SHA256 = 'd13569f6a98159de37e92e9c8ec4dae8f674fbf475f69fe6199b514f756d4364'
FILES = {
    'tensorflow_inception_graph.pb': 'a39b08b826c9d5a5532ff424c03a3a11a202967544e389aca4b06c2bd8aef63f',
    'imagenet_comp_graph_label_strings.txt': 'da2a31ecfe9f212ae8dd07379b11a74cb2d7a110eba12c5fc8c862a65b8e6606',
    'LICENSE': 'f086f362c12f3a0295ba186c8caa1d2778beb6b9a7651c499791f202c2429c0d',
}
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'inception5h'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_model(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    for name, expected in FILES.items():
        path = model_dir / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Missing or altered model file: {path}. Run python download_model.py --model-dir "{model_dir}"')
    return model_dir / 'tensorflow_inception_graph.pb'


def install_archive(archive, model_dir):
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Archive SHA-256 mismatch; refusing to install. Do not bypass this check.')
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=model_dir) as temp:
        stage = Path(temp)
        with zipfile.ZipFile(archive) as z:
            for name, expected in FILES.items():
                # Exact filenames only: never extract archive-controlled paths.
                with z.open(name) as src, (stage / name).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                if sha256(stage / name) != expected:
                    raise ValueError(f'SHA-256 mismatch: {name}')
        # Publish weights last, after the license and labels are in place.
        for name in reversed(FILES):
            (stage / name).replace(model_dir / name)
    return verify_model(model_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--archive', type=Path, help='Install a previously downloaded, hash-checked archive offline')
    args = parser.parse_args()
    try:
        path = verify_model(args.model_dir)
    except ValueError:
        if args.archive:
            path = install_archive(args.archive, args.model_dir)
        else:
            print(f'Downloading {URL}\nUpstream license: Apache-2.0, Copyright 2015 The TensorFlow Authors.')
            with tempfile.TemporaryDirectory() as temp:
                archive = Path(temp) / 'inception5h.zip'
                with urllib.request.urlopen(URL, timeout=60) as src, archive.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                path = install_archive(archive, args.model_dir)
    print(f'Verified: {path}')


if __name__ == '__main__':
    main()
