# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepDream Generator contributors.
"""Reconstructed parallel pair generator; see SOURCE_NOTES.md."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageOps
from download_model import DEFAULT_MODEL_DIR, verify_model

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def build_command(args, input_path, output_path):
    cmd = [sys.executable, str(args.dream_script), '--input', str(input_path),
           '--output', str(output_path), '--layers', *args.layers,
           '--steps', str(args.steps), '--octaves', str(args.octaves),
           '--step-size', str(args.step_size), '--max-size', str(args.dream_max_size),
           '--jpeg-quality', str(args.jpeg_quality), '--min-octave-size', str(args.min_octave_size),
           '--loss-mode', args.loss_mode, '--model-dir', str(args.model_dir)]
    if args.tile_size:
        cmd += ['--tile-size', str(args.tile_size)]
    return cmd


def process_image(source, args):
    relative = source.relative_to(args.source_root)
    # Retain original extension in the name: photo.png and photo.jpg cannot collide.
    name = relative.with_name(relative.name + '.jpg')
    input_path = args.output_root / 'inputs' / name
    target_path = args.output_root / 'targets' / name
    record_path = args.output_root / 'metadata' / name.with_suffix('.json')
    signature = {
        'source': str(source), 'source_size': source.stat().st_size,
        'source_mtime_ns': source.stat().st_mtime_ns,
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                     if k not in {'workers', 'limit', 'skip_existing'}},
    }
    if args.skip_existing and all(p.is_file() for p in (input_path, target_path, record_path)):
        if json.loads(record_path.read_text(encoding='utf-8')).get('signature') == signature:
            return f'SKIP {relative}'
    if any(p.exists() for p in (input_path, target_path, record_path)):
        raise FileExistsError(f'Existing/incomplete pair or changed settings: {relative}; use a new output-root.')
    with tempfile.TemporaryDirectory(prefix='deepdream-') as temp:
        staged_input = Path(temp) / 'input.jpg'
        staged_target = Path(temp) / 'target.jpg'
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            if args.input_long_side:
                image.thumbnail((args.input_long_side, args.input_long_side), Image.Resampling.LANCZOS)
            image.save(staged_input, quality=args.jpeg_quality, subsampling=0, optimize=True)
        subprocess.run(build_command(args, staged_input, staged_target), check=True)
        with Image.open(staged_target) as target:
            if target.size != image.size:
                raise ValueError(f'Pair size mismatch for {relative}; use --dream-max-size 0.')
            target.verify()
        metadata = json.loads(staged_target.with_suffix('.json').read_text(encoding='utf-8'))
        metadata.update(input=str(input_path), output=str(target_path),
                        input_image=str(input_path), output_image=str(target_path))
        record = {'signature': signature, 'input': str(input_path), 'target': str(target_path), 'dream': metadata}
        # The metadata is the completion marker; partial pairs are never silently skipped.
        for path in (input_path, target_path, record_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(staged_input.read_bytes())
        target_path.write_bytes(staged_target.read_bytes())
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    return f'OK {relative}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dream-script', type=Path, default=Path(__file__).with_name('dream.py'))
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--preset-name', default='mixed4c')
    parser.add_argument('--layers', nargs='+', default=['mixed4c'])
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--octaves', type=int, default=4)
    parser.add_argument('--step-size', type=float, default=0.75)
    parser.add_argument('--input-long-side', type=int, default=1536)
    parser.add_argument('--dream-max-size', type=int, default=0)
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--min-octave-size', type=int, default=64)
    parser.add_argument('--loss-mode', choices=['mean', 'square', 'relu'], default='mean')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    for name in ('source_root', 'output_root', 'dream_script', 'model_dir'):
        setattr(args, name, getattr(args, name).resolve())
    if (args.workers < 1 or args.limit < 0 or args.input_long_side < 0 or args.dream_max_size < 0
            or args.steps < 1 or not 0 <= args.octaves <= 100
            or args.tile_size != 0 and args.tile_size < 64 or args.min_octave_size < 64
            or not math.isfinite(args.step_size) or args.step_size <= 0
            or not 1 <= args.jpeg_quality <= 100):
        parser.error('Invalid numeric setting; see README for ranges.')
    if not args.source_root.is_dir() or not args.dream_script.is_file():
        parser.error('source-root must be a directory and dream-script must be a file')
    if args.output_root.is_relative_to(args.source_root):
        parser.error('output-root must be outside source-root')
    verify_model(args.model_dir)
    sources = sorted(p for p in args.source_root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        parser.error('No supported images found')
    def worker(source):
        try:
            print(process_image(source, args), flush=True)
            return True
        except Exception as exc:
            print(f'ERROR {source}: {exc}', file=sys.stderr, flush=True)
            return False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker, sources))
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
