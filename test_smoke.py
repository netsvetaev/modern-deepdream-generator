# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepDream Generator contributors.
"""Run with installed requirements; add --model-dir for real inference/CLI checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image, JpegImagePlugin
import dream
import download_model
import make_deepdream_pairs_parallel as pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    names, channels = dream.parse_layer_specs(['mixed4c:1,2', 'mixed4d'])
    assert names == ['mixed4c', 'mixed4d'] and channels == {'mixed4c': [1, 2], 'mixed4d': None}
    for specs in (['mixed4c:-1'], ['mixed4c:'], ['mixed4c', 'mixed4c']):
        try:
            dream.parse_layer_specs(specs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid layers: {specs}')
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        bad = root / 'bad.zip'
        bad.write_bytes(b'not the upstream archive')
        try:
            download_model.install_archive(bad, root / 'rejected')
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted corrupt model')
        assert not (root / 'rejected').exists()
        image = np.random.default_rng(42).uniform(0, 255, (129, 129, 3)).astype('float32')
        input_path = root / 'sources' / 'input.jpg'
        input_path.parent.mkdir()
        dream.save_image(image, input_path)
        with Image.open(input_path) as saved:
            assert JpegImagePlugin.get_sampling(saved) == 0
            assert saved.quantization[0][0] == 2  # JPEG quality 95 luminance table.
        sidecar = root / 'result.json'
        sidecar.write_text('{}')
        assert dream.get_unique_path(root / 'result.jpg').name == 'result_1.jpg'
        # Force wrapper child failure: no completed or partial pair may be published.
        settings = argparse.Namespace(source_root=input_path.parent, output_root=root / 'failed',
            dream_script=Path(dream.__file__), model_dir=root, layers=['mixed4c:1,2'],
            steps=1, octaves=0, step_size=0.75, dream_max_size=0, jpeg_quality=95,
            min_octave_size=64, loss_mode='mean', tile_size=0, input_long_side=0, skip_existing=False)
        command = pairs.build_command(settings, input_path, root / 'target.jpg')
        assert '--channels' not in command and 'mixed4c:1,2' in command and '--tile-size' not in command
        with patch.object(pairs.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, command)):
            try:
                pairs.process_image(input_path, settings)
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError('Child failure was ignored')
        assert not settings.output_root.exists()
        if args.model_dir:
            model_dir = args.model_dir.resolve()
            download_model.verify_model(model_dir)
            dream.selected_layer_names, dream.selected_layer_channels = dream.parse_layer_specs(['mixed4c:1,2', 'mixed4d:3'])
            dream.deepdream_model = dream.get_inception_v1_model(dream.selected_layer_names, model_dir)
            dream.inspect_and_validate_layers(dream.deepdream_model, dream.selected_layer_names, dream.selected_layer_channels)
            output = dream.run_deep_dream(image, steps=1, octaves=2, tile_size=64)
            assert output.shape == image.shape and np.isfinite(output).all()
            assert np.max(np.abs(output-image)) > 0.01
            try:
                dream.run_deep_dream(image[:32, :32], steps=1, octaves=3)
            except ValueError:
                pass
            else:
                raise AssertionError('Tiny image was accepted')
            # Two distinct source extensions must survive parallel naming and resume.
            dream.save_image(image, input_path.with_suffix('.png'))
            cli = [sys.executable, str(Path(pairs.__file__)), '--source-root', str(input_path.parent),
                '--output-root', str(root / 'pairs'), '--model-dir', str(model_dir),
                '--layers', 'mixed4c:1,2', '--steps', '1', '--octaves', '0', '--workers', '3']
            subprocess.run(cli, check=True)
            targets = sorted((root / 'pairs' / 'targets').glob('*.jpg'))
            assert len(targets) == 2
            before = [p.stat().st_mtime_ns for p in targets]
            subprocess.run(cli + ['--skip-existing'], check=True)
            assert [p.stat().st_mtime_ns for p in targets] == before
            for record in (root / 'pairs' / 'metadata').glob('*.json'):
                data = json.loads(record.read_text())
                with Image.open(data['input']) as src, Image.open(data['target']) as dst:
                    assert src.size == dst.size
                    assert JpegImagePlugin.get_sampling(dst) == 0
            tiny = root / 'tiny.png'
            dream.save_image(image[:32, :32], tiny)
            result = subprocess.run([sys.executable, str(Path(dream.__file__)), '--input', str(tiny),
                '--model-dir', str(model_dir), '--steps', '1', '--octaves', '0'])
            assert result.returncode == 1
    print('PASS: smoke checks' + (' including real model inference and parallel CLI' if args.model_dir else ' (model inference not requested)'))


if __name__ == '__main__':
    main()
