# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepDream Generator contributors.
# Modified for publication: verified model bootstrap, JPEG settings, octave guards,
# CLI validation, safe output naming and failure exit status.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import numpy as np
import argparse
import json
import math
from pathlib import Path
from PIL import Image, ImageOps
from download_model import DEFAULT_MODEL_DIR, verify_model


def load_image(image_path, max_dim=None):
    with Image.open(image_path) as source:
        img = ImageOps.exif_transpose(source).convert('RGB')
    if max_dim and max_dim > 0:
        img.thumbnail((max_dim, max_dim))
    # Для InceptionV1 оставляем пиксели в диапазоне [0, 255]
    return np.array(img, dtype=np.float32)


def save_image(img_array, file_name, quality=95):
    img = Image.fromarray(np.clip(img_array, 0, 255).astype('uint8'))
    options = {"quality": quality, "subsampling": 0, "optimize": True} if Path(file_name).suffix.lower() in {".jpg", ".jpeg"} else {}
    img.save(file_name, **options)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_unique_path(path):
    """
    Возвращает свободный путь без перезаписи существующих файлов.

    Если result/photo.jpg уже существует, вернёт:
    result/photo_1.jpg, потом result/photo_2.jpg и т.д.
    """
    path = Path(path)
    if not path.exists() and not path.with_suffix(".json").exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1

    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists() and not candidate.with_suffix(".json").exists():
            return candidate
        i += 1


def collect_input_images(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"Файл не похож на поддерживаемое изображение: {input_path}")
        return [input_path]

    if input_path.is_dir():
        images = sorted([p for p in input_path.iterdir() if is_image_file(p)])
        if not images:
            raise ValueError(
                f"В папке {input_path} не найдено изображений. "
                f"Поддерживаемые расширения: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )
        return images

    raise ValueError(f"Путь не найден: {input_path}")


def resolve_output_path(input_image_path, input_root_path, output_arg):
    """
    Для одного файла:
      --input image.jpg --output out.jpg -> out.jpg
      --input image.jpg без --output -> image/result/image.jpg

    Для папки:
      --input folder -> folder/result/original_name.jpg
      --input folder --output custom_result -> custom_result/original_name.jpg

    Существующие файлы не перезаписываются: добавляется _1, _2, _3...
    """
    input_image_path = Path(input_image_path)
    input_root_path = Path(input_root_path)

    if input_root_path.is_dir():
        result_dir = Path(output_arg) if output_arg else input_root_path / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        return get_unique_path(result_dir / input_image_path.name)

    # input — одиночный файл
    if output_arg:
        output_path = Path(output_arg)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return get_unique_path(output_path)

    result_dir = input_image_path.parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    return get_unique_path(result_dir / input_image_path.name)


def parse_layer_specs(layer_specs):
    """
    Поддерживает два формата:

    --layers mixed4c mixed4d
        Использовать все каналы в слоях mixed4c и mixed4d.

    --layers mixed4c:123,124 mixed4d:88,91
        Использовать только указанные каналы в каждом слое.

    Можно смешивать:

    --layers mixed4c:123,124 mixed4d mixed5a:10
        mixed4c -> каналы 123,124
        mixed4d -> все каналы
        mixed5a -> канал 10
    """
    layer_names = []
    layer_channel_map = {}

    for spec in layer_specs:
        spec = spec.strip()
        if not spec:
            continue

        if ':' in spec:
            layer_name, channel_part = spec.split(':', 1)
            layer_name = layer_name.strip()
            channel_part = channel_part.strip()

            if not layer_name:
                raise ValueError(f"Некорректный layer spec: {spec!r}. Нет имени слоя до ':'")
            if not channel_part:
                raise ValueError(f"Некорректный layer spec: {spec!r}. Нет каналов после ':'")

            channels = []
            for raw_channel in channel_part.split(','):
                raw_channel = raw_channel.strip()
                if raw_channel == '':
                    continue
                try:
                    channel = int(raw_channel)
                except ValueError as exc:
                    raise ValueError(
                        f"Некорректный канал {raw_channel!r} в {spec!r}. Каналы должны быть целыми числами."
                    ) from exc
                if channel < 0:
                    raise ValueError(f"Некорректный канал {channel} в {spec!r}. Каналы нумеруются с 0.")
                channels.append(channel)

            if not channels:
                raise ValueError(f"Некорректный layer spec: {spec!r}. Список каналов пуст.")
        else:
            layer_name = spec
            channels = None  # None = все каналы слоя

        if layer_name in layer_channel_map:
            raise ValueError(
                f"Слой {layer_name!r} указан несколько раз. "
                f"Объедини каналы в одном spec, например {layer_name}:1,2,3"
            )

        layer_names.append(layer_name)
        layer_channel_map[layer_name] = channels

    if not layer_names:
        raise ValueError("Не указано ни одного слоя.")

    return layer_names, layer_channel_map


def get_inception_v1_model(layer_names, model_dir=DEFAULT_MODEL_DIR):
    pb_path = verify_model(Path(model_dir))

    # 2. Загружаем старый граф
    with tf.io.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Обрезаем граф до нужных слоёв, чтобы TF не проверял ненужные глубокие ветки
    graph_def = tf.compat.v1.graph_util.extract_sub_graph(graph_def, layer_names)

    # 3. Обертка для совместимости с TensorFlow 2.x
    def import_graph(img):
        # InceptionV1 ожидает на входе картинку минус среднее значение 117.0
        img_centered = img - tf.constant([117.0, 117.0, 117.0], dtype=tf.float32)
        img_batch = tf.expand_dims(img_centered, 0)  # Добавляем батч

        returns = tf.import_graph_def(
            graph_def,
            input_map={'input:0': img_batch},
            return_elements=[name + ':0' for name in layer_names],
            name=''
        )
        return returns

    wrapped = tf.compat.v1.wrap_function(
        import_graph,
        [tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32)]
    )
    return wrapped


# Глобальные настройки для tf.function
# deepdream_model(img) возвращает активации в том же порядке, что selected_layer_names.
deepdream_model = None
selected_layer_names = None
selected_layer_channels = None
loss_mode = "mean"


def calc_loss(img):
    layer_activations = deepdream_model(img)
    if not isinstance(layer_activations, (list, tuple)):
        layer_activations = [layer_activations]

    losses = []
    for layer_name, act in zip(selected_layer_names, layer_activations):
        channels = selected_layer_channels.get(layer_name)

        if channels is not None:
            # act shape: [1, H, W, C]
            # axis=-1 = канальная размерность C
            act = tf.gather(act, channels, axis=-1)

        if loss_mode == "mean":
            layer_loss = tf.reduce_mean(act)
        elif loss_mode == "square":
            layer_loss = tf.reduce_mean(tf.square(act))
        elif loss_mode == "relu":
            layer_loss = tf.reduce_mean(tf.nn.relu(act))
        else:
            # На случай некорректного значения, хотя argparse это уже проверяет.
            layer_loss = tf.reduce_mean(act)

        losses.append(layer_loss)

    return tf.reduce_sum(losses)


@tf.function(reduce_retracing=True)
def compute_tile_gradient(tile):
    with tf.GradientTape() as tape:
        tape.watch(tile)
        loss = calc_loss(tile)
    return loss, tape.gradient(loss, tile)


def tile_bounds(length, tile_size):
    # Merge a short final strip into its neighbor to avoid tiny graph inputs.
    bounds = list(range(0, length, tile_size))
    if len(bounds) > 1 and length - bounds[-1] < 64:
        bounds.pop()
    return list(zip(bounds, bounds[1:] + [length]))


def get_tiled_gradients(img, tile_size):
    h, w = img.shape[0], img.shape[1]
    grad_rows = []
    total_loss = 0.0

    for y, y_end in tile_bounds(h, tile_size):
        grad_cols = []
        for x, x_end in tile_bounds(w, tile_size):
            tile = img[y:y_end, x:x_end, :]
            loss, grad = compute_tile_gradient(tile)
            total_loss += loss
            grad_cols.append(grad)
        grad_rows.append(tf.concat(grad_cols, axis=1))

    gradients = tf.concat(grad_rows, axis=0)
    return total_loss, gradients


def deepdream_step_tiled(img, steps, step_size, tile_size, max_jitter):
    for _ in range(steps):
        shift_y = tf.random.uniform([], -max_jitter, max_jitter, dtype=tf.int32)
        shift_x = tf.random.uniform([], -max_jitter, max_jitter, dtype=tf.int32)
        img_rolled = tf.roll(img, shift=[shift_y, shift_x], axis=[0, 1])

        loss, gradients = get_tiled_gradients(img_rolled, tile_size)

        gradients /= tf.math.reduce_std(gradients) + 1e-8
        gradients = tf.roll(gradients, shift=[-shift_y, -shift_x], axis=[0, 1])

        img = img + gradients * step_size
        # В InceptionV1 картинка не в [-1, 1], а в [0, 255]
        img = tf.clip_by_value(img, 0.0, 255.0)

    return loss, img


def run_deep_dream(img, steps=50, step_size=1.5, octaves=4, octave_scale=1.4, tile_size=512, min_octave_size=64):
    img = tf.convert_to_tensor(img)
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)

    completed = 0
    for n in range(-octaves, 1):
        new_shape = tf.cast(float_base_shape * (octave_scale ** n), tf.int32)
        if int(tf.reduce_min(new_shape)) < min_octave_size:
            print(f"[SKIP] Octave {n}: below {min_octave_size}px")
            continue
        completed += 1
        img = tf.image.resize(img, new_shape)
        current_jitter = int(max(32, new_shape[0].numpy() // 10))

        loss, img = deepdream_step_tiled(img, steps, step_size, tile_size, max_jitter=current_jitter)
        print(f"Octave {n + octaves} (Size: {new_shape[0]}x{new_shape[1]}) - Loss: {loss:.2f}")

    if not completed:
        raise ValueError("No valid octaves: use a larger input or smaller min-octave-size (at least 64).")
    img = tf.image.resize(img, base_shape)
    return img.numpy()


def inspect_and_validate_layers(model, layer_names, layer_channel_map):
    """Печатает shape слоёв и проверяет, что указанные каналы существуют."""
    test_img = tf.zeros([224, 224, 3], dtype=tf.float32)
    acts = model(test_img)
    if not isinstance(acts, (list, tuple)):
        acts = [acts]

    print("[*] Выбранные слои и каналы:")
    for layer_name, act in zip(layer_names, acts):
        shape = act.shape.as_list()
        channel_count = shape[-1]
        channels = layer_channel_map.get(layer_name)

        if channel_count is None:
            print(f"    {layer_name}: shape={act.shape}, каналы проверить статически не удалось")
            continue

        if channels is None:
            print(f"    {layer_name}: shape={act.shape}, channels=ALL (0..{channel_count - 1})")
            continue

        invalid = [c for c in channels if c >= channel_count]
        if invalid:
            raise ValueError(
                f"В слое {layer_name!r} всего {channel_count} каналов: 0..{channel_count - 1}. "
                f"Некорректные каналы: {invalid}"
            )

        print(f"    {layer_name}: shape={act.shape}, channels={channels}")


def build_metadata(args, layer_names, layer_channel_map):
    return {
        **vars(args),
        "parsed_layers": layer_names,
        "parsed_layer_channels": layer_channel_map,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Файл изображения или папка с изображениями")
    parser.add_argument("--output", default=None, help="Для файла: путь результата. Для папки: папка результата. По умолчанию: ./result рядом с input")
    parser.add_argument(
        "--layers",
        nargs='+',
        default=['mixed4c'],
        help=(
            "Слои или слои с каналами. Примеры: "
            "--layers mixed4c mixed4d или "
            "--layers mixed4c:123,124 mixed4d:88,91"
        )
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octaves", type=int, default=5)
    # Размер шага сильно больше, так как пиксели теперь от 0 до 255
    parser.add_argument("--step-size", type=float, default=1.5)
    parser.add_argument("--max-size", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--loss-mode",
        choices=["mean", "square", "relu"],
        default="mean",
        help=(
            "Как считать loss по выбранным активациям. "
            "mean = как раньше; square = сильнее разгоняет активные каналы; relu = только положительные активации."
        )
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-octave-size", type=int, default=64)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()
    if (args.steps < 1 or not 0 <= args.octaves <= 100 or args.tile_size < 64
            or args.min_octave_size < 64 or args.max_size < 0
            or not math.isfinite(args.step_size) or args.step_size <= 0
            or not 1 <= args.jpeg_quality <= 100):
        parser.error("steps >= 1; octaves 0..100; tile/min-octave-size >= 64; "
                     "max-size >= 0; finite step-size > 0; jpeg-quality 1..100 required")

    layer_names, layer_channel_map = parse_layer_specs(args.layers)

    print("[*] Подготовка InceptionV1...")
    deepdream_model = get_inception_v1_model(layer_names, args.model_dir)
    selected_layer_names = layer_names
    selected_layer_channels = layer_channel_map
    loss_mode = args.loss_mode

    inspect_and_validate_layers(deepdream_model, layer_names, layer_channel_map)

    input_root = Path(args.input)
    input_images = collect_input_images(input_root)

    if input_root.is_dir():
        result_dir = Path(args.output) if args.output else input_root / "result"
        print(f"[*] Batch-режим: найдено изображений: {len(input_images)}")
        print(f"[*] Папка результата: {result_dir}")
    else:
        print("[*] Single-image режим")

    failures = 0
    for index, input_image in enumerate(input_images, start=1):
        output_path = resolve_output_path(input_image, input_root, args.output)

        print(f"\n[*] [{index}/{len(input_images)}] Обработка изображения: {input_image}")
        print(f"[*] Выходной файл: {output_path}")

        try:
            original_img = load_image(input_image, max_dim=args.max_size)
            print(f"[*] Размер исходника: {original_img.shape}")

            dream_img = run_deep_dream(
                original_img,
                steps=args.steps,
                step_size=args.step_size,
                octaves=args.octaves,
                tile_size=args.tile_size,
                min_octave_size=args.min_octave_size
            )

            save_image(dream_img, output_path, quality=args.jpeg_quality)

            meta_path = output_path.with_suffix(".json")
            metadata = build_metadata(args, layer_names, layer_channel_map)
            metadata.update({
                "input_image": str(input_image),
                "output_image": str(output_path),
                "batch_index": index,
                "batch_total": len(input_images),
            })

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            print(f"[*] Готово: {output_path}")

        except Exception as exc:
            failures += 1
            print(f"[!] Ошибка при обработке {input_image}: {exc}")
            continue

    print("\n[*] Обработка завершена.")

    raise SystemExit(1 if failures else 0)
