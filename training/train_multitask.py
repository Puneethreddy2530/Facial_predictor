"""
Multitask transfer-learning trainer for age, gender, race, and emotion.
- Uses UTKFace for age/gender/race
- Uses FER2013 for emotion
- Saves Keras model + metadata for backend inference
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.AUTOTUNE
RACE_LABELS = ["White", "Black", "Asian", "Indian", "Others"]
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def _find_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return None


def load_utkface(utk_dir: Path, limit: int = None) -> List[Dict]:
    files = list(utk_dir.rglob("*.jpg")) + list(utk_dir.rglob("*.png"))
    random.shuffle(files)
    if limit:
        files = files[:limit]

    samples = []
    for f in files:
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
        except ValueError:
            continue
        if age < 0 or age > 116 or gender not in (0, 1) or race not in range(5):
            continue
        # Use neutral (index 6) for emotion when FER is absent
        samples.append({
            "source": "utk",
            "path": str(f),
            "age": float(age),
            "gender": int(gender),
            "race": int(race),
            "emotion": 6
        })
    return samples


def load_fer2013(fer_csv: Path, usage_filter: Tuple[str, ...] = ("Training", "PublicTest"), limit: int = None) -> List[Dict]:
    samples = []
    with fer_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if usage_filter and row.get("Usage") not in usage_filter:
                continue
            try:
                emotion = int(row["emotion"])
            except (KeyError, ValueError):
                continue
            pixel_str = row.get("pixels", "")
            if not pixel_str:
                continue
            arr = np.fromstring(pixel_str, dtype=np.uint8, sep=" ")
            if arr.size != 48 * 48:
                continue
            img = arr.reshape(48, 48)
            samples.append({
                "source": "fer",
                "image": img,
                "age": 0.0,
                "gender": 0,
                "race": 0,
                "emotion": emotion,
            })
            if limit and len(samples) >= limit:
                break
    return samples


def _augment_np(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1, :])
    if rng.random() < 0.3:
        img = np.clip(img * rng.uniform(0.85, 1.15), 0.0, 1.0)
    if rng.random() < 0.25:
        noise = rng.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
    return img


def _sample_generator(samples: List[Dict], img_size: Tuple[int, int], augment: bool):
    rng = np.random.default_rng(42)
    for s in samples:
        if s.get("source") == "utk":
            img = cv2.imread(s["path"])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            gray = s["image"]
            img = np.stack([gray, gray, gray], axis=-1)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        if augment:
            img = _augment_np(img, rng)

        labels = {
            "age": np.float32(s["age"]),
            "gender": np.int32(s["gender"]),
            "race": np.int32(s["race"]),
            "emotion": np.int32(s["emotion"]),
        }
        weights = {
            "age": np.float32(1.0 if s["source"] == "utk" else 0.0),
            "gender": np.float32(1.0 if s["source"] == "utk" else 0.0),
            "race": np.float32(1.0 if s["source"] == "utk" else 0.0),
            "emotion": np.float32(1.0 if s["source"] == "fer" else 0.3),
        }
        yield img, labels, weights


def build_dataset(samples: List[Dict], img_size: Tuple[int, int], batch_size: int, augment: bool):
    spec_img = tf.TensorSpec(shape=(img_size[1], img_size[0], 3), dtype=tf.float32)
    spec_labels = {
        "age": tf.TensorSpec(shape=(), dtype=tf.float32),
        "gender": tf.TensorSpec(shape=(), dtype=tf.int32),
        "race": tf.TensorSpec(shape=(), dtype=tf.int32),
        "emotion": tf.TensorSpec(shape=(), dtype=tf.int32),
    }
    spec_weights = {
        "age": tf.TensorSpec(shape=(), dtype=tf.float32),
        "gender": tf.TensorSpec(shape=(), dtype=tf.float32),
        "race": tf.TensorSpec(shape=(), dtype=tf.float32),
        "emotion": tf.TensorSpec(shape=(), dtype=tf.float32),
    }
    ds = tf.data.Dataset.from_generator(
        lambda: _sample_generator(samples, img_size, augment),
        output_signature=(spec_img, spec_labels, spec_weights),
    )
    if augment:
        ds = ds.shuffle(min(len(samples), 2000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_model(img_size: Tuple[int, int]):
    inputs = keras.Input(shape=(img_size[1], img_size[0], 3))
    base = keras.applications.EfficientNetB0(
        include_top=False,
        input_tensor=inputs,
        weights="imagenet",
        drop_connect_rate=0.2,
    )
    base.trainable = False

    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(0.25)(x)

    age_out = keras.layers.Dense(1, name="age")(x)
    gender_out = keras.layers.Dense(1, activation="sigmoid", name="gender")(x)
    race_out = keras.layers.Dense(len(RACE_LABELS), activation="softmax", name="race")(x)
    emotion_out = keras.layers.Dense(len(EMOTION_LABELS), activation="softmax", name="emotion")(x)

    model = keras.Model(inputs=inputs, outputs={
        "age": age_out,
        "gender": gender_out,
        "race": race_out,
        "emotion": emotion_out,
    })

    losses = {
        "age": keras.losses.MeanAbsoluteError(),
        "gender": keras.losses.BinaryCrossentropy(),
        "race": keras.losses.SparseCategoricalCrossentropy(),
        "emotion": keras.losses.SparseCategoricalCrossentropy(),
    }
    metrics = {
        "age": [keras.metrics.MeanAbsoluteError(name="mae")],
        "gender": [keras.metrics.BinaryAccuracy(name="acc")],
        "race": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
        "emotion": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
    }
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=losses,
        loss_weights={"age": 0.35, "gender": 1.0, "race": 1.0, "emotion": 1.0},
        metrics=metrics,
    )
    return model


def train(args):
    img_size = (args.image_size, args.image_size)

    utk_candidates = [
        Path(args.utkface_dir),
        Path("datasets/utkface_aligned_cropped/UTKFace"),
        Path("datasets/UTKFace"),
    ]
    fer_candidates = [
        Path(args.fer_csv),
        Path("datasets/fer2013/fer2013.csv"),
        Path("datasets/fer2013.csv"),
    ]

    utk_dir = _find_existing(utk_candidates)
    fer_csv = _find_existing(fer_candidates)

    if not utk_dir:
        raise SystemExit("UTKFace directory not found. Set --utkface-dir to the folder with UTKFace images.")
    if not fer_csv and not args.skip_fer:
        raise SystemExit("FER2013 CSV not found. Set --fer-csv to fer2013.csv from Kaggle or use --skip-fer to train without emotion data.")

    print(f"Using UTKFace from: {utk_dir}")
    if fer_csv:
        print(f"Using FER2013 from: {fer_csv}")
    else:
        print("Skipping FER2013 (emotion head will train on neutral-only placeholders)")

    utk_samples = load_utkface(utk_dir, limit=args.limit_utk)
    fer_samples = load_fer2013(fer_csv, limit=args.limit_fer) if fer_csv else []

    if not utk_samples:
        raise SystemExit("No UTKFace samples loaded. Check dataset path.")
    if not fer_samples and not args.skip_fer:
        raise SystemExit("No FER2013 samples loaded. Check dataset path or use --skip-fer.")

    random.seed(42)
    random.shuffle(utk_samples)
    random.shuffle(fer_samples)

    val_utk = max(1, int(len(utk_samples) * 0.15))
    val_fer = max(1, int(len(fer_samples) * 0.15)) if fer_samples else 0

    train_samples = utk_samples[val_utk:] + (fer_samples[val_fer:] if fer_samples else [])
    val_samples = utk_samples[:val_utk] + (fer_samples[:val_fer] if fer_samples else [])
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"Train samples: {len(train_samples)} (UTK {len(utk_samples) - val_utk} / FER {(len(fer_samples) - val_fer) if fer_samples else 0})")
    print(f"Val samples:   {len(val_samples)} (UTK {val_utk} / FER {val_fer})")

    train_ds = build_dataset(train_samples, img_size, args.batch_size, augment=True)
    val_ds = build_dataset(val_samples, img_size, args.batch_size, augment=False)

    model = build_model(img_size)

    ckpt_path = Path(args.output_dir) / "multitask_model.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    meta = {
        "image_size": args.image_size,
        "race_labels": RACE_LABELS,
        "emotion_labels": EMOTION_LABELS,
        "utk_samples": len(utk_samples),
        "fer_samples": len(fer_samples),
        "trained_heads": {
            "age": True,
            "gender": True,
            "race": True,
            "emotion": True,
        },
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "multitask_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(Path(args.output_dir) / "multitask_history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    print("\nSaved model to", ckpt_path)
    print("Saved metadata to", Path(args.output_dir) / "multitask_meta.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Train multitask facial predictor (UTKFace + FER2013)")
    parser.add_argument("--utkface-dir", default="datasets/UTKFace", help="Folder containing UTKFace images")
    parser.add_argument("--fer-csv", default="datasets/fer2013.csv", help="Path to fer2013.csv")
    parser.add_argument("--output-dir", default="models", help="Where to save the model and metadata")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size for training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--limit-utk", type=int, default=None, help="Optional cap on UTKFace samples for quick tests")
    parser.add_argument("--limit-fer", type=int, default=None, help="Optional cap on FER2013 samples for quick tests")
    parser.add_argument("--skip-fer", action="store_true", help="Train without FER2013 (emotion head gets neutral-only data)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except Exception as exc:
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)
