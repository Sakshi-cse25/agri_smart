# ============================================================
# src/preprocessing/preprocess_images.py
#
# PURPOSE: Prepares the PlantVillage leaf disease image dataset
# for CNN training. It reads images from class-named folders,
# resizes them, normalizes pixel values, and creates TensorFlow
# data pipelines (tf.data.Dataset) — the modern, memory-efficient
# way to feed images to a Keras model.
#
# EXPECTED FOLDER STRUCTURE (PlantVillage):
#   data/raw/PlantVillage/
#     ├── Tomato_Early_blight/
#     │   ├── image001.jpg
#     │   └── ...
#     ├── Tomato_Healthy/
#     └── Potato_Late_blight/
#         └── ...
#
# EVS IMPACT:
# Early and accurate disease detection means farmers spray
# fungicides/pesticides ONLY when needed (precision agriculture),
# reducing chemical load on soil and nearby water bodies by up to 40%.
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Configuration ────────────────────────────────────────────
IMG_DIR       = os.path.join("data", "raw", "PlantVillage")
PROCESSED_DIR = os.path.join("data", "processed")
IMG_SIZE      = (128, 128)   # Width × Height fed into CNN
BATCH_SIZE    = 32           # Images processed per gradient step
AUTOTUNE      = tf.data.AUTOTUNE
os.makedirs(PROCESSED_DIR, exist_ok=True)


def get_class_names(root_dir: str) -> list[str]:
    """
    Scans the root directory and returns sorted folder names.
    Each folder = one disease class (or 'Healthy').
    """
    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    print(f"[Classes] Found {len(classes)} disease categories:")
    for i, c in enumerate(classes):
        print(f"  {i:>3}: {c}")
    return classes


def build_tf_dataset(root_dir: str, class_names: list[str],
                     validation_split: float = 0.20,
                     seed: int = 42):
    """
    Uses tf.keras.utils.image_dataset_from_directory — the cleanest
    way to create train/val datasets directly from folders.

    HOW IT WORKS:
    1. Recursively finds all .jpg/.png files
    2. Assigns the integer label = index of its parent folder in class_names
    3. Splits into training (80%) and validation (20%) sets
    4. Returns lazy tf.data.Dataset objects (images loaded on-demand)
    """
    common_args = dict(
        directory   = root_dir,
        labels      = "inferred",       # Labels come from folder names
        label_mode  = "categorical",    # One-hot vectors (for softmax output)
        class_names = class_names,
        image_size  = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        seed        = seed,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split = validation_split,
        subset           = "training",
        **common_args
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split = validation_split,
        subset           = "validation",
        **common_args
    )
    return train_ds, val_ds


def preprocess_pipeline(dataset: tf.data.Dataset,
                         augment: bool = False) -> tf.data.Dataset:
    """
    Applies:
    (a) NORMALIZATION: pixel values [0,255] → [0,1]
        Neural networks train faster when inputs are small floats.
    (b) DATA AUGMENTATION (training only):
        Randomly flip & rotate images so the model generalizes to
        leaves photographed from different angles — critical for
        real-world farm use where lighting and angle vary wildly.
    (c) PREFETCH: loads the next batch while GPU trains on current one.
    """
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    augmentation_layers = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.10),
    ])

    def apply_norm(images, labels):
        return normalization_layer(images), labels

    def apply_augment(images, labels):
        return augmentation_layers(images, training=True), labels

    ds = dataset.map(apply_norm, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(apply_augment, num_parallel_calls=AUTOTUNE)

    # cache() stores dataset in RAM after first epoch (speeds up training)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)


def save_class_map(class_names: list[str]):
    """
    Saves the index→class_name mapping as JSON.
    The Streamlit app uses this to convert the model's numeric
    prediction back into a human-readable disease name.
    """
    class_map = {str(i): name for i, name in enumerate(class_names)}
    out_path = os.path.join(PROCESSED_DIR, "disease_class_map.json")
    with open(out_path, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"\n[Saved] Class map → {out_path}")
    return class_map


if __name__ == "__main__":
    class_names    = get_class_names(IMG_DIR)
    train_ds, val_ds = build_tf_dataset(IMG_DIR, class_names)

    train_ds = preprocess_pipeline(train_ds, augment=True)
    val_ds   = preprocess_pipeline(val_ds,   augment=False)

    class_map = save_class_map(class_names)

    # Quick sanity check — print one batch shape
    for images, labels in train_ds.take(1):
        print(f"\n[Sanity] Batch shape: images={images.shape}, labels={labels.shape}")

    print("\n✅  Image preprocessing pipeline ready!")
    print(f"   → Training dataset   : {train_ds}")
    print(f"   → Validation dataset : {val_ds}")
