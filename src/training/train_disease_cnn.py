# ============================================================
# src/training/train_disease_cnn.py
#
# PURPOSE: Defines, compiles, and trains a CNN for plant leaf
# disease classification on the PlantVillage dataset.
#
# MODEL ARCHITECTURE: Transfer Learning with MobileNetV2
# ──────────────────────────────────────────────────────────
# Instead of training a CNN from scratch (which needs millions
# of images), we use MobileNetV2 — a lightweight CNN pre-trained
# on ImageNet (1.4M images, 1000 classes).
#
# We "freeze" its feature-extraction layers and only train
# the final classification "head" on our plant images.
# This is called TRANSFER LEARNING. It:
#   • Reaches high accuracy with far fewer plant images
#   • Trains in minutes instead of hours
#   • Requires less computing power (runs on a laptop GPU)
#
# WHY MobileNetV2?
# It was designed for mobile/edge devices. A farmer with a
# basic Android phone can eventually run this model locally —
# no internet needed in remote fields.
#
# EVS IMPACT NOTE:
# Detecting Tomato Late Blight early (before 30% leaf damage)
# allows targeted fungicide spray. Studies show this cuts
# chemical use by 35–45% vs. calendar-based spraying schedules.
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Import our preprocessing utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing.preprocess_images import (
    get_class_names, build_tf_dataset, preprocess_pipeline, PROCESSED_DIR
)

# ── Paths & Config ───────────────────────────────────────────
IMG_DIR    = os.path.join("data", "raw", "PlantVillage")
MODELS_DIR = os.path.join("models", "saved")
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

IMG_SIZE   = (128, 128)
EPOCHS     = 25          # Max epochs (EarlyStopping will halt earlier if needed)
FINE_TUNE_EPOCHS = 10    # Additional epochs after unfreezing top layers


# ─────────────────────────────────────────────────────────────
# 1. BUILD THE MODEL
# ─────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """
    Architecture Overview:
    ┌─────────────────────────────┐
    │  Input: 128×128×3 image     │
    ├─────────────────────────────┤
    │  MobileNetV2 (frozen)       │  ← Pre-trained feature extractor
    │  → 4×4×1280 feature map     │
    ├─────────────────────────────┤
    │  GlobalAveragePooling2D     │  ← Collapses spatial dims to 1280-d vector
    ├─────────────────────────────┤
    │  Dropout(0.3)               │  ← Prevents over-reliance on any one feature
    ├─────────────────────────────┤
    │  Dense(256, relu)           │  ← Learns disease-specific combinations
    ├─────────────────────────────┤
    │  Dropout(0.3)               │
    ├─────────────────────────────┤
    │  Dense(num_classes, softmax)│  ← Probability for each disease class
    └─────────────────────────────┘
    """
    # Load MobileNetV2 without its top classification layer
    # weights='imagenet' downloads the pre-trained weights automatically
    base_model = tf.keras.applications.MobileNetV2(
        input_shape = (*IMG_SIZE, 3),
        include_top = False,
        weights     = "imagenet"
    )

    # PHASE 1: Freeze the base — only train our custom head
    base_model.trainable = False
    print(f"[Model] MobileNetV2 base: {len(base_model.layers)} layers (frozen)")

    # Build the full model using Functional API for clarity
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="leaf_image")

    # Pass through frozen base (batch_norm layers still run in inference mode)
    x = base_model(inputs, training=False)

    # Global Average Pooling: takes the mean of each feature map
    # More robust than Flatten — helps generalize to different image sizes
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.30, name="dropout_1")(x)

    # Dense classification head
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dropout(0.30, name="dropout_2")(x)

    # Output: softmax gives a probability distribution over all disease classes
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="AgriSmart_DiseaseDetector")

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "categorical_crossentropy",  # Multi-class classification
        metrics   = ["accuracy",
                     tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )

    print(model.summary())
    return model, base_model


# ─────────────────────────────────────────────────────────────
# 2. TRAINING CALLBACKS
# ─────────────────────────────────────────────────────────────
def get_callbacks(model_path: str) -> list:
    """
    Callbacks run automatically at the end of each epoch:

    ModelCheckpoint: Saves the model only when validation accuracy improves.
    EarlyStopping:   Stops training if val_accuracy doesn't improve for 5
                     consecutive epochs — saves time and prevents overfitting.
    ReduceLROnPlateau: If val_loss stalls for 3 epochs, halves the learning rate.
                       This helps the model "fine-tune" itself when stuck.
    """
    return [
        callbacks.ModelCheckpoint(
            filepath        = model_path,
            monitor         = "val_accuracy",
            save_best_only  = True,
            verbose         = 1
        ),
        callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1
        ),
        callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-6,
            verbose  = 1
        ),
    ]


# ─────────────────────────────────────────────────────────────
# 3. FINE-TUNING (PHASE 2)
# ─────────────────────────────────────────────────────────────
def fine_tune(model, base_model, train_ds, val_ds, model_path: str):
    """
    After Phase 1 training, we unfreeze the TOP layers of MobileNetV2
    and train them with a very low learning rate.

    WHY FINE-TUNE?
    The top layers of the base model learned general features (edges, textures).
    Unfreezing them lets the model adapt those features specifically to PLANT
    LEAF textures — more relevant than ImageNet (which has cars, dogs, etc.).
    """
    # Unfreeze the top 30 layers of the base model
    base_model.trainable = True
    fine_tune_at         = len(base_model.layers) - 30

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n[Fine-Tune] Unfrozen layers in base: {trainable}")

    # Use a much smaller learning rate to avoid destroying pre-trained weights
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy",
                     tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )

    history_ft = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = FINE_TUNE_EPOCHS,
        callbacks       = get_callbacks(model_path)
    )
    return history_ft


# ─────────────────────────────────────────────────────────────
# 4. PLOT TRAINING CURVES
# ─────────────────────────────────────────────────────────────
def plot_history(h1, h2=None):
    """
    Plots accuracy and loss across epochs for both training phases.
    A healthy training curve shows: train ≈ val accuracy (no overfitting),
    and both curves going UP steadily.
    """
    acc      = h1.history["accuracy"]
    val_acc  = h1.history["val_accuracy"]
    loss     = h1.history["loss"]
    val_loss = h1.history["val_loss"]

    if h2:
        acc     += h2.history["accuracy"]
        val_acc += h2.history["val_accuracy"]
        loss    += h2.history["loss"]
        val_loss += h2.history["val_loss"]

    epochs_range = range(len(acc))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, acc,     label="Train Accuracy",  color="#2d6a4f")
    ax1.plot(epochs_range, val_acc, label="Val Accuracy",    color="#95d5b2", linestyle="--")
    if h2:
        ax1.axvline(x=len(h1.history["accuracy"]), color="gray",
                    linestyle=":", label="Fine-tune start")
    ax1.set_title("Model Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs_range, loss,     label="Train Loss",  color="#e76f51")
    ax2.plot(epochs_range, val_loss, label="Val Loss",    color="#f4a261", linestyle="--")
    ax2.set_title("Model Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("CNN Training History – Plant Disease Detector", fontsize=13, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "cnn_training_history.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training history saved → {plot_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load class names and datasets
    class_names      = get_class_names(IMG_DIR)
    num_classes      = len(class_names)
    train_ds, val_ds = build_tf_dataset(IMG_DIR, class_names)
    train_ds         = preprocess_pipeline(train_ds, augment=True)
    val_ds           = preprocess_pipeline(val_ds,   augment=False)

    # Build model
    model_path = os.path.join(MODELS_DIR, "cnn_disease_detector.keras")
    model, base_model = build_model(num_classes)

    # Phase 1: Train classification head only
    print("\n[Phase 1] Training classification head...")
    h1 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = get_callbacks(model_path)
    )

    # Phase 2: Fine-tune top base layers
    print("\n[Phase 2] Fine-tuning top MobileNetV2 layers...")
    h2 = fine_tune(model, base_model, train_ds, val_ds, model_path)

    # Plot and wrap up
    plot_history(h1, h2)

    # Final evaluation
    print("\n[Eval] Final validation metrics:")
    model.evaluate(val_ds)

    print(f"\n✅  CNN training complete! Model saved → {model_path}")
