# ============================================================
# src/training/train_crop_model.py
#
# PURPOSE: Trains a Random Forest Classifier on the preprocessed
# Crop Recommendation dataset and saves the trained model.
#
# WHY RANDOM FOREST?
# ─────────────────
# A Random Forest is an "ensemble" method: it builds many decision
# trees on random subsets of the data, then takes a majority vote.
# This gives it:
#   • High accuracy on tabular data (soil, climate features)
#   • Built-in feature importance scores (tells us WHICH soil
#     parameter matters most for each crop)
#   • Robustness to outliers and noisy sensor readings
#     (important for real-world farm conditions)
#
# EVS IMPACT NOTE:
# By knowing WHICH crop suits a field's exact NPK and pH,
# farmers avoid growing mismatched crops that drain soil nutrients
# faster, causing land degradation and increased fertilizer use.
# ============================================================

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ── Paths ────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models", "saved")
REPORTS_DIR   = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. LOAD PRE-PROCESSED DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    le      = joblib.load(os.path.join(PROCESSED_DIR, "label_encoder.pkl"))
    scaler  = joblib.load(os.path.join(PROCESSED_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(PROCESSED_DIR, "feature_cols.pkl"))

    print(f"[Data] Train={X_train.shape[0]} | Val={X_val.shape[0]} | Test={X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, feature_cols


# ─────────────────────────────────────────────────────────────
# 2. BUILD & TRAIN THE RANDOM FOREST
# ─────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_val, y_val):
    """
    KEY HYPERPARAMETERS EXPLAINED:
    ─────────────────────────────
    n_estimators=200  → Build 200 decision trees. More trees = more stable
                        predictions (but slower training). 200 is a good balance.

    max_depth=None    → Let each tree grow until leaves are pure. Works well
                        when features (soil, climate) interact non-linearly.

    min_samples_leaf=2 → Each leaf must have ≥2 samples. Prevents overfitting
                         to a single noisy data point.

    class_weight='balanced' → Adjusts for any class imbalance in the dataset
                               (e.g., if 'rice' has 5× more samples than 'coffee').

    n_jobs=-1         → Use ALL available CPU cores for parallel tree building.
    """
    print("\n[Train] Fitting Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators    = 200,
        max_depth       = None,
        min_samples_leaf = 2,
        max_features    = "sqrt",    # Each split considers √(n_features) features
        class_weight    = "balanced",
        random_state    = 42,
        n_jobs          = -1
    )
    rf.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, rf.predict(X_val))
    print(f"[Train] Validation Accuracy: {val_acc * 100:.2f}%")
    return rf


# ─────────────────────────────────────────────────────────────
# 3. EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────
def evaluate(rf, X_test, y_test, le):
    y_pred  = rf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    classes = le.classes_

    print("\n" + "=" * 55)
    print(f"TEST ACCURACY: {acc * 100:.2f}%")
    print("=" * 55)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # ── Confusion Matrix ──────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=classes, yticklabels=classes, ax=ax
    )
    ax.set_title("Random Forest – Crop Recommendation Confusion Matrix",
                 fontsize=14, pad=15)
    ax.set_xlabel("Predicted Crop")
    ax.set_ylabel("Actual Crop")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "rf_confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"[Plot] Confusion matrix saved → {cm_path}")
    return acc


# ─────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────────────────────
def plot_feature_importance(rf, feature_cols):
    """
    Feature importance tells us WHICH soil/climate parameter
    the model relies on most. High importance of 'K' (Potassium),
    for instance, means choosing crops that don't over-extract K
    can preserve long-term soil health.
    """
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        [feature_cols[i] for i in indices],
        importances[indices],
        color=["#2d6a4f", "#40916c", "#52b788", "#74c69d",
               "#95d5b2", "#b7e4c7", "#d8f3dc"]
    )
    ax.set_title("Feature Importances – What Drives Crop Recommendation?",
                 fontsize=13)
    ax.set_ylabel("Importance Score")
    ax.set_xlabel("Soil / Climate Feature")
    plt.tight_layout()
    fi_path = os.path.join(REPORTS_DIR, "rf_feature_importance.png")
    plt.savefig(fi_path, dpi=120)
    plt.close()
    print(f"[Plot] Feature importance saved → {fi_path}")


# ─────────────────────────────────────────────────────────────
# 5. SAVE MODEL
# ─────────────────────────────────────────────────────────────
def save_model(rf):
    model_path = os.path.join(MODELS_DIR, "random_forest_crop.pkl")
    joblib.dump(rf, model_path)
    print(f"\n[Saved] Model → {model_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, feature_cols = load_data()
    rf   = train_random_forest(X_train, y_train, X_val, y_val)
    acc  = evaluate(rf, X_test, y_test, le)
    plot_feature_importance(rf, feature_cols)
    save_model(rf)

    print(f"\n✅  Random Forest training complete! Final test accuracy: {acc*100:.2f}%")
