# ============================================================
# src/preprocessing/preprocess_crop.py
#
# PURPOSE: Loads the Crop Recommendation CSV dataset, performs
# exploratory analysis, cleans the data, and saves a processed
# version ready for model training.
#
# DATASET: Crop_recommendation.csv (from Kaggle)
# Features: N, P, K, temperature, humidity, ph, rainfall
# Target:   label (crop name, e.g., 'rice', 'maize', etc.)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Paths ────────────────────────────────────────────────────
RAW_CSV      = os.path.join("data", "raw", "Crop_recommendation.csv")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_and_inspect(path: str) -> pd.DataFrame:
    """
    Step 1 – Load data and print a quick health-check.
    This helps us understand: How many rows? Any missing values?
    What do the feature distributions look like?
    """
    df = pd.read_csv(path)

    print("=" * 55)
    print("AGRI-SMART | Crop Dataset Inspection")
    print("=" * 55)
    print(f"Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"\nFeature stats:\n{df.describe().round(2)}")
    print(f"\nUnique crops   : {df['label'].nunique()}")
    print(f"Crop classes   : {sorted(df['label'].unique())}")
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2 – Data Cleaning & Feature Engineering.

    WHY THIS MATTERS (EVS Note):
    Clean data means the model learns real patterns, not noise.
    A well-trained model reduces over-fertilization by giving
    precise NPK recommendations — directly cutting chemical runoff
    into rivers and groundwater.
    """
    # Drop duplicates (if any)
    before = len(df)
    df = df.drop_duplicates()
    print(f"\n[Clean] Dropped {before - len(df)} duplicate rows.")

    # Clip extreme outliers using the IQR method per numeric column
    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        print(f"[Clip] {col}: {clipped} outliers clipped to [{lower:.2f}, {upper:.2f}]")

    return df


def encode_and_scale(df: pd.DataFrame):
    """
    Step 3 – Encode labels + Scale features.

    WHY SCALING?
    Random Forests don't strictly need scaling, but it helps
    when we later compare feature importances on the same axis.
    It also future-proofs the pipeline for other algorithms.

    WHY LABEL ENCODING?
    The model works with numbers. 'rice' → 0, 'maize' → 1, etc.
    We save the encoder so the app can reverse this mapping and
    show the farmer a human-readable crop name.
    """
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    X = df[feature_cols].values
    y_raw = df["label"].values

    # Encode crop names → integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Scale features to zero mean, unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / Validation / Test split  (70 / 15 / 15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"\n[Split] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, feature_cols


def save_artifacts(X_train, X_val, X_test, y_train, y_val, y_test,
                   le, scaler, feature_cols):
    """
    Step 4 – Persist processed data and helper objects to disk.
    The Streamlit app and training scripts will load these later.
    """
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)

    joblib.dump(le,     os.path.join(PROCESSED_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(PROCESSED_DIR, "feature_cols.pkl"))

    print(f"\n[Saved] All processed files written to '{PROCESSED_DIR}/'")


if __name__ == "__main__":
    df = load_and_inspect(RAW_CSV)
    df = clean_and engineer(df)
    outputs = encode_and_scale(df)
    save_artifacts(*outputs)
    print("\n✅  Crop data preprocessing complete!")
