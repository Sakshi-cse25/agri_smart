# ============================================================
# demo_without_training.py
#
# PURPOSE: A self-contained demo that generates SYNTHETIC data
# and trains a quick (small) version of both models so you can
# see the full pipeline working WITHOUT downloading the datasets.
#
# Run with: python demo_without_training.py
# Then:     streamlit run app/dashboard.py
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

print("=" * 55)
print("  Agri-Smart DEMO — Generating synthetic data & training")
print("=" * 55)

# ── Directory setup ───────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)

# ── 1. Generate Synthetic Crop Data ──────────────────────────
CROPS = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
    "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
    "banana", "mango", "grapes", "watermelon", "muskmelon",
    "apple", "orange", "papaya", "coconut", "cotton",
    "jute", "coffee"
]

# Define rough soil/climate "signatures" per crop
CROP_PROFILES = {
    "rice":        dict(N=(60,120),  P=(30,60),   K=(30,60),   temp=(20,30), humid=(70,90), ph=(5.5,7),   rain=(1200,2500)),
    "maize":       dict(N=(60,120),  P=(30,60),   K=(15,40),   temp=(18,28), humid=(55,75), ph=(5.5,7.5), rain=(500,1000)),
    "chickpea":    dict(N=(20,60),   P=(40,100),  K=(20,50),   temp=(15,25), humid=(30,60), ph=(6,8),     rain=(300,700)),
    "kidneybeans": dict(N=(15,40),   P=(60,100),  K=(15,40),   temp=(18,27), humid=(50,70), ph=(6,7.5),   rain=(400,700)),
    "pigeonpeas":  dict(N=(10,30),   P=(30,70),   K=(15,40),   temp=(20,30), humid=(50,70), ph=(6,7.5),   rain=(400,700)),
    "mothbeans":   dict(N=(10,25),   P=(30,60),   K=(15,35),   temp=(25,35), humid=(30,50), ph=(6,8),     rain=(200,500)),
    "mungbean":    dict(N=(10,30),   P=(30,60),   K=(15,40),   temp=(25,35), humid=(50,70), ph=(6,7.5),   rain=(300,600)),
    "blackgram":   dict(N=(20,50),   P=(40,70),   K=(15,35),   temp=(22,32), humid=(55,75), ph=(6,7.5),   rain=(400,700)),
    "lentil":      dict(N=(10,30),   P=(50,100),  K=(20,50),   temp=(15,25), humid=(40,65), ph=(6,7.5),   rain=(300,600)),
    "pomegranate": dict(N=(30,90),   P=(20,60),   K=(30,80),   temp=(20,35), humid=(40,65), ph=(6,7.5),   rain=(400,700)),
    "banana":      dict(N=(100,200), P=(70,140),  K=(150,250), temp=(20,30), humid=(70,90), ph=(5.5,7),   rain=(900,2000)),
    "mango":       dict(N=(15,40),   P=(10,30),   K=(15,40),   temp=(24,35), humid=(40,70), ph=(5.5,7.5), rain=(700,1500)),
    "grapes":      dict(N=(20,40),   P=(80,140),  K=(60,120),  temp=(15,30), humid=(55,75), ph=(6,7.5),   rain=(400,900)),
    "watermelon":  dict(N=(60,100),  P=(50,80),   K=(40,80),   temp=(24,34), humid=(60,80), ph=(6,7),     rain=(400,800)),
    "muskmelon":   dict(N=(60,100),  P=(50,80),   K=(40,80),   temp=(24,34), humid=(60,80), ph=(6,7),     rain=(400,800)),
    "apple":       dict(N=(20,50),   P=(100,150), K=(140,200), temp=(10,22), humid=(50,70), ph=(5.5,7),   rain=(700,1200)),
    "orange":      dict(N=(15,35),   P=(5,20),    K=(5,20),    temp=(22,32), humid=(60,80), ph=(6,7.5),   rain=(700,1500)),
    "papaya":      dict(N=(50,120),  P=(30,80),   K=(30,80),   temp=(22,33), humid=(60,80), ph=(6,7),     rain=(800,1800)),
    "coconut":     dict(N=(5,20),    P=(5,20),    K=(5,20),    temp=(25,35), humid=(80,100),ph=(5.5,7),   rain=(1200,2400)),
    "cotton":      dict(N=(100,160), P=(30,60),   K=(15,40),   temp=(24,35), humid=(50,70), ph=(6,7.5),   rain=(500,900)),
    "jute":        dict(N=(60,100),  P=(30,60),   K=(30,60),   temp=(25,35), humid=(70,90), ph=(6,7),     rain=(1200,2500)),
    "coffee":      dict(N=(100,140), P=(60,100),  K=(150,200), temp=(15,25), humid=(70,90), ph=(6,7),     rain=(1200,2000)),
}

N_SAMPLES = 100  # 100 samples per crop = 2200 total
rows = []
for crop, p in CROP_PROFILES.items():
    for _ in range(N_SAMPLES):
        rows.append({
            "N":           np.random.uniform(*p["N"]),
            "P":           np.random.uniform(*p["P"]),
            "K":           np.random.uniform(*p["K"]),
            "temperature": np.random.uniform(*p["temp"]),
            "humidity":    np.random.uniform(*p["humid"]),
            "ph":          np.random.uniform(*p["ph"]),
            "rainfall":    np.random.uniform(*p["rain"]),
            "label":       crop
        })

df = pd.DataFrame(rows)
df.to_csv("data/raw/Crop_recommendation.csv", index=False)
print(f"[Demo] Generated synthetic crop CSV: {len(df)} rows, {df['label'].nunique()} classes")

# ── 2. Train Quick Random Forest ─────────────────────────────
feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[feature_cols].values
le = LabelEncoder()
y = le.fit_transform(df["label"].values)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
acc = rf.score(X_test, y_test)
print(f"[Demo] Random Forest trained. Test accuracy: {acc*100:.1f}%")

# Save everything
joblib.dump(rf,           "models/saved/random_forest_crop.pkl")
joblib.dump(le,           "data/processed/label_encoder.pkl")
joblib.dump(scaler,       "data/processed/scaler.pkl")
joblib.dump(feature_cols, "data/processed/feature_cols.pkl")
print("[Demo] Crop model artifacts saved.")

# ── 3. Create a dummy disease class map (since we skip CNN training) ──
# In a real run, this is created by preprocess_images.py
DEMO_DISEASES = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
]
class_map = {str(i): name for i, name in enumerate(DEMO_DISEASES)}
with open("data/processed/disease_class_map.json", "w") as f:
    json.dump(class_map, f, indent=2)
print(f"[Demo] Disease class map saved ({len(class_map)} classes).")

print("\n" + "=" * 55)
print("  ✅  Demo setup complete!")
print("  → Run the dashboard with:")
print("     streamlit run app/dashboard.py")
print("=" * 55)
print("\nNOTE: The Disease tab requires a trained CNN model.")
print("For demo purposes, train the CNN with:")
print("  python src/training/train_disease_cnn.py")
print("(Requires PlantVillage dataset from Kaggle)")
