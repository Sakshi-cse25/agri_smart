# 🌿 Agri-Smart: AI-Powered Sustainable Agriculture Dashboard

> *An EVS Project integrating Machine Learning with Environmental Science*

---

## 📋 Project Overview

**Agri-Smart** is a two-in-one AI dashboard that helps farmers make
smarter, greener decisions:

| Feature | Technology | Purpose |
|---|---|---|
| 🌾 Crop Recommendation | Random Forest Classifier | Match crops to soil & climate |
| 🔬 Disease Diagnosis | CNN (MobileNetV2) | Identify leaf diseases from photos |
| 🌍 EVS Impact Notes | Curated Database | Explain eco-friendly alternatives |

---

## 📁 Project Structure

```
agri_smart/
│
├── app/
│   └── dashboard.py              ← Main Streamlit web application
│
├── data/
│   ├── raw/
│   │   ├── Crop_recommendation.csv    ← Download from Kaggle
│   │   └── PlantVillage/              ← Download from Kaggle
│   │       ├── Tomato_Early_blight/
│   │       ├── Tomato_Healthy/
│   │       └── Potato_Late_blight/
│   └── processed/                ← Auto-created during preprocessing
│       ├── X_train.npy
│       ├── label_encoder.pkl
│       ├── scaler.pkl
│       └── disease_class_map.json
│
├── models/
│   └── saved/
│       ├── random_forest_crop.pkl     ← Saved after training
│       └── cnn_disease_detector.keras ← Saved after training
│
├── src/
│   ├── preprocessing/
│   │   ├── preprocess_crop.py    ← Step 1a: Prepare CSV data
│   │   └── preprocess_images.py  ← Step 1b: Prepare image data
│   ├── training/
│   │   ├── train_crop_model.py   ← Step 2a: Train Random Forest
│   │   └── train_disease_cnn.py  ← Step 2b: Train CNN
│   └── utils/
│       └── evs_notes.py          ← EVS sustainability database
│
├── reports/
│   ├── rf_confusion_matrix.png   ← Auto-generated
│   ├── rf_feature_importance.png ← Auto-generated
│   └── cnn_training_history.png  ← Auto-generated
│
├── notebooks/
│   └── exploration.ipynb         ← Optional: EDA notebook
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Guide

### Step 0: Prerequisites

- Python 3.10 or newer (`python --version`)
- pip package manager
- 4GB RAM minimum (8GB recommended for CNN training)
- Optional: NVIDIA GPU with CUDA (speeds up CNN training 10×)

### Step 1: Clone / Download the Project

```bash
# If using git:
git clone https://github.com/yourusername/agri-smart.git
cd agri-smart

# Or simply extract the ZIP and cd into it
cd agri_smart
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create isolated environment so packages don't conflict with system Python
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs (with explanation):
- `streamlit` — The web dashboard framework
- `scikit-learn` — Random Forest model
- `tensorflow` — CNN model
- `pandas`, `numpy` — Data manipulation
- `plotly`, `seaborn` — Visualizations
- `Pillow` — Image handling
- `joblib` — Model file persistence

### Step 4: Download Datasets

**Crop Recommendation Dataset:**
1. Visit: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Download `Crop_recommendation.csv`
3. Place in: `data/raw/Crop_recommendation.csv`

**PlantVillage Disease Dataset:**
1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download the "color" version (~1.5GB)
3. Extract so folder structure is: `data/raw/PlantVillage/[ClassName]/[images...]`

---

## 🚀 Running the Project

### Step 5: Preprocess the Data

```bash
# Process the crop CSV (creates data/processed/ files)
python src/preprocessing/preprocess_crop.py

# Process the PlantVillage images (creates disease_class_map.json)
python src/preprocessing/preprocess_images.py
```

### Step 6: Train the Models

```bash
# Train Random Forest (~2-5 minutes on CPU)
python src/training/train_crop_model.py

# Train CNN (~20-60 minutes on CPU, ~5 min on GPU)
python src/training/train_disease_cnn.py
```

After training, check `reports/` folder for accuracy plots.

### Step 7: Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

Open your browser to: **http://localhost:8501**

---

## 🧪 How the Models Work (For Your EVS Report)

### Random Forest — Crop Recommendation

```
Farmer inputs soil test results:
  N=80 (Nitrogen), P=40 (Phosphorus), K=40 (Potassium)
  Temperature=20°C, Humidity=80%, pH=6.5, Rainfall=200mm
            ↓
  200 Decision Trees analyze the data independently
  (each tree trained on a random subset of the training data)
            ↓
  Each tree votes: "Rice", "Maize", "Wheat", ...
            ↓
  Majority vote determines the recommendation
  Probability = fraction of trees that agreed
            ↓
  Output: "Plant RICE — Confidence: 87%"
  + EVS Note about rice's water consumption
```

### Convolutional Neural Network — Disease Detection

```
Farmer photographs a leaf → uploads to dashboard
            ↓
  Image resized to 128×128 pixels
  Pixel values normalized from [0,255] to [0,1]
            ↓
  MobileNetV2 (53 layers) extracts visual features:
  Layer 1-10: Detects edges and color gradients
  Layer 11-30: Detects textures and lesion patterns
  Layer 31-53: Detects disease-specific visual signatures
            ↓
  Custom classification head assigns probabilities
  to each of 38 disease categories
            ↓
  Output: "Tomato Early Blight — Confidence: 92%"
  + Eco-friendly neem oil treatment recommendation
```

---

## 🌍 EVS Impact — Why This Matters

| Problem (Conventional Farming) | Agri-Smart Solution | Environmental Benefit |
|---|---|---|
| Calendar-based pesticide spraying | AI detects disease early | 25-40% less chemical use |
| Guessing which crop to plant | Soil-matched crop recommendation | Prevents soil nutrient depletion |
| Over-irrigation | Rainfall data + crop-specific water needs | 30-50% water savings |
| Broad-spectrum fungicides | Targeted disease ID → precise treatment | Less chemical runoff into rivers |
| Monocropping | Recommends rotation-friendly crops | Preserves biodiversity |

---

## 📈 Expected Model Performance

| Model | Dataset | Expected Accuracy |
|---|---|---|
| Random Forest (Crop) | 2,200 samples, 22 classes | ~99% test accuracy |
| CNN (Disease) | 54,000+ images, 38 classes | ~94-96% val accuracy |

---

## 🤝 Credits

- **Crop Dataset:** Atharva Ingle (Kaggle)
- **PlantVillage Dataset:** Hughes & Salathé, Penn State (2016)
- **Base CNN:** MobileNetV2, Howard et al., Google (2018)
- **EVS Framework:** Integrated Pest Management (IPM) guidelines, FAO

---

*Built with ❤️ for Environmental Science education and sustainable farming.*
