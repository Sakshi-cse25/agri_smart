# ============================================================
# app/dashboard.py
#
# PURPOSE: The main Streamlit web application for Agri-Smart.
# Provides two tabs:
#   1. Crop Recommendation (Random Forest)
#   2. Plant Disease Diagnosis (CNN)
#
# HOW TO RUN:
#   streamlit run app/dashboard.py
#
# STREAMLIT BASICS (for your EVS report):
# Streamlit turns a Python script into a web app automatically.
# Every time a user changes an input widget (slider, upload),
# the entire script re-runs top-to-bottom — this "reactive"
# model means no complex event handling code is needed.
# ============================================================

import os, sys
import json
import numpy as np
import joblib
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path so we can import our utility modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils.evs_notes import (
    get_crop_evs_note,
    get_disease_evs_note,
    sustainability_score_bar
)

# ── Page Configuration (must be first Streamlit call) ────────
st.set_page_config(
    page_title  = "Agri-Smart Dashboard",
    page_icon   = "🌿",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
# We inject CSS to override Streamlit defaults with an
# earthy, agricultural color palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --green-dark:   #1b4332;
        --green-mid:    #2d6a4f;
        --green-light:  #52b788;
        --green-pale:   #d8f3dc;
        --amber:        #e9c46a;
        --earth:        #6b4226;
        --bg:           #f8faf7;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg);
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: var(--green-dark);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--green-pale);
        border-radius: 8px 8px 0 0;
        color: var(--green-dark);
        font-weight: 600;
        padding: 10px 24px;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--green-mid) !important;
        color: white !important;
    }

    .evs-box {
        background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
        border-left: 5px solid #2d6a4f;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }

    .evs-box h4 {
        color: #1b4332;
        margin-bottom: 8px;
        font-family: 'Playfair Display', serif;
    }

    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe0a3 100%);
        border-left: 5px solid #e9c46a;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }

    .danger-box {
        background: linear-gradient(135deg, #fde8e8 0%, #f5c6c6 100%);
        border-left: 5px solid #e63946;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }

    .prediction-banner {
        background: linear-gradient(90deg, #1b4332, #2d6a4f);
        color: white;
        border-radius: 12px;
        padding: 20px 28px;
        margin: 16px 0;
        font-size: 1.3rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MODEL LOADING (cached so they only load once per session)
# ════════════════════════════════════════════════════════════

# @st.cache_resource
def load_crop_model():
    """
    @st.cache_resource caches the model object in memory.
    Without caching, the 200-tree Random Forest would be
    re-loaded from disk on every user interaction — very slow!
    """
    try:
        rf      = joblib.load(os.path.join("models", "saved", "random_forest_crop.pkl"))
        le      = joblib.load(os.path.join("data",   "processed", "label_encoder.pkl"))
        scaler  = joblib.load(os.path.join("data",   "processed", "scaler.pkl"))
        return rf, le, scaler
    except FileNotFoundError:
        return None, None, None


# @st.cache_resource
# def load_disease_model():
#     try:
#         import tensorflow as tf
#         model = tf.keras.models.load_model(
#             os.path.join("models/saved/cnn_disease_detector.keras")
#         )
#         with open(os.path.join("data", "processed", "disease_class_map.json")) as f:
#             class_map = json.load(f)
#         return model, class_map
#     except Exception:
#         return None, None
def load_disease_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            os.path.abspath("models/saved/cnn_disease_detector.keras")
        )
        with open(os.path.join("data", "processed", "disease_class_map.json")) as f:
            class_map = json.load(f)
        return model, class_map
    except Exception as e:
        st.write(e)   # 👈 IMPORTANT (error show karega)
        return None, None


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/2d6a4f/ffffff?text=🌿+Agri-Smart",
             use_container_width=True)
    st.markdown("---")
    st.markdown("### 🌍 About Agri-Smart")
    st.markdown("""
    **Agri-Smart** is an AI-powered precision agriculture dashboard that helps farmers:

    - 🌾 **Choose the right crop** for their soil & climate
    - 🔬 **Diagnose plant diseases** from leaf photos
    - 🌿 **Understand environmental impact** of every decision

    *Built for the EVS Project — Sustainable Agriculture Module*
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Status")

    rf, le, scaler = load_crop_model()
    disease_model, class_map = load_disease_model()

    if rf:
        st.success("✅ Crop Recommender: Loaded")
    else:
        st.warning("⚠️ Crop model not found. Run `train_crop_model.py` first.")

    if disease_model:
        st.success("✅ Disease Detector: Loaded")
    else:
        st.warning("⚠️ Disease model not found. Run `train_disease_cnn.py` first.")

    st.markdown("---")
    st.caption("🔬 Powered by Random Forest + MobileNetV2 CNN")
    st.caption("📚 Data: Crop Recommendation Dataset + PlantVillage")


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px;'>
    <h1 style='font-size:2.8rem; color:#1b4332; margin-bottom:4px;'>
        🌿 Agri-Smart Dashboard
    </h1>
    <p style='color:#555; font-size:1.1rem;'>
        AI-Powered Crop Intelligence with Environmental Sustainability Focus
    </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TABS — Main navigation between the two features
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🌾 Crop Recommendation",
    "🔬 Disease Diagnosis",
    "📊 About the Models"
])


# ────────────────────────────────────────────────────────────
# TAB 1: CROP RECOMMENDATION
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 🌾 Crop Recommendation Engine")
    st.markdown("""
    Enter your soil test values and local climate data.
    The AI will recommend the most suitable crop and explain
    the environmental sustainability of that choice.
    """)

    # ── Input Form ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧪 Soil Nutrients (mg/kg)")
        N = st.slider("Nitrogen (N)", 0, 200, 50,
                      help="Higher N favors leafy crops like rice and maize.")
        P = st.slider("Phosphorus (P)", 0, 150, 50,
                      help="P supports root development and flowering.")
        K = st.slider("Potassium (K)", 0, 250, 50,
                      help="K improves drought and disease resistance.")

    with col2:
        st.markdown("### 🌡️ Climate Parameters")
        temperature = st.slider("Temperature (°C)", 5.0, 50.0, 25.0, step=0.5)
        humidity    = st.slider("Humidity (%)", 10.0, 100.0, 70.0, step=1.0)
        rainfall    = st.slider("Rainfall (mm/year)", 0.0, 3000.0, 800.0, step=10.0)

    with col3:
        st.markdown("### 🌍 Soil Quality")
        ph = st.slider("Soil pH", 3.0, 10.0, 6.5, step=0.1,
                       help="Most crops prefer pH 6–7. <6 = acidic, >7 = alkaline.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**pH Guide:**")
        st.markdown("- 🔴 <5.5: Very acidic")
        st.markdown("- 🟡 5.5–6.0: Slightly acidic")
        st.markdown("- 🟢 6.0–7.0: Optimal for most crops")
        st.markdown("- 🔵 7.0–8.0: Alkaline")

    # ── Prediction Button ─────────────────────────────────────
    st.markdown("---")
    predict_btn = st.button(
        "🔍 Find Best Crop for My Field",
        type="primary",
        use_container_width=True
    )

    if predict_btn:
        if rf is None:
            st.error("❌ Crop model not loaded. Please train the model first.")
        else:
            # Prepare features in the same order as training
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            features_scaled = scaler.transform(features)

            # Get prediction and probability scores
            prediction   = rf.predict(features_scaled)[0]
            probabilities = rf.predict_proba(features_scaled)[0]
            crop_name    = le.inverse_transform([prediction])[0]

            # Top 5 crop probabilities
            top5_idx     = np.argsort(probabilities)[::-1][:5]
            top5_crops   = le.inverse_transform(top5_idx)
            top5_probs   = probabilities[top5_idx] * 100

            # ── Result Banner ─────────────────────────────────
            evs = get_crop_evs_note(crop_name)
            st.markdown(f"""
            <div class='prediction-banner'>
                {evs.get('icon','🌱')} Recommended Crop:
                <span style='font-size:1.6rem'>{crop_name.upper()}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Confidence: {top5_probs[0]:.1f}%
            </div>
            """, unsafe_allow_html=True)

            # ── Results in two columns ─────────────────────────
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                st.markdown("#### 🔝 Top 5 Crop Matches")
                fig = px.bar(
                    x=top5_probs, y=top5_crops,
                    orientation="h",
                    color=top5_probs,
                    color_continuous_scale=["#95d5b2", "#2d6a4f"],
                    labels={"x": "Confidence (%)", "y": "Crop"},
                    text=[f"{p:.1f}%" for p in top5_probs]
                )
                fig.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=280,
                    yaxis=dict(autorange="reversed")
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            with res_col2:
                st.markdown("#### 📊 Your Soil Profile")
                radar_fig = go.Figure(data=go.Scatterpolar(
                    r=[
                        min(N / 200, 1) * 100,
                        min(P / 150, 1) * 100,
                        min(K / 250, 1) * 100,
                        min(ph / 10, 1) * 100,
                        humidity,
                        min(rainfall / 3000, 1) * 100,
                        min(temperature / 50, 1) * 100,
                    ],
                    theta=["N", "P", "K", "pH", "Humidity", "Rainfall", "Temp"],
                    fill="toself",
                    fillcolor="rgba(52,183,135,0.25)",
                    line_color="#2d6a4f"
                ))
                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    height=280,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(radar_fig, use_container_width=True)

            # ── EVS Impact Note ───────────────────────────────
            st.markdown("### 🌍 Environmental Sustainability Analysis")
            evs = get_crop_evs_note(crop_name)

            score_bar = sustainability_score_bar(evs.get("sustainability_score", 6))

            box_class = (
                "evs-box" if evs.get("sustainability_score", 6) >= 7
                else "warning-box" if evs.get("sustainability_score", 6) >= 4
                else "danger-box"
            )

            st.markdown(f"""
            <div class='{box_class}'>
                <h4>♻️ EVS Impact Note — {crop_name.title()}</h4>
                <p><b>💧 Water Usage:</b> {evs.get('water_usage', 'N/A')}</p>
                <p><b>🌡️ Carbon Impact:</b> {evs.get('carbon_impact', 'N/A')}</p>
                <p><b>🌿 Sustainability Score:</b> <code>{score_bar}</code></p>
                <hr style='border:1px solid rgba(0,0,0,0.1); margin:10px 0;'>
                <p><b>💡 Farmer Action Tip:</b> {evs.get('evs_tip', '')}</p>
            </div>
            """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# TAB 2: DISEASE DIAGNOSIS
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🔬 Plant Disease Diagnosis")
    st.markdown("""
    Upload a clear photo of a plant leaf. The CNN will analyze
    visual patterns (lesions, discoloration, texture changes)
    to identify the disease — and suggest eco-friendly treatments.
    """)

    # ── Tips for good photos ──────────────────────────────────
    with st.expander("📸 Tips for best diagnosis accuracy"):
        st.markdown("""
        - 📷 **Good lighting**: Natural daylight works best. Avoid harsh shadows.
        - 🍃 **Single leaf**: Photograph one clear leaf against a plain background.
        - 🔍 **Show the affected area**: Include both healthy and diseased portions.
        - 📐 **Distance**: 10–30cm from the leaf — fill most of the frame.
        - 🚫 **Avoid**: Blurry images, wet leaves (reflective), or heavily shaded shots.
        """)

    # ── File Upload ───────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a leaf image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Supports most common leaf images. Max file size: 10MB."
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        # Show the image
        img_col, result_col = st.columns([1, 1.2])

        with img_col:
            st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
            st.caption(f"Image size: {img.size[0]}×{img.size[1]}px")

        with result_col:
            if disease_model is None:
                st.error("❌ Disease model not loaded. Please train the CNN first.")
            else:
                import tensorflow as tf

                # Preprocess image for CNN
                # 1. Resize to 128×128 (what the model expects)
                # 2. Convert to numpy array → shape (128, 128, 3)
                # 3. Normalize pixels to [0, 1]
                # 4. Add batch dimension → shape (1, 128, 128, 3)
                img_resized   = img.resize((128, 128))
                img_array     = np.array(img_resized) / 255.0
                img_batch     = np.expand_dims(img_array, axis=0)

                # Run inference
                with st.spinner("🔍 Analyzing leaf patterns..."):
                    predictions = disease_model.predict(img_batch, verbose=0)[0]

                # Get top 3 predictions
                top3_idx      = np.argsort(predictions)[::-1][:3]
                top3_labels   = [class_map[str(i)] for i in top3_idx]
                top3_confs    = predictions[top3_idx] * 100

                predicted_label = top3_labels[0]
                confidence      = top3_confs[0]

                # ── Result Display ────────────────────────────
                evs_d = get_disease_evs_note(predicted_label)
                icon  = evs_d.get("icon", "🔬")

                # Color banner based on severity
                sev = evs_d.get("severity", "")
                banner_color = (
                    "#1b4332" if "✅" in sev else
                    "#c77b14" if "⚠️" in sev else
                    "#8b0000"
                )

                # Clean label for display
                display_label = predicted_label.replace("___", " → ").replace("_", " ")

                st.markdown(f"""
                <div style='background:{banner_color}; color:white; border-radius:10px;
                            padding:16px 20px; margin-bottom:16px;'>
                    <div style='font-size:1.1rem; opacity:0.8;'>Diagnosis Result</div>
                    <div style='font-size:1.5rem; font-weight:700;'>{icon} {display_label}</div>
                    <div style='font-size:1rem; opacity:0.85; margin-top:4px;'>
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence chart for top 3
                st.markdown("**Top 3 Predictions:**")
                for label, conf in zip(top3_labels, top3_confs):
                    clean = label.replace("___", " → ").replace("_", " ")
                    st.progress(int(conf), text=f"{clean}: {conf:.1f}%")

        # ── EVS Disease Note ──────────────────────────────────
        if disease_model is not None and uploaded_file is not None:
            st.markdown("---")
            st.markdown("### 🌍 Environmental Treatment Guidance")

            evs_d      = get_disease_evs_note(predicted_label)
            box_class  = (
                "evs-box" if "✅" in evs_d.get("severity", "")
                else "warning-box" if "⚠️" in evs_d.get("severity", "")
                else "danger-box"
            )

            st.markdown(f"""
            <div class='{box_class}'>
                <h4>♻️ EVS Impact Note — Disease Management</h4>
                <p><b>⚠️ Severity:</b> {evs_d.get('severity', 'Unknown')}</p>
                <p><b>☣️ Chemical Risk (conventional treatment):</b>
                    {evs_d.get('chemical_risk', 'Assess before use')}</p>
                <hr style='border:1px solid rgba(0,0,0,0.1); margin:10px 0;'>
                <p><b>🌿 Eco-Friendly Alternative:</b> {evs_d.get('evs_tip', '')}</p>
                <p><b>✅ Recommended Action:</b> {evs_d.get('action', '')}</p>
            </div>
            """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# TAB 3: ABOUT THE MODELS
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📊 Model Architecture & Educational Notes")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("""
        ### 🌾 Random Forest — Crop Recommender

        **What it is:** An ensemble of 200 decision trees that each
        vote on the best crop. The majority vote wins.

        **Why it works for soil data:**
        - Handles both numerical (N, P, K) and climatic (temperature)
          features without needing complex transformations
        - Captures non-linear interactions (e.g., high N + high humidity
          = high disease risk for some crops)
        - Built-in feature importance scores educate farmers on WHICH
          soil factor matters most

        **Training data:** Crop Recommendation Dataset (Kaggle)
        - 2,200 samples across 22 crops
        - 7 input features per sample

        **Performance target:** ~99% accuracy on test set

        ---

        **How it makes a prediction:**
        ```
        Input: [N=80, P=40, K=40, temp=20°C,
                humidity=80%, pH=6.5, rain=200mm]
            ↓
        200 Trees each vote independently
            ↓
        Majority vote → "Rice" (164/200 trees)
            ↓
        Confidence: 82%
        ```
        """)

    with tech_col2:
        st.markdown("""
        ### 🔬 MobileNetV2 CNN — Disease Detector

        **What it is:** A deep Convolutional Neural Network
        (53 layers) pre-trained on 1.4 million images, fine-tuned
        on PlantVillage leaf photos.

        **Why CNNs work for images:**
        - Convolutional layers detect local patterns: edges → textures
          → lesion shapes → disease signatures
        - Deeper layers combine these into abstract "disease concepts"
        - Transfer learning means we benefit from ImageNet training
          even with limited plant disease images

        **Training data:** PlantVillage Dataset
        - 54,000+ leaf images across 38 disease classes
        - 14 plant species including tomato, potato, corn

        **Performance target:** ~95%+ validation accuracy

        ---

        **How it makes a prediction:**
        ```
        Input: 128×128×3 pixel image
            ↓
        MobileNetV2 extracts 1280 visual features
            ↓
        Dense layers classify features
            ↓
        Softmax → probability for each disease
            ↓
        "Tomato Early Blight" (92% confidence)
        ```
        """)

    st.markdown("---")
    st.markdown("### 🌍 Sustainability Philosophy")
    st.info("""
    **Why combine AI with EVS (Environmental Studies)?**

    Traditional farming uses **calendar-based** chemical schedules:
    spray on Day 14, Day 28, regardless of actual need. This wastes
    chemicals and harms ecosystems.

    **Precision agriculture** (what Agri-Smart enables) uses data to
    spray ONLY when needed, plant ONLY what the soil supports, and
    alert farmers to diseases BEFORE widespread infection.

    Research shows precision agriculture can:
    - Reduce fertilizer use by **15–20%**
    - Reduce pesticide use by **25–40%**
    - Reduce irrigation water use by **30–50%**
    - Increase yield by **10–15%** through better crop-soil matching

    This dashboard is designed to make that knowledge accessible to
    every farmer — from large commercial operations to small family plots.
    """)
