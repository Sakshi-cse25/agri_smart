# ============================================================
# src/utils/evs_notes.py
#
# PURPOSE: Provides "EVS Impact Notes" — curated, evidence-based
# sustainability insights that appear alongside every AI prediction.
#
# PHILOSOPHY:
# The Agri-Smart dashboard is not just a prediction tool. It is
# an EDUCATIONAL PLATFORM that helps farmers understand the
# environmental consequences of crop and chemical choices.
# Every recommendation comes with a sustainability dimension.
# ============================================================

from typing import Optional

# ─────────────────────────────────────────────────────────────
# CROP RECOMMENDATION EVS NOTES
# Key: crop name (lowercase, matches label_encoder classes)
# ─────────────────────────────────────────────────────────────
CROP_EVS_NOTES: dict[str, dict] = {
    "rice": {
        "water_usage":    "HIGH (1,200–2,000 L per kg produced)",
        "carbon_impact":  "Paddy fields emit methane (CH₄) — a greenhouse gas 28× more potent than CO₂.",
        "evs_tip":        "💧 Use System of Rice Intensification (SRI) to cut water use by up to 50% and reduce methane by 70%. Alternate Wetting and Drying (AWD) is a proven technique.",
        "sustainability_score": 5,
        "icon": "🌾"
    },
    "maize": {
        "water_usage":    "MODERATE (500–800 L per kg)",
        "carbon_impact":  "Maize residue left in soil builds organic carbon, improving soil health.",
        "evs_tip":        "🌽 Intercropping maize with legumes (e.g., beans) fixes atmospheric nitrogen naturally, reducing synthetic fertilizer need by 30%.",
        "sustainability_score": 7,
        "icon": "🌽"
    },
    "chickpea": {
        "water_usage":    "LOW (300–500 L per kg)",
        "carbon_impact":  "Nitrogen-fixing legume — adds 40–80 kg N/ha to soil naturally.",
        "evs_tip":        "♻️ Growing chickpea before wheat is a classic crop rotation strategy. It reduces wheat's fertilizer demand and breaks pest cycles, cutting pesticide use by ~25%.",
        "sustainability_score": 9,
        "icon": "🫘"
    },
    "kidneybeans": {
        "water_usage":    "LOW (300–500 L per kg)",
        "carbon_impact":  "Legume — fixes nitrogen, reduces synthetic fertilizer dependency.",
        "evs_tip":        "🌿 Excellent cover crop. Prevents soil erosion during off-seasons and adds biomass to soil, boosting earthworm activity.",
        "sustainability_score": 9,
        "icon": "🫘"
    },
    "pigeonpeas": {
        "water_usage":    "LOW (250–450 L per kg)",
        "carbon_impact":  "Deep-rooted — improves soil structure and breaks hardpan layers.",
        "evs_tip":        "🌱 Pigeon peas are drought-tolerant and can grow on degraded land, making them ideal for reclaiming eroded soils without heavy inputs.",
        "sustainability_score": 10,
        "icon": "🌿"
    },
    "mothbeans": {
        "water_usage":    "VERY LOW (200–350 L per kg)",
        "carbon_impact":  "Arid-land legume — prevents desertification in dry zones.",
        "evs_tip":        "🏜️ Moth beans thrive in semi-arid conditions where other crops fail. Choosing this crop preserves groundwater and avoids over-irrigation of marginal land.",
        "sustainability_score": 10,
        "icon": "🌿"
    },
    "mungbean": {
        "water_usage":    "LOW (300–400 L per kg)",
        "carbon_impact":  "Short-duration crop (60 days) — ideal for catch-cropping between main seasons.",
        "evs_tip":        "⏱️ Short growth cycle means less water exposure and faster soil cover. Reduces weed pressure, lowering herbicide needs.",
        "sustainability_score": 9,
        "icon": "🫘"
    },
    "blackgram": {
        "water_usage":    "LOW (300–500 L per kg)",
        "carbon_impact":  "Nitrogen-fixer with significant organic matter contribution.",
        "evs_tip":        "🌍 Blackgram improves soil microbial diversity. Rich microbial soil needs fewer fungicide applications over time.",
        "sustainability_score": 9,
        "icon": "🫘"
    },
    "lentil": {
        "water_usage":    "LOW (300–500 L per kg)",
        "carbon_impact":  "Low carbon footprint crop — ranked among the most climate-friendly proteins.",
        "evs_tip":        "🌡️ Lentils have one of the lowest greenhouse gas emissions per gram of protein (0.9 kg CO₂ eq/kg) vs beef (27 kg CO₂ eq/kg).",
        "sustainability_score": 10,
        "icon": "🫘"
    },
    "cotton": {
        "water_usage":    "VERY HIGH (8,000–10,000 L per kg of lint!)",
        "carbon_impact":  "Conventionally among the most pesticide-intensive crops globally.",
        "evs_tip":        "⚠️ Consider Bt cotton varieties (pest-resistant GMO) to cut insecticide use by 50%. Drip irrigation can reduce water consumption by 40% vs flood irrigation.",
        "sustainability_score": 3,
        "icon": "🌿"
    },
    "jute": {
        "water_usage":    "MODERATE (700–900 L per kg)",
        "carbon_impact":  "Sequesters ~1.5 tonnes of CO₂ per acre during its growth cycle.",
        "evs_tip":        "♻️ Jute is 100% biodegradable. Replacing synthetic packaging with jute reduces plastic pollution. Its fallen leaves also enrich soil naturally.",
        "sustainability_score": 8,
        "icon": "🌿"
    },
    "coffee": {
        "water_usage":    "HIGH (140 L per cup!)",
        "carbon_impact":  "Shade-grown coffee preserves biodiversity in tropical forest margins.",
        "evs_tip":        "🌳 Agroforestry coffee systems (coffee under tree canopies) support bird diversity, reduce erosion, and cut synthetic fertilizer need by 35%.",
        "sustainability_score": 6,
        "icon": "☕"
    },
    "banana": {
        "water_usage":    "HIGH (790 L per kg)",
        "carbon_impact":  "Banana leaves and pseudostem can be composted, returning nutrients to soil.",
        "evs_tip":        "🍌 Zero-waste farming: banana peels make excellent compost or bio-pesticide (potassium-rich spray to deter aphids), eliminating chemical use.",
        "sustainability_score": 6,
        "icon": "🍌"
    },
    "mango": {
        "water_usage":    "MODERATE (500–700 L per kg)",
        "carbon_impact":  "Long-lived perennial tree — sequesters carbon for 30–50 years.",
        "evs_tip":        "🌳 Mango orchards double as carbon sinks. Intercropping with turmeric or ginger below the canopy maximizes land use and reduces surface runoff.",
        "sustainability_score": 8,
        "icon": "🥭"
    },
    "grapes": {
        "water_usage":    "MODERATE (400–600 L per kg)",
        "carbon_impact":  "Vineyards can use cover crops between rows to prevent soil erosion.",
        "evs_tip":        "🍇 Use precision drip irrigation in grape cultivation. Deficit irrigation (intentional mild water stress) can actually improve fruit quality while saving 30% water.",
        "sustainability_score": 7,
        "icon": "🍇"
    },
    "watermelon": {
        "water_usage":    "MODERATE (235 L per kg — efficient given fruit water content)",
        "carbon_impact":  "Fast-growing, short-season crop with minimal inputs.",
        "evs_tip":        "🍉 Grow on raised beds with plastic mulch to retain soil moisture, reduce weeding, and cut water use by 25%.",
        "sustainability_score": 7,
        "icon": "🍉"
    },
    "muskmelon": {
        "water_usage":    "MODERATE (250 L per kg)",
        "carbon_impact":  "Short-season crop — fits well in rotation to break disease cycles.",
        "evs_tip":        "🫐 Muskmelon grown on trellises improves air circulation, reducing fungal disease risk and cutting fungicide spray frequency.",
        "sustainability_score": 7,
        "icon": "🍈"
    },
    "apple": {
        "water_usage":    "MODERATE (700 L per kg)",
        "carbon_impact":  "Perennial orchard — long-term carbon storage in woody biomass.",
        "evs_tip":        "🍎 Integrated Pest Management (IPM) in apple orchards uses pheromone traps and beneficial insects to reduce insecticide use by up to 80%.",
        "sustainability_score": 7,
        "icon": "🍎"
    },
    "orange": {
        "water_usage":    "MODERATE (560 L per kg)",
        "carbon_impact":  "Citrus groves support pollinators — important for ecosystem services.",
        "evs_tip":        "🍊 Composted citrus pulp (after juicing) makes excellent biofertilizer, closing the nutrient loop and reducing waste.",
        "sustainability_score": 7,
        "icon": "🍊"
    },
    "papaya": {
        "water_usage":    "MODERATE (300–400 L per kg)",
        "carbon_impact":  "Fast-growing with high biomass return to soil.",
        "evs_tip":        "🫧 Papaya leaves contain papain — a natural bio-pesticide effective against nematodes, reducing soil fumigant chemical use.",
        "sustainability_score": 8,
        "icon": "🍈"
    },
    "coconut": {
        "water_usage":    "MODERATE (2,000 L per kg of oil — but perennial with many byproducts)",
        "carbon_impact":  "Every part of the coconut tree is usable — zero waste crop.",
        "evs_tip":        "🌴 Coconut husk (coir) is a sustainable growing medium replacing peat moss in horticulture, protecting peatland ecosystems from extraction.",
        "sustainability_score": 8,
        "icon": "🥥"
    },
    "pomegranate": {
        "water_usage":    "LOW (400–500 L per kg — drought-tolerant)",
        "carbon_impact":  "Grows in arid climates where few crops can, preventing land abandonment.",
        "evs_tip":        "🌵 Pomegranate is one of the most drought-resistant fruit crops. It can use brackish water (slightly saline) — reducing pressure on freshwater sources.",
        "sustainability_score": 9,
        "icon": "🍎"
    },
}

# Generic fallback note for any crop not in the dictionary
_DEFAULT_NOTE = {
    "water_usage": "Data not available — check local agricultural extension guidelines.",
    "carbon_impact": "Monitor soil organic matter and reduce tillage to sequester carbon.",
    "evs_tip": "🌱 Adopt Integrated Nutrient Management (INM): combine organic (compost, FYM) with minimal inorganic fertilizers to maintain soil health and reduce chemical dependency.",
    "sustainability_score": 6,
    "icon": "🌱"
}


# ─────────────────────────────────────────────────────────────
# DISEASE DETECTION EVS NOTES
# ─────────────────────────────────────────────────────────────
DISEASE_EVS_NOTES: dict[str, dict] = {
    "healthy": {
        "severity":      "None ✅",
        "chemical_risk": "None",
        "evs_tip":       "🌿 Your plant is healthy! Maintain soil health with organic mulch and avoid prophylactic (preventive) pesticide spraying — it disrupts beneficial insects like bees and predatory beetles.",
        "action":        "No treatment needed. Continue monitoring weekly.",
        "icon":          "✅"
    },
    "early_blight": {
        "severity":      "Moderate ⚠️",
        "chemical_risk": "Medium — often treated with mancozeb or chlorothalonil fungicides.",
        "evs_tip":       "🍃 Biological alternative: Neem oil spray (3ml/L water) applied every 7–10 days shows 60–70% efficacy against early blight with ZERO chemical residue in soil or water.",
        "action":        "Remove infected lower leaves. Apply copper-based fungicide or neem oil. Ensure proper plant spacing for air circulation.",
        "icon":          "⚠️"
    },
    "late_blight": {
        "severity":      "Severe 🔴",
        "chemical_risk": "HIGH — requires systemic fungicides (metalaxyl). Overuse causes fungicide resistance.",
        "evs_tip":       "💊 Use fungicides only when 3+ consecutive days of rain+humidity >90% are forecast (disease-favorable conditions). Weather-based spray decisions reduce chemical use by 40%.",
        "action":        "Act immediately. Apply systemic fungicide. Remove and destroy (do not compost) heavily infected plants.",
        "icon":          "🔴"
    },
    "leaf_curl": {
        "severity":      "Moderate ⚠️",
        "chemical_risk": "Medium — typically caused by whiteflies (viral vector). Insecticide overuse harms pollinators.",
        "evs_tip":       "🐝 Use yellow sticky traps and reflective mulch to deter whiteflies WITHOUT insecticides. This protects bee populations which are essential for crop pollination.",
        "action":        "Remove and destroy infected plants. Use silver reflective mulch. Introduce Encarsia formosa (natural whitefly predator) if available.",
        "icon":          "⚠️"
    },
    "powdery_mildew": {
        "severity":      "Moderate ⚠️",
        "chemical_risk": "Medium — sulfur-based fungicides are common but can acidify soil.",
        "evs_tip":       "🧴 Baking soda spray (5g/L water + a drop of dish soap) raises leaf pH, inhibiting mildew growth. Studies show 70% effectiveness with NO soil or water contamination.",
        "action":        "Improve air circulation. Apply potassium bicarbonate or dilute neem oil. Avoid overhead watering.",
        "icon":          "⚠️"
    },
    "bacterial_spot": {
        "severity":      "Moderate ⚠️",
        "chemical_risk": "Medium — copper bactericides are standard but toxic to aquatic organisms.",
        "evs_tip":       "💧 Avoid overhead irrigation — water splashing spreads bacterial spot dramatically. Switching to drip irrigation saves water AND reduces disease spread by up to 60%.",
        "action":        "Apply copper-based bactericide early. Practice crop rotation (do not plant solanums in the same plot for 2 years).",
        "icon":          "⚠️"
    },
    "leaf_scorch": {
        "severity":      "Mild to Moderate",
        "chemical_risk": "LOW — usually abiotic (heat/drought stress), not infection-based.",
        "evs_tip":       "☀️ Leaf scorch is often a WATER STRESS signal, not a disease! Mulching with 5–8cm of straw around the base reduces soil temperature and retains moisture, potentially eliminating the 'disease'.",
        "action":        "Check for drought stress first. Apply organic mulch. Ensure adequate irrigation. Test soil for potassium deficiency.",
        "icon":          "🟡"
    },
    "mosaic_virus": {
        "severity":      "Severe (incurable) 🔴",
        "chemical_risk": "Viruses cannot be treated — removal is the only option.",
        "evs_tip":       "🛡️ Prevention is the only strategy. Certified virus-free seed is the single most effective intervention — avoiding synthetic viral protectants entirely.",
        "action":        "Remove and destroy infected plants immediately (bag and bin — do not compost). Control aphid vectors with neem oil. Use virus-resistant varieties next season.",
        "icon":          "🔴"
    },
}

_DEFAULT_DISEASE_NOTE = {
    "severity":      "Unknown — consult local agricultural extension officer.",
    "chemical_risk": "Assess before any chemical application.",
    "evs_tip":       "🔍 Before spraying any chemical, identify the exact pathogen. Using the wrong pesticide is both environmentally harmful AND economically wasteful.",
    "action":        "Collect a leaf sample and send to a plant disease diagnostic lab.",
    "icon":          "❓"
}


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def get_crop_evs_note(crop_name: str) -> dict:
    """Returns EVS impact data for a given crop name."""
    key = crop_name.lower().replace(" ", "").replace("-", "")
    return CROP_EVS_NOTES.get(key, _DEFAULT_NOTE)


def get_disease_evs_note(disease_label: str) -> dict:
    """
    Maps a PlantVillage-style label (e.g., 'Tomato___Early_blight')
    to an EVS note. Extracts the disease keyword from the label.
    """
    label_lower = disease_label.lower()

    # Check for healthy first
    if "healthy" in label_lower:
        return DISEASE_EVS_NOTES["healthy"]

    # Match against known disease keywords
    for key in DISEASE_EVS_NOTES:
        if key in label_lower.replace("_", "_"):
            return DISEASE_EVS_NOTES[key]

    # Fuzzy match on partial words
    for key in DISEASE_EVS_NOTES:
        words = key.split("_")
        if any(w in label_lower for w in words if len(w) > 4):
            return DISEASE_EVS_NOTES[key]

    return _DEFAULT_DISEASE_NOTE


def sustainability_score_bar(score: int, max_score: int = 10) -> str:
    """Creates a simple text-based sustainability score bar for display."""
    filled = "█" * score
    empty  = "░" * (max_score - score)
    return f"{filled}{empty} {score}/{max_score}"
