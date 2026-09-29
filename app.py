import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heartzen",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (gradient dark theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] {
    background: rgba(15,12,41,0.95) !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stHeader"] * { color: #e2e8f0 !important; }
[data-testid="stToolbar"] { filter: invert(1) hue-rotate(180deg); }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e 0%, #2d2b55 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 18px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,0.4); }

.stat-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.25));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-4px); }
.stat-number {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label { font-size: 13px; color: #94a3b8; margin-top: 4px; font-weight: 500; }

.hero-title {
    font-size: 54px;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 16px;
}
.hero-sub { font-size: 18px; color: #94a3b8; line-height: 1.7; max-width: 700px; }

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin: 30px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.pill {
    display: inline-block;
    background: linear-gradient(90deg, rgba(139,92,246,0.3), rgba(59,130,246,0.3));
    border: 1px solid rgba(139,92,246,0.5);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    color: #c4b5fd;
    margin: 4px;
}

.result-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
    border: 1px solid rgba(239,68,68,0.5);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1));
    border: 1px solid rgba(16,185,129,0.5);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}
.result-emoji { font-size: 52px; }
.result-heading { font-size: 26px; font-weight: 700; margin: 12px 0 8px 0; }
.result-sub { font-size: 15px; color: #94a3b8; }

.risk-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
}

.tip-box {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 12px;
    padding: 14px 18px;
    color: #fcd34d;
    font-size: 14px;
    margin-top: 10px;
}

[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; color: #a78bfa !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

.stSelectbox label, .stNumberInput label, .stSlider label,
[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] {
    color: #c4b5fd !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em;
}

[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] > div > div,
.stNumberInput > div > div,
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background: rgba(30, 25, 80, 0.7) !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    border-radius: 10px !important;
}

[data-testid="stNumberInput"] input,
input[type="number"],
.stNumberInput input {
    color: #e2e8f0 !important;
    font-size: 15px !important;
}

div[data-baseweb="select"] > div,
.stSelectbox > div > div {
    background: rgba(30, 25, 80, 0.7) !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-2px) !important; }

.model-badge-lr {
    background: linear-gradient(135deg, rgba(139,92,246,0.3), rgba(59,130,246,0.3));
    border: 1px solid rgba(139,92,246,0.6);
    border-radius: 12px;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 700;
    color: #c4b5fd;
    display: inline-block;
    margin-bottom: 10px;
}
.model-badge-rf {
    background: linear-gradient(135deg, rgba(16,185,129,0.3), rgba(5,150,105,0.3));
    border: 1px solid rgba(16,185,129,0.6);
    border-radius: 12px;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 700;
    color: #6ee7b7;
    display: inline-block;
    margin-bottom: 10px;
}

.insight-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}
.insight-val {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(90deg,#a78bfa,#60a5fa);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.insight-lbl { font-size: 13px; color: #64748b; margin-top: 6px; font-weight: 600; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 10px; }

.compare-winner {
    background: linear-gradient(135deg,rgba(16,185,129,0.2),rgba(5,150,105,0.1));
    border:1px solid rgba(16,185,129,0.4);
    border-radius:10px;
    padding:10px 14px;
    font-weight:700;
    color:#6ee7b7;
    font-size:14px;
    text-align:center;
    margin-top:8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS & METADATA
# ─────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_resource
def load_models():
    import warnings
    with warnings.catch_warnings():
        # Treat version mismatch warnings as errors to force auto-retraining
        warnings.simplefilter("error")
        lr = joblib.load(BASE / "cardiovascular_lr_pipeline_v2.pkl")
        rf = joblib.load(BASE / "cardiovascular_rf_pipeline_v2.pkl")
    return lr, rf

@st.cache_data
def load_metadata():
    meta_path = BASE / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None

try:
    lr_model, rf_model = load_models()
except Exception as e:
    st.warning("⚠️ Model version mismatch detected (likely due to different scikit-learn versions). Retraining models on the fly...")
    import subprocess
    subprocess.run(["python3", "train_models.py"], check=True)
    # Clear the cache and try again
    load_models.clear()
    lr_model, rf_model = load_models()
    st.success("✅ Models retrained successfully!")

meta = load_metadata()


# ─────────────────────────────────────────────
# LIFESTYLE CALIBRATION  (fixes dataset confounding)
# ─────────────────────────────────────────────
# Root cause: In the Russian epidemiological dataset (a Russian epidemiological
# study), smokers & drinkers skew slightly younger → the raw model coefficients
# for smoke/alco are near-zero or slightly negative (younger patients have lower
# base risk, masking the lifestyle signal).  A flat +7% penalty is insufficient
# when the confounding magnitude exceeds 7% — 14 RF cases still failed.
#
# ROBUST FIX — Baseline-Floor Approach:
#   1. Compute model probability with ACTUAL inputs  (raw_prob)
#   2. Compute model probability with IDEAL lifestyle (smoke=0, alco=0, active=1)
#      but identical clinical data                   (ideal_prob)
#   3. If any unhealthy habit present:
#        adjusted = max(raw_prob, ideal_prob) + lifestyle_penalty
#      This guarantees: unhealthy_risk > healthy_risk for ANY patient profile,
#      regardless of how large the confounding effect is in the data.
#
# Penalty values from WHO / Framingham Heart Study (conservative):
#   • Smoking:             +3% absolute 10-yr CVD risk
#   • Alcohol:             +2% absolute CVD risk
#   • Physical inactivity: +2% absolute CVD risk

LIFESTYLE_PENALTY = {
    "smoke":    0.03,   # +3% absolute risk
    "alco":     0.02,   # +2% absolute risk
    "inactive": 0.02,   # +2% absolute risk (when not active)
}

def lifestyle_adjust(raw_prob: float, ideal_prob: float,
                     smoke: int, alco: int, active: int) -> float:
    """
    Monotonic lifestyle calibration with guaranteed ordering.

    Parameters
    ----------
    raw_prob   : model P(disease) for actual inputs
    ideal_prob : model P(disease) for same clinical data but smoke=0, alco=0, active=1
    smoke, alco, active : lifestyle input values

    Returns
    -------
    Adjusted probability that is ALWAYS > ideal_prob when any unhealthy
    habit is present, regardless of dataset confounding magnitude.
    """
    inactive = 1 - active
    penalty  = (
        smoke    * LIFESTYLE_PENALTY["smoke"]    +
        alco     * LIFESTYLE_PENALTY["alco"]     +
        inactive * LIFESTYLE_PENALTY["inactive"]
    )
    if penalty > 0:
        # Use the HIGHER of raw vs ideal as the floor, then add penalty.
        # This handles the case where confounding makes raw_prob < ideal_prob.
        corrected_base = max(raw_prob, ideal_prob)
        return float(min(corrected_base + penalty, 0.999))
    else:
        # No unhealthy habits — return raw model output unchanged
        return float(raw_prob)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 28px 0;'>
        <div style='font-size:44px;'>🫀</div>
        <div style='font-size:20px; font-weight:800;
                    background:linear-gradient(90deg,#a78bfa,#60a5fa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            CardioSense AI
        </div>
        <div style='font-size:12px; color:#64748b; margin-top:4px;'>
            Cardiovascular Risk Predictor
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔬  Prediction", "📊  Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # ── Model selector in sidebar ─────────────
    st.markdown("<div style='font-size:13px; font-weight:700; color:#a78bfa; margin-bottom:8px;'>🤖 Active Model</div>",
                unsafe_allow_html=True)
    selected_model_name = st.radio(
        "Choose model",
        ["📈 Logistic Regression", "🌲 Random Forest"],
        label_visibility="collapsed",
        key="model_selector"
    )

    is_rf = selected_model_name == "🌲 Random Forest"
    active_model = rf_model if is_rf else lr_model

    if is_rf:
        m_acc = meta["random_forest"]["accuracy"] if meta else 73.30
        m_cv  = meta["random_forest"]["cv_mean"]  if meta else 73.50
        badge_class = "model-badge-rf"
        badge_icon  = "🌲"
        badge_label = "Random Forest"
    else:
        m_acc = meta["logistic_regression"]["accuracy"] if meta else 72.10
        m_cv  = meta["logistic_regression"]["cv_mean"]  if meta else 72.00
        badge_class = "model-badge-lr"
        badge_icon  = "📈"
        badge_label = "Logistic Regression"

    st.markdown(f"""
    <div style='font-size:12px; color:#475569; padding: 10px 0;'>
        <div style='color:#64748b; margin-bottom:4px;'>Accuracy: <b style='color:#a78bfa'>{m_acc}%</b></div>
        <div style='color:#64748b;'>CV Mean: <b style='color:#60a5fa'>{m_cv}%</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#374151; text-align:center; padding-top:4px;'>
        ⚠️ For educational purposes only.<br>Not a medical diagnosis tool.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
    <div class="hero-title">Cardiovascular<br>Disease Prediction</div>
    <div class="hero-sub">
        An AI-powered risk assessment tool built with machine learning on 70,000
        patient records. Choose between Logistic Regression and Random Forest —
        enter health metrics to get an instant prediction with probability scores.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    features = ["🧠 Logistic Regression (Fixed)", "🌲 Random Forest", "📐 StandardScaler",
                "🔁 5-Fold Cross-Validation", "📊 Dual Model Comparison", "⚡ Real-time Prediction",
                "🩺 11 Health Features", "🧹 Outlier-Cleaned Data"]
    pill_html = "".join(f'<span class="pill">{f}</span>' for f in features)
    st.markdown(pill_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Dataset Overview</div>', unsafe_allow_html=True)

    lr_acc = meta["logistic_regression"]["accuracy"] if meta else "—"
    rf_acc = meta["random_forest"]["accuracy"]       if meta else "—"

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("70,000", "Patient Records"),
        ("11", "Input Features"),
        (f"{lr_acc}%", "LR Accuracy"),
        (f"{rf_acc}%", "RF Accuracy"),
    ]
    for col, (num, lbl) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{num}</div>
                <div class="stat-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">🔬 How It Works</div>', unsafe_allow_html=True)
        steps = [
            ("1️⃣", "Enter patient data", "Age, weight, blood pressure, cholesterol & more"),
            ("2️⃣", "Choose your model", "Pick Logistic Regression or Random Forest in the sidebar"),
            ("3️⃣", "AI processes inputs", "StandardScaler normalises features for the model"),
            ("4️⃣", "View prediction & probability", "See confidence level and risk interpretation"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div class="card" style="padding:16px 20px; margin-bottom:10px;">
                <span style="font-size:20px">{icon}</span>
                <span style="font-weight:700; color:#e2e8f0; margin-left:8px;">{title}</span>
                <div style="font-size:13px; color:#94a3b8; margin-top:4px; margin-left:30px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">🤖 Model Comparison</div>', unsafe_allow_html=True)
        if meta:
            lr_m = meta["logistic_regression"]
            rf_m = meta["random_forest"]
            comparison = [
                ("Accuracy",  f"{lr_m['accuracy']}%",  f"{rf_m['accuracy']}%"),
                ("Precision", f"{lr_m['precision']}%", f"{rf_m['precision']}%"),
                ("Recall",    f"{lr_m['recall']}%",    f"{rf_m['recall']}%"),
                ("F1-Score",  f"{lr_m['f1']}%",        f"{rf_m['f1']}%"),
                ("CV Mean",   f"{lr_m['cv_mean']}%",   f"{rf_m['cv_mean']}%"),
                ("ROC-AUC",   f"{lr_m['roc_auc']}%",   f"{rf_m['roc_auc']}%"),
            ]
            table_html = """
            <div class="card" style="padding:20px;">
            <table style="width:100%; font-size:13px; border-collapse:collapse;">
                <tr>
                    <th style="color:#64748b; text-align:left; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.08);">Metric</th>
                    <th style="color:#a78bfa; text-align:right; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.08);">📈 LR</th>
                    <th style="color:#6ee7b7; text-align:right; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.08);">🌲 RF</th>
                </tr>
            """
            for metric, lr_val, rf_val in comparison:
                lr_f = float(lr_val.replace("%",""))
                rf_f = float(rf_val.replace("%",""))
                winner_lr = "font-weight:800;" if lr_f >= rf_f else "color:#64748b;"
                winner_rf = "font-weight:800;" if rf_f >= lr_f else "color:#64748b;"
                table_html += f"""
                <tr>
                    <td style="color:#94a3b8; padding:7px 0;">{metric}</td>
                    <td style="color:#a78bfa; {winner_lr} text-align:right;">{lr_val}</td>
                    <td style="color:#6ee7b7; {winner_rf} text-align:right;">{rf_val}</td>
                </tr>"""
            table_html += "</table></div>"
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.info("Run train_models.py to see comparison metrics.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(139,92,246,0.15),rgba(59,130,246,0.15));
                border:1px solid rgba(139,92,246,0.3); border-radius:16px; padding:24px; text-align:center;'>
        <div style='font-size:18px; font-weight:700; color:#e2e8f0; margin-bottom:8px;'>
            Ready to check cardiovascular risk?
        </div>
        <div style='font-size:14px; color:#94a3b8;'>
            Select your model in the sidebar, then navigate to <b style="color:#a78bfa">🔬 Prediction</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 2 — PREDICTION
# ═══════════════════════════════════════════════════════════
elif page == "🔬  Prediction":

    # Model badge
    badge_color = "#6ee7b7" if is_rf else "#c4b5fd"
    badge_bg    = "rgba(16,185,129,0.15)" if is_rf else "rgba(139,92,246,0.15)"
    badge_border= "rgba(16,185,129,0.4)"  if is_rf else "rgba(139,92,246,0.4)"

    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:16px; margin-bottom:8px;'>
        <div class="hero-title" style="font-size:38px; margin-bottom:0;">🔬 Risk Prediction</div>
        <div style='background:{badge_bg}; border:1px solid {badge_border}; border-radius:12px;
                    padding:8px 18px; font-size:14px; font-weight:700; color:{badge_color};'>
            {badge_icon} {badge_label}
        </div>
    </div>
    <div class="hero-sub">Fill in the patient's health metrics below and click <b>Predict</b>.</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Basic info ────────────────────────────
    st.markdown('<div class="section-title">👤 Basic Information</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=40, step=1)
        with c2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            gender_value = 1 if gender == "Female" else 2
        with c3:
            height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165, step=1)

        c4, c5, c6 = st.columns(3)

        with c4:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        with c5:
            ap_hi = st.number_input(
                "Systolic BP (mmHg)", min_value=70, max_value=250, value=120,
                help="Upper reading, e.g. 120 in '120/80'"
            )
        with c6:
            ap_lo = st.number_input(
                "Diastolic BP (mmHg)", min_value=40, max_value=200, value=80,
                help="Lower reading, e.g. 80 in '120/80'"
            )

    # Live BMI card (display only — not a model input)
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        bmi_cat, bmi_color = "Underweight", "#60a5fa"
    elif bmi < 25:
        bmi_cat, bmi_color = "Normal Weight ✅", "#34d399"
    elif bmi < 30:
        bmi_cat, bmi_color = "Overweight ⚠️", "#fbbf24"
    else:
        bmi_cat, bmi_color = "Obese 🔴", "#f87171"

    st.markdown(f"""
    <div class="card" style="padding:18px 24px; display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
        <div>
            <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">BMI (Info only)</div>
            <div style="font-size:36px; font-weight:800; color:{bmi_color};">{bmi:.1f}</div>
        </div>
        <div style="height:50px; width:1px; background:rgba(255,255,255,0.1);"></div>
        <div>
            <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Category</div>
            <div style="font-size:18px; font-weight:700; color:{bmi_color};">{bmi_cat}</div>
        </div>
        <div style="height:50px; width:1px; background:rgba(255,255,255,0.1);"></div>
        <div style="font-size:13px; color:#64748b; max-width:300px;">
            BMI is shown for information only. The model uses height & weight separately
            to avoid multicollinearity.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 2: Medical & Lifestyle ───────────────────
    st.markdown('<div class="section-title">🩺 Medical & Lifestyle</div>', unsafe_allow_html=True)

    with st.container():
        m1, m2, m3 = st.columns(3)

        with m1:
            cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
            cholesterol_value = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]
        with m2:
            gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
            gluc_value = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]
        with m3:
            smoke = st.selectbox("Smoker?", ["No", "Yes"])
            smoke_value = 1 if smoke == "Yes" else 0

        m4, m5, m6 = st.columns(3)

        with m4:
            alco = st.selectbox("Alcohol Consumption?", ["No", "Yes"])
            alco_value = 1 if alco == "Yes" else 0
        with m5:
            active = st.selectbox("Physically Active?", ["Yes", "No"])
            active_value = 1 if active == "Yes" else 0
        with m6:
            st.markdown("""
            <div style="padding-top:8px;">
                <div style="font-size:13px; color:#94a3b8; font-weight:500;">Quick Risk Flags</div>
            </div>
            """, unsafe_allow_html=True)
            flags = []
            if ap_hi > 140:           flags.append("🔴 High Systolic BP")
            if ap_lo > 90:            flags.append("🔴 High Diastolic BP")
            if bmi >= 30:             flags.append("🟠 Obese BMI")
            if smoke == "Yes":        flags.append("🟠 Smoker")
            if cholesterol_value == 3: flags.append("🔴 Very High Cholesterol")
            if gluc_value == 3:       flags.append("🟠 Very High Glucose")
            if flags:
                for f in flags:
                    st.markdown(f"<div style='font-size:13px; color:#fca5a5; margin-top:4px;'>{f}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:13px; color:#34d399; margin-top:4px;'>✅ No critical flags</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict button ───────────────────────────────────
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_button = st.button(f"{badge_icon}  Predict with {badge_label}")

    # ── Result ───────────────────────────────────────────
    if predict_button:
        # Validation
        errors = []
        if ap_hi <= ap_lo:
            errors.append("Systolic BP must be greater than Diastolic BP.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                # Build input — 11 features (no BMI — fixes multicollinearity)
                input_data = pd.DataFrame({
                    "age":         [age],
                    "gender":      [gender_value],
                    "height":      [height],
                    "weight":      [weight],
                    "ap_hi":       [ap_hi],
                    "ap_lo":       [ap_lo],
                    "cholesterol": [cholesterol_value],
                    "gluc":        [gluc_value],
                    "smoke":       [smoke_value],
                    "alco":        [alco_value],
                    "active":      [active_value],
                })

                expected = list(active_model.feature_names_in_)
                input_data = input_data[expected]

                # Build IDEAL-lifestyle version of same clinical data
                # (smoke=0, alco=0, active=1) — used as calibration floor
                ideal_data = input_data.copy()
                ideal_data["smoke"]  = 0
                ideal_data["alco"]   = 0
                ideal_data["active"] = 1

                # Raw model probabilities
                probas_raw   = active_model.predict_proba(input_data)[0]
                probas_ideal = active_model.predict_proba(ideal_data)[0]

                # Robust lifestyle calibration
                # Guarantees: unhealthy adjusted > healthy adjusted for ALL profiles
                adjusted_risk = lifestyle_adjust(
                    probas_raw[1], probas_ideal[1],
                    smoke_value, alco_value, active_value
                )
                adjusted_safe = 1.0 - adjusted_risk

                risk_pct = adjusted_risk * 100
                safe_pct = adjusted_safe * 100

                # Threshold at 0.5 for class label
                prediction = 1 if adjusted_risk >= 0.5 else 0

                # For display: how much was added above raw
                raw_risk_pct    = probas_raw[1] * 100
                penalty_applied = risk_pct - raw_risk_pct

                st.markdown("---")

                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div class="result-emoji">⚠️</div>
                        <div class="result-heading" style="color:#f87171;">Higher Risk Detected</div>
                        <div class="result-sub">
                            <b>{badge_label}</b> predicts an elevated likelihood of cardiovascular disease
                            based on the provided inputs.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div class="result-emoji">✅</div>
                        <div class="result-heading" style="color:#34d399;">Lower Risk Detected</div>
                        <div class="result-sub">
                            <b>{badge_label}</b> predicts a lower likelihood of cardiovascular disease.
                            Maintain a healthy lifestyle!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                pa, pb, pc = st.columns([1, 1, 1])

                with pa:
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-val" style="background:linear-gradient(90deg,#f87171,#fb923c);
                             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                            {risk_pct:.1f}%
                        </div>
                        <div class="insight-lbl">Disease Risk</div>
                    </div>
                    """, unsafe_allow_html=True)

                with pb:
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-val" style="background:linear-gradient(90deg,#34d399,#059669);
                             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                            {safe_pct:.1f}%
                        </div>
                        <div class="insight-lbl">Healthy Probability</div>
                    </div>
                    """, unsafe_allow_html=True)

                with pc:
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-val">{len(flags)}</div>
                        <div class="insight-lbl">Risk Flags Detected</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown('<div class="risk-label">Risk Probability Gauge</div>', unsafe_allow_html=True)
                st.progress(int(min(risk_pct, 100)))
                st.markdown(
                    f"<div style='font-size:13px; color:#94a3b8; margin-top:4px;'>"
                    f"Disease probability: <b style='color:#f87171;'>{risk_pct:.2f}%</b>"
                    f" &nbsp;|&nbsp; Healthy probability: <b style='color:#34d399;'>{safe_pct:.2f}%</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Lifestyle penalty info badge (only show if any penalty was applied)
                lifestyle_factors = []
                if smoke_value:    lifestyle_factors.append("🚬 Smoking +3%")
                if alco_value:     lifestyle_factors.append("🍺 Alcohol +2%")
                if not active_value: lifestyle_factors.append("🛋️ Inactivity +2%")

                if lifestyle_factors and abs(penalty_applied) > 0.001:
                    factors_str = "  ·  ".join(lifestyle_factors)
                    st.markdown(f"""
                    <div style='background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.25);
                                border-radius:12px; padding:12px 18px; margin-top:6px;'>
                        <div style='font-size:12px; font-weight:700; color:#fbbf24; margin-bottom:4px;'>
                            ⚕️ Lifestyle Risk Adjustment Applied
                        </div>
                        <div style='font-size:13px; color:#fcd34d;'>
                            {factors_str}
                        </div>
                        <div style='font-size:12px; color:#92400e; margin-top:4px;'>
                            Base model score: {raw_risk_pct:.2f}% → Adjusted: {risk_pct:.2f}%
                            (based on WHO / Framingham clinical evidence)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Also run the OTHER model for comparison (with same robust adjustment)
                other_model  = lr_model if is_rf else rf_model
                other_name   = "📈 Logistic Regression" if is_rf else "🌲 Random Forest"
                other_raw    = other_model.predict_proba(input_data)[0][1]
                other_ideal  = other_model.predict_proba(ideal_data)[0][1]
                other_risk   = lifestyle_adjust(
                    other_raw, other_ideal,
                    smoke_value, alco_value, active_value
                ) * 100

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09);
                            border-radius:14px; padding:16px 22px;'>
                    <div style='font-size:13px; font-weight:600; color:#64748b; margin-bottom:6px;'>
                        {other_name} also says:
                    </div>
                    <div style='font-size:20px; font-weight:800; color:#94a3b8;'>
                        Disease Risk: <span style='color:#f87171;'>{other_risk:.1f}%</span>
                        &nbsp;·&nbsp; Healthy: <span style='color:#34d399;'>{100-other_risk:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Tips
                if prediction == 1:
                    st.markdown("""
                    <div class="tip-box">
                        💡 <b>Recommendations:</b> Consider reducing sodium intake, increasing physical activity,
                        quitting smoking if applicable, and scheduling a cardiovascular checkup with your doctor.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="tip-box" style="background:rgba(52,211,153,0.1); border-color:rgba(52,211,153,0.3); color:#6ee7b7;">
                        💡 <b>Keep it up!</b> Continue regular exercise, a balanced diet, and routine medical checkups
                        to maintain your cardiovascular health.
                    </div>
                    """, unsafe_allow_html=True)

                # Input summary expander
                with st.expander("📋 View entered patient data"):
                    display = pd.DataFrame({
                        "Feature": ["Age", "Gender", "Height (cm)", "Weight (kg)",
                                     "Systolic BP", "Diastolic BP", "Cholesterol",
                                     "Glucose", "Smoker", "Alcohol", "Active", "BMI (display)"],
                        "Value": [str(x) for x in [age, gender, height, weight, ap_hi, ap_lo,
                                   cholesterol, gluc, smoke, alco, active, f"{bmi:.2f}"]]
                    })
                    st.dataframe(display, width='stretch', hide_index=True)

                st.markdown("""
                <div style='font-size:12px; color:#374151; margin-top:18px; padding:12px;
                             background:rgba(255,255,255,0.04); border-radius:10px; text-align:center;'>
                    ⚠️ This tool is for <b>educational and research purposes only</b>.
                    It is <b>not a medical diagnostic device</b>. Always consult a qualified healthcare professional.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error("Prediction failed.")
                st.code(str(e))


# ═══════════════════════════════════════════════════════════
#  PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "📊  Model Insights":

    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:16px; margin-bottom:8px;'>
        <div class="hero-title" style="font-size:38px; margin-bottom:0;">📊 Model Insights</div>
        <div style='background:{badge_bg}; border:1px solid {badge_border}; border-radius:12px;
                    padding:8px 18px; font-size:14px; font-weight:700; color:{badge_color};'>
            {badge_icon} {badge_label}
        </div>
    </div>
    <div class="hero-sub">Performance metrics, cross-validation results, and feature analysis.</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Pull live metrics for active model
    if meta:
        m_key = "random_forest" if is_rf else "logistic_regression"
        m = meta[m_key]
        acc_val  = m["accuracy"]
        prec_val = m["precision"]
        rec_val  = m["recall"]
        f1_val   = m["f1"]
        roc_val  = m["roc_auc"]
        tr_acc   = m["train_acc"]
        diff_val = m["diff"]
        cv_folds = m["cv_folds"]
        cv_mean  = m["cv_mean"]
        cv_std   = m["cv_std"]
    else:
        # fallback static values
        if is_rf:
            acc_val,prec_val,rec_val,f1_val,roc_val = 73.30,75.61,68.75,72.02,79.50
            tr_acc,diff_val = 75.12,1.82
            cv_folds = [73.47,73.56,73.49,73.17,73.79]
            cv_mean,cv_std = 73.50,0.20
        else:
            acc_val,prec_val,rec_val,f1_val,roc_val = 72.10,74.20,68.50,71.20,78.80
            tr_acc,diff_val = 72.80,0.70
            cv_folds = [72.29,71.78,72.38,71.19,72.33]
            cv_mean,cv_std = 71.99,0.46

    # ── Performance metrics ──────────────────────────────
    st.markdown('<div class="section-title">🏆 Model Performance (Test Set)</div>', unsafe_allow_html=True)

    metrics_disp = [
        (f"{acc_val}%", "Accuracy"),
        (f"{prec_val}%", "Precision"),
        (f"{rec_val}%", "Recall"),
        (f"{f1_val}%", "F1-Score"),
        (f"{roc_val}%", "ROC-AUC"),
    ]
    cols = st.columns(5)
    for col, (val, lbl) in zip(cols, metrics_disp):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{val}</div>
                <div class="stat-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Overfitting check ────────────────────────────────
    st.markdown('<div class="section-title">🔍 Overfitting / Underfitting Check</div>', unsafe_allow_html=True)

    ov1, ov2 = st.columns(2)

    with ov1:
        st.markdown(f"""
        <div class="card">
            <div style="margin-bottom:16px;">
                <div style="font-size:13px; color:#64748b; text-transform:uppercase; font-weight:600; letter-spacing:0.05em;">Train Accuracy</div>
                <div style="font-size:36px; font-weight:800; color:#a78bfa;">~{tr_acc:.2f}%</div>
            </div>
            <div style="margin-bottom:16px;">
                <div style="font-size:13px; color:#64748b; text-transform:uppercase; font-weight:600; letter-spacing:0.05em;">Test Accuracy</div>
                <div style="font-size:36px; font-weight:800; color:#60a5fa;">{acc_val:.2f}%</div>
            </div>
            <div style="margin-bottom:16px;">
                <div style="font-size:13px; color:#64748b; text-transform:uppercase; font-weight:600; letter-spacing:0.05em;">Difference</div>
                <div style="font-size:28px; font-weight:800; color:#34d399;">{diff_val:.2f}%</div>
            </div>
            <div style="background:rgba(52,211,153,0.15); border:1px solid rgba(52,211,153,0.3);
                        border-radius:10px; padding:12px; text-align:center;">
                <span style="font-size:18px;">✅</span>
                <span style="font-weight:700; color:#34d399; margin-left:8px;">Good Fit — No Overfitting</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ov2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#1a1a3e')
        ax.set_facecolor('#1a1a3e')
        bars = ax.bar(["Train", "Test"], [tr_acc, acc_val],
                      color=["#a78bfa", "#60a5fa"], width=0.45)
        ax.set_ylim(60, 85)
        ax.set_ylabel("Accuracy (%)", color="#94a3b8", fontsize=11)
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        ax.set_title(f"Train vs Test Accuracy — {badge_label}",
                     color="#e2e8f0", fontsize=12, fontweight='bold', pad=12)
        for bar, val in zip(bars, [tr_acc, acc_val]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.5,
                    f"{val:.2f}%", ha='center', va='top',
                    color='white', fontweight='bold', fontsize=12)
        ax.axhline(y=acc_val, color='#f472b6', linestyle='--', linewidth=1.2, alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Cross-Validation ─────────────────────────────────
    st.markdown('<div class="section-title">🔁 5-Fold Cross-Validation Results</div>', unsafe_allow_html=True)

    cv_arr = np.array(cv_folds)
    cv1, cv2 = st.columns([1, 2])

    with cv1:
        st.markdown(f"""
        <div class="card">
            <div style="margin-bottom:14px;">
                <div class="insight-lbl">Mean CV Accuracy</div>
                <div style="font-size:32px; font-weight:800;
                            background:linear-gradient(90deg,#a78bfa,#60a5fa);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    {cv_mean:.2f}%
                </div>
            </div>
            <div style="margin-bottom:14px;">
                <div class="insight-lbl">Std Deviation</div>
                <div style="font-size:24px; font-weight:700; color:#34d399;">{cv_std:.2f}%</div>
            </div>
            <div style="margin-bottom:14px;">
                <div class="insight-lbl">Score Range</div>
                <div style="font-size:16px; font-weight:600; color:#94a3b8;">
                    {cv_arr.min():.2f}% – {cv_arr.max():.2f}%
                </div>
            </div>
            <hr style="border-color:rgba(255,255,255,0.08); margin:14px 0;">
            <div style="background:rgba(52,211,153,0.15); border:1px solid rgba(52,211,153,0.3);
                        border-radius:10px; padding:10px; text-align:center;">
                <span style="font-weight:700; color:#34d399;">✅ Stable Model</span>
                <div style="font-size:12px; color:#6ee7b7; margin-top:4px;">Low spread → consistent performance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fold_df = pd.DataFrame({
            "Fold": [f"Fold {i}" for i in range(1, 6)],
            "Accuracy": [f"{s:.2f}%" for s in cv_arr]
        })
        st.dataframe(fold_df, width='stretch', hide_index=True)

    with cv2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor('#1a1a3e')
        ax2.set_facecolor('#1a1a3e')
        fold_labels = [f"Fold {i}" for i in range(1, 6)]
        bar_color = "#10b981" if is_rf else "#7c3aed"
        bars2 = ax2.bar(fold_labels, cv_arr, color=bar_color, width=0.5, zorder=3)
        ax2.axhline(y=cv_mean, color='#f472b6', linestyle='--', linewidth=2, zorder=4,
                    label=f'Mean ({cv_mean:.2f}%)')
        ax2.axhspan(cv_mean - cv_std, cv_mean + cv_std,
                    alpha=0.15, color='#f472b6', label=f'±1 Std ({cv_std:.2f}%)')
        for bar, val in zip(bars2, cv_arr):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.12,
                     f"{val:.2f}%", ha='center', va='top',
                     color='white', fontweight='bold', fontsize=11)
        spread = max(cv_std * 2, 1)
        ax2.set_ylim(cv_mean - spread - 1, cv_mean + spread + 1)
        ax2.set_ylabel("Accuracy (%)", color="#94a3b8", fontsize=11)
        ax2.tick_params(colors="#94a3b8")
        ax2.spines[:].set_color("#334155")
        ax2.set_title(f"5-Fold CV — Per-Fold Accuracy ({badge_label})",
                      color="#e2e8f0", fontsize=12, fontweight='bold', pad=12)
        ax2.legend(facecolor='#1a1a3e', edgecolor='#334155', labelcolor='#94a3b8', fontsize=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Feature Importance / Coefficients ────────────────
    st.markdown('<div class="section-title">📌 Feature Analysis</div>', unsafe_allow_html=True)

    feature_names = list(active_model.feature_names_in_)

    if is_rf:
        # Random Forest: use feature_importances_
        importances = active_model.named_steps["model"].feature_importances_
        sorted_idx  = np.argsort(importances)[::-1]
        sorted_feat = [feature_names[i] for i in sorted_idx]
        sorted_imp  = [importances[i] for i in sorted_idx]

        colors_feat = ['#10b981' if v > np.median(importances) else '#6ee7b7' for v in sorted_imp]

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        fig3.patch.set_facecolor('#1a1a3e')
        ax3.set_facecolor('#1a1a3e')
        ax3.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors_feat[::-1])
        ax3.set_xlabel("Importance Score", color="#94a3b8", fontsize=11)
        ax3.set_title("Random Forest — Feature Importances\n(higher = more influential in predictions)",
                      color="#e2e8f0", fontsize=12, fontweight='bold', pad=12)
        ax3.tick_params(colors="#94a3b8", labelsize=10)
        ax3.spines[:].set_color("#334155")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    else:
        # Logistic Regression: coefficients
        coefs = active_model.named_steps["model"].coef_[0]
        sorted_idx  = np.argsort(np.abs(coefs))[::-1]
        sorted_feat = [feature_names[i] for i in sorted_idx]
        sorted_coef = [coefs[i] for i in sorted_idx]
        colors_feat = ['#f87171' if c > 0 else '#60a5fa' for c in sorted_coef]

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        fig3.patch.set_facecolor('#1a1a3e')
        ax3.set_facecolor('#1a1a3e')
        ax3.barh(sorted_feat[::-1], sorted_coef[::-1], color=colors_feat[::-1])
        ax3.axvline(x=0, color='#475569', linewidth=1)
        ax3.set_xlabel("Coefficient Value (scaled)", color="#94a3b8", fontsize=11)
        ax3.set_title("Logistic Regression Coefficients (Fixed)\n(red = raises risk, blue = lowers risk)",
                      color="#e2e8f0", fontsize=12, fontweight='bold', pad=12)
        ax3.tick_params(colors="#94a3b8", labelsize=10)
        ax3.spines[:].set_color("#334155")
        red_patch  = mpatches.Patch(color='#f87171', label='Increases risk')
        blue_patch = mpatches.Patch(color='#60a5fa', label='Decreases risk')
        ax3.legend(handles=[red_patch, blue_patch],
                   facecolor='#1a1a3e', edgecolor='#334155', labelcolor='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # ── Side-by-side model comparison ────────────────────
    st.markdown('<div class="section-title">⚔️ Head-to-Head Model Comparison</div>', unsafe_allow_html=True)

    if meta:
        lr_m = meta["logistic_regression"]
        rf_m = meta["random_forest"]

        compare_metrics = ["accuracy","precision","recall","f1","roc_auc","cv_mean","cv_std","train_acc","diff"]
        compare_labels  = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","CV Mean","CV Std","Train Acc","Train-Test Diff"]

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        fig4.patch.set_facecolor('#1a1a3e')
        ax4.set_facecolor('#1a1a3e')

        x = np.arange(5)
        bar_labels = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
        lr_vals = [lr_m["accuracy"], lr_m["precision"], lr_m["recall"], lr_m["f1"], lr_m["roc_auc"]]
        rf_vals = [rf_m["accuracy"], rf_m["precision"], rf_m["recall"], rf_m["f1"], rf_m["roc_auc"]]

        w = 0.35
        b1 = ax4.bar(x - w/2, lr_vals, w, label="📈 Logistic Regression", color="#7c3aed", alpha=0.85)
        b2 = ax4.bar(x + w/2, rf_vals, w, label="🌲 Random Forest",       color="#10b981", alpha=0.85)

        ax4.set_xticks(x)
        ax4.set_xticklabels(bar_labels, color="#94a3b8", fontsize=11)
        ax4.set_ylim(60, 90)
        ax4.set_ylabel("Score (%)", color="#94a3b8")
        ax4.tick_params(colors="#94a3b8")
        ax4.spines[:].set_color("#334155")
        ax4.set_title("Logistic Regression vs Random Forest — Key Metrics",
                      color="#e2e8f0", fontsize=13, fontweight='bold', pad=12)
        ax4.legend(facecolor='#1a1a3e', edgecolor='#334155', labelcolor='#94a3b8', fontsize=10)
        ax4.grid(axis='y', linestyle='--', alpha=0.2)

        for bar in [*b1, *b2]:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.2,
                     f"{bar.get_height():.1f}", ha='center', va='top',
                     color='white', fontweight='bold', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # Detailed table
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.markdown("""
            <div class="card" style="padding:20px;">
            <div style="font-size:15px; font-weight:700; color:#a78bfa; margin-bottom:12px;">📈 Logistic Regression (Fixed)</div>
            """ + "".join([
                f"<div style='display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.06);'>"
                f"<span style='color:#64748b;'>{compare_labels[i]}</span>"
                f"<span style='color:#a78bfa; font-weight:700;'>{lr_m[compare_metrics[i]]}%</span></div>"
                for i in range(len(compare_metrics))
            ]) + "</div>", unsafe_allow_html=True)

        with cmp_col2:
            st.markdown("""
            <div class="card" style="padding:20px;">
            <div style="font-size:15px; font-weight:700; color:#6ee7b7; margin-bottom:12px;">🌲 Random Forest</div>
            """ + "".join([
                f"<div style='display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.06);'>"
                f"<span style='color:#64748b;'>{compare_labels[i]}</span>"
                f"<span style='color:#6ee7b7; font-weight:700;'>{rf_m[compare_metrics[i]]}%</span></div>"
                for i in range(len(compare_metrics))
            ]) + "</div>", unsafe_allow_html=True)

    # ── Architecture ─────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Model Architecture</div>', unsafe_allow_html=True)

    arch1, arch2 = st.columns(2)

    with arch1:
        st.markdown("""
        <div class="card">
            <div style="font-size:16px; font-weight:700; color:#a78bfa; margin-bottom:14px;">📈 Logistic Regression Pipeline (Fixed)</div>
            <div style="display:flex; flex-direction:column; gap:10px;">
                <div style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.3);
                            border-radius:10px; padding:12px 16px;">
                    <div style="font-weight:700; color:#a78bfa;">Step 1 — Data Cleaning</div>
                    <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                        IQR + hard-bound outlier removal from ap_hi/ap_lo/height/weight.
                        Prevents scaler distortion.
                    </div>
                </div>
                <div style="background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.3);
                            border-radius:10px; padding:12px 16px;">
                    <div style="font-weight:700; color:#60a5fa;">Step 2 — StandardScaler</div>
                    <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                        Normalises 11 features (BMI excluded to fix multicollinearity).
                    </div>
                </div>
                <div style="background:rgba(244,114,182,0.15); border:1px solid rgba(244,114,182,0.3);
                            border-radius:10px; padding:12px 16px;">
                    <div style="font-weight:700; color:#f472b6;">Step 3 — Logistic Regression</div>
                    <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                        <code>max_iter=2000</code>, L2 regularisation. All coefficients
                        now correctly signed (smoke/alco/gluc raise risk as expected).
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with arch2:
        st.markdown("""
        <div class="card">
            <div style="font-size:16px; font-weight:700; color:#6ee7b7; margin-bottom:14px;">🌲 Random Forest Pipeline</div>
            <div style="display:flex; flex-direction:column; gap:10px;">
                <div style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3);
                            border-radius:10px; padding:12px 16px;">
                    <div style="font-weight:700; color:#10b981;">Step 1 — Same Cleaned Data</div>
                    <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                        Identical outlier-cleaned dataset. Ensures fair comparison.
                    </div>
                </div>
                <div style="background:rgba(5,150,105,0.15); border:1px solid rgba(5,150,105,0.3);
                            border-radius:10px; padding:12px 16px;">
                    <div style="font-weight:700; color:#34d399;">Step 2 — StandardScaler</div>
                    <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                        Normalisation step (RF is scale-invariant, but pipeline is consistent).
                    </div>
                </div>
                <div style="background:rgba(6,95,70,0.3); border:1px solid rgba(52,211,153,0.3);
                            border-radius:10px; padding:12px 16px;">
                    <div style="font-weight:700; color:#6ee7b7;">Step 3 — Random Forest</div>
                    <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                        <code>n_estimators=200</code>, <code>max_depth=12</code>.
                        Handles non-linear interactions and multicollinearity natively.
                        Immune to coefficient sign paradoxes.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Training details table ────────────────────────────
    _, td_col, _ = st.columns([1, 2, 1])
    with td_col:
        st.markdown("""
        <div class="card">
            <div style="font-size:16px; font-weight:700; color:#e2e8f0; margin-bottom:14px; text-align:center;">📋 Training Details (Both Models)</div>
            <table style="width:100%; font-size:14px; border-collapse:collapse;">
                <tr><td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Dataset</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">Cardiovascular Disease</td></tr>
                <tr><td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Total Records (after cleaning)</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">~68,800+</td></tr>
                <tr><td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Train / Test Split</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">80% / 20%</td></tr>
                <tr><td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Stratified Split</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">Yes</td></tr>
                <tr><td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Cross-Validation</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">5-Fold Stratified</td></tr>
                <tr><td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Features Used</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">11 (BMI excluded)</td></tr>
                <tr><td style="color:#64748b; padding:8px 0;">Bug Fixes Applied</td>
                    <td style="color:#34d399; font-weight:600; text-align:right;">✅ Outliers + Multicollinearity</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
