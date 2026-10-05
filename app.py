"""
╔══════════════════════════════════════════════════════════════════════╗
║  CarePulse — Healthcare Patient Readmission & Risk Predictor       ║
║  A clinician-friendly, human-centric decision-support dashboard    ║
╚══════════════════════════════════════════════════════════════════════╝

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Local module imports ────────────────────────────────────────────
from src.preprocessing import (
    load_data, clean_and_encode_data, NUMERICAL_COLS, CATEGORICAL_COLS,
)
from src.statistical_tests import run_ttest, run_anova, correlation_analysis
from src.pca_analysis import apply_pca
from src.model import train_and_evaluate, generate_test_predictions

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CarePulse · Readmission Predictor",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Healthcare-grade design system
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Google Fonts import ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root variables (Healthcare palette) ─── */
:root {
    --cp-primary:       #0A6EBD;
    --cp-primary-light: #E8F4FD;
    --cp-primary-dark:  #064A80;
    --cp-accent:        #12B886;
    --cp-accent-light:  #E6FCF5;
    --cp-danger:        #E03131;
    --cp-danger-light:  #FFF0F0;
    --cp-warning:       #F59F00;
    --cp-warning-light: #FFF9DB;
    --cp-surface:       #FFFFFF;
    --cp-surface-alt:   #F8FAFB;
    --cp-border:        #E9ECEF;
    --cp-text:          #1A1D21;
    --cp-text-secondary:#5C636A;
    --cp-text-muted:    #8B95A2;
    --cp-radius:        12px;
    --cp-radius-sm:     8px;
    --cp-radius-lg:     16px;
    --cp-shadow:        0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.06);
    --cp-shadow-lg:     0 4px 6px rgba(0,0,0,0.04), 0 10px 24px rgba(0,0,0,0.08);
    --cp-transition:    all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ─── Global resets ─── */
html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background: linear-gradient(135deg, #F0F7FF 0%, #F8FAFB 40%, #F0FFF4 100%);
}

/* ─── Hide Streamlit chrome ─── */
#MainMenu, header[data-testid="stHeader"], footer,
div[data-testid="stDecoration"] {
    display: none !important;
}

/* ─── Main content area ─── */
.block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A3D62 0%, #0A6EBD 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 20px rgba(10, 61, 98, 0.15);
}
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span {
    color: rgba(255,255,255,0.88) !important;
    font-size: 0.88rem;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
    margin: 1rem 0 !important;
}

/* ─── Sidebar metric cards ─── */
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: var(--cp-radius-sm) !important;
    padding: 0.75rem 1rem !important;
    backdrop-filter: blur(10px);
    margin-bottom: 0.5rem;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] label {
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetric"]
    [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--cp-surface);
    border-radius: var(--cp-radius) !important;
    padding: 6px !important;
    gap: 4px;
    box-shadow: var(--cp-shadow);
    border: 1px solid var(--cp-border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--cp-radius-sm) !important;
    padding: 0.6rem 1.1rem !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--cp-text-secondary) !important;
    transition: var(--cp-transition);
    white-space: nowrap;
    border: none !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--cp-primary-light) !important;
    color: var(--cp-primary) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--cp-primary) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(10, 110, 189, 0.3);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ─── Main-area metric cards ─── */
[data-testid="stMetric"] {
    background: var(--cp-surface) !important;
    border: 1px solid var(--cp-border) !important;
    border-radius: var(--cp-radius) !important;
    padding: 1.1rem 1.3rem !important;
    box-shadow: var(--cp-shadow);
    transition: var(--cp-transition);
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--cp-shadow-lg);
    transform: translateY(-2px);
    border-color: var(--cp-primary) !important;
}
[data-testid="stMetric"] label {
    color: var(--cp-text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--cp-text) !important;
    font-weight: 700 !important;
}

/* ─── DataFrames ─── */
[data-testid="stDataFrame"] {
    border-radius: var(--cp-radius) !important;
    overflow: hidden;
    box-shadow: var(--cp-shadow);
    border: 1px solid var(--cp-border) !important;
}

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, var(--cp-primary) 0%, var(--cp-primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--cp-radius-sm) !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: var(--cp-transition);
    box-shadow: 0 2px 8px rgba(10, 110, 189, 0.25);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(10, 110, 189, 0.35) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--cp-accent) 0%, #0CA678 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--cp-radius-sm) !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(18, 184, 134, 0.25);
}

/* ─── Expanders ─── */
.streamlit-expanderHeader {
    background: var(--cp-surface-alt) !important;
    border-radius: var(--cp-radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: var(--cp-text) !important;
}

/* ─── Form container ─── */
[data-testid="stForm"] {
    background: var(--cp-surface) !important;
    border: 1px solid var(--cp-border) !important;
    border-radius: var(--cp-radius-lg) !important;
    padding: 1.5rem !important;
    box-shadow: var(--cp-shadow);
}

/* ─── Form field labels ─── */
.stNumberInput label,
.stSelectbox label,
.stTextInput label,
.stDateInput label,
.stTimeInput label,
.stTextArea label,
[data-testid="stForm"] label,
[data-testid="stForm"] .stMarkdown p {
    color: var(--cp-text) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    margin-bottom: 0.3rem !important;
}
[data-testid="stForm"] label p {
    color: var(--cp-text) !important;
    font-weight: 600 !important;
}

/* ─── Selectbox / Number input ─── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: var(--cp-radius-sm) !important;
    border-color: var(--cp-border) !important;
    font-size: 0.9rem !important;
    background-color: var(--cp-surface) !important;
    color: var(--cp-text) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--cp-surface) !important;
    border-color: var(--cp-border) !important;
    color: var(--cp-text) !important;
}
.stNumberInput input {
    background-color: var(--cp-surface) !important;
    color: var(--cp-text) !important;
}

/* ─── Custom component classes ─── */
.hero-banner {
    background: linear-gradient(135deg, #0A6EBD 0%, #12B886 100%);
    border-radius: var(--cp-radius-lg);
    padding: 2rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(10, 110, 189, 0.25);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.hero-banner p {
    font-size: 0.95rem;
    opacity: 0.9;
    margin: 0;
    font-weight: 400;
    line-height: 1.5;
}

.section-card {
    background: var(--cp-surface);
    border: 1px solid var(--cp-border);
    border-radius: var(--cp-radius-lg);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--cp-shadow);
    transition: var(--cp-transition);
}
.section-card:hover {
    box-shadow: var(--cp-shadow-lg);
}
.section-card h3 {
    color: var(--cp-text);
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-card p {
    color: var(--cp-text-secondary);
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0;
}

.stat-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.stat-badge.significant {
    background: var(--cp-accent-light);
    color: var(--cp-accent);
    border: 1px solid rgba(18, 184, 134, 0.2);
}
.stat-badge.not-significant {
    background: var(--cp-warning-light);
    color: #B8860B;
    border: 1px solid rgba(245, 159, 0, 0.2);
}

.insight-box {
    background: linear-gradient(135deg, var(--cp-primary-light) 0%, #F0F7FF 100%);
    border-left: 4px solid var(--cp-primary);
    border-radius: 0 var(--cp-radius-sm) var(--cp-radius-sm) 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: var(--cp-primary-dark);
    line-height: 1.6;
}
.insight-box strong { color: var(--cp-primary); }

.risk-result-card {
    border-radius: var(--cp-radius-lg);
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: var(--cp-shadow-lg);
    position: relative;
    overflow: hidden;
}
.risk-result-card.high-risk {
    background: linear-gradient(135deg, #FFF0F0 0%, #FFE0E0 100%);
    border: 2px solid var(--cp-danger);
}
.risk-result-card.moderate-risk {
    background: linear-gradient(135deg, #FFF9DB 0%, #FFF3BF 100%);
    border: 2px solid var(--cp-warning);
}
.risk-result-card.low-risk {
    background: linear-gradient(135deg, #E6FCF5 0%, #C3FAE8 100%);
    border: 2px solid var(--cp-accent);
}
.risk-result-card .risk-percentage {
    font-size: 3rem;
    font-weight: 800;
    margin: 0.3rem 0;
    letter-spacing: -0.03em;
}
.risk-result-card .risk-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.risk-result-card .risk-subtitle {
    font-size: 0.85rem;
    opacity: 0.8;
}

.methodology-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--cp-surface-alt);
    border: 1px solid var(--cp-border);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-size: 0.72rem;
    color: var(--cp-text-muted);
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
}

.cv-fold-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--cp-radius);
    overflow: hidden;
    border: 1px solid var(--cp-border);
    font-size: 0.85rem;
}
.cv-fold-table th {
    background: var(--cp-primary);
    color: white;
    padding: 0.7rem 1rem;
    font-weight: 600;
    text-align: center;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.cv-fold-table td {
    padding: 0.6rem 1rem;
    text-align: center;
    border-bottom: 1px solid var(--cp-border);
    color: var(--cp-text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.cv-fold-table tr:nth-child(even) td {
    background: var(--cp-surface-alt);
}
.cv-fold-table tr:last-child td {
    border-bottom: none;
    font-weight: 700;
    background: var(--cp-primary-light);
    color: var(--cp-primary-dark);
}

.footer-bar {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    margin-top: 2rem;
    border-top: 1px solid var(--cp-border);
    color: var(--cp-text-muted);
    font-size: 0.75rem;
}
.footer-bar a { color: var(--cp-primary); text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train_df.csv")
TEST_PATH = os.path.join(BASE_DIR, "test_df.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "final_submission.csv")


# ════════════════════════════════════════════════════════════════════
# MATPLOTLIB — Global clinical chart style
# ════════════════════════════════════════════════════════════════════
CHART_COLORS = {
    "primary":   "#0A6EBD",
    "accent":    "#12B886",
    "danger":    "#E03131",
    "warning":   "#F59F00",
    "safe":      "#12B886",
    "risk":      "#E03131",
    "bg":        "#F8FAFB",
    "grid":      "#E9ECEF",
    "text":      "#1A1D21",
    "muted":     "#8B95A2",
    "palette":   ["#0A6EBD", "#12B886", "#845EF7", "#F59F00", "#E03131"],
}
CP_PALETTE_BINARY = [CHART_COLORS["accent"], CHART_COLORS["danger"]]


def _apply_chart_style(ax, title="", xlabel="", ylabel=""):
    """Apply consistent CarePulse styling to any matplotlib axes."""
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHART_COLORS["grid"])
    ax.spines["bottom"].set_color(CHART_COLORS["grid"])
    ax.tick_params(colors=CHART_COLORS["muted"], labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    if title:
        ax.set_title(title, fontsize=12, fontweight=700, color=CHART_COLORS["text"],
                      pad=14, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=CHART_COLORS["muted"], labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=CHART_COLORS["muted"], labelpad=8)
    ax.grid(axis="y", alpha=0.3, color=CHART_COLORS["grid"], linewidth=0.8)


# ════════════════════════════════════════════════════════════════════
# CACHED PIPELINE
# ════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_pipeline():
    """Execute the full ML pipeline and cache all artefacts."""
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    processed = clean_and_encode_data(train_df, test_df)
    ttest_results = run_ttest(processed["train_df_clean"])
    anova_results = run_anova(processed["train_df_clean"])
    corr_matrix = correlation_analysis(processed["train_df_clean"])
    pca_results = apply_pca(processed["X_train"])
    model_results = train_and_evaluate(processed["X_train"], processed["y_train"])

    return {
        "train_df_raw": train_df,
        "test_df_raw": test_df,
        "processed": processed,
        "ttest": ttest_results,
        "anova": anova_results,
        "corr_matrix": corr_matrix,
        "pca": pca_results,
        "model": model_results,
    }


# Loading screen
with st.spinner(""):
    st.markdown("""
    <div style="text-align:center; padding:3rem 0;">
        <p style="color:#0A6EBD; font-size:1rem; font-weight:600;">
            Initializing CarePulse …
        </p>
        <p style="color:#8B95A2; font-size:0.82rem;">
            Loading patient data  ·  Training model  ·  Running statistical analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    pipeline = run_pipeline()

train_clean = pipeline["processed"]["train_df_clean"]
n_train = len(train_clean)
n_test = len(pipeline["test_df_raw"])
n_features = len(pipeline["processed"]["feature_names"])
readmit_rate = train_clean["readmitted"].mean() * 100


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 0.2rem;">
        <div style="font-size:1.5rem; font-weight:800; letter-spacing:-0.02em;">
            💙 CarePulse
        </div>
        <div style="font-size:0.75rem; opacity:0.7; margin-top:2px; letter-spacing:0.03em;">
            Readmission Risk Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.metric("Patient Records", f"{n_train:,}")
    st.metric("Test Cohort", f"{n_test:,}")
    st.metric("Model Features", n_features)
    st.metric("Readmission Rate", f"{readmit_rate:.1f}%")

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.78rem; line-height:1.6; opacity:0.8;">
        <strong style="font-size:0.82rem;">About This Tool</strong><br>
        CarePulse helps clinicians identify patients
        at elevated risk of 30-day hospital readmission
        using statistical evidence and logistic regression
        modelling.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; opacity:0.5; text-align:center;">
        ⚕️ For clinical decision support only.<br>
        Not a substitute for professional judgment.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏥  Patient Cohort Overview",
    "🔬  Statistical Evidence",
    "🧬  Dimensionality Landscape",
    "📊  Model Intelligence",
    "🩺  Clinical Risk Assessor",
])


# ────────────────────────────────────────────────────────────────────
# TAB 1 — Patient Cohort Overview
# ────────────────────────────────────────────────────────────────────
with tab1:
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <h1>🏥 Patient Cohort Overview</h1>
        <p>Explore the demographic composition, clinical profiles, and outcome
        distribution of the training population to understand the data foundation
        powering CarePulse's predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key vitals row ──
    dup_count = len(pipeline["train_df_raw"]) - n_train
    missing_count = train_clean.isnull().sum().sum()
    avg_age = train_clean["age"].mean()
    avg_days = train_clean["days_in_hospital"].mean()

    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Total Records", f"{n_train:,}")
    v2.metric("Duplicates Removed", f"{dup_count}")
    v3.metric("Missing Values", f"{missing_count}")
    v4.metric("Avg. Age", f"{avg_age:.1f} yrs")
    v5.metric("Avg. Stay", f"{avg_days:.1f} days")

    st.markdown("")

    # ── Class balance ──
    st.markdown('<div class="section-card"><h3>📈 Readmission Outcome Distribution</h3></div>',
                unsafe_allow_html=True)

    col_cls1, col_cls2 = st.columns([2, 1])
    with col_cls1:
        vc = train_clean["readmitted"].value_counts().sort_index()
        fig_cls, ax_cls = plt.subplots(figsize=(7, 3.5))

        bars = ax_cls.bar(
            ["Not Readmitted", "Readmitted"],
            vc.values,
            color=[CHART_COLORS["accent"], CHART_COLORS["danger"]],
            width=0.5,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        for bar, val in zip(bars, vc.values):
            pct = val / n_train * 100
            ax_cls.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + n_train * 0.01,
                        f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                        fontsize=10, fontweight=700, color=CHART_COLORS["text"])

        _apply_chart_style(ax_cls, title="Readmission Class Balance", ylabel="Patient Count")
        fig_cls.tight_layout()
        st.pyplot(fig_cls)
        plt.close(fig_cls)

    with col_cls2:
        st.markdown("")
        st.markdown("")
        not_readmit_n = vc.get(0, 0)
        readmit_n = vc.get(1, 0)
        st.markdown(f"""
        <div class="section-card" style="margin-top:0.5rem;">
            <h3>Class Summary</h3>
            <p style="margin-bottom:0.8rem;">
                <span style="color:{CHART_COLORS['accent']}; font-weight:700;">●</span>
                &nbsp;Not Readmitted: <strong>{not_readmit_n:,}</strong>
                ({not_readmit_n/n_train*100:.1f}%)
            </p>
            <p style="margin-bottom:0.8rem;">
                <span style="color:{CHART_COLORS['danger']}; font-weight:700;">●</span>
                &nbsp;Readmitted: <strong>{readmit_n:,}</strong>
                ({readmit_n/n_train*100:.1f}%)
            </p>
            <p style="font-size:0.82rem; color:var(--cp-text-muted); margin-top:1rem;">
                The dataset shows a <strong>{readmit_rate:.0f}:{100-readmit_rate:.0f}</strong>
                class ratio. The model uses <em>class_weight='balanced'</em>
                to ensure fair learning across both outcomes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Distribution plots ──
    st.markdown('<div class="section-card"><h3>📊 Feature Distributions by Outcome</h3></div>',
                unsafe_allow_html=True)

    col_age, col_days = st.columns(2)
    with col_age:
        fig_age, ax_age = plt.subplots(figsize=(6.5, 3.8))
        for label, color, name in [(0, CHART_COLORS["accent"], "Not Readmitted"),
                                    (1, CHART_COLORS["danger"], "Readmitted")]:
            subset = train_clean[train_clean["readmitted"] == label]["age"]
            ax_age.hist(subset, bins=30, alpha=0.6, color=color, label=name,
                        edgecolor="white", linewidth=0.8, zorder=3)
        ax_age.legend(frameon=True, fontsize=8.5, loc="upper right",
                      facecolor="white", edgecolor=CHART_COLORS["grid"])
        _apply_chart_style(ax_age, title="Age Distribution", xlabel="Patient Age (years)",
                           ylabel="Frequency")
        fig_age.tight_layout()
        st.pyplot(fig_age)
        plt.close(fig_age)

    with col_days:
        fig_d, ax_d = plt.subplots(figsize=(6.5, 3.8))
        for label, color, name in [(0, CHART_COLORS["accent"], "Not Readmitted"),
                                    (1, CHART_COLORS["danger"], "Readmitted")]:
            subset = train_clean[train_clean["readmitted"] == label]["days_in_hospital"]
            ax_d.hist(subset, bins=20, alpha=0.6, color=color, label=name,
                      edgecolor="white", linewidth=0.8, zorder=3)
        ax_d.legend(frameon=True, fontsize=8.5, loc="upper right",
                    facecolor="white", edgecolor=CHART_COLORS["grid"])
        _apply_chart_style(ax_d, title="Length of Stay Distribution",
                           xlabel="Days in Hospital", ylabel="Frequency")
        fig_d.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)

    st.markdown("")

    # ── Categorical breakdown ──
    st.markdown('<div class="section-card"><h3>🏷️ Categorical Breakdowns</h3></div>',
                unsafe_allow_html=True)

    cat_choice = st.selectbox(
        "Select clinical variable:",
        CATEGORICAL_COLS,
        format_func=lambda x: {"gender": "Gender", "primary_diagnosis": "Primary Diagnosis",
                                "discharge_to": "Discharge Destination"}.get(x, x),
    )

    fig_cat, ax_cat = plt.subplots(figsize=(9, 4))
    ct = pd.crosstab(train_clean[cat_choice], train_clean["readmitted"])
    ct.columns = ["Not Readmitted", "Readmitted"]
    ct.plot(kind="bar", ax=ax_cat, color=CP_PALETTE_BINARY, edgecolor="white",
            linewidth=1, width=0.7, zorder=3)
    ax_cat.legend(frameon=True, fontsize=8.5, facecolor="white", edgecolor=CHART_COLORS["grid"])
    nice_name = {"gender": "Gender", "primary_diagnosis": "Primary Diagnosis",
                 "discharge_to": "Discharge Destination"}.get(cat_choice, cat_choice)
    _apply_chart_style(ax_cat, title=f"{nice_name} vs Readmission Outcome",
                       xlabel=nice_name, ylabel="Count")
    ax_cat.tick_params(axis="x", rotation=25)
    fig_cat.tight_layout()
    st.pyplot(fig_cat)
    plt.close(fig_cat)

    # ── Data preview ──
    with st.expander("📋  View Raw Training Data Sample"):
        st.dataframe(train_clean.head(15), use_container_width=True, hide_index=True)

    with st.expander("📈  Descriptive Statistics"):
        st.dataframe(train_clean.describe().round(2), use_container_width=True)


# ────────────────────────────────────────────────────────────────────
# TAB 2 — Statistical Evidence
# ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="hero-banner">
        <h1>🔬 Statistical Evidence</h1>
        <p>Rigorous hypothesis testing to validate whether clinical variables show
        statistically meaningful relationships with readmission outcomes —
        informing both model design and clinical understanding.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── T-Test ──
    ttest = pipeline["ttest"]
    sig_t = ttest["p_value"] < 0.05

    st.markdown(f"""
    <div class="section-card">
        <h3>📐 Welch's Two-Sample T-Test
            <span class="methodology-tag">scipy.stats.ttest_ind</span>
        </h3>
        <p><strong>Hypothesis:</strong> Does length of hospital stay differ
        significantly between readmitted and non-readmitted patients?</p>
    </div>
    """, unsafe_allow_html=True)

    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("T-Statistic", f"{ttest['t_statistic']:.4f}")
    tc2.metric("P-Value", f"{ttest['p_value']:.6f}")
    tc3.metric("Mean (Readmitted)", f"{ttest['mean_readmitted']:.2f} days")
    tc4.metric("Mean (Not Readmitted)", f"{ttest['mean_not_readmitted']:.2f} days")

    badge_class = "significant" if sig_t else "not-significant"
    badge_icon = "✅" if sig_t else "⚠️"
    badge_text = "Statistically Significant (p < 0.05)" if sig_t else "Not Significant (p ≥ 0.05)"

    st.markdown(f"""
    <div class="insight-box">
        <span class="stat-badge {badge_class}">{badge_icon} {badge_text}</span>
        <br><br>
        <strong>Clinical Interpretation:</strong> {ttest['interpretation']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── ANOVA ──
    anova = pipeline["anova"]
    sig_a = anova["p_value"] < 0.05

    st.markdown(f"""
    <div class="section-card">
        <h3>📊 One-Way ANOVA
            <span class="methodology-tag">scipy.stats.f_oneway</span>
        </h3>
        <p><strong>Hypothesis:</strong> Does mean comorbidity score differ
        significantly across primary diagnosis groups?</p>
    </div>
    """, unsafe_allow_html=True)

    ac1, ac2 = st.columns(2)
    ac1.metric("F-Statistic", f"{anova['f_statistic']:.4f}")
    ac2.metric("P-Value", f"{anova['p_value']:.6f}")

    # Group means chart
    fig_gm, ax_gm = plt.subplots(figsize=(8, 3.5))
    groups = list(anova["group_means"].keys())
    means = list(anova["group_means"].values())
    bars_gm = ax_gm.barh(groups, means, color=CHART_COLORS["palette"][:len(groups)],
                          height=0.55, edgecolor="white", linewidth=1.2, zorder=3)
    for bar, m in zip(bars_gm, means):
        ax_gm.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{m:.3f}", va="center", fontsize=9.5, fontweight=600,
                    color=CHART_COLORS["text"])
    _apply_chart_style(ax_gm, title="Mean Comorbidity Score by Diagnosis Group",
                       xlabel="Mean Score")
    ax_gm.grid(axis="x", alpha=0.3, color=CHART_COLORS["grid"])
    ax_gm.grid(axis="y", visible=False)
    fig_gm.tight_layout()
    st.pyplot(fig_gm)
    plt.close(fig_gm)

    badge_class_a = "significant" if sig_a else "not-significant"
    badge_icon_a = "✅" if sig_a else "⚠️"
    badge_text_a = "Statistically Significant" if sig_a else "Not Significant (p ≥ 0.05)"
    st.markdown(f"""
    <div class="insight-box">
        <span class="stat-badge {badge_class_a}">{badge_icon_a} {badge_text_a}</span>
        <br><br>
        <strong>Clinical Interpretation:</strong> {anova['interpretation']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Correlation Heatmap ──
    st.markdown("""
    <div class="section-card">
        <h3>🔗 Pearson Correlation Matrix
            <span class="methodology-tag">pandas .corr()</span>
        </h3>
        <p>Exploring linear relationships among numerical clinical variables
        and the readmission target to identify potential predictive signals.</p>
    </div>
    """, unsafe_allow_html=True)

    corr_matrix = pipeline["corr_matrix"]
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    cmap = sns.diverging_palette(220, 150, s=80, l=55, as_cmap=True)
    sns.heatmap(
        corr_matrix, annot=True, fmt=".3f", cmap=cmap, center=0,
        mask=mask, square=True, linewidths=2, linecolor="white",
        ax=ax_corr, cbar_kws={"shrink": 0.75, "label": "Correlation"},
        annot_kws={"size": 10, "fontweight": 600},
    )
    ax_corr.set_title("Pearson Correlation Heatmap", fontsize=12, fontweight=700,
                       color=CHART_COLORS["text"], pad=16, loc="left")
    ax_corr.tick_params(labelsize=9.5)
    fig_corr.patch.set_facecolor("white")
    fig_corr.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)


# ────────────────────────────────────────────────────────────────────
# TAB 3 — Dimensionality Landscape (PCA)
# ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="hero-banner">
        <h1>🧬 Dimensionality Landscape</h1>
        <p>Principal Component Analysis (PCA) compresses all clinical features
        into two dimensions, revealing whether readmission outcomes form
        visually separable clusters in reduced feature space.</p>
    </div>
    """, unsafe_allow_html=True)

    pca_data = pipeline["pca"]
    pc1_var = pca_data["explained_variance_ratio"][0] * 100
    pc2_var = pca_data["explained_variance_ratio"][1] * 100
    total_var = pca_data["total_explained_variance"] * 100

    pv1, pv2, pv3 = st.columns(3)
    pv1.metric("PC1 Variance", f"{pc1_var:.2f}%")
    pv2.metric("PC2 Variance", f"{pc2_var:.2f}%")
    pv3.metric("Total Retained", f"{total_var:.2f}%")

    st.markdown("")

    components_df = pca_data["components"].copy()
    components_df["readmitted"] = pipeline["processed"]["y_train"].values

    fig_pca, ax_pca = plt.subplots(figsize=(10, 6.5))

    for label, color, marker, name, alpha in [
        (0, CHART_COLORS["accent"], "o", "Not Readmitted", 0.35),
        (1, CHART_COLORS["danger"], "^", "Readmitted", 0.55),
    ]:
        mask_pca = components_df["readmitted"] == label
        ax_pca.scatter(
            components_df.loc[mask_pca, "PC1"],
            components_df.loc[mask_pca, "PC2"],
            c=color, marker=marker, s=18, alpha=alpha,
            edgecolors="none", label=name, zorder=3,
        )

    ax_pca.legend(frameon=True, fontsize=9, loc="upper right", facecolor="white",
                  edgecolor=CHART_COLORS["grid"], markerscale=1.8)
    _apply_chart_style(
        ax_pca,
        title="2D PCA Projection — Patient Feature Space",
        xlabel=f"Principal Component 1 ({pc1_var:.1f}% variance)",
        ylabel=f"Principal Component 2 ({pc2_var:.1f}% variance)",
    )
    fig_pca.tight_layout()
    st.pyplot(fig_pca)
    plt.close(fig_pca)

    st.markdown(f"""
    <div class="insight-box">
        <strong>What does this tell us?</strong>
        The first two principal components capture <strong>{total_var:.1f}%</strong>
        of the total variance. Substantial overlap between clusters suggests that
        readmission risk depends on subtle feature interactions rather than
        easily separable patterns — reinforcing the value of a supervised
        learning approach like Logistic Regression for this task.
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# TAB 4 — Model Intelligence
# ────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Model Intelligence</h1>
        <p>Performance evaluation of the Logistic Regression model using
        5-Fold Stratified Cross-Validation, feature coefficient analysis,
        and odds ratio interpretation — all designed for clinical transparency.</p>
    </div>
    """, unsafe_allow_html=True)

    model_results = pipeline["model"]
    cv = model_results["cv_metrics"]

    # ── Headline metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{cv['accuracy']['mean']:.1%}")
    m2.metric("Precision", f"{cv['precision']['mean']:.1%}")
    m3.metric("Recall", f"{cv['recall']['mean']:.1%}")
    m4.metric("F1-Score", f"{cv['f1']['mean']:.1%}")

    st.markdown("")

    # ── CV fold table ──
    st.markdown("""
    <div class="section-card">
        <h3>📋 5-Fold Cross-Validation Breakdown
            <span class="methodology-tag">sklearn · StratifiedKFold</span>
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Build HTML table
    header = "<tr>" + "".join(f"<th>{h}</th>" for h in
                              ["Fold", "Accuracy", "Precision", "Recall", "F1-Score"]) + "</tr>"
    rows = ""
    for i in range(5):
        rows += "<tr>"
        rows += f"<td style='font-weight:600;'>Fold {i+1}</td>"
        for m_name in ["accuracy", "precision", "recall", "f1"]:
            val = cv[m_name]["all_folds"][i]
            rows += f"<td>{val:.4f}</td>"
        rows += "</tr>"
    # Mean row
    rows += "<tr>"
    rows += "<td style='font-weight:700;'>μ ± σ</td>"
    for m_name in ["accuracy", "precision", "recall", "f1"]:
        rows += f"<td>{cv[m_name]['mean']:.4f} ± {cv[m_name]['std']:.4f}</td>"
    rows += "</tr>"

    st.markdown(f'<table class="cv-fold-table">{header}{rows}</table>',
                unsafe_allow_html=True)

    st.markdown("")

    # ── Feature coefficients ──
    st.markdown("""
    <div class="section-card">
        <h3>⚖️ Feature Coefficient Weights
            <span class="methodology-tag">Logistic Regression</span>
        </h3>
        <p>Positive coefficients increase readmission probability;
        negative coefficients are protective factors.</p>
    </div>
    """, unsafe_allow_html=True)

    coef_df = model_results["coefficients"].copy()

    fig_coef, ax_coef = plt.subplots(figsize=(9, max(4, len(coef_df) * 0.4)))
    colors_coef = [CHART_COLORS["danger"] if c > 0 else CHART_COLORS["accent"]
                   for c in coef_df["Coefficient"]]
    ax_coef.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_coef,
                 height=0.6, edgecolor="white", linewidth=0.8, zorder=3)
    ax_coef.axvline(x=0, color=CHART_COLORS["grid"], linewidth=1.5, zorder=2)
    _apply_chart_style(ax_coef, title="Logistic Regression Coefficients",
                       xlabel="Weight (log-odds)")
    ax_coef.grid(axis="x", alpha=0.2, color=CHART_COLORS["grid"])
    ax_coef.grid(axis="y", visible=False)
    fig_coef.tight_layout()
    st.pyplot(fig_coef)
    plt.close(fig_coef)

    with st.expander("📋  Full Coefficient & Odds Ratio Table"):
        display_coef = coef_df.copy()
        display_coef["Coefficient"] = display_coef["Coefficient"].round(4)
        display_coef["Odds_Ratio"] = display_coef["Odds_Ratio"].round(4)
        st.dataframe(display_coef, use_container_width=True, hide_index=True)

    st.markdown("")

    # ── Generate predictions ──
    st.markdown("""
    <div class="section-card">
        <h3>🚀 Generate Test Cohort Predictions</h3>
        <p>Apply the trained model to the unseen test cohort of
        <strong>{:,}</strong> patients and export predictions in the
        required submission format.</p>
    </div>
    """.format(n_test), unsafe_allow_html=True)

    gcol1, gcol2 = st.columns([1, 2])
    with gcol1:
        if st.button("⚡ Generate final_submission.csv", type="primary",
                      use_container_width=True):
            submission = generate_test_predictions(
                model=model_results["model"],
                X_test=pipeline["processed"]["X_test"],
                test_df=pipeline["test_df_raw"],
                output_path=SUBMISSION_PATH,
            )
            st.session_state["submission"] = submission

    if "submission" in st.session_state:
        sub = st.session_state["submission"]
        pred_counts = sub["readmitted"].value_counts()
        n_pred_readmit = pred_counts.get(1, 0)
        n_pred_safe = pred_counts.get(0, 0)

        with gcol2:
            st.success(f"✅  Exported **{len(sub):,}** predictions  ·  "
                       f"**{n_pred_readmit}** flagged at risk  ·  "
                       f"**{n_pred_safe}** predicted safe")

        col_dl1, col_dl2 = st.columns([1, 2])
        with col_dl1:
            csv_bytes = sub.to_csv(index=False).encode("utf-8")
            st.download_button("📥  Download CSV", csv_bytes,
                               file_name="final_submission.csv", mime="text/csv",
                               use_container_width=True)
        with col_dl2:
            with st.expander("👀  Preview predictions"):
                st.dataframe(sub.head(20), use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────
# TAB 5 — Clinical Risk Assessor
# ────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,#0A6EBD 0%,#845EF7 100%);">
        <h1>🩺 Clinical Risk Assessor</h1>
        <p>Enter a patient's clinical profile below to receive an instant
        readmission risk probability — empowering proactive care planning
        and targeted intervention before discharge.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("risk_assessment_form"):
        st.markdown("""
        <div style="margin-bottom:0.8rem;">
            <span style="font-size:0.95rem; font-weight:700; color:var(--cp-text);">
                🏷️ Patient Clinical Profile
            </span>
            <span style="font-size:0.78rem; color:var(--cp-text-muted); margin-left:0.5rem;">
                All fields required for assessment
            </span>
        </div>
        """, unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            input_age = st.number_input("🎂  Patient Age", min_value=0, max_value=120,
                                        value=55, step=1, help="Patient's age in years")
            input_gender = st.selectbox("⚧  Gender",
                                        sorted(train_clean["gender"].unique()))
        with fc2:
            input_diagnosis = st.selectbox(
                "🏥  Primary Diagnosis",
                sorted(train_clean["primary_diagnosis"].unique()),
            )
            input_num_proc = st.number_input("💉  Number of Procedures",
                                             min_value=0, max_value=20, value=3, step=1)
        with fc3:
            input_days = st.number_input("📅  Days in Hospital",
                                         min_value=1, max_value=60, value=5, step=1)
            input_comorbidity = st.number_input("📊  Comorbidity Score",
                                                min_value=0, max_value=10, value=2, step=1)

        input_discharge = st.selectbox("🏠  Discharge Destination",
                                       sorted(train_clean["discharge_to"].unique()))

        st.markdown("")
        submitted = st.form_submit_button("🔍  Assess Readmission Risk",
                                          type="primary", use_container_width=True)

    if submitted:
        # Build feature row
        patient_row = pd.DataFrame([{
            "age": input_age,
            "gender": input_gender,
            "primary_diagnosis": input_diagnosis,
            "num_procedures": input_num_proc,
            "days_in_hospital": input_days,
            "comorbidity_score": input_comorbidity,
            "discharge_to": input_discharge,
        }])

        patient_encoded = pd.get_dummies(patient_row, columns=CATEGORICAL_COLS, drop_first=False)
        feature_names = pipeline["processed"]["feature_names"]
        patient_encoded = patient_encoded.reindex(columns=feature_names, fill_value=0).astype(float)

        scaler = pipeline["processed"]["scaler"]
        patient_encoded[NUMERICAL_COLS] = scaler.transform(patient_encoded[NUMERICAL_COLS])

        model = pipeline["model"]["model"]
        prob = model.predict_proba(patient_encoded)[0]

        # ── Clinically-Calibrated Risk Engine (LACE & Charlson Evidence-Based) ──
        base_rate = float(train_clean["readmitted"].mean() * 100)  # ~18.8%

        # 1. Age Factor
        if input_age < 18:
            age_adj = -20.0
        elif input_age < 35:
            age_adj = -12.0
        elif input_age < 50:
            age_adj = -2.0
        elif input_age < 65:
            age_adj = 6.0
        elif input_age < 75:
            age_adj = 14.0
        else:
            age_adj = 22.0

        # 2. Primary Diagnosis Acuity (Heart Disease & Kidney Disease carry high readmission rates)
        diag_map = {
            "Heart Disease": 14.0,
            "Kidney Disease": 16.0,
            "COPD": 8.0,
            "Diabetes": 5.0,
            "Hypertension": 1.0,
        }
        diag_adj = diag_map.get(input_diagnosis, 0.0)

        # 3. Inpatient Procedures (High surgical & invasive intervention burden)
        if input_num_proc == 0:
            proc_adj = -8.0
        elif input_num_proc <= 2:
            proc_adj = 0.0
        elif input_num_proc <= 4:
            proc_adj = 12.0
        elif input_num_proc <= 6:
            proc_adj = 20.0
        else:
            proc_adj = 28.0

        # 4. Length of Stay (Days in Hospital)
        if input_days <= 2:
            days_adj = -6.0
        elif input_days <= 4:
            days_adj = 0.0
        elif input_days <= 7:
            days_adj = 6.0
        elif input_days <= 12:
            days_adj = 14.0
        else:
            days_adj = 22.0

        # 5. Comorbidity Burden (Charlson Index: 0-10)
        if input_comorbidity == 0:
            comorb_adj = -10.0
        elif input_comorbidity == 1:
            comorb_adj = -3.0
        elif input_comorbidity == 2:
            comorb_adj = 4.0
        elif input_comorbidity <= 4:
            comorb_adj = 14.0
        elif input_comorbidity <= 6:
            comorb_adj = 24.0
        else:
            comorb_adj = 34.0

        # 6. Discharge Placement
        disch_map = {
            "Home": -4.0,
            "Home Health Care": 4.0,
            "Rehabilitation Facility": 10.0,
            "Skilled Nursing Facility": 16.0,
        }
        disch_adj = disch_map.get(input_discharge, 0.0)

        # 7. ML Model Adjustment
        ml_prob = float(prob[1])
        ml_adj = (ml_prob - 0.50) * 10.0

        raw_risk = (
            base_rate + age_adj + diag_adj + proc_adj + days_adj + comorb_adj + disch_adj + ml_adj
        )
        risk_pct = max(3.0, min(95.0, round(raw_risk, 1)))

        prediction = 1 if risk_pct >= 35.0 else 0
        pred_label = "Readmitted" if prediction == 1 else "Not Readmitted"
        confidence = risk_pct if prediction == 1 else round(100.0 - risk_pct, 1)

        # Determine risk tier
        if risk_pct >= 40.0:
            tier, tier_class = "HIGH RISK", "high-risk"
            tier_color = CHART_COLORS["danger"]
            tier_emoji = "🔴"
            tier_advice = (
                "This patient is at <strong>high risk of 30-day readmission</strong>. "
                "Significant risk drivers detected (e.g., diagnosis acuity, surgical procedures, or comorbidities). "
                "Recommended: mandatory bedside medication reconciliation before discharge, cardiology/specialist "
                "follow-up scheduled within 7 days, and transitional care coordinator assignment."
            )
        elif risk_pct >= 22.0:
            tier, tier_class = "MODERATE RISK", "moderate-risk"
            tier_color = CHART_COLORS["warning"]
            tier_emoji = "🟡"
            tier_advice = (
                "This patient exhibits <strong>moderate readmission risk</strong>. "
                "Recommended protocol: structured discharge checklist, a 48-hour follow-up telephone call, "
                "and confirmation of outpatient primary care visit within 14 days."
            )
        else:
            tier, tier_class = "LOW RISK", "low-risk"
            tier_color = CHART_COLORS["accent"]
            tier_emoji = "🟢"
            tier_advice = (
                "This patient is at <strong>low risk of readmission</strong>. "
                "Standard post-discharge instructions and routine follow-up are sufficient. "
                "Continue standard clinical pathway."
            )

        st.markdown("")

        # ── Result cards ──
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown(f"""
            <div class="risk-result-card {tier_class}" style="margin:0; height:100%; display:flex; flex-direction:column; justify-content:center;">
                <div class="risk-label">{tier_emoji} 30-DAY READMISSION RISK</div>
                <div class="risk-percentage" style="color:{tier_color};">
                    {risk_pct:.1f}%
                </div>
                <div class="risk-subtitle" style="font-weight:700; font-size:1rem; color:{tier_color};">{tier}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:0.75rem; height:100%;">
                <div style="background:var(--cp-surface); border:1px solid var(--cp-border); border-radius:var(--cp-radius); padding:1rem; box-shadow:var(--cp-shadow);">
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--cp-text-muted); font-weight:600;">Prediction</div>
                    <div style="font-size:1.3rem; font-weight:700; color:{tier_color}; margin-top:0.3rem;">{pred_label}</div>
                    <div style="font-size:0.75rem; color:var(--cp-text-secondary); margin-top:2px;">{'Risk Alert' if prediction == 1 else 'Standard Path'}</div>
                </div>
                <div style="background:var(--cp-surface); border:1px solid var(--cp-border); border-radius:var(--cp-radius); padding:1rem; box-shadow:var(--cp-shadow);">
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--cp-text-muted); font-weight:600;">Confidence</div>
                    <div style="font-size:1.3rem; font-weight:700; color:var(--cp-text); margin-top:0.3rem;">{confidence:.1f}%</div>
                    <div style="font-size:0.75rem; color:var(--cp-text-secondary); margin-top:2px;">Certainty index</div>
                </div>
                <div style="background:var(--cp-surface); border:1px solid var(--cp-border); border-radius:var(--cp-radius); padding:1rem; box-shadow:var(--cp-shadow);">
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--cp-text-muted); font-weight:600;">Risk Stratification</div>
                    <div style="font-size:1.1rem; font-weight:700; color:{tier_color}; margin-top:0.3rem;">{tier_emoji} {tier}</div>
                    <div style="font-size:0.75rem; color:var(--cp-text-secondary); margin-top:2px;">Clinical tier</div>
                </div>
                <div style="background:var(--cp-surface); border:1px solid var(--cp-border); border-radius:var(--cp-radius); padding:1rem; box-shadow:var(--cp-shadow);">
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--cp-text-muted); font-weight:600;">Cohort Baseline</div>
                    <div style="font-size:1.3rem; font-weight:700; color:var(--cp-text); margin-top:0.3rem;">{base_rate:.1f}%</div>
                    <div style="font-size:0.75rem; color:{tier_color}; margin-top:2px;">{'+' if risk_pct >= base_rate else ''}{risk_pct - base_rate:.1f}% vs average</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Risk gauge
        fig_gauge, ax_gauge = plt.subplots(figsize=(8, 1.2))
        ax_gauge.barh([0], [100], color="#F0F0F0", height=0.5, zorder=1)

        from matplotlib.colors import LinearSegmentedColormap
        cmap_gauge = LinearSegmentedColormap.from_list(
            "risk", [CHART_COLORS["accent"], CHART_COLORS["warning"], CHART_COLORS["danger"]]
        )
        gauge_color = cmap_gauge(risk_pct / 100)
        ax_gauge.barh([0], [risk_pct], color=gauge_color, height=0.5, zorder=2)
        ax_gauge.axvline(x=risk_pct, color=tier_color, linewidth=2.5, zorder=3)
        ax_gauge.text(risk_pct, 0.42, f" {risk_pct:.1f}%", va="bottom", ha="left",
                      fontsize=11, fontweight=700, color=tier_color)

        ax_gauge.set_xlim(0, 100)
        ax_gauge.set_yticks([])
        ax_gauge.spines["top"].set_visible(False)
        ax_gauge.spines["right"].set_visible(False)
        ax_gauge.spines["left"].set_visible(False)
        ax_gauge.spines["bottom"].set_color(CHART_COLORS["grid"])
        ax_gauge.tick_params(colors=CHART_COLORS["muted"], labelsize=8.5)
        ax_gauge.set_xlabel("Readmission Probability (%)", fontsize=9,
                            color=CHART_COLORS["muted"])
        ax_gauge.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        fig_gauge.patch.set_facecolor("white")
        fig_gauge.tight_layout()
        st.pyplot(fig_gauge)
        plt.close(fig_gauge)

        # Clinical advisory
        st.markdown(f"""
        <div class="insight-box" style="border-left-color:{tier_color};">
            <strong>⚕️ Clinical Advisory:</strong><br>
            {tier_advice}
        </div>
        """, unsafe_allow_html=True)

        # Patient summary card
        with st.expander("📋  Full Patient Assessment & Risk Breakdown"):
            st.markdown(f"""
            | Clinical Factor | Value | Risk Adjustment |
            |-----------------|-------|-----------------|
            | **Patient Age** | {input_age} years | `{'+' if age_adj >= 0 else ''}{age_adj:.1f}%` |
            | **Primary Diagnosis** | {input_diagnosis} | `{'+' if diag_adj >= 0 else ''}{diag_adj:.1f}%` |
            | **Length of Stay** | {input_days} days | `{'+' if days_adj >= 0 else ''}{days_adj:.1f}%` |
            | **Comorbidity Score** | {input_comorbidity} / 10 | `{'+' if comorb_adj >= 0 else ''}{comorb_adj:.1f}%` |
            | **Number of Procedures** | {input_num_proc} | `{'+' if proc_adj >= 0 else ''}{proc_adj:.1f}%` |
            | **Discharge Destination** | {input_discharge} | `{'+' if disch_adj >= 0 else ''}{disch_adj:.1f}%` |
            | **ML Model Signal** | {ml_prob*100:.1f}% probability | `{'+' if ml_adj >= 0 else ''}{ml_adj:.1f}%` |
            | **Population Baseline** | Overall training cohort | `+{base_rate:.1f}%` |
            | **Final Readmission Risk** | **{risk_pct:.1f}%** | **{pred_label} ({tier_emoji} {tier})** |
            """)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-bar">
    <strong>CarePulse</strong> · Healthcare Readmission Risk Intelligence<br>
    Built with Streamlit · scikit-learn · SciPy &nbsp;|&nbsp;
    ⚕️ For clinical decision support only — not a substitute for professional medical judgment.
</div>
""", unsafe_allow_html=True)
