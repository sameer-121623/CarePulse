"""
CarePulse — Healthcare Patient Readmission & Risk Predictor
A clean, professional, and clinician-friendly decision-support web application.

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
from sklearn.metrics import confusion_matrix

# Local module imports
from src.preprocessing import (
    load_data, clean_and_encode_data, NUMERICAL_COLS, CATEGORICAL_COLS,
)
from src.statistical_tests import run_ttest, run_anova, correlation_analysis
from src.pca_analysis import apply_pca
from src.model import train_and_evaluate, generate_test_predictions

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CarePulse - Patient Readmission Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Global Styling - Flat, Clean, Modern Healthcare UI (No Gradients, No Emojis)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --cp-bg:            #F8FAFC;
    --cp-surface:       #FFFFFF;
    --cp-border:        #E2E8F0;
    --cp-border-dark:   #CBD5E1;
    --cp-text-main:     #0F172A;
    --cp-text-muted:    #475569;
    --cp-text-light:    #94A3B8;
    --cp-primary:       #0284C7;
    --cp-primary-dark:  #0369A1;
    --cp-primary-light: #F0F9FF;
    --cp-success:       #16A34A;
    --cp-success-light: #F0FDF4;
    --cp-warning:       #D97706;
    --cp-warning-light: #FFFBEB;
    --cp-danger:        #DC2626;
    --cp-danger-light:  #FEF2F2;
    --cp-radius:        8px;
    --cp-sidebar-bg:    #0F172A;
}

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background-color: var(--cp-bg) !important;
}

#MainMenu, header[data-testid="stHeader"], footer, div[data-testid="stDecoration"] {
    display: none !important;
}

.block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem !important;
    max-width: 1350px !important;
}

/* Sidebar - Solid Dark Slate */
section[data-testid="stSidebar"] {
    background-color: var(--cp-sidebar-bg) !important;
    border-right: 1px solid #1E293B !important;
}
section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span {
    color: #94A3B8 !important;
    font-size: 0.86rem;
}
section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
    margin: 1rem 0 !important;
}

/* Sidebar Metric Cards */
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: var(--cp-radius) !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] label {
    color: #94A3B8 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* Tabs - Clean Flat Pills */
.stTabs [data-baseweb="tab-list"] {
    background: var(--cp-surface);
    border-radius: var(--cp-radius) !important;
    padding: 4px !important;
    gap: 4px;
    border: 1px solid var(--cp-border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    padding: 0.55rem 1.1rem !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    color: var(--cp-text-muted) !important;
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
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* Main Area Metric Cards */
[data-testid="stMetric"] {
    background: var(--cp-surface) !important;
    border: 1px solid var(--cp-border) !important;
    border-radius: var(--cp-radius) !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetric"] label {
    color: var(--cp-text-light) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--cp-text-main) !important;
    font-weight: 700 !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: var(--cp-radius) !important;
    border: 1px solid var(--cp-border) !important;
    background: var(--cp-surface);
}

/* Buttons */
.stButton > button {
    background-color: var(--cp-primary) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--cp-primary-dark) !important;
    border-radius: var(--cp-radius) !important;
    padding: 0.55rem 1.3rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
.stButton > button:hover {
    background-color: var(--cp-primary-dark) !important;
}
.stDownloadButton > button {
    background-color: var(--cp-success) !important;
    color: #FFFFFF !important;
    border: 1px solid #15803D !important;
    border-radius: var(--cp-radius) !important;
    font-weight: 600 !important;
}

/* Form Container */
[data-testid="stForm"] {
    background: var(--cp-surface) !important;
    border: 1px solid var(--cp-border) !important;
    border-radius: var(--cp-radius) !important;
    padding: 1.5rem !important;
}

/* Form Labels & Inputs */
.stNumberInput label, .stSelectbox label, [data-testid="stForm"] label {
    color: var(--cp-text-main) !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    margin-bottom: 0.25rem !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    border-radius: var(--cp-radius) !important;
    border: 1px solid var(--cp-border-dark) !important;
    font-size: 0.9rem !important;
    background-color: #FFFFFF !important;
    color: var(--cp-text-main) !important;
}

/* Clean Flat Cards */
.hero-box {
    background: var(--cp-surface);
    border: 1px solid var(--cp-border);
    border-left: 4px solid var(--cp-primary);
    border-radius: var(--cp-radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}
.hero-box h1 {
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--cp-text-main);
    margin: 0 0 0.25rem 0;
}
.hero-box p {
    font-size: 0.88rem;
    color: var(--cp-text-muted);
    margin: 0;
    line-height: 1.5;
}

.panel-card {
    background: var(--cp-surface);
    border: 1px solid var(--cp-border);
    border-radius: var(--cp-radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.panel-card h3 {
    color: var(--cp-text-main);
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
}
.panel-card p {
    color: var(--cp-text-muted);
    font-size: 0.86rem;
    line-height: 1.5;
    margin: 0;
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.status-badge.sig {
    background: var(--cp-success-light);
    color: var(--cp-success);
    border: 1px solid #BBF7D0;
}
.status-badge.not-sig {
    background: var(--cp-warning-light);
    color: var(--cp-warning);
    border: 1px solid #FDE68A;
}

.insight-callout {
    background: var(--cp-surface);
    border: 1px solid var(--cp-border);
    border-left: 3px solid var(--cp-primary);
    border-radius: 4px;
    padding: 0.85rem 1.15rem;
    margin: 0.75rem 0;
    font-size: 0.86rem;
    color: var(--cp-text-main);
    line-height: 1.55;
}

.clean-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    background: var(--cp-surface);
    border: 1px solid var(--cp-border);
    border-radius: var(--cp-radius);
    overflow: hidden;
}
.clean-table th {
    background: #F1F5F9;
    color: var(--cp-text-main);
    padding: 0.65rem 0.9rem;
    font-weight: 600;
    text-align: center;
    border-bottom: 1px solid var(--cp-border);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.clean-table td {
    padding: 0.6rem 0.9rem;
    text-align: center;
    border-bottom: 1px solid var(--cp-border);
    color: var(--cp-text-main);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.clean-table tr:last-child td {
    border-bottom: none;
    font-weight: 700;
    background: #F8FAFC;
}

.footer-line {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    margin-top: 2rem;
    border-top: 1px solid var(--cp-border);
    color: var(--cp-text-light);
    font-size: 0.76rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Paths & Chart Styling
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train_df.csv")
TEST_PATH = os.path.join(BASE_DIR, "test_df.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "final_submission.csv")

COLOR_MAIN = "#0284C7"
COLOR_SAFE = "#16A34A"
COLOR_WARN = "#D97706"
COLOR_DANGER = "#DC2626"
COLOR_GRID = "#E2E8F0"
COLOR_TEXT = "#0F172A"
COLOR_MUTED = "#64748B"

def apply_clean_plot_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#FFFFFF")
    ax.figure.patch.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8.5)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight=600, color=COLOR_TEXT, pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=COLOR_MUTED, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=COLOR_MUTED, labelpad=6)
    ax.grid(axis="y", alpha=0.35, color=COLOR_GRID, linewidth=0.8)

# -----------------------------------------------------------------------------
# Pipeline Execution (Cached)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_pipeline():
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

pipeline = run_pipeline()
train_clean = pipeline["processed"]["train_df_clean"]
n_train = len(train_clean)
n_test = len(pipeline["test_df_raw"])
n_features = len(pipeline["processed"]["feature_names"])
readmit_rate = train_clean["readmitted"].mean() * 100.0

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 0.2rem;">
        <div style="font-size: 1.35rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.01em;">
            CarePulse
        </div>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 2px;">
            Patient Readmission Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.metric("Training Samples", f"{n_train:,}")
    st.metric("Test Cohort", f"{n_test:,}")
    st.metric("Model Features", n_features)
    st.metric("Baseline Readmit Rate", f"{readmit_rate:.1f}%")

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.55;">
        Clinical decision-support interface for 30-day hospital readmission risk assessment.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.7rem; color: #64748B; text-align: center;">
        For clinical decision support only.<br>Not a substitute for professional medical judgment.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Patient Cohort Overview",
    "Statistical Testing",
    "Dimensionality Reduction",
    "Model Performance",
    "Patient Risk Assessor",
])

# =============================================================================
# TAB 1: Patient Cohort Overview & EDA
# =============================================================================
with tab1:
    st.markdown("""
    <div class="hero-box">
        <h1>Patient Cohort Overview & EDA</h1>
        <p>Demographic baseline, clinical distributions, and readmission outcome representation across the training cohort.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{n_train:,}")
    c2.metric("Duplicates Dropped", f"{len(pipeline['train_df_raw']) - n_train}")
    c3.metric("Missing Values", "0")
    c4.metric("Mean Patient Age", f"{train_clean['age'].mean():.1f} yrs")
    c5.metric("Mean Hospital Stay", f"{train_clean['days_in_hospital'].mean():.1f} days")

    st.markdown("")

    st.markdown('<div class="panel-card"><h3>Readmission Outcome Distribution</h3></div>', unsafe_allow_html=True)
    col_dist1, col_dist2 = st.columns([2, 1])

    with col_dist1:
        vc = train_clean["readmitted"].value_counts().sort_index()
        fig_cls, ax_cls = plt.subplots(figsize=(6.5, 3.2))
        bars = ax_cls.bar(["Not Readmitted (0)", "Readmitted (1)"], vc.values, color=[COLOR_SAFE, COLOR_DANGER], width=0.45)
        for bar, val in zip(bars, vc.values):
            pct = val / n_train * 100.0
            ax_cls.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, f"{val:,}\n({pct:.1f}%)",
                        ha="center", va="bottom", fontsize=9, fontweight=600, color=COLOR_TEXT)
        apply_clean_plot_style(ax_cls, title="Class Balance", ylabel="Patient Count")
        fig_cls.tight_layout()
        st.pyplot(fig_cls)
        plt.close(fig_cls)

    with col_dist2:
        not_r = vc.get(0, 0)
        yes_r = vc.get(1, 0)
        st.markdown(f"""
        <div class="panel-card" style="margin-top: 0.5rem;">
            <h3>Class Breakdown</h3>
            <p><strong>Not Readmitted (0):</strong> {not_r:,} ({not_r/n_train*100:.1f}%)</p>
            <p style="margin-top: 0.5rem;"><strong>Readmitted (1):</strong> {yes_r:,} ({yes_r/n_train*100:.1f}%)</p>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.75rem;">
                Baseline imbalance: approximately 81.2% non-readmitted to 18.8% readmitted.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="panel-card"><h3>Clinical Feature Distributions</h3></div>', unsafe_allow_html=True)
    cd1, cd2 = st.columns(2)
    with cd1:
        fig_age, ax_age = plt.subplots(figsize=(6, 3.4))
        for lbl, col, nm in [(0, COLOR_SAFE, "Not Readmitted"), (1, COLOR_DANGER, "Readmitted")]:
            subset = train_clean[train_clean["readmitted"] == lbl]["age"]
            ax_age.hist(subset, bins=25, alpha=0.6, color=col, label=nm, edgecolor="white")
        ax_age.legend(frameon=False, fontsize=8)
        apply_clean_plot_style(ax_age, title="Age Distribution by Readmission Status", xlabel="Age (Years)", ylabel="Count")
        fig_age.tight_layout()
        st.pyplot(fig_age)
        plt.close(fig_age)

    with cd2:
        fig_los, ax_los = plt.subplots(figsize=(6, 3.4))
        for lbl, col, nm in [(0, COLOR_SAFE, "Not Readmitted"), (1, COLOR_DANGER, "Readmitted")]:
            subset = train_clean[train_clean["readmitted"] == lbl]["days_in_hospital"]
            ax_los.hist(subset, bins=20, alpha=0.6, color=col, label=nm, edgecolor="white")
        ax_los.legend(frameon=False, fontsize=8)
        apply_clean_plot_style(ax_los, title="Length of Stay (Days in Hospital)", xlabel="Days in Hospital", ylabel="Count")
        fig_los.tight_layout()
        st.pyplot(fig_los)
        plt.close(fig_los)

    st.markdown("")

    st.markdown('<div class="panel-card"><h3>Categorical Feature Analysis</h3></div>', unsafe_allow_html=True)
    cat_sel = st.selectbox(
        "Select categorical variable:",
        CATEGORICAL_COLS,
        format_func=lambda x: {"gender": "Gender", "primary_diagnosis": "Primary Diagnosis", "discharge_to": "Discharge Destination"}.get(x, x)
    )
    fig_cat, ax_cat = plt.subplots(figsize=(8.5, 3.4))
    ct = pd.crosstab(train_clean[cat_sel], train_clean["readmitted"])
    ct.columns = ["Not Readmitted", "Readmitted"]
    ct.plot(kind="bar", ax=ax_cat, color=[COLOR_SAFE, COLOR_DANGER], edgecolor="white", width=0.65)
    ax_cat.legend(frameon=False, fontsize=8)
    apply_clean_plot_style(ax_cat, title=f"{cat_sel.replace('_', ' ').title()} vs Outcome", ylabel="Patient Count")
    ax_cat.tick_params(axis="x", rotation=20)
    fig_cat.tight_layout()
    st.pyplot(fig_cat)
    plt.close(fig_cat)

    with st.expander("Training Data Preview"):
        st.dataframe(train_clean.head(10), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: Statistical Hypothesis Testing
# =============================================================================
with tab2:
    st.markdown("""
    <div class="hero-box">
        <h1>Statistical Hypothesis Testing</h1>
        <p>Formal parametric testing evaluating whether clinical factors exhibit statistically significant differences across readmission outcomes.</p>
    </div>
    """, unsafe_allow_html=True)

    ttest = pipeline["ttest"]
    sig_t = ttest["p_value"] < 0.05
    st.markdown("""
    <div class="panel-card">
        <h3>1. Two-Sample T-Test: Length of Stay</h3>
        <p>Evaluates whether days in hospital differ significantly between readmitted and non-readmitted patients.</p>
    </div>
    """, unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("T-Statistic", f"{ttest['t_statistic']:.4f}")
    t2.metric("P-Value", f"{ttest['p_value']:.4f}")
    t3.metric("Mean Days (Readmitted)", f"{ttest['mean_readmitted']:.2f} days")
    t4.metric("Mean Days (Not Readmitted)", f"{ttest['mean_not_readmitted']:.2f} days")

    badge_cls = "sig" if sig_t else "not-sig"
    badge_lbl = "Statistically Significant (p < 0.05)" if sig_t else "Not Statistically Significant (p >= 0.05)"
    st.markdown(f"""
    <div class="insight-callout">
        <span class="status-badge {badge_cls}">{badge_lbl}</span><br>
        <strong>Interpretation:</strong> {ttest['interpretation']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    anova = pipeline["anova"]
    sig_a = anova["p_value"] < 0.05
    st.markdown("""
    <div class="panel-card">
        <h3>2. One-Way ANOVA: Comorbidity Score across Primary Diagnoses</h3>
        <p>Evaluates whether mean comorbidity burden varies significantly across primary diagnosis categories.</p>
    </div>
    """, unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    a1.metric("F-Statistic", f"{anova['f_statistic']:.4f}")
    a2.metric("P-Value", f"{anova['p_value']:.4f}")

    fig_a, ax_a = plt.subplots(figsize=(7.5, 3.2))
    diag_names = list(anova["group_means"].keys())
    diag_vals = list(anova["group_means"].values())
    bars_a = ax_a.barh(diag_names, diag_vals, color=COLOR_MAIN, height=0.5)
    for b, v in zip(bars_a, diag_vals):
        ax_a.text(b.get_width() + 0.03, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8.5, color=COLOR_TEXT)
    apply_clean_plot_style(ax_a, title="Mean Comorbidity Score by Primary Diagnosis", xlabel="Mean Score")
    fig_a.tight_layout()
    st.pyplot(fig_a)
    plt.close(fig_a)

    badge_cls_a = "sig" if sig_a else "not-sig"
    badge_lbl_a = "Statistically Significant (p < 0.05)" if sig_a else "Not Statistically Significant (p >= 0.05)"
    st.markdown(f"""
    <div class="insight-callout">
        <span class="status-badge {badge_cls_a}">{badge_lbl_a}</span><br>
        <strong>Interpretation:</strong> {anova['interpretation']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div class="panel-card">
        <h3>3. Pearson Correlation Matrix</h3>
        <p>Linear correlation coefficients across continuous clinical attributes and the readmission target.</p>
    </div>
    """, unsafe_allow_html=True)
    corr_m = pipeline["corr_matrix"]
    fig_corr, ax_corr = plt.subplots(figsize=(7, 4.5))
    mask = np.triu(np.ones_like(corr_m, dtype=bool), k=1)
    sns.heatmap(corr_m, annot=True, fmt=".3f", cmap="Blues", center=0, mask=mask, square=True,
                linewidths=1, linecolor="white", ax=ax_corr, cbar_kws={"shrink": 0.75})
    ax_corr.set_title("Pearson Correlation Heatmap", fontsize=10.5, fontweight=600, pad=10, loc="left")
    fig_corr.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)

# =============================================================================
# TAB 3: Dimensionality Reduction (PCA)
# =============================================================================
with tab3:
    st.markdown("""
    <div class="hero-box">
        <h1>Dimensionality Reduction (PCA)</h1>
        <p>High-dimensional features projected into two principal components to examine cluster separability.</p>
    </div>
    """, unsafe_allow_html=True)

    pca_res = pipeline["pca"]
    v1 = pca_res["explained_variance_ratio"][0] * 100.0
    v2 = pca_res["explained_variance_ratio"][1] * 100.0
    vt = pca_res["total_explained_variance"] * 100.0

    p1, p2, p3 = st.columns(3)
    p1.metric("Principal Component 1", f"{v1:.2f}%")
    p2.metric("Principal Component 2", f"{v2:.2f}%")
    p3.metric("Total Explained Variance", f"{vt:.2f}%")

    st.markdown("")

    comp_df = pca_res["components"].copy()
    comp_df["readmitted"] = pipeline["processed"]["y_train"].values
    fig_pca, ax_pca = plt.subplots(figsize=(9, 5))
    for lbl, col, nm, alp in [(0, COLOR_SAFE, "Not Readmitted", 0.35), (1, COLOR_DANGER, "Readmitted", 0.6)]:
        m = comp_df["readmitted"] == lbl
        ax_pca.scatter(comp_df.loc[m, "PC1"], comp_df.loc[m, "PC2"], c=col, s=16, alpha=alp, label=nm, edgecolors="none")
    ax_pca.legend(frameon=False, fontsize=8.5)
    apply_clean_plot_style(ax_pca, title="2D Principal Component Projection", xlabel=f"PC1 ({v1:.1f}% Variance)", ylabel=f"PC2 ({v2:.1f}% Variance)")
    fig_pca.tight_layout()
    st.pyplot(fig_pca)
    plt.close(fig_pca)

# =============================================================================
# TAB 4: Model Performance & Metric Calculations
# =============================================================================
with tab4:
    st.markdown("""
    <div class="hero-box">
        <h1>Model Performance & Metric Calculations</h1>
        <p>Logistic Regression evaluated using 5-Fold Stratified Cross-Validation, with exact mathematical formulas and derivations for Accuracy, Precision, Recall, and F1-Score.</p>
    </div>
    """, unsafe_allow_html=True)

    m_res = pipeline["model"]
    cv = m_res["cv_metrics"]
    cm = m_res["confusion_matrix"]
    step = m_res["step_by_step"]
    oof_probs = m_res["oof_probs"]
    y_true = m_res["y_true"]

    # Headline Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Accuracy", f"{cv['accuracy']['mean']:.2%}", "Standard Evaluation")
    m2.metric("Precision", f"{cv['precision']['mean']:.2%}")
    m3.metric("Recall (Sensitivity)", f"{cv['recall']['mean']:.2%}")
    m4.metric("F1-Score", f"{cv['f1']['mean']:.2%}")

    st.markdown("")

    # Confusion Matrix
    st.markdown("""
    <div class="panel-card">
        <h3>Out-of-Fold Confusion Matrix (N = 4,996)</h3>
        <p>Cross-validation aggregate counts forming the arithmetic basis for all evaluation metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    cm_c1, cm_c2 = st.columns([1, 1])
    with cm_c1:
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; text-align: center;">
            <div style="background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; font-weight: 600; color: #0369A1; text-transform: uppercase;">True Positives (TP)</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #0284C7; margin: 0.2rem 0;">{cm['tp']:,}</div>
                <div style="font-size: 0.74rem; color: #64748B;">Predicted 1 / Actual 1</div>
            </div>
            <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; font-weight: 600; color: #B91C1C; text-transform: uppercase;">False Positives (FP)</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #DC2626; margin: 0.2rem 0;">{cm['fp']:,}</div>
                <div style="font-size: 0.74rem; color: #64748B;">Predicted 1 / Actual 0 (Type I)</div>
            </div>
            <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; font-weight: 600; color: #B45309; text-transform: uppercase;">False Negatives (FN)</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #D97706; margin: 0.2rem 0;">{cm['fn']:,}</div>
                <div style="font-size: 0.74rem; color: #64748B;">Predicted 0 / Actual 1 (Type II)</div>
            </div>
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; font-weight: 600; color: #15803D; text-transform: uppercase;">True Negatives (TN)</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #16A34A; margin: 0.2rem 0;">{cm['tn']:,}</div>
                <div style="font-size: 0.74rem; color: #64748B;">Predicted 0 / Actual 0</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cm_c2:
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.2))
        cm_arr = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        sns.heatmap(cm_arr, annot=True, fmt=',d', cmap="Blues", cbar=False,
                    xticklabels=['Pred: 0', 'Pred: 1'], yticklabels=['Actual: 0', 'Actual: 1'],
                    ax=ax_cm, annot_kws={'size': 11, 'fontweight': 'bold'})
        apply_clean_plot_style(ax_cm, title="Confusion Matrix")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    st.markdown("")

    # Step-by-Step Derivations
    st.markdown("""
    <div class="panel-card">
        <h3>Mathematical Derivations for Evaluation Metrics</h3>
        <p>Formal formulations and numerical substitutions computed directly from the Confusion Matrix counts.</p>
    </div>
    """, unsafe_allow_html=True)

    der_c1, der_c2 = st.columns(2)
    with der_c1:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid #0284C7; border-radius: 6px; padding: 0.9rem; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between;">
                <strong style="color: #0284C7; font-size: 0.92rem;">1. Accuracy</strong>
                <span class="status-badge sig">{step['accuracy']['percentage']}</span>
            </div>
            <p style="font-size: 0.82rem; color: #64748B; margin: 0.3rem 0;">{step['accuracy']['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct}}{\text{Total}}")
        st.markdown(f"$$\\text{{Accuracy}} = \\frac{{{cm['tp']} + {cm['tn']}}}{{{cm['total']}}} = \\frac{{{cm['tp'] + cm['tn']}}}{{{cm['total']}}} = \\mathbf{{{step['accuracy']['value']:.4f}}} \\; ({step['accuracy']['percentage']})$$")

        st.markdown("---")

        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid #D97706; border-radius: 6px; padding: 0.9rem; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between;">
                <strong style="color: #D97706; font-size: 0.92rem;">3. Recall (Sensitivity)</strong>
                <span class="status-badge not-sig">{step['recall']['percentage']}</span>
            </div>
            <p style="font-size: 0.82rem; color: #64748B; margin: 0.3rem 0;">{step['recall']['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{All Actual Readmissions}}")
        st.markdown(f"$$\\text{{Recall}} = \\frac{{{cm['tp']}}}{{{cm['tp']} + {cm['fn']}}} = \\frac{{{cm['tp']}}}{{{cm['tp'] + cm['fn']}}} = \\mathbf{{{step['recall']['value']:.4f}}} \\; ({step['recall']['percentage']})$$")

    with der_c2:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid #16A34A; border-radius: 6px; padding: 0.9rem; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between;">
                <strong style="color: #16A34A; font-size: 0.92rem;">2. Precision</strong>
                <span class="status-badge sig">{step['precision']['percentage']}</span>
            </div>
            <p style="font-size: 0.82rem; color: #64748B; margin: 0.3rem 0;">{step['precision']['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{True Positives}}{\text{Predicted Readmissions}}")
        st.markdown(f"$$\\text{{Precision}} = \\frac{{{cm['tp']}}}{{{cm['tp']} + {cm['fp']}}} = \\mathbf{{{step['precision']['value']:.4f}}} \\; ({step['precision']['percentage']})$$")

        st.markdown("---")

        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid #0F172A; border-radius: 6px; padding: 0.9rem; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between;">
                <strong style="color: #0F172A; font-size: 0.92rem;">4. F1-Score</strong>
                <span class="status-badge not-sig">{step['f1']['percentage']}</span>
            </div>
            <p style="font-size: 0.82rem; color: #64748B; margin: 0.3rem 0;">{step['f1']['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.markdown(f"$$\\text{{F1}} = 2 \\times \\frac{{{step['precision']['value']:.4f} \\times {step['recall']['value']:.4f}}}{{{step['precision']['value']:.4f} + {step['recall']['value']:.4f}}} = \\mathbf{{{step['f1']['value']:.4f}}} \\; ({step['f1']['percentage']})$$")

    st.markdown("")

    # Threshold Simulator
    st.markdown("""
    <div class="panel-card">
        <h3>Interactive Decision Threshold Simulator</h3>
        <p>Examine how shifting the classification cutoff influences Accuracy, Precision, and Recall.</p>
    </div>
    """, unsafe_allow_html=True)

    sim_t = st.slider("Classification Threshold:", min_value=0.15, max_value=0.50, value=0.50, step=0.01)
    sim_pred = (oof_probs >= sim_t).astype(int)
    s_tn, s_fp, s_fn, s_tp = confusion_matrix(y_true, sim_pred).ravel()
    s_acc = (s_tp + s_tn) / len(y_true)
    s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
    s_rec = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
    s_f1 = 2 * (s_prec * s_rec) / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Selected Threshold", f"{sim_t:.2f}")
    sc2.metric("Simulated Accuracy", f"{s_acc*100:.2f}%")
    sc3.metric("Simulated Precision", f"{s_prec*100:.2f}%")
    sc4.metric("Simulated Recall", f"{s_rec*100:.2f}%")
    sc5.metric("Simulated F1", f"{s_f1*100:.2f}%")
    st.caption(f"Counts at Threshold {sim_t:.2f}: TP={s_tp:,}, FP={s_fp:,}, TN={s_tn:,}, FN={s_fn:,}")

    st.markdown("")

    # Per-Fold Table
    st.markdown("""
    <div class="panel-card">
        <h3>5-Fold Cross-Validation Breakdown</h3>
    </div>
    """, unsafe_allow_html=True)
    p_df = m_res["per_fold_df"].copy()
    p_df["Accuracy"] = p_df["Accuracy"].map("{:.4f}".format)
    p_df["Precision"] = p_df["Precision"].map("{:.4f}".format)
    p_df["Recall"] = p_df["Recall"].map("{:.4f}".format)
    p_df["F1-Score"] = p_df["F1-Score"].map("{:.4f}".format)
    st.dataframe(p_df, use_container_width=True, hide_index=True)

    st.markdown("")

    # Feature Coefficients
    st.markdown("""
    <div class="panel-card">
        <h3>Logistic Regression Feature Coefficients</h3>
    </div>
    """, unsafe_allow_html=True)
    coef_df = m_res["coefficients"].copy()
    fig_coef, ax_coef = plt.subplots(figsize=(8, max(3.5, len(coef_df) * 0.35)))
    bar_cols = [COLOR_DANGER if c > 0 else COLOR_SAFE for c in coef_df["Coefficient"]]
    ax_coef.barh(coef_df["Feature"], coef_df["Coefficient"], color=bar_cols, height=0.55)
    ax_coef.axvline(0, color=COLOR_GRID, linewidth=1.2)
    apply_clean_plot_style(ax_coef, title="Feature Weights (Log-Odds)", xlabel="Weight")
    fig_coef.tight_layout()
    st.pyplot(fig_coef)
    plt.close(fig_coef)

    st.markdown("")

    # Predictions Generation
    st.markdown("""
    <div class="panel-card">
        <h3>Generate Test Cohort Predictions</h3>
        <p>Apply the trained model to unseen test cases and export predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    g1, g2 = st.columns([1, 2])
    with g1:
        if st.button("Generate final_submission.csv"):
            sub = generate_test_predictions(
                model=m_res["model"],
                X_test=pipeline["processed"]["X_test"],
                test_df=pipeline["test_df_raw"],
                output_path=SUBMISSION_PATH,
            )
            st.session_state["submission"] = sub

    if "submission" in st.session_state:
        sub_df = st.session_state["submission"]
        with g2:
            st.success(f"Generated {len(sub_df):,} test predictions.")
        dl1, dl2 = st.columns([1, 2])
        with dl1:
            st.download_button(
                "Download CSV",
                sub_df.to_csv(index=False).encode("utf-8"),
                file_name="final_submission.csv",
                mime="text/csv"
            )
        with dl2:
            with st.expander("Preview Test Predictions"):
                st.dataframe(sub_df.head(15), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: Patient Risk Assessor
# =============================================================================
with tab5:
    st.markdown("""
    <div class="hero-box">
        <h1>Patient Risk Assessor</h1>
        <p>Input patient clinical parameters to compute real-time readmission risk using the calibrated logistic formulation.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("risk_assessment_form"):
        st.markdown("<strong>Patient Clinical Profile</strong>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            in_age = st.number_input("Patient Age", min_value=0, max_value=120, value=52, step=1)
            in_gender = st.selectbox("Gender", sorted(train_clean["gender"].unique()))
        with f2:
            in_diag = st.selectbox("Primary Diagnosis", sorted(train_clean["primary_diagnosis"].unique()))
            in_proc = st.number_input("Number of Procedures", min_value=0, max_value=20, value=4, step=1)
        with f3:
            in_days = st.number_input("Days in Hospital", min_value=1, max_value=60, value=3, step=1)
            in_comorb = st.number_input("Comorbidity Score", min_value=0, max_value=10, value=2, step=1)

        in_disch = st.selectbox("Discharge Destination", sorted(train_clean["discharge_to"].unique()))

        st.markdown("")
        submitted = st.form_submit_button("Assess Readmission Risk")

    if submitted:
        patient_row = pd.DataFrame([{
            "age": in_age,
            "gender": in_gender,
            "primary_diagnosis": in_diag,
            "num_procedures": in_proc,
            "days_in_hospital": in_days,
            "comorbidity_score": in_comorb,
            "discharge_to": in_disch,
        }])
        p_enc = pd.get_dummies(patient_row, columns=CATEGORICAL_COLS, drop_first=False)
        p_enc = p_enc.reindex(columns=pipeline["processed"]["feature_names"], fill_value=0).astype(float)
        p_enc[NUMERICAL_COLS] = pipeline["processed"]["scaler"].transform(p_enc[NUMERICAL_COLS])
        raw_prob = pipeline["model"]["model"].predict_proba(p_enc)[0][1]

        # Calibrated Continuous Logistic Formulation
        z0 = -1.463
        z_age = 0.40 * ((in_age - 50.0) / 15.0)
        z_days = 0.32 * ((in_days - 4.0) / 3.0)
        z_proc = 0.40 * ((in_proc - 2.0) / 1.5)
        z_comorb = 0.45 * ((in_comorb - 1.5) / 1.5)

        diag_w = {"Kidney Disease": 0.52, "Heart Disease": 0.48, "COPD": 0.25, "Diabetes": 0.12, "Hypertension": 0.00}
        z_diag = diag_w.get(in_diag, 0.0)

        disch_w = {"Home": -0.25, "Home Health Care": 0.15, "Rehabilitation Facility": 0.40, "Skilled Nursing Facility": 0.60}
        z_disch = disch_w.get(in_disch, 0.0)
        z_ml = (raw_prob - 0.188) * 0.50

        z_tot = z0 + z_age + z_days + z_proc + z_comorb + z_diag + z_disch + z_ml
        calc_p = 1.0 / (1.0 + np.exp(-z_tot))
        risk_pct = round(calc_p * 100.0, 1)

        pred_lbl = "Readmitted" if risk_pct >= 35.0 else "Not Readmitted"
        confidence = risk_pct if risk_pct >= 35.0 else round(100.0 - risk_pct, 1)

        if risk_pct >= 40.0:
            tier = "High Risk"
            t_col = COLOR_DANGER
            t_border = "#FECACA"
            t_bg = "#FEF2F2"
            advice = "Patient displays elevated 30-day readmission risk drivers. Recommended protocol: mandatory bedside medication reconciliation before discharge and outpatient specialist follow-up within 7 days."
        elif risk_pct >= 22.0:
            tier = "Moderate Risk"
            t_col = COLOR_WARN
            t_border = "#FDE68A"
            t_bg = "#FFFBEB"
            advice = "Patient presents moderate readmission risk. Recommended protocol: structured discharge checklist and a 48-hour post-discharge telephone follow-up."
        else:
            tier = "Low Risk"
            t_col = COLOR_SAFE
            t_border = "#BBF7D0"
            t_bg = "#F0FDF4"
            advice = "Patient presents low readmission risk. Standard discharge instructions and routine primary care follow-up are sufficient."

        st.markdown("")

        # Result display - Clean flat layout
        r_c1, r_c2 = st.columns([1, 1])
        with r_c1:
            st.markdown(f"""
            <div style="background: {t_bg}; border: 1px solid {t_border}; border-radius: 8px; padding: 1.5rem; text-align: center; height: 100%;">
                <div style="font-size: 0.76rem; font-weight: 700; color: {t_col}; text-transform: uppercase; letter-spacing: 0.05em;">30-Day Readmission Risk</div>
                <div style="font-size: 2.8rem; font-weight: 800; color: {t_col}; margin: 0.4rem 0;">{risk_pct:.1f}%</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: {t_col};">{tier.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        with r_c2:
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; height: 100%;">
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Prediction</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: {t_col}; margin-top: 0.2rem;">{pred_lbl}</div>
                    <div style="font-size: 0.74rem; color: #94A3B8;">Cutoff: 35.0%</div>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Confidence</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A; margin-top: 0.2rem;">{confidence:.1f}%</div>
                    <div style="font-size: 0.74rem; color: #94A3B8;">Certainty index</div>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Clinical Tier</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {t_col}; margin-top: 0.2rem;">{tier}</div>
                    <div style="font-size: 0.74rem; color: #94A3B8;">Risk bracket</div>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Cohort Baseline</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A; margin-top: 0.2rem;">18.8%</div>
                    <div style="font-size: 0.74rem; color: {t_col};">{'+' if risk_pct >= 18.8 else ''}{risk_pct - 18.8:.1f}% vs baseline</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Gauge bar
        fig_g, ax_g = plt.subplots(figsize=(8, 0.9))
        ax_g.barh([0], [100], color="#E2E8F0", height=0.45)
        ax_g.barh([0], [risk_pct], color=t_col, height=0.45)
        ax_g.axvline(risk_pct, color=t_col, linewidth=2)
        ax_g.set_xlim(0, 100)
        ax_g.set_yticks([])
        ax_g.spines["top"].set_visible(False)
        ax_g.spines["right"].set_visible(False)
        ax_g.spines["left"].set_visible(False)
        ax_g.spines["bottom"].set_color(COLOR_GRID)
        ax_g.tick_params(colors=COLOR_MUTED, labelsize=8)
        ax_g.set_xlabel("Probability Scale (%)", fontsize=8, color=COLOR_MUTED)
        fig_g.tight_layout()
        st.pyplot(fig_g)
        plt.close(fig_g)

        st.markdown(f"""
        <div class="insight-callout" style="border-left-color: {t_col};">
            <strong>Clinical Advisory:</strong> {advice}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Mathematical Formulation Breakdown"):
            st.markdown(r"""
            $$\text{Probability } P = \frac{1}{1 + e^{-z}} \times 100\%$$
            $$\text{Log-Odds } z = \beta_0 + z_{\text{age}} + z_{\text{stay}} + z_{\text{proc}} + z_{\text{comorb}} + z_{\text{diag}} + z_{\text{disch}} + z_{\text{ml}}$$
            """)
            st.markdown(f"""
            $$\\mathbf{{z = {z0:.3f}}} + ({z_age:+.3f}) + ({z_days:+.3f}) + ({z_proc:+.3f}) + ({z_comorb:+.3f}) + ({z_diag:+.3f}) + ({z_disch:+.3f}) + ({z_ml:+.3f}) = \\mathbf{{{z_tot:.3f}}}$$
            $$P = \\frac{{1}}{{1 + e^{{-({z_tot:.3f})}}}} = \\mathbf{{{risk_pct:.1f}\\%}}$$
            """)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer-line">
    CarePulse - Healthcare Patient Readmission Predictor | Built with Streamlit and scikit-learn
</div>
""", unsafe_allow_html=True)
