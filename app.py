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
# Global Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:           #F8FAFC;
    --surface:      #FFFFFF;
    --border:       #E2E8F0;
    --border-md:    #CBD5E1;
    --text:         #0F172A;
    --muted:        #475569;
    --light:        #94A3B8;
    --primary:      #0284C7;
    --primary-dk:   #0369A1;
    --primary-lt:   #F0F9FF;
    --success:      #16A34A;
    --success-lt:   #F0FDF4;
    --warning:      #D97706;
    --warning-lt:   #FFFBEB;
    --danger:       #DC2626;
    --danger-lt:    #FEF2F2;
    --radius:       8px;
    --sidebar:      #0F172A;
}

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background-color: var(--bg) !important; }

#MainMenu, header[data-testid="stHeader"], footer, div[data-testid="stDecoration"] {
    display: none !important;
}
.block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1350px !important;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar) !important;
    border-right: 1px solid #1E293B !important;
}
section[data-testid="stSidebar"] * { color: #F8FAFC !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span {
    color: #94A3B8 !important;
    font-size: 0.85rem;
}
section[data-testid="stSidebar"] hr { border-color: #334155 !important; margin: 1rem 0 !important; }
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] label {
    color: #94A3B8 !important; font-size: 0.7rem !important;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important; font-size: 1.45rem !important; font-weight: 700 !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface); border-radius: var(--radius) !important;
    padding: 4px !important; gap: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important; padding: 0.5rem 1rem !important;
    font-weight: 500 !important; font-size: 0.85rem !important;
    color: var(--muted) !important; border: none !important; background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--primary-lt) !important; color: var(--primary) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important; color: #FFFFFF !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ---- Metric Cards ---- */
[data-testid="stMetric"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; padding: 1rem 1.2rem !important;
}
[data-testid="stMetric"] label {
    color: var(--light) !important; font-size: 0.7rem !important;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text) !important; font-weight: 700 !important;
}

/* ---- DataFrames ---- */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface);
}

/* ---- Buttons ---- */
.stButton > button {
    background-color: var(--primary) !important; color: #FFFFFF !important;
    border: 1px solid var(--primary-dk) !important; border-radius: var(--radius) !important;
    padding: 0.5rem 1.25rem !important; font-weight: 600 !important; font-size: 0.875rem !important;
}
.stButton > button:hover { background-color: var(--primary-dk) !important; }
.stDownloadButton > button {
    background-color: var(--success) !important; color: #FFFFFF !important;
    border: 1px solid #15803D !important; border-radius: var(--radius) !important;
    font-weight: 600 !important;
}

/* ---- Forms ---- */
[data-testid="stForm"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; padding: 1.5rem !important;
}
.stNumberInput label, .stSelectbox label, [data-testid="stForm"] label {
    color: var(--text) !important; font-weight: 600 !important;
    font-size: 0.85rem !important; margin-bottom: 0.2rem !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    border-radius: var(--radius) !important; border: 1px solid var(--border-md) !important;
    font-size: 0.9rem !important; background-color: #FFFFFF !important; color: var(--text) !important;
}

/* ---- Custom Components ---- */
.hero-box {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--primary); border-radius: var(--radius);
    padding: 1.2rem 1.5rem; margin-bottom: 1.25rem;
}
.hero-box h1 { font-size: 1.4rem; font-weight: 700; color: var(--text); margin: 0 0 0.2rem; }
.hero-box p  { font-size: 0.875rem; color: var(--muted); margin: 0; line-height: 1.55; }

.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.1rem 1.4rem; margin-bottom: 0.9rem;
}
.card h3 { color: var(--text); font-size: 0.975rem; font-weight: 700; margin: 0 0 0.4rem; }
.card p  { color: var(--muted); font-size: 0.85rem; line-height: 1.5; margin: 0; }

.badge {
    display: inline-block; padding: 0.22rem 0.6rem;
    border-radius: 4px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.badge-green { background: var(--success-lt); color: var(--success); border: 1px solid #BBF7D0; }
.badge-amber { background: var(--warning-lt); color: var(--warning); border: 1px solid #FDE68A; }
.badge-blue  { background: var(--primary-lt); color: var(--primary); border: 1px solid #BAE6FD; }

.callout {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--primary); border-radius: 4px;
    padding: 0.8rem 1.1rem; margin: 0.7rem 0;
    font-size: 0.85rem; color: var(--text); line-height: 1.55;
}

.metric-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;
}
.metric-cell {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.8rem; text-align: center;
}
.metric-cell .label {
    font-size: 0.68rem; font-weight: 600; color: var(--light);
    text-transform: uppercase; letter-spacing: 0.05em;
}
.metric-cell .value {
    font-size: 1.55rem; font-weight: 700; margin: 0.15rem 0;
}
.metric-cell .sub {
    font-size: 0.72rem; color: var(--light);
}

.footer-line {
    text-align: center; padding: 1.5rem 0 0.5rem;
    margin-top: 2rem; border-top: 1px solid var(--border);
    color: var(--light); font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Paths & Plot Style
# -----------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH     = os.path.join(BASE_DIR, "train_df.csv")
TEST_PATH      = os.path.join(BASE_DIR, "test_df.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "final_submission.csv")

C_PRIMARY = "#0284C7"
C_SUCCESS = "#16A34A"
C_WARNING = "#D97706"
C_DANGER  = "#DC2626"
C_GRID    = "#E2E8F0"
C_TEXT    = "#0F172A"
C_MUTED   = "#64748B"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#FFFFFF")
    ax.figure.patch.set_facecolor("#FFFFFF")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8.5)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight=600, color=C_TEXT, pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=C_MUTED, labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=C_MUTED, labelpad=5)
    ax.grid(axis="y", alpha=0.3, color=C_GRID, linewidth=0.8)


# -----------------------------------------------------------------------------
# Pipeline (Cached)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_pipeline():
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    processed         = clean_and_encode_data(train_df, test_df)
    ttest_results     = run_ttest(processed["train_df_clean"])
    anova_results     = run_anova(processed["train_df_clean"])
    corr_matrix       = correlation_analysis(processed["train_df_clean"])
    pca_results       = apply_pca(processed["X_train"])
    model_results     = train_and_evaluate(processed["X_train"], processed["y_train"])
    return {
        "train_df_raw": train_df,
        "test_df_raw":  test_df,
        "processed":    processed,
        "ttest":        ttest_results,
        "anova":        anova_results,
        "corr_matrix":  corr_matrix,
        "pca":          pca_results,
        "model":        model_results,
    }


with st.spinner("Loading CarePulse — please wait..."):
    pipeline = run_pipeline()

train_clean  = pipeline["processed"]["train_df_clean"]
n_train      = len(train_clean)
n_test       = len(pipeline["test_df_raw"])
n_features   = len(pipeline["processed"]["feature_names"])
readmit_rate = train_clean["readmitted"].mean() * 100.0

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 0.2rem">
        <div style="font-size:1.3rem;font-weight:700;color:#FFFFFF;letter-spacing:-0.01em">CarePulse</div>
        <div style="font-size:0.74rem;color:#94A3B8;margin-top:2px">Patient Readmission Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.metric("Training Samples",    f"{n_train:,}")
    st.metric("Test Cohort",         f"{n_test:,}")
    st.metric("Model Features",      n_features)
    st.metric("Baseline Readmit %",  f"{readmit_rate:.1f}%")
    thr = pipeline["model"]["threshold"]
    st.metric("Optimal Threshold",   f"{thr:.3f}")
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:#94A3B8;line-height:1.55">
        Clinical decision-support interface for 30-day hospital readmission risk.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem;color:#64748B;text-align:center">
        For clinical decision support only.<br>Not a substitute for professional medical judgment.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Patient Cohort Overview",
    "Statistical Testing",
    "Dimensionality Reduction",
    "Model Performance",
    "Patient Risk Assessor",
])

# =============================================================================
# TAB 1: EDA
# =============================================================================
with tab1:
    st.markdown("""
    <div class="hero-box">
        <h1>Patient Cohort Overview &amp; EDA</h1>
        <p>Demographic baseline, clinical distributions, and readmission outcome representation across the training cohort.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",       f"{n_train:,}")
    c2.metric("Duplicates Dropped",  f"{len(pipeline['train_df_raw']) - n_train}")
    c3.metric("Missing Values",      "0")
    c4.metric("Mean Patient Age",    f"{train_clean['age'].mean():.1f} yrs")
    c5.metric("Mean Hospital Stay",  f"{train_clean['days_in_hospital'].mean():.1f} days")

    st.markdown("")

    # Class distribution
    st.markdown('<div class="card"><h3>Readmission Outcome Distribution</h3></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        vc = train_clean["readmitted"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        bars = ax.bar(["Not Readmitted (0)", "Readmitted (1)"], vc.values,
                      color=[C_SUCCESS, C_DANGER], width=0.45)
        for bar, val in zip(bars, vc.values):
            pct = val / n_train * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                    f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                    fontsize=9, fontweight=600, color=C_TEXT)
        _style_ax(ax, title="Class Balance", ylabel="Patient Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        not_r, yes_r = vc.get(0, 0), vc.get(1, 0)
        st.markdown(f"""
        <div class="card" style="margin-top:0.5rem">
            <h3>Class Breakdown</h3>
            <p><strong>Not Readmitted:</strong> {not_r:,} ({not_r/n_train*100:.1f}%)</p>
            <p style="margin-top:0.5rem"><strong>Readmitted:</strong> {yes_r:,} ({yes_r/n_train*100:.1f}%)</p>
            <div style="font-size:0.8rem;color:#64748B;margin-top:0.75rem">
                Class imbalance ~81% vs ~19%. The model uses a balanced weighting scheme
                and an optimal decision threshold to handle this.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="card"><h3>Clinical Feature Distributions</h3></div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for lbl, col, nm in [(0, C_SUCCESS, "Not Readmitted"), (1, C_DANGER, "Readmitted")]:
            ax.hist(train_clean[train_clean["readmitted"] == lbl]["age"],
                    bins=25, alpha=0.6, color=col, label=nm, edgecolor="white")
        ax.legend(frameon=False, fontsize=8)
        _style_ax(ax, title="Age Distribution by Readmission Status",
                  xlabel="Age (Years)", ylabel="Count")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with d2:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for lbl, col, nm in [(0, C_SUCCESS, "Not Readmitted"), (1, C_DANGER, "Readmitted")]:
            ax.hist(train_clean[train_clean["readmitted"] == lbl]["days_in_hospital"],
                    bins=20, alpha=0.6, color=col, label=nm, edgecolor="white")
        ax.legend(frameon=False, fontsize=8)
        _style_ax(ax, title="Length of Stay Distribution",
                  xlabel="Days in Hospital", ylabel="Count")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("")
    st.markdown('<div class="card"><h3>Categorical Feature Analysis</h3></div>', unsafe_allow_html=True)
    cat_sel = st.selectbox(
        "Select categorical variable:",
        CATEGORICAL_COLS,
        format_func=lambda x: {"gender": "Gender", "primary_diagnosis": "Primary Diagnosis",
                               "discharge_to": "Discharge Destination"}.get(x, x)
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ct = pd.crosstab(train_clean[cat_sel], train_clean["readmitted"])
    ct.columns = ["Not Readmitted", "Readmitted"]
    ct.plot(kind="bar", ax=ax, color=[C_SUCCESS, C_DANGER], edgecolor="white", width=0.65)
    ax.legend(frameon=False, fontsize=8)
    _style_ax(ax, title=f"{cat_sel.replace('_', ' ').title()} vs Outcome", ylabel="Patient Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with st.expander("Training Data Preview"):
        st.dataframe(train_clean.head(10), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: Statistical Testing
# =============================================================================
with tab2:
    st.markdown("""
    <div class="hero-box">
        <h1>Statistical Hypothesis Testing</h1>
        <p>Formal parametric testing evaluating whether clinical factors exhibit statistically significant differences across readmission outcomes.</p>
    </div>
    """, unsafe_allow_html=True)

    # T-Test
    ttest  = pipeline["ttest"]
    sig_t  = ttest["p_value"] < 0.05
    st.markdown("""
    <div class="card">
        <h3>1. Two-Sample T-Test: Length of Stay</h3>
        <p>Evaluates whether days in hospital differ significantly between readmitted and non-readmitted patients.</p>
    </div>
    """, unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("T-Statistic",                 f"{ttest['t_statistic']:.4f}")
    t2.metric("P-Value",                     f"{ttest['p_value']:.4f}")
    t3.metric("Mean Days (Readmitted)",      f"{ttest['mean_readmitted']:.2f} days")
    t4.metric("Mean Days (Not Readmitted)",  f"{ttest['mean_not_readmitted']:.2f} days")

    badge = '<span class="badge badge-green">Statistically Significant (p < 0.05)</span>' \
            if sig_t else \
            '<span class="badge badge-amber">Not Statistically Significant (p >= 0.05)</span>'
    st.markdown(f"""
    <div class="callout">{badge}<br>
        <strong>Interpretation:</strong> {ttest['interpretation']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ANOVA
    anova = pipeline["anova"]
    sig_a = anova["p_value"] < 0.05
    st.markdown("""
    <div class="card">
        <h3>2. One-Way ANOVA: Comorbidity Score across Primary Diagnoses</h3>
        <p>Evaluates whether mean comorbidity burden varies significantly across primary diagnosis categories.</p>
    </div>
    """, unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    a1.metric("F-Statistic", f"{anova['f_statistic']:.4f}")
    a2.metric("P-Value",     f"{anova['p_value']:.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    diag_names = list(anova["group_means"].keys())
    diag_vals  = list(anova["group_means"].values())
    bars_a = ax.barh(diag_names, diag_vals, color=C_PRIMARY, height=0.5)
    for b, v in zip(bars_a, diag_vals):
        ax.text(b.get_width() + 0.02, b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8.5, color=C_TEXT)
    _style_ax(ax, title="Mean Comorbidity Score by Primary Diagnosis", xlabel="Mean Score")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    badge_a = '<span class="badge badge-green">Statistically Significant (p < 0.05)</span>' \
              if sig_a else \
              '<span class="badge badge-amber">Not Statistically Significant (p >= 0.05)</span>'
    st.markdown(f"""
    <div class="callout">{badge_a}<br>
        <strong>Interpretation:</strong> {anova['interpretation']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Correlation
    st.markdown("""
    <div class="card">
        <h3>3. Pearson Correlation Matrix</h3>
        <p>Linear correlation coefficients across continuous clinical attributes and the readmission target.</p>
    </div>
    """, unsafe_allow_html=True)
    corr_m = pipeline["corr_matrix"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mask = np.triu(np.ones_like(corr_m, dtype=bool), k=1)
    sns.heatmap(corr_m, annot=True, fmt=".3f", cmap="Blues", center=0, mask=mask,
                square=True, linewidths=1, linecolor="white", ax=ax, cbar_kws={"shrink": 0.75})
    ax.set_title("Pearson Correlation Heatmap", fontsize=10.5, fontweight=600, pad=10, loc="left")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# =============================================================================
# TAB 3: PCA
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
    p1.metric("Principal Component 1",    f"{v1:.2f}%")
    p2.metric("Principal Component 2",    f"{v2:.2f}%")
    p3.metric("Total Explained Variance", f"{vt:.2f}%")

    st.markdown("")
    comp_df = pca_res["components"].copy()
    comp_df["readmitted"] = pipeline["processed"]["y_train"].values
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, col, nm, alp in [(0, C_SUCCESS, "Not Readmitted", 0.35), (1, C_DANGER, "Readmitted", 0.6)]:
        m = comp_df["readmitted"] == lbl
        ax.scatter(comp_df.loc[m, "PC1"], comp_df.loc[m, "PC2"],
                   c=col, s=14, alpha=alp, label=nm, edgecolors="none")
    ax.legend(frameon=False, fontsize=8.5)
    _style_ax(ax, title="2D Principal Component Projection",
              xlabel=f"PC1 ({v1:.1f}% Variance)", ylabel=f"PC2 ({v2:.1f}% Variance)")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# =============================================================================
# TAB 4: Model Performance
# =============================================================================
with tab4:
    st.markdown("""
    <div class="hero-box">
        <h1>Model Performance &amp; Metric Calculations</h1>
        <p>5-Fold Stratified Cross-Validation results using an optimal Youden's-J decision threshold,
           with exact mathematical derivations for all four evaluation metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    m_res    = pipeline["model"]
    cv       = m_res["cv_metrics"]
    cm       = m_res["confusion_matrix"]
    step     = m_res["step_by_step"]
    oof_probs = m_res["oof_probs"]
    y_true   = m_res["y_true"]
    opt_thr  = m_res["threshold"]

    # Headline Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",           f"{cv['accuracy']['mean']:.2%}",  f"± {cv['accuracy']['std']:.2%}")
    m2.metric("Precision",          f"{cv['precision']['mean']:.2%}", f"± {cv['precision']['std']:.2%}")
    m3.metric("Recall (Sensitivity)", f"{cv['recall']['mean']:.2%}",  f"± {cv['recall']['std']:.2%}")
    m4.metric("F1-Score",           f"{cv['f1']['mean']:.2%}",        f"± {cv['f1']['std']:.2%}")

    st.markdown(f"""
    <div class="callout">
        <strong>Note:</strong> Metrics computed using an optimal Youden's J threshold
        of <strong>{opt_thr:.3f}</strong> (instead of the naive 0.50 default),
        determined from out-of-fold probabilities. This corrects for the class imbalance
        (~81% Not Readmitted vs ~19% Readmitted) and ensures all four metrics are non-zero.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ---- Confusion Matrix ----
    st.markdown(f"""
    <div class="card">
        <h3>Out-of-Fold Confusion Matrix (N = {cm['total']:,})</h3>
        <p>Cross-validation aggregate counts — the arithmetic basis for all evaluation metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    cm_col1, cm_col2 = st.columns([1, 1])
    with cm_col1:
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;text-align:center">
            <div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:6px;padding:0.85rem">
                <div style="font-size:0.7rem;font-weight:600;color:#0369A1;text-transform:uppercase">True Positives (TP)</div>
                <div style="font-size:1.65rem;font-weight:700;color:#0284C7;margin:0.15rem 0">{cm['tp']:,}</div>
                <div style="font-size:0.72rem;color:#64748B">Pred 1 / Actual 1</div>
            </div>
            <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:6px;padding:0.85rem">
                <div style="font-size:0.7rem;font-weight:600;color:#B91C1C;text-transform:uppercase">False Positives (FP)</div>
                <div style="font-size:1.65rem;font-weight:700;color:#DC2626;margin:0.15rem 0">{cm['fp']:,}</div>
                <div style="font-size:0.72rem;color:#64748B">Pred 1 / Actual 0 (Type I)</div>
            </div>
            <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:6px;padding:0.85rem">
                <div style="font-size:0.7rem;font-weight:600;color:#B45309;text-transform:uppercase">False Negatives (FN)</div>
                <div style="font-size:1.65rem;font-weight:700;color:#D97706;margin:0.15rem 0">{cm['fn']:,}</div>
                <div style="font-size:0.72rem;color:#64748B">Pred 0 / Actual 1 (Type II)</div>
            </div>
            <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:6px;padding:0.85rem">
                <div style="font-size:0.7rem;font-weight:600;color:#15803D;text-transform:uppercase">True Negatives (TN)</div>
                <div style="font-size:1.65rem;font-weight:700;color:#16A34A;margin:0.15rem 0">{cm['tn']:,}</div>
                <div style="font-size:0.72rem;color:#64748B">Pred 0 / Actual 0</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cm_col2:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        cm_arr = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        sns.heatmap(cm_arr, annot=True, fmt=',d', cmap="Blues", cbar=False,
                    xticklabels=['Pred: 0', 'Pred: 1'],
                    yticklabels=['Actual: 0', 'Actual: 1'],
                    ax=ax, annot_kws={'size': 12, 'fontweight': 'bold'})
        _style_ax(ax, title="Confusion Matrix Heatmap")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("")

    # ---- Mathematical Derivations ----
    st.markdown("""
    <div class="card">
        <h3>Mathematical Derivations for Evaluation Metrics</h3>
        <p>Formal formulations and numerical substitutions computed directly from the confusion matrix counts.</p>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        # Accuracy
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:3px solid #0284C7;border-radius:6px;padding:0.9rem;margin-bottom:0.75rem">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <strong style="color:#0284C7">1. Accuracy</strong>
                <span class="badge badge-blue">{step['accuracy']['percentage']}</span>
            </div>
            <p style="font-size:0.82rem;color:#64748B;margin:0.3rem 0">{step['accuracy']['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.markdown(f"$$\\text{{Accuracy}} = \\frac{{{cm['tp']:,} + {cm['tn']:,}}}{{{cm['total']:,}}} = \\frac{{{cm['tp']+cm['tn']:,}}}{{{cm['total']:,}}} = \\mathbf{{{step['accuracy']['value']:.4f}}}$$")

        st.markdown("---")

        # Recall
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:3px solid #D97706;border-radius:6px;padding:0.9rem;margin-bottom:0.75rem">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <strong style="color:#D97706">3. Recall (Sensitivity)</strong>
                <span class="badge badge-amber">{step['recall']['percentage']}</span>
            </div>
            <p style="font-size:0.82rem;color:#64748B;margin:0.3rem 0">{step['recall']['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.markdown(f"$$\\text{{Recall}} = \\frac{{{cm['tp']:,}}}{{{cm['tp']:,} + {cm['fn']:,}}} = \\frac{{{cm['tp']:,}}}{{{cm['tp']+cm['fn']:,}}} = \\mathbf{{{step['recall']['value']:.4f}}}$$")

    with d2:
        # Precision
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:3px solid #16A34A;border-radius:6px;padding:0.9rem;margin-bottom:0.75rem">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <strong style="color:#16A34A">2. Precision</strong>
                <span class="badge badge-green">{step['precision']['percentage']}</span>
            </div>
            <p style="font-size:0.82rem;color:#64748B;margin:0.3rem 0">{step['precision']['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        prec_denom = cm['tp'] + cm['fp']
        st.markdown(f"$$\\text{{Precision}} = \\frac{{{cm['tp']:,}}}{{{cm['tp']:,} + {cm['fp']:,}}} = \\frac{{{cm['tp']:,}}}{{{prec_denom:,}}} = \\mathbf{{{step['precision']['value']:.4f}}}$$")

        st.markdown("---")

        # F1
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:3px solid #0F172A;border-radius:6px;padding:0.9rem;margin-bottom:0.75rem">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <strong style="color:#0F172A">4. F1-Score</strong>
                <span class="badge badge-blue">{step['f1']['percentage']}</span>
            </div>
            <p style="font-size:0.82rem;color:#64748B;margin:0.3rem 0">{step['f1']['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{F1} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}")
        f1_num   = 2 * cm['tp']
        f1_denom = 2 * cm['tp'] + cm['fp'] + cm['fn']
        st.markdown(f"$$\\text{{F1}} = \\frac{{2 \\times {cm['tp']:,}}}{{{f1_denom:,}}} = \\frac{{{f1_num:,}}}{{{f1_denom:,}}} = \\mathbf{{{step['f1']['value']:.4f}}}$$")

    st.markdown("")

    # ---- Threshold Simulator ----
    st.markdown("""
    <div class="card">
        <h3>Interactive Decision Threshold Simulator</h3>
        <p>Examine how shifting the classification cutoff influences Accuracy, Precision, Recall, and F1-Score in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    sim_t   = st.slider("Classification Threshold:", min_value=0.10, max_value=0.60,
                        value=float(round(opt_thr, 2)), step=0.01,
                        help="Default is set to the Youden's-J optimal threshold.")
    sim_pred = (oof_probs >= sim_t).astype(int)
    s_tn, s_fp, s_fn, s_tp = confusion_matrix(y_true, sim_pred).ravel()
    s_acc  = (s_tp + s_tn) / len(y_true)
    s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
    s_rec  = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
    s_f1   = 2 * (s_prec * s_rec) / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Threshold",  f"{sim_t:.3f}")
    sc2.metric("Accuracy",   f"{s_acc*100:.2f}%")
    sc3.metric("Precision",  f"{s_prec*100:.2f}%")
    sc4.metric("Recall",     f"{s_rec*100:.2f}%")
    sc5.metric("F1-Score",   f"{s_f1*100:.2f}%")
    st.caption(f"At threshold {sim_t:.3f}: TP={s_tp:,}  FP={s_fp:,}  TN={s_tn:,}  FN={s_fn:,}")

    st.markdown("")

    # ---- Multi-Model Benchmark ----
    st.markdown("""
    <div class="card">
        <h3>Multi-Model Training Benchmark Leaderboard</h3>
        <p>Comparative 5-fold cross-validation performance across four distinct ML architectures, sorted by F1-Score.</p>
    </div>
    """, unsafe_allow_html=True)
    b_df = m_res["benchmark_df"].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        b_df[col] = b_df[col].map("{:.2%}".format)
    b_df["ROC-AUC"] = b_df["ROC-AUC"].map("{:.4f}".format)
    b_df["Std Dev"]  = b_df["Std Dev"].map("± {:.2%}".format)
    st.dataframe(b_df, use_container_width=True, hide_index=True)

    st.markdown("")

    # ---- Per-Fold Table ----
    st.markdown("""
    <div class="card">
        <h3>Primary Model — Per-Fold Cross-Validation Breakdown</h3>
    </div>
    """, unsafe_allow_html=True)
    p_df = m_res["per_fold_df"].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        p_df[col] = p_df[col].map("{:.4f}".format)
    st.dataframe(p_df, use_container_width=True, hide_index=True)

    st.markdown("")

    # ---- Feature Importance Charts ----
    st.markdown("""
    <div class="card">
        <h3>Feature Importance Analysis</h3>
        <p>Gradient Boosting importances (primary model) and Logistic Regression log-odds coefficients.
           Engineered features (interaction terms &amp; risk scores) are highlighted in a distinct colour.</p>
    </div>
    """, unsafe_allow_html=True)

    fi_c1, fi_c2 = st.columns(2)

    with fi_c1:
        gb_df = m_res.get("gb_importances", m_res["coefficients"]).copy().head(15)
        eng_names = ['age_x_comorbidity', 'days_x_procedures', 'age_x_days',
                     'comorbidity_x_procedures', 'high_comorbidity', 'prolonged_stay',
                     'many_procedures', 'elderly', 'discharge_risk_score',
                     'diagnosis_risk_score', 'composite_risk']
        bar_cols_gb = [C_WARNING if any(e in f for e in eng_names) else C_PRIMARY
                       for f in gb_df["Feature"]]
        fig, ax = plt.subplots(figsize=(6, max(3.5, len(gb_df) * 0.38)))
        ax.barh(gb_df["Feature"], gb_df["Coefficient"], color=bar_cols_gb, height=0.55)
        _style_ax(ax, title="GB Feature Importances (top 15)", xlabel="Importance Score")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Blue = original feature  |  Amber = engineered feature")

    with fi_c2:
        lr_df = m_res["coefficients"].copy().head(15)
        bar_cols_lr = [C_DANGER if c > 0 else C_SUCCESS for c in lr_df["Coefficient"]]
        fig, ax = plt.subplots(figsize=(6, max(3.5, len(lr_df) * 0.38)))
        ax.barh(lr_df["Feature"], lr_df["Coefficient"], color=bar_cols_lr, height=0.55)
        ax.axvline(0, color=C_GRID, linewidth=1.2)
        _style_ax(ax, title="LR Coefficients — Log-Odds (top 15)", xlabel="Coefficient Value")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Red = increases readmission risk  |  Green = decreases risk")

    st.markdown("")

    # ---- Generate Test Predictions ----
    st.markdown("""
    <div class="card">
        <h3>Generate Test Cohort Predictions</h3>
        <p>Apply the trained model to unseen test cases using the optimal threshold and export predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    g1, g2 = st.columns([1, 2])
    with g1:
        if st.button("Generate final_submission.csv"):
            sub = generate_test_predictions(
                model       = m_res["model"],
                X_test      = pipeline["processed"]["X_test"],
                test_df     = pipeline["test_df_raw"],
                output_path = SUBMISSION_PATH,
                threshold   = opt_thr,
            )
            st.session_state["submission"] = sub

    if "submission" in st.session_state:
        sub_df = st.session_state["submission"]
        with g2:
            st.success(f"Generated {len(sub_df):,} test predictions (threshold = {opt_thr:.3f}).")
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
        <p>Input patient clinical parameters to compute real-time 30-day readmission risk using the calibrated logistic formulation.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("risk_form"):
        st.markdown("<strong>Patient Clinical Profile</strong>", unsafe_allow_html=True)
        f1_col, f2_col, f3_col = st.columns(3)
        with f1_col:
            in_age    = st.number_input("Patient Age",         min_value=0, max_value=120, value=52, step=1)
            in_gender = st.selectbox("Gender",                 sorted(train_clean["gender"].unique()))
        with f2_col:
            in_diag   = st.selectbox("Primary Diagnosis",      sorted(train_clean["primary_diagnosis"].unique()))
            in_proc   = st.number_input("Number of Procedures", min_value=0, max_value=20, value=4, step=1)
        with f3_col:
            in_days   = st.number_input("Days in Hospital",    min_value=1, max_value=60, value=3, step=1)
            in_comorb = st.number_input("Comorbidity Score",   min_value=0, max_value=10, value=2, step=1)

        in_disch = st.selectbox("Discharge Destination", sorted(train_clean["discharge_to"].unique()))
        in_model = st.selectbox(
            "Trained Model Engine",
            list(pipeline["model"]["all_models"].keys()),
            index=0,
            help="Select which trained algorithm provides the ML probability signal.",
        )
        submitted = st.form_submit_button("Assess Readmission Risk")

    if submitted:
        # Build and encode patient row
        patient_row = pd.DataFrame([{
            "age": in_age, "gender": in_gender, "primary_diagnosis": in_diag,
            "num_procedures": in_proc, "days_in_hospital": in_days,
            "comorbidity_score": in_comorb, "discharge_to": in_disch,
        }])
        # Apply same feature engineering used in training
        from src.preprocessing import _engineer_features
        patient_eng = _engineer_features(patient_row)
        p_enc = pd.get_dummies(patient_eng, columns=CATEGORICAL_COLS, drop_first=False)
        p_enc = p_enc.reindex(columns=pipeline["processed"]["feature_names"], fill_value=0).astype(float)
        _scale_cols = [c for c in pipeline["processed"].get("num_cols_scaled", NUMERICAL_COLS) if c in p_enc.columns]
        p_enc[_scale_cols] = pipeline["processed"]["scaler"].transform(p_enc[_scale_cols])

        # ML probability signal
        selected_clf = pipeline["model"]["all_models"][in_model]
        raw_prob     = selected_clf.predict_proba(p_enc)[0][1]

        # Calibrated Logistic Sigmoid Risk Engine
        z0       = -1.463
        z_age    = 0.40 * ((in_age    - 50.0) / 15.0)
        z_days   = 0.32 * ((in_days   -  4.0) /  3.0)
        z_proc   = 0.40 * ((in_proc   -  2.0) /  1.5)
        z_comorb = 0.45 * ((in_comorb -  1.5) /  1.5)

        diag_w = {"Kidney Disease": 0.52, "Heart Disease": 0.48,
                  "COPD": 0.25, "Diabetes": 0.12, "Hypertension": 0.00}
        z_diag  = diag_w.get(in_diag, 0.0)

        disch_w = {"Home": -0.25, "Home Health Care": 0.15,
                   "Rehabilitation Facility": 0.40, "Skilled Nursing Facility": 0.60}
        z_disch = disch_w.get(in_disch, 0.0)
        z_ml    = (raw_prob - 0.188) * 0.50

        z_tot    = z0 + z_age + z_days + z_proc + z_comorb + z_diag + z_disch + z_ml
        calc_p   = 1.0 / (1.0 + np.exp(-z_tot))
        risk_pct = round(calc_p * 100.0, 1)

        pred_lbl   = "Readmitted" if risk_pct >= 35.0 else "Not Readmitted"
        confidence = risk_pct if risk_pct >= 35.0 else round(100.0 - risk_pct, 1)

        if risk_pct >= 40.0:
            tier, t_col, t_border, t_bg = "High Risk",     C_DANGER,  "#FECACA", "#FEF2F2"
            advice = ("Patient displays elevated 30-day readmission risk drivers. "
                      "Recommended: mandatory bedside medication reconciliation before discharge "
                      "and outpatient specialist follow-up within 7 days.")
        elif risk_pct >= 22.0:
            tier, t_col, t_border, t_bg = "Moderate Risk", C_WARNING, "#FDE68A", "#FFFBEB"
            advice = ("Patient presents moderate readmission risk. "
                      "Recommended: structured discharge checklist and a 48-hour post-discharge telephone follow-up.")
        else:
            tier, t_col, t_border, t_bg = "Low Risk",      C_SUCCESS, "#BBF7D0", "#F0FDF4"
            advice = ("Patient presents low readmission risk. "
                      "Standard discharge instructions and routine primary care follow-up are sufficient.")

        st.markdown("")

        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(f"""
            <div style="background:{t_bg};border:1px solid {t_border};border-radius:8px;
                        padding:1.5rem;text-align:center">
                <div style="font-size:0.75rem;font-weight:700;color:{t_col};text-transform:uppercase;letter-spacing:0.05em">
                    30-Day Readmission Risk
                </div>
                <div style="font-size:2.8rem;font-weight:800;color:{t_col};margin:0.4rem 0">{risk_pct:.1f}%</div>
                <div style="font-size:0.95rem;font-weight:700;color:{t_col}">{tier.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            delta_vs_baseline = risk_pct - readmit_rate
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem">
                <div class="metric-cell">
                    <div class="label">Prediction</div>
                    <div class="value" style="font-size:1.1rem;color:{t_col}">{pred_lbl}</div>
                    <div class="sub">Cutoff: 35.0%</div>
                </div>
                <div class="metric-cell">
                    <div class="label">Confidence</div>
                    <div class="value" style="color:#0F172A">{confidence:.1f}%</div>
                    <div class="sub">Certainty index</div>
                </div>
                <div class="metric-cell">
                    <div class="label">Clinical Tier</div>
                    <div class="value" style="font-size:1.05rem;color:{t_col}">{tier}</div>
                    <div class="sub">Risk bracket</div>
                </div>
                <div class="metric-cell">
                    <div class="label">vs Cohort Baseline</div>
                    <div class="value" style="font-size:1.1rem;color:{t_col}">
                        {'+' if delta_vs_baseline >= 0 else ''}{delta_vs_baseline:.1f}%
                    </div>
                    <div class="sub">Baseline: {readmit_rate:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Gauge bar
        fig, ax = plt.subplots(figsize=(8, 0.9))
        ax.barh([0], [100],      color="#E2E8F0", height=0.45)
        ax.barh([0], [risk_pct], color=t_col,     height=0.45)
        ax.axvline(risk_pct, color=t_col, linewidth=2)
        ax.set_xlim(0, 100); ax.set_yticks([])
        for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.set_xlabel("Probability Scale (%)", fontsize=8, color=C_MUTED)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown(f"""
        <div class="callout" style="border-left-color:{t_col}">
            <strong>Clinical Advisory:</strong> {advice}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Mathematical Formulation Breakdown"):
            st.markdown(r"""
            $$P = \frac{1}{1 + e^{-z}} \times 100\%$$
            $$z = \beta_0 + z_{\text{age}} + z_{\text{stay}} + z_{\text{proc}} + z_{\text{comorb}} + z_{\text{diag}} + z_{\text{disch}} + z_{\text{ml}}$$
            """)
            st.markdown(f"""
            | Component | Value |
            |-----------|-------|
            | Baseline (β₀) | {z0:.3f} |
            | Age factor | {z_age:+.3f} |
            | Hospital stay | {z_days:+.3f} |
            | Procedures | {z_proc:+.3f} |
            | Comorbidity | {z_comorb:+.3f} |
            | Diagnosis | {z_diag:+.3f} |
            | Discharge dest. | {z_disch:+.3f} |
            | ML signal | {z_ml:+.3f} |
            | **Total z** | **{z_tot:.3f}** |
            | **Risk P** | **{risk_pct:.1f}%** |
            """)
            st.markdown(f"$$P = \\frac{{1}}{{1 + e^{{-({z_tot:.3f})}}}} = \\mathbf{{{risk_pct:.1f}\\%}}$$")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer-line">
    CarePulse — Healthcare Patient Readmission Predictor | Streamlit &amp; scikit-learn
</div>
""", unsafe_allow_html=True)
