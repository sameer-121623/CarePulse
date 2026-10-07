"""
Generate CarePulse Project Documentation as a Word (.docx) file.
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

OUTPUT = r"c:\Users\hp\OneDrive\Desktop\AIML\said\project\CarePulse_Project_Documentation.docx"

doc = Document()

# ── Page margins ─────────────────────────────────────────────────────────────
section = doc.sections[0]
section.top_margin    = Cm(2.0)
section.bottom_margin = Cm(2.0)
section.left_margin   = Cm(2.5)
section.right_margin  = Cm(2.5)

# ── Helper functions ──────────────────────────────────────────────────────────
def add_heading(doc, text, level=1, color=None):
    p = doc.add_heading(text, level=level)
    if color:
        for run in p.runs:
            run.font.color.rgb = RGBColor(*color)
    return p

def add_body(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style='List Bullet')
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    return p

def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_row = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
        cell._tc.get_or_add_tcPr()
        shading = OxmlElement('w:shd')
        shading.set(qn('w:fill'), '0F172A')
        shading.set(qn('w:val'), 'clear')
        cell._tc.tcPr.append(shading)
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            row.cells[c_idx].text = str(val)
    return table

def add_formula_box(doc, formula_text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Inches(0.5)
    p.paragraph_format.right_indent = Inches(0.5)
    run = p.add_run(formula_text)
    run.font.name  = 'Courier New'
    run.font.size  = Pt(12)
    run.font.bold  = True
    run.font.color.rgb = RGBColor(2, 132, 199)
    return p

# =============================================================================
# TITLE PAGE
# =============================================================================
doc.add_paragraph("")
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = title.add_run("CarePulse")
title_run.bold = True
title_run.font.size = Pt(36)
title_run.font.color.rgb = RGBColor(2, 132, 199)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_run = subtitle.add_run("Patient Readmission Prediction System")
sub_run.font.size = Pt(18)
sub_run.font.color.rgb = RGBColor(71, 85, 105)

doc.add_paragraph("")
meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run("Machine Learning | Healthcare Analytics | Python | Streamlit\n").font.size = Pt(12)
meta.add_run("GitHub: github.com/sameer-121623/CarePulse").font.size = Pt(11)

doc.add_page_break()

# =============================================================================
# 1. PROJECT OVERVIEW
# =============================================================================
add_heading(doc, "1. Project Overview", 1, color=(2, 132, 199))

add_body(doc,
    "CarePulse is an end-to-end Machine Learning web application that predicts whether "
    "a hospital patient is at HIGH RISK of being readmitted within 30 days of their initial "
    "discharge. The system is built entirely in Python and deployed as an interactive dashboard "
    "using the Streamlit framework.",
    size=11
)

add_body(doc, "\nWhy does this matter?", bold=True, size=11)
add_bullet(doc, "In the US alone, ~3.5 million patients are readmitted within 30 days each year, costing hospitals billions in Medicare/Medicaid penalties.")
add_bullet(doc, "By identifying HIGH-RISK patients before discharge, doctors can arrange follow-up care, home nursing, and medication reviews.")
add_bullet(doc, "CarePulse acts as a clinical decision-support tool — it gives doctors a data-driven second opinion.")

doc.add_paragraph("")

# =============================================================================
# 2. CORE CONCEPT
# =============================================================================
add_heading(doc, "2. Core Machine Learning Concept", 1, color=(2, 132, 199))

add_body(doc,
    "CarePulse uses Supervised Binary Classification — a type of machine learning where the "
    "model studies thousands of past patient records (each labelled 'Readmitted' or 'Not Readmitted') "
    "and learns to predict the label for NEW, unseen patients.",
    size=11
)

add_body(doc, "\nThe Two Classes:", bold=True)
add_bullet(doc, "Class 0 — NOT Readmitted (81.2% of records): Patient recovered successfully and did not return within 30 days.")
add_bullet(doc, "Class 1 — READMITTED (18.8% of records): Patient had to come back to the hospital within 30 days.")

add_body(doc, "\nThe Challenge — Class Imbalance:", bold=True)
add_body(doc,
    "Since only 18.8% of patients are readmitted, a naive model could just predict 'Not Readmitted' "
    "for everyone and still achieve 81% accuracy — but it would never catch a single at-risk patient! "
    "CarePulse solves this using class_weight='balanced' (penalising the model harder for missing "
    "readmitted patients) and an accuracy-anchored threshold selection algorithm."
)

doc.add_paragraph("")

# =============================================================================
# 3. PROJECT FLOW
# =============================================================================
add_heading(doc, "3. Project Flow (Step-by-Step Pipeline)", 1, color=(2, 132, 199))

steps = [
    ("Step 1", "Data Loading", "train_df.csv (5,000 records) and test_df.csv (2,000 records) are loaded into memory."),
    ("Step 2", "Feature Engineering", "11 new clinical features are mathematically derived from the raw 7 columns — interaction terms (age × comorbidity), risk flags (prolonged_stay > 7 days), and domain risk scores."),
    ("Step 3", "Encoding & Scaling", "Text categories (gender, diagnosis, discharge_to) are One-Hot Encoded into numeric columns. All numeric columns are Standard Scaled (mean=0, std=1)."),
    ("Step 4", "Model Training (5-Fold CV)", "The dataset is split into 5 equal chunks. The model trains on 4 chunks and is tested on the 5th. This is repeated 5 times to produce stable, generalizable metrics."),
    ("Step 5", "OOF Probability Collection", "Out-Of-Fold (OOF) probabilities are collected — for every patient, the model's estimated probability that they will be readmitted."),
    ("Step 6", "Threshold Optimization", "Instead of using a naive 0.5 cutoff, the system scans 400 probability thresholds to find the one that guarantees ~80% accuracy while keeping Precision/Recall non-zero."),
    ("Step 7", "Final Model Fitting", "All 4 models (LR, RF, GB, Ensemble) are fitted on the complete training set and made available for the interactive Risk Assessor."),
    ("Step 8", "Dashboard Rendering", "Results are displayed in a 5-tab Streamlit dashboard with charts, tables, math breakdowns, and the interactive patient risk calculator."),
]

add_table(doc,
    ["Step", "Name", "Description"],
    [(s[0], s[1], s[2]) for s in steps]
)
doc.add_paragraph("")

# =============================================================================
# 4. INPUT PARAMETERS (FEATURES)
# =============================================================================
add_heading(doc, "4. Input Parameters & Prediction Factors", 1, color=(2, 132, 199))

add_heading(doc, "4.1 Raw Input Features (from the dataset)", 2)
raw_features = [
    ("age",               "Numeric (years)",      "Patient age. Older patients have longer recovery times and higher complication rates."),
    ("gender",            "Categorical",           "Male / Female. Included for demographic baselining."),
    ("primary_diagnosis", "Categorical (5 types)", "The main illness: Heart Disease, Diabetes, COPD, Hypertension, Kidney Disease."),
    ("num_procedures",    "Numeric (count)",       "Number of medical procedures performed during the hospital stay."),
    ("days_in_hospital",  "Numeric (days)",        "Total length of the initial hospital admission."),
    ("comorbidity_score", "Numeric (1–10)",        "A score representing how many co-existing medical conditions the patient has. Higher = sicker."),
    ("discharge_to",      "Categorical (4 types)", "Where the patient went after leaving: Home, Home Health Care, Rehabilitation Facility, Skilled Nursing Facility."),
]

add_table(doc,
    ["Feature Name", "Data Type", "Clinical Meaning"],
    raw_features
)

doc.add_paragraph("")
add_heading(doc, "4.2 Engineered Features (Auto-calculated by CarePulse)", 2)
add_body(doc,
    "Because the raw features alone have weak individual correlations with readmission, "
    "CarePulse automatically creates 11 additional features before training:"
)

eng_features = [
    ("age_x_comorbidity",        "Interaction", "age × comorbidity_score. An elderly patient with high comorbidity is exponentially riskier."),
    ("days_x_procedures",        "Interaction", "days_in_hospital × num_procedures. Long stays with many procedures indicate severity."),
    ("age_x_days",               "Interaction", "age × days_in_hospital. Prolonged stays are more dangerous for older patients."),
    ("comorbidity_x_procedures", "Interaction", "comorbidity_score × num_procedures."),
    ("high_comorbidity",         "Binary Flag", "1 if comorbidity_score ≥ 4, else 0."),
    ("prolonged_stay",           "Binary Flag", "1 if days_in_hospital ≥ 7, else 0."),
    ("many_procedures",          "Binary Flag", "1 if num_procedures ≥ 5, else 0."),
    ("elderly",                  "Binary Flag", "1 if age ≥ 65, else 0."),
    ("discharge_risk_score",     "Risk Lookup", "Numerical risk: Home=0.0, Home Health=0.3, Rehab=0.6, Skilled Nursing=1.0."),
    ("diagnosis_risk_score",     "Risk Lookup", "Numerical risk: Hypertension=0.0, Diabetes=0.2, COPD=0.5, Heart=0.8, Kidney=1.0."),
    ("composite_risk",           "Weighted Sum", "0.30×diagnosis + 0.25×discharge + 0.20×comorbidity/10 + 0.15×days/30 + 0.10×age/100."),
]

add_table(doc,
    ["Feature Name", "Type", "Description"],
    eng_features
)
doc.add_paragraph("")

# =============================================================================
# 5. MATHEMATICAL FORMULAS
# =============================================================================
add_heading(doc, "5. Mathematical Formulas", 1, color=(2, 132, 199))

add_heading(doc, "5.1 Confusion Matrix (The Foundation)", 2)
add_body(doc,
    "Every metric is computed from the Confusion Matrix — a 2×2 grid comparing predictions vs reality:"
)
cm_data = [
    ("True Positive (TP)",  "Model said READMITTED — Patient WAS readmitted.      (Correct Positive)"),
    ("True Negative (TN)",  "Model said NOT READMITTED — Patient was NOT.          (Correct Negative)"),
    ("False Positive (FP)", "Model said READMITTED — Patient was NOT. (False Alarm)"),
    ("False Negative (FN)", "Model said NOT READMITTED — Patient WAS.  (Missed Case — most dangerous)"),
]
add_table(doc, ["Term", "Meaning"], cm_data)
doc.add_paragraph("")

add_heading(doc, "5.2 Accuracy", 2)
add_body(doc, "What percentage of ALL predictions were correct?")
add_formula_box(doc, "Accuracy  =  (TP + TN)  /  (TP + TN + FP + FN)")
add_body(doc, "Example: TP=28, TN=3932, FP=124, FN=912 → (28+3932)/(5000) = 79.2%", italic=True)

doc.add_paragraph("")
add_heading(doc, "5.3 Precision", 2)
add_body(doc, "When the model raises a READMISSION ALERT, how often is it actually correct?")
add_formula_box(doc, "Precision  =  TP  /  (TP + FP)")
add_body(doc, "If Precision = 18%, that means 18% of the patients the model flagged were truly readmitted.", italic=True)

doc.add_paragraph("")
add_heading(doc, "5.4 Recall (Sensitivity)", 2)
add_body(doc, "Out of ALL the patients who WERE readmitted, what percentage did the model successfully catch?")
add_formula_box(doc, "Recall  =  TP  /  (TP + FN)")
add_body(doc, "This is the most important metric in healthcare — a missed readmission (FN) is dangerous.", italic=True)

doc.add_paragraph("")
add_heading(doc, "5.5 F1-Score", 2)
add_body(doc, "The harmonic mean of Precision and Recall. Balances both — critical for imbalanced datasets.")
add_formula_box(doc, "F1  =  2 * (Precision * Recall)  /  (Precision + Recall)")
add_formula_box(doc, "   =  2 * TP  /  (2*TP + FP + FN)")

doc.add_paragraph("")
add_heading(doc, "5.6 Risk Assessor — Logistic Sigmoid Formula (Tab 5)", 2)
add_body(doc,
    "When you use the interactive sliders in Tab 5, the risk probability is calculated "
    "using a Logistic Sigmoid function that combines clinical risk weights:"
)
add_formula_box(doc, "Z_total  =  Z0 + Z_age + Z_days + Z_procedures + Z_comorbidity + Z_diagnosis + Z_discharge + Z_ml")
add_formula_box(doc, "P(Readmission)  =  1  /  (1 + e^(-Z_total))")
add_body(doc, "Where e = 2.71828 (Euler's number). Output is always between 0% and 100%.", italic=True)

doc.add_paragraph("")
add_heading(doc, "5.7 Standard Scaling Formula", 2)
add_body(doc, "All numeric features are scaled before model training:")
add_formula_box(doc, "Z  =  (X - mean)  /  standard_deviation")
add_body(doc, "This ensures age (0-100) and comorbidity (1-10) are on the same mathematical scale.", italic=True)

doc.add_paragraph("")

# =============================================================================
# 6. THE MODELS
# =============================================================================
add_heading(doc, "6. Machine Learning Models Used", 1, color=(2, 132, 199))

models_info = [
    ("Logistic Regression",
     "Primary",
     "A statistical model that calculates a probability using a linear equation. Very fast and interpretable — it produces feature coefficients (log-odds) that tell you exactly HOW MUCH each factor contributes to readmission risk.",
     "Fast, transparent, good for understanding feature importance."),
    ("Random Forest",
     "Benchmark",
     "Builds 120 individual decision trees, each seeing a random subset of the data and features. The final prediction is a vote across all 120 trees. Handles non-linear relationships well.",
     "Resistant to overfitting, handles missing/noisy data."),
    ("Gradient Boosting (Primary OOF model)",
     "Primary / Benchmark",
     "Builds trees sequentially — each new tree specifically tries to correct the errors of the previous one. The most powerful model for tabular data. Used for the Out-Of-Fold probability collection that drives threshold optimization.",
     "Highest accuracy, best probability calibration."),
    ("Voting Ensemble",
     "Benchmark",
     "Combines all three models (LR + RF + GB) using 'soft voting' — it averages their probability outputs. Typically the most robust because it benefits from the strengths of each individual model.",
     "Most stable, minimizes any single model's weaknesses."),
]

for name, role, description, benefit in models_info:
    add_body(doc, f"\n{name} ({role})", bold=True)
    add_body(doc, description)
    add_body(doc, f"Key Benefit: {benefit}", italic=True)

doc.add_paragraph("")

# =============================================================================
# 7. FILE STRUCTURE
# =============================================================================
add_heading(doc, "7. Project File Structure & Responsibilities", 1, color=(2, 132, 199))

files = [
    ("app.py",
     "Frontend Dashboard",
     "The main web server. Uses Streamlit to render 5 interactive tabs: Data Overview, Statistical Tests, PCA, Model Performance, and Risk Assessor. Calls all other modules and displays results."),
    ("src/preprocessing.py",
     "Data Engineering Module",
     "Contains load_data(), _engineer_features(), and clean_and_encode_data(). Responsible for all data transformation: One-Hot Encoding, Standard Scaling, and generating the 11 engineered features."),
    ("src/model.py",
     "ML Training Engine",
     "Contains train_and_evaluate() and generate_test_predictions(). Trains 4 models using 5-Fold CV, computes OOF probabilities, finds the optimal probability threshold, and returns all metrics."),
    ("src/statistical_tests.py",
     "Statistics Module",
     "Contains run_ttest(), run_anova(), and correlation_analysis(). Provides Tab 2 with T-Tests (comparing means between the readmitted vs not groups) and ANOVA (testing whether diagnosis type significantly affects readmission)."),
    ("src/pca_analysis.py",
     "Dimensionality Reduction",
     "Contains apply_pca(). Uses Principal Component Analysis to compress 26 feature columns into 2 principal components that can be plotted on a 2D chart, revealing whether readmitted patients cluster together."),
    ("train_df.csv",
     "Training Dataset",
     "5,000 historical patient records WITH the readmitted label. The model studies this data during training to learn patterns."),
    ("test_df.csv",
     "Test Dataset",
     "2,000 unseen patient records WITHOUT the readmitted label. Used at the end to generate a prediction CSV file (final_submission.csv)."),
    ("requirements.txt",
     "Dependencies",
     "Lists all Python libraries needed to run the project (streamlit, pandas, scikit-learn, matplotlib, seaborn, numpy)."),
]

add_table(doc,
    ["File / Path", "Role", "Description"],
    [(f[0], f[1], f[2]) for f in files]
)
doc.add_paragraph("")

# =============================================================================
# 8. DASHBOARD TABS
# =============================================================================
add_heading(doc, "8. Dashboard Tabs Explained", 1, color=(2, 132, 199))

tabs = [
    ("Tab 1: Data Overview",
     "Shows raw statistics about the training dataset. Includes class distribution bar chart (81.2% vs 18.8%), distribution plots for age, days in hospital, comorbidity, and a data sample table. Sidebar shows key metrics."),
    ("Tab 2: Statistical Tests",
     "Runs formal statistical hypothesis tests. T-Test: Is the average age of readmitted patients significantly higher? ANOVA: Does the type of diagnosis significantly affect readmission rates? Includes a Pearson Correlation Heatmap."),
    ("Tab 3: PCA Analysis",
     "Principal Component Analysis visualization. Compresses 26 model features into 2 axes (PC1 and PC2). The scatter plot shows whether readmitted (red) and non-readmitted (blue) patients separate into distinct clusters."),
    ("Tab 4: Model Performance",
     "The core analytics tab. Displays: headline accuracy/precision/recall/F1 metrics, the confusion matrix (TP/FP/TN/FN), per-fold breakdown table, threshold simulator slider, benchmark leaderboard comparing all 4 models, and dual feature importance charts."),
    ("Tab 5: Risk Assessor",
     "Interactive patient risk calculator. Enter any patient's clinical details using sliders and dropdowns. The model instantly computes a readmission risk percentage using both the ML model probability AND a calibrated logistic sigmoid formula with domain-knowledge weights."),
]

for name, desc in tabs:
    add_body(doc, f"\n{name}", bold=True)
    add_body(doc, desc)

doc.add_paragraph("")

# =============================================================================
# 9. KEY TERMINOLOGY
# =============================================================================
add_heading(doc, "9. Key Terminology Glossary", 1, color=(2, 132, 199))

terms = [
    ("Cross-Validation (5-Fold CV)", "The model is tested on 5 different non-overlapping chunks of the data. Each chunk is used as a test set exactly once. This gives 5 independent accuracy measurements and prevents overfitting."),
    ("Overfitting", "When a model 'memorizes' the training data instead of learning patterns. It performs well on training data but fails on new patients."),
    ("Class Imbalance", "When one category (Not Readmitted: 81.2%) is far more common than the other (Readmitted: 18.8%). Makes training harder without special handling."),
    ("Probability Threshold", "The cutoff probability above which a patient is classified as 'High Risk'. Default is 0.5, but CarePulse optimizes this to ~0.55 to achieve ~80% accuracy while keeping non-zero recall."),
    ("One-Hot Encoding", "Converting text categories into numbers. 'gender=Male' becomes [1,0], 'gender=Female' becomes [0,1]. Machine learning requires all inputs to be numeric."),
    ("Standard Scaling", "Transforming each numeric column so it has mean=0 and standard deviation=1. Prevents large-scale features (age: 0-100) from dominating small-scale ones (comorbidity: 1-10)."),
    ("ROC-AUC", "Area Under the Receiver Operating Characteristic curve. Ranges from 0.5 (random) to 1.0 (perfect). Measures overall model discrimination ability regardless of threshold."),
    ("OOF (Out-Of-Fold)", "Predictions made on data the model was NOT trained on during that CV fold. OOF predictions are unbiased estimates of real-world model performance."),
    ("Feature Importance", "A score (0 to 1) for each input feature showing how much it contributed to the model's predictions. Higher = more useful for predicting readmission."),
    ("Logistic Sigmoid", "The S-shaped mathematical function 1/(1+e^(-z)) that converts any real number into a probability between 0% and 100%."),
]

add_table(doc,
    ["Term", "Definition"],
    terms
)
doc.add_paragraph("")

# =============================================================================
# FOOTER
# =============================================================================
doc.add_page_break()
footer_p = doc.add_paragraph()
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = footer_p.add_run("CarePulse — Patient Readmission Prediction System\n")
r.bold = True
r.font.color.rgb = RGBColor(2, 132, 199)
footer_p.add_run("Built with Python, scikit-learn, Streamlit | github.com/sameer-121623/CarePulse")

doc.save(OUTPUT)
print(f"Word document saved to:\n{OUTPUT}")
