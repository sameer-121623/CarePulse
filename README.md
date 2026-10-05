# 💙 CarePulse — Healthcare Patient Readmission & Risk Predictor

CarePulse is an end-to-end clinical decision-support application built with **Python**, **scikit-learn**, and **Streamlit**. It predicts and stratifies 30-day patient hospital readmission risk using clinical data, statistical hypothesis testing, dimensionality reduction (PCA), and machine learning.

---

## 🌟 Key Features

1. **📊 Patient Cohort Overview & EDA**:
   - Demographic composition, clinical distributions (age, length of stay, procedures, comorbidities).
   - Readmission outcome distribution and categorical breakdowns.
2. **🔬 Statistical Hypothesis Testing**:
   - Welch's Two-Sample T-Test on length of stay.
   - One-Way ANOVA across primary diagnosis categories.
   - Pearson correlation matrix and clinical significance interpretations.
3. **🧬 Dimensionality Landscape (PCA)**:
   - 2D feature projection capturing dominant variance across patient records.
4. **📊 Model Intelligence & Cross-Validation**:
   - Logistic Regression trained on clinical features.
   - 5-Fold Stratified Cross-Validation evaluation.
   - Feature coefficient weights and odds ratio interpretation.
   - One-click prediction generation for unseen test cohorts (`final_submission.csv`).
5. **🩺 Clinical Risk Assessor**:
   - Real-time patient readmission risk calculator.
   - Evidence-based risk stratification (🟢 Low, 🟡 Moderate, 🔴 High).
   - Actionable clinical advisories for discharge planning.

---

## 📁 Project Architecture

```
project/
├── app.py                      # Main Streamlit clinical dashboard
├── requirements.txt            # Project dependencies
├── train_df.csv                # Training cohort dataset (5,000 records)
├── test_df.csv                 # Unseen test cohort (2,000 records)
├── final_submission.csv        # Generated test predictions
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── src/
    ├── __init__.py
    ├── preprocessing.py        # Data cleaning, alignment & standard scaling
    ├── statistical_tests.py    # T-test, ANOVA & Pearson correlation
    ├── pca_analysis.py         # Principal Component Analysis
    └── model.py                # Logistic Regression & 5-Fold CV pipeline
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/carepulse.git
cd carepulse
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit web application
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **Machine Learning**: scikit-learn
- **Scientific Computing**: NumPy, SciPy, pandas
- **Data Visualization**: Matplotlib, Seaborn
