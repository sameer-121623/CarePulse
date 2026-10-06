import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


NUMERICAL_COLS   = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
CATEGORICAL_COLS = ['gender', 'primary_diagnosis', 'discharge_to']

# Engineered feature names (raw, before scaling)
ENGINEERED_COLS  = [
    'age_x_comorbidity', 'days_x_procedures', 'age_x_days',
    'comorbidity_x_procedures',
    'high_comorbidity', 'prolonged_stay', 'many_procedures', 'elderly',
    'discharge_risk_score', 'diagnosis_risk_score',
    'composite_risk',
]


# ── Domain-knowledge risk look-ups ──────────────────────────────────────────
_DISCHARGE_RISK = {
    'Home':                     0.0,
    'Home Health Care':         0.3,
    'Rehabilitation Facility':  0.6,
    'Skilled Nursing Facility': 1.0,
}
_DIAGNOSIS_RISK = {
    'Hypertension':  0.0,
    'Diabetes':      0.2,
    'COPD':          0.5,
    'Heart Disease': 0.8,
    'Kidney Disease':1.0,
}


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically-motivated interaction and risk features.

    All new columns are appended to the DataFrame. The original raw columns are
    kept so one-hot encoding still works for categorical variables.

    New features
    ------------
    Interaction terms:
        age_x_comorbidity       age × comorbidity_score
        days_x_procedures       days_in_hospital × num_procedures
        age_x_days              age × days_in_hospital
        comorbidity_x_procedures comorbidity_score × num_procedures

    Binary risk flags (0/1):
        high_comorbidity        comorbidity_score >= 4
        prolonged_stay          days_in_hospital >= 7
        many_procedures         num_procedures >= 5
        elderly                 age >= 65

    Ordinal look-up scores (0-1 range):
        discharge_risk_score    from _DISCHARGE_RISK map
        diagnosis_risk_score    from _DIAGNOSIS_RISK map

    Composite score:
        composite_risk          weighted sum of all risk signals
    """
    d = df.copy()

    # Interaction terms
    d['age_x_comorbidity']        = d['age'] * d['comorbidity_score']
    d['days_x_procedures']        = d['days_in_hospital'] * d['num_procedures']
    d['age_x_days']               = d['age'] * d['days_in_hospital']
    d['comorbidity_x_procedures'] = d['comorbidity_score'] * d['num_procedures']

    # Binary risk flags
    d['high_comorbidity'] = (d['comorbidity_score'] >= 4).astype(float)
    d['prolonged_stay']   = (d['days_in_hospital']  >= 7).astype(float)
    d['many_procedures']  = (d['num_procedures']     >= 5).astype(float)
    d['elderly']          = (d['age']               >= 65).astype(float)

    # Domain risk look-ups
    d['discharge_risk_score'] = d['discharge_to'].map(_DISCHARGE_RISK).fillna(0.3)
    d['diagnosis_risk_score'] = d['primary_diagnosis'].map(_DIAGNOSIS_RISK).fillna(0.3)

    # Composite risk score (hand-calibrated weights)
    d['composite_risk'] = (
        0.30 * d['diagnosis_risk_score']
        + 0.25 * d['discharge_risk_score']
        + 0.20 * (d['comorbidity_score'] / 10.0)
        + 0.15 * (d['days_in_hospital']  / 30.0)
        + 0.10 * (d['age']               / 100.0)
    )

    return d


def load_data(train_path: str, test_path: str) -> tuple:
    """Load train and test CSV files.

    Args:
        train_path: Path to training CSV.
        test_path:  Path to test CSV.

    Returns:
        Tuple of (train_df, test_df) DataFrames.
    """
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df


def clean_and_encode_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Clean, engineer features, encode, and scale data for modeling.

    Pipeline
    --------
    1.  Drop duplicate rows from training data.
    2.  Separate target variable 'readmitted'.
    3.  Engineer interaction / risk features (see _engineer_features).
    4.  One-Hot Encode categorical columns.
    5.  Align train/test columns (train as reference).
    6.  StandardScale all numerical + engineered columns.

    Args:
        train_df: Training DataFrame (with 'readmitted' column).
        test_df:  Test DataFrame (without 'readmitted' column).

    Returns:
        Dictionary with keys:
            'X_train'        : Processed training feature DataFrame
            'y_train'        : Target Series
            'X_test'         : Processed test feature DataFrame
            'scaler'         : Fitted StandardScaler
            'feature_names'  : List of final column names
            'train_df_clean' : Cleaned training DataFrame (pre-encoding, for EDA)
            'num_cols_scaled': All numerical columns that were scaled
    """
    # 1. Drop duplicates
    train_clean = train_df.drop_duplicates().reset_index(drop=True)

    # 2. Separate target
    y_train = train_clean['readmitted'].copy()
    X_raw   = train_clean.drop(columns=['readmitted'])
    X_test_raw = test_df.copy()

    # 3. Feature engineering
    X_eng      = _engineer_features(X_raw)
    X_test_eng = _engineer_features(X_test_raw)

    # 4. One-Hot Encode categorical columns
    X_enc      = pd.get_dummies(X_eng,      columns=CATEGORICAL_COLS, drop_first=False)
    X_test_enc = pd.get_dummies(X_test_eng, columns=CATEGORICAL_COLS, drop_first=False)

    # 5. Align columns
    X_enc, X_test_enc = X_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

    X_enc      = X_enc.astype(float)
    X_test_enc = X_test_enc.astype(float)

    # 6. Scale: original numerical + engineered numerical columns
    continuous_to_scale = [c for c in NUMERICAL_COLS + ENGINEERED_COLS
                           if c in X_enc.columns]
    scaler = StandardScaler()
    X_enc[continuous_to_scale]      = scaler.fit_transform(X_enc[continuous_to_scale])
    X_test_enc[continuous_to_scale] = scaler.transform(X_test_enc[continuous_to_scale])

    return {
        'X_train':         X_enc,
        'y_train':         y_train,
        'X_test':          X_test_enc,
        'scaler':          scaler,
        'feature_names':   list(X_enc.columns),
        'train_df_clean':  train_clean,
        'num_cols_scaled': continuous_to_scale,
    }
