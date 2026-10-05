import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


NUMERICAL_COLS = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
CATEGORICAL_COLS = ['gender', 'primary_diagnosis', 'discharge_to']


def load_data(train_path: str, test_path: str) -> tuple:
    """Load train and test CSV files.
    
    Args:
        train_path: Path to training CSV.
        test_path: Path to test CSV.
        
    Returns:
        Tuple of (train_df, test_df) DataFrames.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def clean_and_encode_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Clean, encode, and scale data for modeling.
    
    Steps:
    1. Drop duplicate rows from training data.
    2. Separate target variable 'readmitted' from training features.
    3. One-Hot Encode categorical columns (gender, primary_diagnosis, discharge_to)
       using pd.get_dummies with drop_first=False.
    4. Align train and test DataFrames to ensure identical feature columns,
       filling missing columns with 0.
    5. Scale numerical columns using StandardScaler (fit on train, transform both).
    
    Args:
        train_df: Training DataFrame with 'readmitted' column.
        test_df: Test DataFrame without 'readmitted' column.
        
    Returns:
        Dictionary with keys:
        - 'X_train': Processed training feature DataFrame (all columns scaled/encoded)
        - 'y_train': Target Series
        - 'X_test': Processed test feature DataFrame
        - 'scaler': Fitted StandardScaler object
        - 'feature_names': List of final feature column names
        - 'train_df_clean': Cleaned training DataFrame (before encoding, for EDA)
    """
    # Step 1: Drop duplicates
    train_clean = train_df.drop_duplicates().reset_index(drop=True)
    
    # Step 2: Separate target
    y_train = train_clean['readmitted'].copy()
    X_train = train_clean.drop(columns=['readmitted'])
    X_test = test_df.copy()
    
    # Step 3: One-Hot Encode categorical columns
    X_train_encoded = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=False)
    X_test_encoded = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, drop_first=False)
    
    # Step 4: Align columns (train as reference)
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded, join='left', axis=1, fill_value=0
    )
    
    # Ensure all values are numeric
    X_train_encoded = X_train_encoded.astype(float)
    X_test_encoded = X_test_encoded.astype(float)
    
    # Step 5: Scale numerical columns
    scaler = StandardScaler()
    X_train_encoded[NUMERICAL_COLS] = scaler.fit_transform(X_train_encoded[NUMERICAL_COLS])
    X_test_encoded[NUMERICAL_COLS] = scaler.transform(X_test_encoded[NUMERICAL_COLS])
    
    feature_names = list(X_train_encoded.columns)
    
    return {
        'X_train': X_train_encoded,
        'y_train': y_train,
        'X_test': X_test_encoded,
        'scaler': scaler,
        'feature_names': feature_names,
        'train_df_clean': train_clean
    }
