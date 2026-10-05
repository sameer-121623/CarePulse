import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train Logistic Regression and evaluate with 5-Fold Cross Validation.
    
    Steps:
    1. Train LogisticRegression with max_iter=1000, random_state=42,
       class_weight='balanced' (to handle class imbalance).
    2. Extract odds ratios (exp of coefficients) and feature importance.
    3. Run 5-Fold CV for accuracy, precision, recall, and f1.
    
    Args:
        X_train: Processed training feature matrix.
        y_train: Binary target variable (0/1).
        
    Returns:
        Dictionary with:
        - 'model': Trained LogisticRegression model
        - 'coefficients': DataFrame with feature names, coefficients, and odds ratios
        - 'cv_metrics': Dictionary with mean and std for accuracy, precision, recall, f1
    """
    # Train model with balanced class weights to handle imbalanced data
    model = LogisticRegression(
        max_iter=1000, random_state=42, solver='lbfgs', class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Feature coefficients and odds ratios
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0],
        'Odds_Ratio': np.exp(model.coef_[0])
    }).sort_values(by='Coefficient', ascending=False, key=abs).reset_index(drop=True)
    
    # 5-Fold Cross Validation with custom scorers
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
    }
    
    cv_results = {}
    for metric_name, scorer in scoring_metrics.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
        cv_results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'all_folds': scores.tolist()
        }
    
    return {
        'model': model,
        'coefficients': coef_df,
        'cv_metrics': cv_results
    }


def generate_test_predictions(model, X_test: pd.DataFrame, test_df: pd.DataFrame,
                              output_path: str = 'final_submission.csv') -> pd.DataFrame:
    """Generate predictions for test data and save to CSV.
    
    Creates a submission file with Patient_ID and readmitted columns,
    matching the sample_submission.csv format.
    
    Args:
        model: Trained model with predict method.
        X_test: Processed test feature matrix.
        test_df: Original test DataFrame (for Patient_ID generation).
        output_path: Path to save the submission CSV.
        
    Returns:
        Submission DataFrame with Patient_ID and readmitted columns.
    """
    predictions = model.predict(X_test)
    
    submission = pd.DataFrame({
        'Patient_ID': [f'P{i+1}' for i in range(len(predictions))],
        'readmitted': predictions.astype(int)
    })
    
    submission.to_csv(output_path, index=False)
    
    return submission
