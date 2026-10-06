import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train Logistic Regression and evaluate with 5-Fold Cross Validation.
    
    Computes comprehensive evaluation metrics, per-fold confusion matrices,
    and step-by-step mathematical formulas for Accuracy, Precision, Recall, and F1-Score.
    
    Args:
        X_train: Processed training feature matrix.
        y_train: Binary target variable (0/1).
        
    Returns:
        Dictionary with:
        - 'model': Trained LogisticRegression model on full training set
        - 'coefficients': DataFrame with feature names, coefficients, and odds ratios
        - 'cv_metrics': Dictionary with mean and std for accuracy, precision, recall, f1
        - 'confusion_matrix': Global out-of-fold confusion matrix {tp, fp, tn, fn, total}
        - 'per_fold_df': DataFrame with TP, FP, TN, FN and metrics per fold
        - 'step_by_step': Dict with LaTeX / text formulas and actual substituted calculations
        - 'oof_probs': Out-of-fold predicted probabilities for threshold analysis
        - 'y_true': True target labels
    """
    # 1. Train full model with balanced class weights
    model = LogisticRegression(
        max_iter=1000, random_state=42, solver='lbfgs', class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # 2. Feature coefficients and odds ratios
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0],
        'Odds_Ratio': np.exp(model.coef_[0])
    }).sort_values(by='Coefficient', ascending=False, key=abs).reset_index(drop=True)
    
    # 3. 5-Fold Stratified Cross Validation with explicit metric calculation per fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros(len(y_train))
    oof_preds = np.zeros(len(y_train))
    fold_records = []
    
    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        fold_model = LogisticRegression(
            max_iter=1000, random_state=42, solver='lbfgs', class_weight='balanced'
        )
        fold_model.fit(X_tr, y_tr)
        
        val_probs = fold_model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)
        
        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds
        
        tn, fp, fn, tp = confusion_matrix(y_val, val_preds).ravel()
        n_val = len(y_val)
        
        acc = (tp + tn) / n_val
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        acc_scores.append(acc)
        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)
        
        fold_records.append({
            'Fold': f'Fold {fold + 1}',
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn),
            'Total': int(n_val),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
    
    per_fold_df = pd.DataFrame(fold_records)
    
    # 4. Out-of-fold global confusion matrix
    global_tn, global_fp, global_fn, global_tp = confusion_matrix(y_train, oof_preds).ravel()
    total_samples = len(y_train)
    
    global_acc = (global_tp + global_tn) / total_samples
    global_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    global_f1 = (
        2 * (global_prec * global_rec) / (global_prec + global_rec)
        if (global_prec + global_rec) > 0 else 0.0
    )
    
    cv_results = {
        'accuracy': {
            'mean': float(np.mean(acc_scores)),
            'std': float(np.std(acc_scores)),
            'all_folds': [float(x) for x in acc_scores]
        },
        'precision': {
            'mean': float(np.mean(prec_scores)),
            'std': float(np.std(prec_scores)),
            'all_folds': [float(x) for x in prec_scores]
        },
        'recall': {
            'mean': float(np.mean(rec_scores)),
            'std': float(np.std(rec_scores)),
            'all_folds': [float(x) for x in rec_scores]
        },
        'f1': {
            'mean': float(np.mean(f1_scores)),
            'std': float(np.std(f1_scores)),
            'all_folds': [float(x) for x in f1_scores]
        }
    }
    
    # 5. Formatted step-by-step mathematical calculations
    step_by_step = {
        'accuracy': {
            'formula': r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}",
            'calculation': f"({global_tp:,} + {global_tn:,}) / ({global_tp:,} + {global_tn:,} + {global_fp:,} + {global_fn:,}) = {global_tp + global_tn:,} / {total_samples:,}",
            'value': global_acc,
            'percentage': f"{global_acc * 100:.2f}%",
            'explanation': "Proportion of all patients correctly classified as either readmitted or not readmitted."
        },
        'precision': {
            'formula': r"\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{True Positives}}{\text{All Predicted Positives}}",
            'calculation': f"{global_tp:,} / ({global_tp:,} + {global_fp:,}) = {global_tp:,} / {global_tp + global_fp:,}",
            'value': global_prec,
            'percentage': f"{global_prec * 100:.2f}%",
            'explanation': "When the model flags a patient as 'Readmitted', the probability that they will actually be readmitted."
        },
        'recall': {
            'formula': r"\text{Recall (Sensitivity)} = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{All Actual Positives}}",
            'calculation': f"{global_tp:,} / ({global_tp:,} + {global_fn:,}) = {global_tp:,} / {global_tp + global_fn:,}",
            'value': global_rec,
            'percentage': f"{global_rec * 100:.2f}%",
            'explanation': "Of all patients who were truly readmitted, the percentage that the model successfully caught."
        },
        'f1': {
            'formula': r"\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}",
            'calculation': f"2 * ({global_prec:.4f} * {global_rec:.4f}) / ({global_prec:.4f} + {global_rec:.4f}) = (2 * {global_tp:,}) / (2 * {global_tp:,} + {global_fp:,} + {global_fn:,})",
            'value': global_f1,
            'percentage': f"{global_f1 * 100:.2f}%",
            'explanation': "Harmonic mean of Precision and Recall, balancing false alarms against missed readmissions."
        }
    }
    
    return {
        'model': model,
        'coefficients': coef_df,
        'cv_metrics': cv_results,
        'confusion_matrix': {
            'tp': int(global_tp),
            'fp': int(global_fp),
            'tn': int(global_tn),
            'fn': int(global_fn),
            'total': int(total_samples)
        },
        'per_fold_df': per_fold_df,
        'step_by_step': step_by_step,
        'oof_probs': oof_probs,
        'y_true': y_train.values
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
