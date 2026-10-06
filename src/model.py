import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train multiple advanced machine learning models and evaluate with 5-Fold Cross Validation.
    
    Trains a comprehensive benchmark suite:
    1. Optimized Logistic Regression (L2-Regularized, 2,000 max iterations)
    2. Random Forest Classifier (200 Estimators, Depth-Constrained)
    3. Gradient Boosting Classifier (120 Iterative Boosted Trees)
    4. Soft-Voting Ensemble (Consensus Probability Integration)
    
    Args:
        X_train: Processed training feature matrix.
        y_train: Binary target variable (0/1).
        
    Returns:
        Dictionary with:
        - 'model': Primary trained model (Logistic Regression)
        - 'all_models': Dictionary containing all trained candidate models
        - 'benchmark_df': Performance comparison leaderboard across all models
        - 'coefficients': Feature names, coefficients, and odds ratios
        - 'cv_metrics': Mean and std for accuracy, precision, recall, f1
        - 'confusion_matrix': Global out-of-fold confusion matrix {tp, fp, tn, fn, total}
        - 'per_fold_df': Per-fold evaluation metrics
        - 'step_by_step': Step-by-step arithmetic derivations
        - 'oof_probs': Out-of-fold predicted probabilities
        - 'y_true': True target labels
    """
    # 1. Instantiate multi-model candidate suite
    lr_model = LogisticRegression(C=0.1, max_iter=2000, random_state=42, solver='lbfgs')
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.03, max_depth=3, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[
            ('logistic_regression', lr_model),
            ('random_forest', rf_model),
            ('gradient_boosting', gb_model)
        ],
        voting='soft'
    )
    
    model_suite = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Voting Ensemble': ensemble_model
    }
    
    # 2. Fit all candidate models on the full training set
    fitted_models = {}
    for name, clf in model_suite.items():
        clf.fit(X_train, y_train)
        fitted_models[name] = clf
        
    primary_model = fitted_models['Logistic Regression']
    
    # 3. Feature coefficients for linear interpretation
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': primary_model.coef_[0],
        'Odds_Ratio': np.exp(primary_model.coef_[0])
    }).sort_values(by='Coefficient', ascending=False, key=abs).reset_index(drop=True)
    
    # 4. Multi-Model 5-Fold Stratified Cross-Validation Benchmark
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_records = []
    
    for name, clf in model_suite.items():
        fold_accs, fold_rocs, fold_f1s = [], [], []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Clone and fit fold model
            if name == 'Voting Ensemble':
                f_m = VotingClassifier(
                    estimators=[
                        ('lr', LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
                        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
                        ('gb', GradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42))
                    ],
                    voting='soft'
                )
            elif name == 'Logistic Regression':
                f_m = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
            elif name == 'Random Forest':
                f_m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            else:
                f_m = GradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42)
                
            f_m.fit(X_tr, y_tr)
            preds = f_m.predict(X_val)
            probs = f_m.predict_proba(X_val)[:, 1]
            
            fold_accs.append(accuracy_score(y_val, preds))
            fold_rocs.append(roc_auc_score(y_val, probs))
            fold_f1s.append(f1_score(y_val, preds, zero_division=0))
            
        benchmark_records.append({
            'Model Architecture': name,
            'Training Setup': '2,000 Iterations' if 'Logistic' in name else ('200 Trees' if 'Forest' in name else ('120 Boosted Trees' if 'Gradient' in name else 'Consensus Tri-Ensemble')),
            'Accuracy': float(np.mean(fold_accs)),
            'ROC-AUC': float(np.mean(fold_rocs)),
            'F1-Score': float(np.mean(fold_f1s)),
            'Std Dev': float(np.std(fold_accs))
        })
        
    benchmark_df = pd.DataFrame(benchmark_records)
    
    # 5. Out-of-fold detailed evaluation on primary model
    oof_probs = np.zeros(len(y_train))
    oof_preds = np.zeros(len(y_train))
    fold_records = []
    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        fold_lr = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
        fold_lr.fit(X_tr, y_tr)
        
        val_probs = fold_lr.predict_proba(X_val)[:, 1]
        val_preds = fold_lr.predict(X_val)
        
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
    
    step_by_step = {
        'accuracy': {
            'formula': r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}",
            'calculation': f"({global_tp:,} + {global_tn:,}) / ({global_tp:,} + {global_tn:,} + {global_fp:,} + {global_fn:,}) = {global_tp + global_tn:,} / {total_samples:,}",
            'value': global_acc,
            'percentage': f"{global_acc * 100:.2f}%",
            'explanation': "Proportion of all patient cases correctly classified."
        },
        'precision': {
            'formula': r"\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{True Positives}}{\text{All Predicted Positives}}",
            'calculation': f"{global_tp:,} / ({global_tp:,} + {global_fp:,})" if (global_tp + global_fp) > 0 else "0 / 0 (No positive predictions at default 0.50 threshold)",
            'value': global_prec,
            'percentage': f"{global_prec * 100:.2f}%",
            'explanation': "When the model flags a patient for readmission, the likelihood that they are readmitted."
        },
        'recall': {
            'formula': r"\text{Recall (Sensitivity)} = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{All Actual Positives}}",
            'calculation': f"{global_tp:,} / ({global_tp:,} + {global_fn:,}) = {global_tp:,} / {global_tp + global_fn:,}",
            'value': global_rec,
            'percentage': f"{global_rec * 100:.2f}%",
            'explanation': "Percentage of truly readmitted patients successfully captured by the model."
        },
        'f1': {
            'formula': r"\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}",
            'calculation': f"2 * ({global_prec:.4f} * {global_rec:.4f}) / ({global_prec:.4f} + {global_rec:.4f})" if (global_prec + global_rec) > 0 else "0.0000",
            'value': global_f1,
            'percentage': f"{global_f1 * 100:.2f}%",
            'explanation': "Harmonic mean balancing precision against recall."
        }
    }
    
    return {
        'model': primary_model,
        'all_models': fitted_models,
        'benchmark_df': benchmark_df,
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
    
    Args:
        model: Trained model with predict method.
        X_test: Processed test feature matrix.
        test_df: Original test DataFrame.
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
