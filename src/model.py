import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Threshold Selection
# ---------------------------------------------------------------------------

def _accuracy_anchored_threshold(
    y_true: np.ndarray,
    probs:  np.ndarray,
    target_acc: float = 0.79,
    min_recall: float = 0.02,
    min_tp:     int   = 5,
) -> float:
    """Choose the decision threshold that delivers ≥ target_acc accuracy
    while keeping at least min_recall and min_tp True Positives.
    """
    thresholds = np.linspace(0.30, 0.90, 400)
    candidates = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        cm    = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        n    = len(y_true)
        acc  = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        candidates.append({
            't': t, 'acc': acc, 'prec': prec,
            'rec': rec, 'f1': f1, 'tp': int(tp),
        })

    def _best(pool):
        return max(pool, key=lambda x: x['f1'])['t'] if pool else None

    # Primary: all guards
    pool = [c for c in candidates
            if c['acc'] >= target_acc and c['rec'] >= min_recall and c['tp'] >= min_tp]
    if pool:
        return float(_best(pool))

    # Relax-1: drop accuracy guard
    pool = [c for c in candidates if c['rec'] >= min_recall and c['tp'] >= min_tp]
    if pool:
        return float(_best(pool))

    # Relax-2: just need tp > 0
    pool = [c for c in candidates if c['tp'] > 0]
    if pool:
        return float(_best(pool))

    return 0.50


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train and evaluate with 5-Fold Stratified CV. Optimised for speed.

    Speed improvements vs naive approach:
    - Single OOF pass (no duplicate CV loops)
    - Lightweight estimator counts for benchmark (50 trees instead of 200)
    - Full-dataset models trained only once at the end using the best params
    - Gradient Boosting used for OOF only (best probability calibration)
    """
    y_arr = y_train.values
    n_pos = y_arr.sum()
    n_neg = len(y_arr) - n_pos
    cw    = {0: 1.0, 1: round(n_neg / n_pos, 2)}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -----------------------------------------------------------------------
    # 1. Single OOF pass: collect probs + per-fold scores simultaneously
    # -----------------------------------------------------------------------
    oof_probs    = np.zeros(len(y_train))
    fold_records_pre = []   # store probs + true for per-fold scoring after threshold

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr   = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Primary model: lightweight GB
        fold_gb = GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.08, max_depth=4,
            min_samples_leaf=4, subsample=0.85, random_state=42,
        )
        fold_gb.fit(X_tr, y_tr)
        probs_val = fold_gb.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs_val
        fold_records_pre.append((y_val.values, probs_val))

    # -----------------------------------------------------------------------
    # 2. Find optimal threshold from OOF probs
    # -----------------------------------------------------------------------
    threshold = _accuracy_anchored_threshold(y_arr, oof_probs)

    # -----------------------------------------------------------------------
    # 3. Score each fold with the optimal threshold (no re-training!)
    # -----------------------------------------------------------------------
    fold_records = []
    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    for fold, (y_val, probs_val) in enumerate(fold_records_pre):
        preds = (probs_val >= threshold).astype(int)
        cm_f  = confusion_matrix(y_val, preds, labels=[0, 1])
        tn, fp, fn, tp = cm_f.ravel()
        n_val = len(y_val)
        acc   = (tp + tn) / n_val
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        acc_scores.append(acc);  prec_scores.append(prec)
        rec_scores.append(rec);  f1_scores.append(f1)
        fold_records.append({
            'Fold': f'Fold {fold + 1}',
            'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
            'Total': int(n_val),
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1,
        })

    per_fold_df = pd.DataFrame(fold_records)

    # Global OOF confusion matrix
    oof_preds_global = (oof_probs >= threshold).astype(int)
    cm_g = confusion_matrix(y_arr, oof_preds_global, labels=[0, 1])
    global_tn, global_fp, global_fn, global_tp = cm_g.ravel()
    total_samples = len(y_arr)
    global_acc  = (global_tp + global_tn) / total_samples
    global_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_rec  = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    global_f1   = (
        2 * global_prec * global_rec / (global_prec + global_rec)
        if (global_prec + global_rec) > 0 else 0.0
    )

    cv_results = {
        'accuracy':  {'mean': float(np.mean(acc_scores)),  'std': float(np.std(acc_scores)),  'all_folds': list(map(float, acc_scores))},
        'precision': {'mean': float(np.mean(prec_scores)), 'std': float(np.std(prec_scores)), 'all_folds': list(map(float, prec_scores))},
        'recall':    {'mean': float(np.mean(rec_scores)),  'std': float(np.std(rec_scores)),  'all_folds': list(map(float, rec_scores))},
        'f1':        {'mean': float(np.mean(f1_scores)),   'std': float(np.std(f1_scores)),   'all_folds': list(map(float, f1_scores))},
    }

    # -----------------------------------------------------------------------
    # 4. Step-by-Step derivations
    # -----------------------------------------------------------------------
    step_by_step = {
        'accuracy': {
            'formula':     r'\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}',
            'calculation': f'({global_tp:,} + {global_tn:,}) / {total_samples:,}',
            'value':       global_acc,
            'percentage':  f'{global_acc * 100:.2f}%',
            'explanation': 'Proportion of all patient cases correctly classified.',
        },
        'precision': {
            'formula':     r'\text{Precision} = \frac{TP}{TP + FP}',
            'calculation': f'{global_tp:,} / ({global_tp:,} + {global_fp:,})',
            'value':       global_prec,
            'percentage':  f'{global_prec * 100:.2f}%',
            'explanation': 'When the model flags readmission, how often it is correct.',
        },
        'recall': {
            'formula':     r'\text{Recall} = \frac{TP}{TP + FN}',
            'calculation': f'{global_tp:,} / ({global_tp:,} + {global_fn:,})',
            'value':       global_rec,
            'percentage':  f'{global_rec * 100:.2f}%',
            'explanation': 'Percentage of truly readmitted patients caught by the model.',
        },
        'f1': {
            'formula':     r'\text{F1} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}',
            'calculation': f'2 × {global_tp:,} / (2×{global_tp:,} + {global_fp:,} + {global_fn:,})',
            'value':       global_f1,
            'percentage':  f'{global_f1 * 100:.2f}%',
            'explanation': 'Harmonic mean of Precision and Recall.',
        },
    }

    # -----------------------------------------------------------------------
    # 5. Benchmark: lightweight CV for each model (50 trees, same threshold)
    # -----------------------------------------------------------------------
    model_specs = {
        'Logistic Regression': lambda: LogisticRegression(C=0.5, max_iter=300, random_state=42, class_weight=cw),
        'Random Forest':       lambda: RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1, class_weight=cw),
        'Gradient Boosting':   lambda: GradientBoostingClassifier(n_estimators=50, learning_rate=0.08, max_depth=4, random_state=42),
        'Voting Ensemble':     lambda: VotingClassifier(estimators=[
            ('lr', LogisticRegression(C=0.5, max_iter=200, random_state=42, class_weight=cw)),
            ('rf', RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1, class_weight=cw)),
            ('gb', GradientBoostingClassifier(n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42)),
        ], voting='soft'),
    }

    benchmark_records = []
    for name, make_model in model_specs.items():
        fold_accs, fold_rocs, fold_f1s, fold_precs, fold_recs = [], [], [], [], []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr   = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            f_m = make_model()
            f_m.fit(X_tr, y_tr)
            probs = f_m.predict_proba(X_val)[:, 1]
            preds = (probs >= threshold).astype(int)
            fold_accs.append(accuracy_score(y_val, preds))
            fold_rocs.append(roc_auc_score(y_val, probs))
            fold_f1s.append(f1_score(y_val, preds, zero_division=0))
            fold_precs.append(precision_score(y_val, preds, zero_division=0))
            fold_recs.append(recall_score(y_val, preds, zero_division=0))
        benchmark_records.append({
            'Model Architecture': name,
            'Accuracy':  float(np.mean(fold_accs)),
            'Precision': float(np.mean(fold_precs)),
            'Recall':    float(np.mean(fold_recs)),
            'F1-Score':  float(np.mean(fold_f1s)),
            'ROC-AUC':   float(np.mean(fold_rocs)),
            'Std Dev':   float(np.std(fold_accs)),
        })

    benchmark_df = (
        pd.DataFrame(benchmark_records)
        .sort_values('Accuracy', ascending=False)
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------------
    # 6. Fit final models on FULL training set (once, at the end)
    # -----------------------------------------------------------------------
    final_gb = GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.05, max_depth=5,
        min_samples_leaf=4, subsample=0.85, random_state=42,
    )
    final_rf = RandomForestClassifier(
        n_estimators=120, max_depth=8, min_samples_leaf=2,
        max_features='sqrt', random_state=42, n_jobs=-1, class_weight=cw,
    )
    final_lr = LogisticRegression(C=0.5, max_iter=500, random_state=42, class_weight=cw)
    final_voting = VotingClassifier(estimators=[
        ('lr', LogisticRegression(C=0.5, max_iter=300, random_state=42, class_weight=cw)),
        ('rf', RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1, class_weight=cw)),
        ('gb', GradientBoostingClassifier(n_estimators=60, learning_rate=0.08, max_depth=4, random_state=42)),
    ], voting='soft')

    final_gb.fit(X_train, y_train)
    final_rf.fit(X_train, y_train)
    final_lr.fit(X_train, y_train)
    final_voting.fit(X_train, y_train)

    fitted_models = {
        'Logistic Regression': final_lr,
        'Random Forest':       final_rf,
        'Gradient Boosting':   final_gb,
        'Voting Ensemble':     final_voting,
    }

    # Feature importances
    gb_importances = final_gb.feature_importances_
    coef_df = pd.DataFrame({
        'Feature':     X_train.columns,
        'Coefficient': gb_importances,
        'Odds_Ratio':  np.exp(np.clip(gb_importances, -10, 10)),
    }).sort_values('Coefficient', ascending=False, key=abs).reset_index(drop=True)

    lr_coef_df = pd.DataFrame({
        'Feature':     X_train.columns,
        'Coefficient': final_lr.coef_[0],
        'Odds_Ratio':  np.exp(np.clip(final_lr.coef_[0], -10, 10)),
    }).sort_values('Coefficient', ascending=False, key=abs).reset_index(drop=True)

    return {
        'model':          final_gb,
        'all_models':     fitted_models,
        'benchmark_df':   benchmark_df,
        'coefficients':   lr_coef_df,
        'gb_importances': coef_df,
        'cv_metrics':     cv_results,
        'confusion_matrix': {
            'tp': int(global_tp), 'fp': int(global_fp),
            'tn': int(global_tn), 'fn': int(global_fn),
            'total': int(total_samples),
        },
        'per_fold_df':  per_fold_df,
        'step_by_step': step_by_step,
        'oof_probs':    oof_probs,
        'y_true':       y_arr,
        'threshold':    threshold,
    }


# ---------------------------------------------------------------------------
# Prediction Export
# ---------------------------------------------------------------------------

def generate_test_predictions(
    model,
    X_test:      pd.DataFrame,
    test_df:     pd.DataFrame,
    output_path: str   = 'final_submission.csv',
    threshold:   float = 0.50,
) -> pd.DataFrame:
    """Apply trained model to test set and export predictions."""
    probs       = model.predict_proba(X_test)[:, 1]
    predictions = (probs >= threshold).astype(int)
    submission  = pd.DataFrame({
        'Patient_ID': [f'P{i + 1}' for i in range(len(predictions))],
        'readmitted': predictions,
    })
    submission.to_csv(output_path, index=False)
    return submission
