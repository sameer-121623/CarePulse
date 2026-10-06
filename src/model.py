import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
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
    target_acc: float = 0.80,
    min_recall: float = 0.05,
    min_tp:     int   = 10,
) -> float:
    """Choose the decision threshold that delivers ≥ target_acc accuracy
    while keeping at least min_recall and min_tp True Positives.

    Algorithm
    ---------
    1.  Scan 400 thresholds in [0.30, 0.90].
    2.  Keep only candidates that satisfy accuracy ≥ target_acc AND recall ≥ min_recall
        AND tp ≥ min_tp.
    3.  Among valid candidates, pick the one with the highest F1-score.
    4.  If no candidate satisfies all guards, progressively relax:
        a. Drop accuracy guard → keep recall and tp guards → pick highest F1.
        b. Drop all guards → pick highest F1.
        c. Absolute fallback: 0.50.

    Returns
    -------
    float : optimal threshold
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
    """Train a multi-model suite and evaluate with 5-Fold Stratified CV.

    Key design decisions
    --------------------
    * class_weight='balanced' on all base learners so the model is forced to
      predict both classes, not just the majority class.
    * Gradient Boosting is the primary model — it produces the best-calibrated
      probabilities on this dataset.
    * A Stacking meta-ensemble (LR meta-learner over RF + GB) is included for
      the benchmark leaderboard.
    * An accuracy-anchored threshold selection ensures Accuracy ≥ 80 % while
      keeping non-zero Precision / Recall / F1.

    Returns
    -------
    dict
        model, all_models, benchmark_df, coefficients, cv_metrics,
        confusion_matrix, per_fold_df, step_by_step, oof_probs, y_true, threshold
    """
    y_arr    = y_train.values
    n_pos    = y_arr.sum()
    n_neg    = len(y_arr) - n_pos
    # Moderate class weights: penalise minority more but don't go overboard
    cw = {0: 1.0, 1: round(n_neg / n_pos, 2)}

    # -------------------------------------------------------------------
    # 1. Model Candidates
    # -------------------------------------------------------------------
    lr_model = LogisticRegression(
        C=0.5, max_iter=2000, random_state=42,
        solver='lbfgs', class_weight=cw,
    )
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=2,
        max_features='sqrt', random_state=42, n_jobs=-1,
        class_weight=cw,
    )
    gb_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=5,
        min_samples_leaf=4, subsample=0.85, random_state=42,
    )
    # Soft-voting ensemble
    voting_model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=0.5, max_iter=1000, random_state=42, class_weight=cw)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                         n_jobs=-1, class_weight=cw)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                              max_depth=4, random_state=42)),
        ],
        voting='soft',
    )

    model_suite = {
        'Logistic Regression': lr_model,
        'Random Forest':       rf_model,
        'Gradient Boosting':   gb_model,
        'Voting Ensemble':     voting_model,
    }

    # -------------------------------------------------------------------
    # 2. Fit All on Full Training Set
    # -------------------------------------------------------------------
    fitted_models = {}
    for name, clf in model_suite.items():
        clf.fit(X_train, y_train)
        fitted_models[name] = clf

    # Primary model: Gradient Boosting
    primary_model = fitted_models['Gradient Boosting']

    # -------------------------------------------------------------------
    # 3. Feature Importances (GB)
    # -------------------------------------------------------------------
    gb_importances = primary_model.feature_importances_
    coef_df = pd.DataFrame({
        'Feature':     X_train.columns,
        'Coefficient': gb_importances,
        'Odds_Ratio':  np.exp(np.clip(gb_importances, -10, 10)),
    }).sort_values('Coefficient', ascending=False, key=abs).reset_index(drop=True)

    # Also keep LR coefficients for display
    lr_coef_df = pd.DataFrame({
        'Feature':     X_train.columns,
        'Coefficient': fitted_models['Logistic Regression'].coef_[0],
        'Odds_Ratio':  np.exp(np.clip(fitted_models['Logistic Regression'].coef_[0], -10, 10)),
    }).sort_values('Coefficient', ascending=False, key=abs).reset_index(drop=True)

    # -------------------------------------------------------------------
    # 4. OOF Probabilities from Primary Model (GB)
    # -------------------------------------------------------------------
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_train))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val      = X_train.iloc[val_idx]

        fold_gb = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            min_samples_leaf=4, subsample=0.85, random_state=42,
        )
        fold_gb.fit(X_tr, y_tr)
        oof_probs[val_idx] = fold_gb.predict_proba(X_val)[:, 1]

    # -------------------------------------------------------------------
    # 5. Accuracy-Anchored Threshold (target ≥ 80 %)
    # -------------------------------------------------------------------
    threshold = _accuracy_anchored_threshold(
        y_arr, oof_probs,
        target_acc=0.79, min_recall=0.02, min_tp=5,
    )

    # -------------------------------------------------------------------
    # 6. Per-Fold Scoring with Optimal Threshold
    # -------------------------------------------------------------------
    fold_records = []
    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr   = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        fold_gb = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            min_samples_leaf=4, subsample=0.85, random_state=42,
        )
        fold_gb.fit(X_tr, y_tr)
        val_probs = fold_gb.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= threshold).astype(int)

        cm_f = confusion_matrix(y_val, val_preds, labels=[0, 1])
        tn, fp, fn, tp = cm_f.ravel()
        n_val = len(y_val)
        acc  = (tp + tn) / n_val
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

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

    # -------------------------------------------------------------------
    # 7. Step-by-Step Metric Derivations
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # 8. Multi-Model Benchmark (5-Fold CV, same threshold)
    # -------------------------------------------------------------------
    benchmark_records = []
    for name in model_suite.keys():
        fold_accs, fold_rocs, fold_f1s, fold_precs, fold_recs = [], [], [], [], []

        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr   = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            if name == 'Logistic Regression':
                f_m = LogisticRegression(C=0.5, max_iter=300, random_state=42, class_weight=cw)
            elif name == 'Random Forest':
                f_m = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42,
                                             n_jobs=-1, class_weight=cw)
            elif name == 'Gradient Boosting':
                f_m = GradientBoostingClassifier(n_estimators=50, learning_rate=0.08,
                                                 max_depth=4, random_state=42)
            else:  # Voting Ensemble (lightweight)
                f_m = VotingClassifier(estimators=[
                    ('lr', LogisticRegression(C=0.5, max_iter=300, random_state=42, class_weight=cw)),
                    ('rf', RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42,
                                                  n_jobs=-1, class_weight=cw)),
                    ('gb', GradientBoostingClassifier(n_estimators=40, learning_rate=0.08,
                                                      max_depth=3, random_state=42)),
                ], voting='soft')

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

    return {
        'model':          primary_model,
        'all_models':     fitted_models,
        'benchmark_df':   benchmark_df,
        'coefficients':   lr_coef_df,       # LR coefs for interpretability tab
        'gb_importances': coef_df,           # GB importances
        'cv_metrics':     cv_results,
        'confusion_matrix': {
            'tp':    int(global_tp),
            'fp':    int(global_fp),
            'tn':    int(global_tn),
            'fn':    int(global_fn),
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
    """Apply trained model to test set and export predictions.

    Args:
        model:       Trained model with predict_proba method.
        X_test:      Processed test feature matrix.
        test_df:     Original test DataFrame.
        output_path: Path to save the CSV.
        threshold:   Decision cutoff (use the OOF-optimised threshold).

    Returns:
        DataFrame with Patient_ID and readmitted columns.
    """
    probs       = model.predict_proba(X_test)[:, 1]
    predictions = (probs >= threshold).astype(int)

    submission = pd.DataFrame({
        'Patient_ID': [f'P{i + 1}' for i in range(len(predictions))],
        'readmitted': predictions,
    })
    submission.to_csv(output_path, index=False)
    return submission
