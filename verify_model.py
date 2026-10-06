import numpy as np
from src.preprocessing import load_data, clean_and_encode_data
from src.model import train_and_evaluate
import os

BASE = r'c:\Users\hp\OneDrive\Desktop\AIML\said\project'
train_df, test_df = load_data(
    os.path.join(BASE, 'train_df.csv'),
    os.path.join(BASE, 'test_df.csv')
)
processed = clean_and_encode_data(train_df, test_df)
print(f"Features after engineering: {len(processed['feature_names'])}")
print(f"Scaled columns: {len(processed['num_cols_scaled'])}")

res = train_and_evaluate(processed['X_train'], processed['y_train'])

cv  = res['cv_metrics']
cm  = res['confusion_matrix']
thr = res['threshold']

print(f"\n=== RESULTS ===")
print(f"Optimal Threshold: {thr:.4f}")
print(f"Accuracy:          {cv['accuracy']['mean']:.4f}  ({cv['accuracy']['mean']*100:.2f}%)")
print(f"Precision:         {cv['precision']['mean']:.4f}  ({cv['precision']['mean']*100:.2f}%)")
print(f"Recall:            {cv['recall']['mean']:.4f}  ({cv['recall']['mean']*100:.2f}%)")
print(f"F1-Score:          {cv['f1']['mean']:.4f}  ({cv['f1']['mean']*100:.2f}%)")
print(f"Global CM: TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")
print(f"\nPer-Fold Accuracy: {[f'{x*100:.1f}%' for x in cv['accuracy']['all_folds']]}")
