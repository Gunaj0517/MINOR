import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# -------------------------------------------------------
# STEP 1: Load predictions
# -------------------------------------------------------
csv_path = "outputs/predictions.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {csv_path}")
except FileNotFoundError:
    print("❌ predictions.csv not found. Run main_app.py first.")
    exit()

# -------------------------------------------------------
# STEP 2: (Optional) Add or import ground truth labels
# -------------------------------------------------------
# If you already annotated some frames manually, merge them here.
# Example: df['gt_label'] = ['awake','awake','drowsy',...]  # your labels
# For demo, we create pseudo labels based on EAR threshold.

EAR_THRESHOLD = 0.25  # the same one you used in main_app.py


# -------------------------------------------------------
# STEP 3: Compute evaluation metrics
# -------------------------------------------------------
y_true = df['gt_label']
y_pred = df['pred_label']

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label='drowsy')
rec = recall_score(y_true, y_pred, pos_label='drowsy')
f1 = f1_score(y_true, y_pred, pos_label='drowsy')

print("\n📊 Evaluation Metrics:")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1-score : {f1*100:.2f}%")

print("\nDetailed Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# -------------------------------------------------------
# STEP 4: Analyze EAR / MAR distributions
# -------------------------------------------------------
plt.figure(figsize=(10,5))
plt.hist(df[df['pred_label']=='awake']['avg_ear'], bins=30, alpha=0.6, label='Pred Awake')
plt.hist(df[df['pred_label']=='drowsy']['avg_ear'], bins=30, alpha=0.6, label='Pred Drowsy')
plt.axvline(EAR_THRESHOLD, color='r', linestyle='--', label='EAR Threshold')
plt.title('EAR Distribution by Predicted Label')
plt.xlabel('EAR')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# -------------------------------------------------------
# STEP 5: Threshold tuning suggestion
# -------------------------------------------------------
# ear_values = df['avg_ear']
# possible_thresholds = [round(t, 2) for t in list(set([round(x,2) for x in ear_values]))]
# best_thresh = EAR_THRESHOLD
# best_f1 = 0

# for t in possible_thresholds:
#     preds = ['drowsy' if x < t else 'awake' for x in df['avg_ear']]
#     f1_temp = f1_score(df['gt_label'], preds, pos_label='drowsy')
#     if f1_temp > best_f1:
#         best_f1 = f1_temp
#         best_thresh = t

# print(f"\n🔍 Suggested better EAR threshold (max F1): {best_thresh:.2f}")
# print(f"   Old threshold: {EAR_THRESHOLD}")
