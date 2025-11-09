import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------------
# STEP 1: Load predictions
# -------------------------------------------------------
csv_path = "emotion_predictions.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {csv_path}")
except FileNotFoundError:
    print(f"❌ {csv_path} not found. Run facialExpressions.py first to generate logs.")
    exit()

# -------------------------------------------------------
# STEP 2: Ground Truth Check
# -------------------------------------------------------
# IMPORTANT: Aapko CSV file khol kar 'ground_truth_emotion' column
# manually bharna hoga. Uske baad hi yeh script chalayen.

# Hum unhi rows ko lenge jahan ground truth bhara hua hai
df_labeled = df.dropna(subset=['ground_truth_emotion'])

if len(df_labeled) == 0:
    print("\n❌ Error: 'ground_truth_emotion' column is empty.")
    print("Please open 'emotion_predictions.csv' and manually label your data.")
    exit()
else:
    print(f"\nFound {len(df_labeled)} manually labeled frames to analyze.")

# -------------------------------------------------------
# STEP 3: Compute evaluation metrics
# -------------------------------------------------------
y_true = df_labeled['ground_truth_emotion']
y_pred = df_labeled['predicted_emotion']
labels = sorted(y_true.unique())

acc = accuracy_score(y_true, y_pred)

print("\n📊 Evaluation Metrics:")
print(f"Overall Accuracy: {acc*100:.2f}%")

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, labels=labels))

print("\nConfusion Matrix:")
# Confusion Matrix banate hain
cm = confusion_matrix(y_true, y_pred, labels=labels)
# DataFrame banate hain taaki labels saaf dikhein
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# -------------------------------------------------------
# STEP 4: Visualize Confusion Matrix
# -------------------------------------------------------
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Predicted vs. Ground Truth')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nAnalysis complete.")