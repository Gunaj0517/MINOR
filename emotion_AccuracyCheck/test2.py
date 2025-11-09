# emotion_logger.py
import os
import csv
import time
from logging_config import ensure_csv_has_header

# Nayi file ka naam
EMOTIONS_CSV = "emotion_predictions.csv"
# Naya header
HEADER = [
    "frame_id",
    "timestamp",
    "ground_truth_emotion", # Yeh hum manually bharenge
    "predicted_emotion",  # Classifier ka output
    "smile_score",
    "frown_score",
    "jaw_open_score",
    "brow_down_score",
    "eye_wide_score"
]

# Ensure csv header exists at import time
ensure_csv_has_header(EMOTIONS_CSV, HEADER)

def log_emotion_prediction(
    frame_id: int,
    gt_label: str,
    pred_label: str,
    scores: dict
):
    """
    Appends a single row to emotion_predictions.csv
    """
    row = [
        frame_id,
        time.time(),
        gt_label if gt_label is not None else "",
        pred_label,
        scores.get('mouthSmileLeft', 0) + scores.get('mouthSmileRight', 0) / 2.0,
        scores.get('mouthFrownLeft', 0) + scores.get('mouthFrownRight', 0) / 2.0,
        scores.get('jawOpen', 0),
        scores.get('browDownLeft', 0) + scores.get('browDownRight', 0) / 2.0,
        scores.get('eyeWideLeft', 0) + scores.get('eyeWideRight', 0) / 2.0,
    ]
    with open(EMOTIONS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)