# predictions_logger.py
import os
import csv
import time
from logging_config import ensure_csv_has_header

# path for predictions log
PREDICTIONS_CSV = "predictions.csv"
# Header for predictions.csv
HEADER = [
    "video_id",        # string, can be filename or session id (use "live" for webcam)
    "frame_idx",       # integer
    "timestamp",       # float: epoch seconds
    "gt_label",        # 'awake' or 'drowsy' (use empty string if unknown)
    "pred_label",      # 'awake' or 'drowsy'
    "pred_score",      # numeric score (e.g., avg_ear or probability)
    "avg_ear",         # float
    "mar",             # float
    "gamma_used"       # float or empty
]

# Ensure csv header exists at import time
ensure_csv_has_header(PREDICTIONS_CSV, HEADER)

def log_prediction(
    video_id: str,
    frame_idx: int,
    gt_label: str,
    pred_label: str,
    pred_score: float,
    avg_ear: float,
    mar: float,
    gamma_used: float
):
    """
    Appends a single row to predictions.csv
    """
    row = [
        video_id,
        frame_idx,
        time.time(),
        gt_label if gt_label is not None else "",
        pred_label,
        "" if pred_score is None else float(pred_score),
        "" if avg_ear is None else float(avg_ear),
        "" if mar is None else float(mar),
        "" if gamma_used is None else float(gamma_used),
    ]
    with open(PREDICTIONS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
