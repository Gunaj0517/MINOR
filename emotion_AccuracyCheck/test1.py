# logging_config.py
import os
import csv
from typing import Optional

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ensure_csv_has_header(path, header):
    if not os.path.exists(path):
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)