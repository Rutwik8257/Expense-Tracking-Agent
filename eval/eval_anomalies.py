"""
eval/eval_anomalies.py
Evaluate anomaly detection precision and recall against the 10 known anomalies
in sample_transactions.csv (is_true_anomaly column).

Run: python eval/eval_anomalies.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from agent.tools import detect_anomalies, ingest_csv

Path("results").mkdir(exist_ok=True)
ingest_csv("data/sample_transactions.csv")

# Ground truth: which transaction IDs are truly anomalous
gt_df = pd.read_csv("data/sample_transactions.csv")
# is_true_anomaly may be stored as bool or string
gt_df["is_true_anomaly"] = gt_df["is_true_anomaly"].astype(str).str.lower().isin(["true", "1", "yes"])
true_anomaly_ids = set(gt_df[gt_df["is_true_anomaly"]]["id"].tolist())
print(f"Ground truth anomalies: {len(true_anomaly_ids)} transactions (IDs: {sorted(true_anomaly_ids)})")

# Run detection across all months
months = gt_df["date"].str[:7].unique().tolist()
detected = []
for month in months:
    found = detect_anomalies.invoke({"month": month})
    detected.extend(found)

detected_ids = set(a["transaction"]["id"] for a in detected)
print(f"Detected anomalies: {len(detected_ids)} transactions (IDs: {sorted(detected_ids)})")

# Precision = TP / (TP + FP)
# Recall    = TP / (TP + FN)
tp = len(true_anomaly_ids & detected_ids)
fp = len(detected_ids - true_anomaly_ids)
fn = len(true_anomaly_ids - detected_ids)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

report = (
    f"Anomaly Detection Evaluation\n"
    f"=============================\n"
    f"True anomalies (ground truth): {len(true_anomaly_ids)}\n"
    f"Detected by agent:             {len(detected_ids)}\n"
    f"True Positives:  {tp}\n"
    f"False Positives: {fp}\n"
    f"False Negatives: {fn}\n\n"
    f"Precision: {precision:.4f} ({precision*100:.1f}%)\n"
    f"Recall:    {recall:.4f} ({recall*100:.1f}%)\n"
    f"F1 Score:  {f1:.4f}\n"
)
Path("results/anomaly_eval.txt").write_text(report)
print(f"\n{report}")
print("Saved: results/anomaly_eval.txt")
