"""
eval/eval_accuracy.py
Run categorize_tx() on all 100 ground truth transactions.
Compare predicted vs true category.
Saves: results/accuracy_report.txt, results/predictions_vs_truth.csv

Run: python eval/eval_accuracy.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from agent.tools import categorize_tx, ingest_csv

Path("results").mkdir(exist_ok=True)

# Ensure DB is populated
ingest_csv("data/sample_transactions.csv")

gt = pd.read_csv("data/ground_truth.csv")
predictions = []

print(f"Running categorize_tx on {len(gt)} transactions…")
t_start = time.time()

for _, row in gt.iterrows():
    tx = {"id": row["id"], "description": row["description"], "amount": row["amount"], "date": row["date"]}
    result = categorize_tx.invoke({"transaction": tx})
    predictions.append(result["category"])
    print(f"  [{row.name+1}/{len(gt)}] {row['description'][:30]:30s} → {result['category']}")

elapsed = time.time() - t_start

gt["predicted_category"] = predictions
gt.to_csv("results/predictions_vs_truth.csv", index=False)

y_true = gt["true_category"].tolist()
y_pred = predictions

acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, zero_division=0)

report_text = (
    f"Accuracy Evaluation Report\n"
    f"==========================\n"
    f"Transactions evaluated: {len(gt)}\n"
    f"Total time: {elapsed:.1f}s  ({elapsed/len(gt):.2f}s/tx)\n\n"
    f"Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)\n\n"
    f"Per-Category Report:\n{report}\n"
)

Path("results/accuracy_report.txt").write_text(report_text)
print(f"\n{report_text}")
print("Saved: results/accuracy_report.txt, results/predictions_vs_truth.csv")
