"""
eval/eval_rag_comparison.py
Compare categorization accuracy WITH vs WITHOUT ChromaDB corrections.

Methodology:
  1. Baseline: clear ChromaDB, run categorize_tx on all 100 ground truth rows
  2. RAG: store 20-30 corrections into ChromaDB, re-run on remaining ~80 rows
  
Run: python eval/eval_rag_comparison.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.metrics import accuracy_score

from agent.memory import clear_memory, save_correction
from agent.tools import categorize_tx, ingest_csv

Path("results").mkdir(exist_ok=True)
ingest_csv("data/sample_transactions.csv")
gt = pd.read_csv("data/ground_truth.csv")

# ── Baseline: no corrections ──────────────────────────────────────────────────
print("=== BASELINE (no RAG corrections) ===")
clear_memory()

baseline_preds = []
for _, row in gt.iterrows():
    tx = {"id": row["id"], "description": row["description"], "amount": row["amount"], "date": row["date"]}
    result = categorize_tx.invoke({"transaction": tx})
    baseline_preds.append(result["category"])
    print(f"  {row['description'][:30]:30s} → {result['category']} (true: {row['true_category']})")

baseline_acc = accuracy_score(gt["true_category"].tolist(), baseline_preds)
Path("results/baseline_accuracy.txt").write_text(
    f"Baseline accuracy (no RAG): {baseline_acc:.4f} ({baseline_acc*100:.1f}%)\n"
    f"Evaluated on {len(gt)} transactions\n"
)
print(f"\nBaseline accuracy: {baseline_acc:.4f}")

# ── RAG: inject 20-30 corrections, then re-run on remaining rows ──────────────
print("\n=== RAG (with corrections) ===")
clear_memory()

# Use first 25 rows as "training corrections"
train = gt.head(25)
test = gt.tail(len(gt) - 25)

print(f"Storing {len(train)} corrections into ChromaDB…")
for _, row in train.iterrows():
    tx = {"id": row["id"], "description": row["description"], "amount": row["amount"], "date": row["date"]}
    save_correction(tx, row["true_category"])

print(f"Running categorize_tx on {len(test)} test transactions…")
rag_preds = []
for _, row in test.iterrows():
    tx = {"id": row["id"], "description": row["description"], "amount": row["amount"], "date": row["date"]}
    result = categorize_tx.invoke({"transaction": tx})
    rag_preds.append(result["category"])
    print(f"  {row['description'][:30]:30s} → {result['category']} (true: {row['true_category']})")

rag_acc = accuracy_score(test["true_category"].tolist(), rag_preds)
Path("results/rag_accuracy.txt").write_text(
    f"RAG accuracy (with {len(train)} corrections): {rag_acc:.4f} ({rag_acc*100:.1f}%)\n"
    f"Evaluated on {len(test)} transactions\n"
    f"Delta vs baseline: {(rag_acc - baseline_acc)*100:+.1f}%\n"
)
print(f"\nRAG accuracy: {rag_acc:.4f}")
print(f"Delta: {(rag_acc - baseline_acc)*100:+.1f}%")
print("Saved: results/baseline_accuracy.txt, results/rag_accuracy.txt")
