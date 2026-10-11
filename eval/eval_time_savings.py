"""
eval/eval_time_savings.py
Time 50 categorize_tx calls and log to results/agent_timing.txt.

Run: python eval/eval_time_savings.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from agent.tools import categorize_tx, ingest_csv

Path("results").mkdir(exist_ok=True)
ingest_csv("data/sample_transactions.csv")

df = pd.read_csv("data/sample_transactions.csv").head(50)
timings = []

print(f"Timing categorize_tx across {len(df)} transactions…")
total_start = time.time()

for i, (_, row) in enumerate(df.iterrows()):
    tx = {"id": row["id"], "description": row["description"], "amount": row["amount"], "date": row["date"]}
    t0 = time.time()
    categorize_tx.invoke({"transaction": tx})
    elapsed = time.time() - t0
    timings.append(elapsed)
    print(f"  [{i+1:2d}/50] {row['description'][:30]:30s} {elapsed:.2f}s")

total = time.time() - total_start
avg = sum(timings) / len(timings)
mn, mx = min(timings), max(timings)

report = (
    f"Agent Timing Report — categorize_tx\n"
    f"=====================================\n"
    f"Transactions timed: {len(timings)}\n"
    f"Total time:         {total:.2f}s\n"
    f"Average per tx:     {avg:.2f}s\n"
    f"Min:                {mn:.2f}s\n"
    f"Max:                {mx:.2f}s\n\n"
    f"Manual baseline (human reading + typing category): ~10-15s per transaction\n"
    f"Estimated manual time for 50 tx: {50*12:.0f}s (~{50*12/60:.1f} min)\n"
    f"Agent time for 50 tx:            {total:.1f}s (~{total/60:.1f} min)\n"
    f"Estimated time savings:          {(50*12 - total)/60:.1f} min\n"
)
Path("results/agent_timing.txt").write_text(report)
print(f"\n{report}")
print("Saved: results/agent_timing.txt")
