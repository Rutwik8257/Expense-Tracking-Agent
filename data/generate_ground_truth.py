"""
Generate ground_truth.csv: 100 rows from sample_transactions.csv with
manually-assigned true_category labels based on obvious categorization logic.
Run: python data/generate_ground_truth.py
"""
import csv
import pandas as pd

# Keyword → category mapping (simulates human labeling)
RULES = [
    (["payroll", "salary", "direct deposit", "employer"], "Income"),
    (["freelance", "tax refund", "venmo", "zelle"], lambda desc, amt: "Income" if amt > 0 else "Transfer"),
    (["whole foods", "trader joe", "kroger", "walmart grocery", "aldi", "costco"], "Groceries"),
    (["chipotle", "mcdonald", "starbucks", "olive garden", "domino", "subway", "panera", "chick-fil"], "Food & Dining"),
    (["uber", "lyft", "shell", "bp gas", "metro transit", "parking"], "Transportation"),
    (["amazon", "target", "best buy", "nike", "h&m", "zara"], "Shopping"),
    (["at&t", "comcast", "con edison", "national gas", "netflix", "spotify"], "Bills & Utilities"),
    (["amc", "steam", "ticketmaster", "bowling", "escape room"], "Entertainment"),
    (["cvs", "walgreens", "clinic", "labcorp", "dental"], "Healthcare"),
    (["apartment rent"], "Rent/Housing"),
]

def label(description: str, amount: float) -> str:
    desc_lower = description.lower()
    for keywords, category in RULES:
        if any(k in desc_lower for k in keywords):
            if callable(category):
                return category(desc_lower, amount)
            return category
    return "Other"

df = pd.read_csv("data/sample_transactions.csv")
df_sample = df.sample(n=min(100, len(df)), random_state=42).copy()
df_sample["true_category"] = df_sample.apply(lambda r: label(r["description"], r["amount"]), axis=1)
df_sample.to_csv("data/ground_truth.csv", index=False)
print(f"Generated ground_truth.csv with {len(df_sample)} rows")
print(df_sample["true_category"].value_counts().to_string())
