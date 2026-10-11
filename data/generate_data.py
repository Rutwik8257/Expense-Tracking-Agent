"""
Generate 150 realistic synthetic transactions with 10 deliberate anomalies.
Run once: python data/generate_data.py
"""
import csv
import random
from datetime import date, timedelta

random.seed(42)

CATEGORIES = {
    "Food & Dining": [
        ("Chipotle", 12, 18), ("McDonald's", 8, 15), ("Starbucks", 5, 9),
        ("Olive Garden", 25, 55), ("Domino's Pizza", 18, 35), ("Subway", 8, 14),
        ("Panera Bread", 11, 20), ("Chick-fil-A", 10, 18),
    ],
    "Groceries": [
        ("Whole Foods", 45, 120), ("Trader Joe's", 30, 90), ("Kroger", 40, 110),
        ("Walmart Grocery", 50, 130), ("Costco", 80, 200), ("Aldi", 25, 75),
    ],
    "Transportation": [
        ("Uber", 12, 35), ("Lyft", 10, 30), ("Shell Gas Station", 40, 70),
        ("BP Gas", 38, 68), ("Metro Transit", 3, 5), ("Parking Meter", 2, 8),
    ],
    "Shopping": [
        ("Amazon", 20, 150), ("Target", 30, 120), ("Best Buy", 50, 300),
        ("Nike Store", 60, 180), ("H&M", 25, 80), ("Zara", 40, 120),
    ],
    "Bills & Utilities": [
        ("AT&T Wireless", 65, 95), ("Comcast Internet", 70, 90),
        ("Con Edison Electric", 80, 140), ("National Gas Co", 40, 90),
        ("Netflix", 15, 18), ("Spotify", 10, 11),
    ],
    "Entertainment": [
        ("AMC Theaters", 14, 22), ("Steam Games", 10, 60),
        ("Ticketmaster", 40, 150), ("Bowling Alley", 20, 45),
        ("Escape Room", 25, 40),
    ],
    "Healthcare": [
        ("CVS Pharmacy", 15, 60), ("Walgreens", 12, 55),
        ("Dr. Smith Clinic", 80, 200), ("LabCorp", 50, 150),
        ("Dental Care Plus", 100, 300),
    ],
    "Rent/Housing": [
        ("Apartment Rent", 1200, 1800),
    ],
    "Income": [
        ("Employer Payroll", 2500, 4500), ("Freelance Payment", 500, 2000),
        ("Tax Refund", 300, 1500),
    ],
    "Transfer": [
        ("Venmo Transfer", 20, 200), ("Zelle Payment", 50, 500),
    ],
}

# Normal transaction distribution (140 normal + 10 anomalous)
NORMAL_DIST = {
    "Food & Dining": 28, "Groceries": 20, "Transportation": 22,
    "Shopping": 18, "Bills & Utilities": 14, "Entertainment": 12,
    "Healthcare": 10, "Rent/Housing": 4, "Income": 7, "Transfer": 5,
}

def random_date(start=date(2024, 1, 1), end=date(2024, 6, 30)):
    return start + timedelta(days=random.randint(0, (end - start).days))

rows = []

# Generate normal transactions
for category, count in NORMAL_DIST.items():
    merchants = CATEGORIES[category]
    for _ in range(count):
        merchant, lo, hi = random.choice(merchants)
        amount = round(random.uniform(lo, hi), 2)
        if category not in ("Income", "Transfer"):
            amount = -amount  # expenses are negative
        rows.append({
            "date": random_date().isoformat(),
            "description": merchant,
            "amount": amount,
            "category": "",          # to be filled by agent
            "is_true_anomaly": False,
        })

# 10 deliberate anomalies
anomalies = [
    # 1. Amount > 2.5x category average for Food & Dining (avg ~$20 → flag >$50)
    {"date": "2024-03-15", "description": "Chipotle", "amount": -189.50, "category": "", "is_true_anomaly": True},
    # 2. New merchant > $100
    {"date": "2024-02-20", "description": "Obscure Boutique XYZ", "amount": -340.00, "category": "", "is_true_anomaly": True},
    # 3. Duplicate same-day same-merchant
    {"date": "2024-04-10", "description": "Netflix", "amount": -15.99, "category": "", "is_true_anomaly": False},
    {"date": "2024-04-10", "description": "Netflix", "amount": -15.99, "category": "", "is_true_anomaly": True},
    # 4. Grocery spend 2.5x average (~$80 avg → flag >$200)
    {"date": "2024-05-03", "description": "Whole Foods", "amount": -487.30, "category": "", "is_true_anomaly": True},
    # 5. Healthcare way above average
    {"date": "2024-01-22", "description": "Dr. Smith Clinic", "amount": -950.00, "category": "", "is_true_anomaly": True},
    # 6. New merchant > $100
    {"date": "2024-03-08", "description": "Unknown Electronics Co", "amount": -620.00, "category": "", "is_true_anomaly": True},
    # 7. Transportation 2.5x average (~$30 avg → flag >$75)
    {"date": "2024-06-01", "description": "Uber", "amount": -210.00, "category": "", "is_true_anomaly": True},
    # 8. Duplicate same-day same-merchant (shopping)
    {"date": "2024-05-18", "description": "Amazon", "amount": -89.99, "category": "", "is_true_anomaly": False},
    {"date": "2024-05-18", "description": "Amazon", "amount": -89.99, "category": "", "is_true_anomaly": True},
]
rows.extend(anomalies)

# Shuffle and assign IDs
random.shuffle(rows)
for i, row in enumerate(rows):
    row["id"] = i + 1

fieldnames = ["id", "date", "description", "amount", "category", "is_true_anomaly"]

with open("data/sample_transactions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} transactions ({sum(1 for r in rows if r['is_true_anomaly'])} anomalies)")
