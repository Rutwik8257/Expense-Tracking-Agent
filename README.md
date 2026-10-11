# 💳 Personal Finance / Expense Tracking Agent

An AI agent that ingests bank transaction CSVs, auto-categorizes transactions,
detects spending anomalies, learns from user corrections (RAG), and answers
natural language questions — all through a Streamlit chat interface.

## Architecture

```
CSV Upload
    │
    ▼
ingest_node ──► categorize_node ──► anomaly_node ──► Streamlit UI
                     │                                     │
                 ChromaDB RAG                         query_node
                 (corrections)                      (ReAct loop)
                                                    ┌──────────┐
                                                    │ LLM      │
                                                    │ ↕ tools  │
                                                    └──────────┘
```

### Key design decisions

| Decision | Why |
|---|---|
| LangGraph StateGraph | Explicit state transitions — easy to debug, extend, and test each node independently |
| Pydantic validation on every LLM response | LLMs hallucinate JSON keys; validation + retry prevents silent bad data |
| SQLite for numbers, LLM for language | The LLM never estimates a dollar amount — all totals come from SQL aggregations |
| ChromaDB cosine similarity > 0.85 threshold | Below this, corrections are "hints"; above it, they're strong priors |
| `@tool` decorator | LangChain reads docstrings + type hints to auto-generate the tool schema the LLM sees |

## Project Structure

```
expense-agent/
├── data/
│   ├── generate_data.py          # generates sample_transactions.csv
│   ├── generate_ground_truth.py  # generates ground_truth.csv
│   └── sample_transactions.csv   # 150 synthetic transactions (10 anomalous)
├── agent/
│   ├── tools.py                  # all @tool functions + ingest helpers
│   ├── graph.py                  # LangGraph pipeline + query graphs
│   ├── prompts.py                # ChatPromptTemplate definitions
│   └── memory.py                 # ChromaDB read/write
├── db/
│   ├── transactions.db           # SQLite (auto-created)
│   └── chroma/                   # ChromaDB persistent store (auto-created)
├── eval/
│   ├── eval_accuracy.py
│   ├── eval_rag_comparison.py
│   ├── eval_anomalies.py
│   └── eval_time_savings.py
├── results/                      # eval outputs (auto-created)
├── logs/
│   └── tool_calls.log            # audit trail (auto-created)
├── tests/
│   └── test_tools.py
├── app.py                        # Streamlit entrypoint
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone / open the project
cd expense-agent

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
copy .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 5. Generate synthetic data
python data/generate_data.py
python data/generate_ground_truth.py

# 6. Run the app
streamlit run app.py
```

## Running Tests

```bash
pytest tests/ -v
```

Tests mock the LLM so they run without API keys and are fast.

## Evaluation Scripts

All scripts must be run from the project root after `generate_data.py` has been run.

### Categorization accuracy
```bash
python eval/eval_accuracy.py
# Output: results/accuracy_report.txt, results/predictions_vs_truth.csv
```

### RAG improvement comparison
```bash
python eval/eval_rag_comparison.py
# Output: results/baseline_accuracy.txt, results/rag_accuracy.txt
```

### Anomaly detection precision/recall
```bash
python eval/eval_anomalies.py
# Output: results/anomaly_eval.txt
```

### Timing benchmark
```bash
python eval/eval_time_savings.py
# Output: results/agent_timing.txt
```

---

## Methodology & Results

### Ground Truth

- **Size:** 100 transactions sampled from `sample_transactions.csv`
- **Labeling approach:** Rule-based keyword matching simulating human judgment
  (e.g. "Whole Foods" → Groceries, "Uber" → Transportation). This represents
  what a human would conclude by reading the description — not perfect, but
  consistent and reproducible.
- **File:** `data/ground_truth.csv` (columns: all original + `true_category`)

### Categorization Accuracy (`eval_accuracy.py`)

Runs `categorize_tx()` on all 100 ground truth rows and compares predicted vs
true category using `sklearn.metrics.accuracy_score`.

**What it means:** What % of transactions did the LLM categorize the same way
a human would? Expect 80-95% depending on model and ambiguous merchants.

### RAG Improvement (`eval_rag_comparison.py`)

- **Baseline:** ChromaDB wiped, categorize all 100 rows cold
- **With RAG:** Store 25 corrections, categorize remaining 75

**What it means:** The delta shows how much the agent improves when it has
seen similar transactions before. A positive delta confirms the RAG memory
is working.

### Anomaly Detection (`eval_anomalies.py`)

10 rows in `sample_transactions.csv` have `is_true_anomaly=True`. The eval
runs `detect_anomalies()` across all months and checks which flagged IDs
overlap with the ground truth.

**What it means:**
- **Precision** = of all flagged transactions, what % were actually anomalous
- **Recall** = of all true anomalies, what % did the agent catch
- High recall is more important here (missing a fraud is worse than a false alarm)

### Timing (`eval_time_savings.py`)

Times 50 `categorize_tx()` calls end-to-end (including LLM API latency).
Manual baseline assumes ~12 seconds per transaction (human reads description,
opens spreadsheet, types category).

**What it means:** Quantifies the productivity gain. Even at 3s/tx API latency,
the agent is 4x faster than manual categorization.

### Tool Call Audit Log

Every call to `categorize_tx`, `detect_anomalies`, and `calculate_total`
appends a line to `logs/tool_calls.log`:

```
2024-03-15 10:23:45 | calculate_total | input={"category": "Groceries", "month": "2024-03"} | output=285.50 | latency=0.012s
```

This proves every number in the UI came from a real database query, not LLM estimation.
