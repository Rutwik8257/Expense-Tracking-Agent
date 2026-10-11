"""
Core tool functions exposed to the LangGraph agent.

Architecture pattern:
  Each tool is a pure function (easy to unit-test) decorated with @tool so
  LangChain can bind it to an LLM for function-calling.
  All numeric results come from SQLite/Pandas — the LLM never estimates numbers.
"""
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, ValidationError, field_validator

from agent.memory import find_similar_correction, save_correction as _save_correction
from agent.prompts import anomaly_prompt, categorize_prompt

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
_tool_logger = logging.getLogger("tool_calls")
_tool_logger.setLevel(logging.INFO)
_fh = logging.FileHandler("logs/tool_calls.log")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_tool_logger.addHandler(_fh)

DB_PATH = "db/transactions.db"


def _log(tool_name: str, inputs: dict, output, latency: float):
    _tool_logger.info(
        "%s | input=%s | output=%s | latency=%.3fs",
        tool_name, json.dumps(inputs, default=str),
        json.dumps(output, default=str)[:500], latency,
    )


def _get_llm():
    """Lazy-load LLM so tools.py can be imported without API keys (for unit tests)."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0)


# ── Pydantic schemas for LLM response validation ──────────────────────────────
VALID_CATEGORIES = {
    "Food & Dining", "Groceries", "Transportation", "Shopping",
    "Bills & Utilities", "Entertainment", "Healthcare",
    "Rent/Housing", "Income", "Transfer", "Other",
}

class CategoryResult(BaseModel):
    category: str
    merchant: str
    confidence: float
    reasoning: str

    @field_validator("category")
    @classmethod
    def must_be_valid(cls, v):
        if v not in VALID_CATEGORIES:
            raise ValueError(f"'{v}' not in allowed categories")
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, v))


class AnomalyResult(BaseModel):
    is_anomaly: bool
    severity: str   # "", "low", "medium", "high"
    reason: str


def _parse_llm_json(raw: str, schema: type[BaseModel], fallback: dict) -> BaseModel:
    """
    Parse LLM JSON output against a Pydantic schema.
    Retries once on failure, then returns fallback.
    """
    for attempt in range(2):
        try:
            # Strip markdown code fences if present
            text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return schema.model_validate_json(text)
        except (ValidationError, ValueError, Exception) as e:
            if attempt == 0:
                continue  # retry
            logging.warning("LLM parse failed after retry: %s | raw=%s", e, raw[:200])
    return schema.model_validate(fallback)


# ── DB helpers ────────────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _df(query: str, params=()) -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql_query(query, con, params=params)


# ── Tool 1: get_transactions ──────────────────────────────────────────────────
@tool
def get_transactions(start_date: str, end_date: str, category: Optional[str] = None) -> list[dict]:
    """
    Fetch transactions from the database within a date range.
    Args:
        start_date: ISO format date string (YYYY-MM-DD)
        end_date:   ISO format date string (YYYY-MM-DD)
        category:   Optional category filter (e.g. "Groceries")
    Returns list of transaction dicts.
    """
    t0 = time.time()
    sql = "SELECT * FROM transactions WHERE date BETWEEN ? AND ?"
    params: list = [start_date, end_date]
    if category:
        sql += " AND category = ?"
        params.append(category)
    sql += " ORDER BY date"

    df = _df(sql, params)
    result = df.to_dict(orient="records")
    _log("get_transactions", {"start_date": start_date, "end_date": end_date, "category": category}, f"{len(result)} rows", time.time() - t0)
    return result


# ── Rule-based fast categorization (no LLM needed for known merchants) ────────
_RULES: list[tuple[list[str], str]] = [
    (["whole foods", "trader joe", "kroger", "walmart grocery", "aldi", "costco", "safeway", "publix"], "Groceries"),
    (["chipotle", "mcdonald", "starbucks", "olive garden", "domino", "subway", "panera", "chick-fil", "taco bell", "burger king", "wendy", "pizza", "sushi", "restaurant", "cafe", "diner"], "Food & Dining"),
    (["uber", "lyft", "shell", "bp gas", "chevron", "exxon", "metro transit", "parking", "mta", "bart", "transit"], "Transportation"),
    (["amazon", "target", "best buy", "nike", "h&m", "zara", "walmart", "ebay", "etsy", "shopify"], "Shopping"),
    (["at&t", "verizon", "comcast", "xfinity", "con edison", "national gas", "netflix", "spotify", "hulu", "disney", "electric", "utility", "internet", "wireless"], "Bills & Utilities"),
    (["amc", "steam", "ticketmaster", "bowling", "escape room", "cinema", "theater", "concert", "spotify"], "Entertainment"),
    (["cvs", "walgreens", "clinic", "labcorp", "dental", "pharmacy", "hospital", "doctor", "medical"], "Healthcare"),
    (["apartment rent", "rent payment", "mortgage"], "Rent/Housing"),
    (["payroll", "salary", "direct deposit", "employer", "freelance", "tax refund"], "Income"),
    (["venmo", "zelle", "transfer", "wire"], "Transfer"),
]

def _rule_based_category(description: str, amount: float) -> Optional[str]:
    """Return category if a keyword rule matches, else None."""
    desc_lower = description.lower()
    # Positive amounts are income/credits
    if amount > 0 and any(k in desc_lower for k in ["payroll", "salary", "deposit", "refund", "freelance"]):
        return "Income"
    for keywords, category in _RULES:
        if any(k in desc_lower for k in keywords):
            return category
    return None


# ── Tool 2: categorize_tx ─────────────────────────────────────────────────────
@tool
def categorize_tx(transaction: dict) -> dict:
    """
    Categorize a single transaction using rules + RAG memory + LLM fallback.
    Fast rule-based matching runs first; LLM is only called for unknown merchants.
    Returns: {category, merchant, confidence, reasoning}
    """
    t0 = time.time()
    desc = transaction.get("description", "")
    amount = transaction.get("amount", 0)

    # 1. RAG: check for similar past corrections (highest priority)
    similar = find_similar_correction(desc, n_results=3)
    if similar and similar[0]["similarity"] > 0.85:
        top = similar[0]
        output = {"category": top["corrected_category"], "merchant": desc, "confidence": top["similarity"], "reasoning": f"RAG match: {top['merchant']}"}
        _log("categorize_tx", {"description": desc, "amount": amount}, output, time.time() - t0)
        return output

    # 2. Rule-based fast path (no LLM call)
    rule_cat = _rule_based_category(desc, amount)
    if rule_cat:
        output = {"category": rule_cat, "merchant": desc, "confidence": 0.92, "reasoning": "keyword rule match"}
        _log("categorize_tx", {"description": desc, "amount": amount}, output, time.time() - t0)
        return output

    # 3. LLM fallback for unknown merchants only
    correction_context = ""
    if similar:
        lines = "\n".join(
            f"  - '{s['merchant']}' → {s['corrected_category']} (sim={s['similarity']:.2f})"
            for s in similar
        )
        correction_context = f"Related past corrections for context:\n{lines}\n\n"

    llm = _get_llm()
    chain = categorize_prompt | llm
    raw = chain.invoke({
        "description": desc,
        "amount": amount,
        "date": transaction.get("date", ""),
        "correction_context": correction_context,
    })
    raw_text = raw.content if hasattr(raw, "content") else str(raw)

    result = _parse_llm_json(
        raw_text,
        CategoryResult,
        fallback={"category": "Other", "merchant": desc, "confidence": 0.0, "reasoning": "parse failure"},
    )
    output = result.model_dump()
    _log("categorize_tx", {"description": desc, "amount": amount}, output, time.time() - t0)
    return output


# ── Tool 3: calculate_total ───────────────────────────────────────────────────
@tool
def calculate_total(category: str, month: str) -> float:
    """
    Calculate the total spending for a category in a given month.
    Args:
        category: e.g. "Groceries"
        month:    "YYYY-MM" format, e.g. "2024-03"
    Returns the absolute total as a positive float (expenses are stored negative).
    The LLM must NEVER estimate this — it always comes from the database.
    """
    t0 = time.time()
    df = _df(
        "SELECT amount FROM transactions WHERE category = ? AND strftime('%Y-%m', date) = ?",
        (category, month),
    )
    total = round(abs(df["amount"].sum()), 2) if not df.empty else 0.0
    _log("calculate_total", {"category": category, "month": month}, total, time.time() - t0)
    return total


# ── Tool 4: detect_anomalies ──────────────────────────────────────────────────
@tool
def detect_anomalies(month: str) -> list[dict]:
    """
    Detect anomalous transactions in a given month.
    Flags: amount > 2.5x category average, new merchant > $100, same-day duplicate.
    Args:
        month: "YYYY-MM" format
    Returns list of {transaction, severity, reason}
    """
    t0 = time.time()

    # All transactions for the month
    month_df = _df(
        "SELECT * FROM transactions WHERE strftime('%Y-%m', date) = ?", (month,)
    )
    if month_df.empty:
        return []

    # Category averages across ALL historical data (not just this month)
    all_df = _df("SELECT category, amount FROM transactions")
    cat_avg = (
        all_df[all_df["amount"] < 0]  # expenses only
        .groupby("category")["amount"]
        .mean()
        .abs()
        .to_dict()
    )

    # All known merchants (for "new merchant" detection)
    known_merchants = set(_df("SELECT DISTINCT description FROM transactions")["description"].tolist())

    anomalies = []
    for _, tx in month_df.iterrows():
        tx_dict = tx.to_dict()
        amount_abs = abs(tx_dict["amount"])
        cat = tx_dict.get("category", "Other")
        avg = cat_avg.get(cat, 0)
        desc = tx_dict["description"]

        # Rule checks (deterministic — no LLM needed for the flag decision)
        is_2_5x = avg > 0 and amount_abs > 2.5 * avg
        is_new_big = desc not in known_merchants and amount_abs > 100
        # Duplicate: same merchant, same date, same amount, more than once
        dupes = month_df[
            (month_df["description"] == desc) &
            (month_df["date"] == tx_dict["date"]) &
            (month_df["amount"] == tx_dict["amount"])
        ]
        is_duplicate = len(dupes) > 1

        if not (is_2_5x or is_new_big or is_duplicate):
            continue

        # Use LLM only to generate a human-readable severity + reason
        llm = _get_llm()
        chain = anomaly_prompt | llm
        raw = chain.invoke({
            "description": desc,
            "amount": tx_dict["amount"],
            "date": tx_dict["date"],
            "category": cat,
            "category_avg": f"{avg:.2f}",
            "is_new_merchant": is_new_big,
            "is_duplicate": is_duplicate,
        })
        raw_text = raw.content if hasattr(raw, "content") else str(raw)
        result = _parse_llm_json(
            raw_text,
            AnomalyResult,
            fallback={"is_anomaly": True, "severity": "medium", "reason": "rule-based flag"},
        )
        anomalies.append({"transaction": tx_dict, **result.model_dump()})

    _log("detect_anomalies", {"month": month}, f"{len(anomalies)} anomalies", time.time() - t0)
    return anomalies


# ── Tool 5: save_correction (exposed as tool for the agent) ───────────────────
@tool
def save_correction_tool(transaction: dict, corrected_category: str) -> str:
    """
    Save a user's category correction to ChromaDB memory so future
    categorizations improve. Returns confirmation message.
    """
    if corrected_category not in VALID_CATEGORIES:
        return f"Invalid category '{corrected_category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
    _save_correction(transaction, corrected_category)

    # Also update SQLite
    if tx_id := transaction.get("id"):
        with _conn() as con:
            con.execute("UPDATE transactions SET category = ? WHERE id = ?", (corrected_category, tx_id))
    return f"Correction saved: '{transaction.get('description')}' → {corrected_category}"


# ── Ingest helper (not a LangChain tool — called by graph nodes directly) ─────
def ingest_csv(csv_path: str) -> int:
    """
    Load a CSV into SQLite. Creates the table if needed.
    Returns number of rows inserted.
    """
    df = pd.read_csv(csv_path)
    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"date", "description", "amount"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}. Got: {set(df.columns)}")

    if "category" not in df.columns:
        df["category"] = ""
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["category"] = df["category"].fillna("")

    Path(DB_PATH).parent.mkdir(exist_ok=True)
    with _conn() as con:
        df.to_sql("transactions", con, if_exists="replace", index=False)

    return len(df)


def get_uncategorized() -> list[dict]:
    """Return all transactions with empty category."""
    df = _df("SELECT * FROM transactions WHERE category = '' OR category IS NULL")
    return df.to_dict(orient="records")


def update_category(tx_id: int, category: str):
    """Update a single transaction's category in SQLite."""
    with _conn() as con:
        con.execute("UPDATE transactions SET category = ? WHERE id = ?", (category, tx_id))


ALL_TOOLS = [get_transactions, categorize_tx, calculate_total, detect_anomalies, save_correction_tool]
