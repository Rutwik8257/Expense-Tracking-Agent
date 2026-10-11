"""
Unit tests for tool functions.
These test the pure logic without hitting the LLM or real DB.
Run: pytest tests/ -v
"""
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Create a temp SQLite DB with sample data and patch DB_PATH."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("agent.tools.DB_PATH", db_path)

    con = sqlite3.connect(db_path)
    df = pd.DataFrame([
        {"id": 1, "date": "2024-03-01", "description": "Whole Foods", "amount": -85.0, "category": "Groceries"},
        {"id": 2, "date": "2024-03-05", "description": "Starbucks", "amount": -6.5, "category": "Food & Dining"},
        {"id": 3, "date": "2024-03-10", "description": "Uber", "amount": -22.0, "category": "Transportation"},
        {"id": 4, "date": "2024-03-15", "description": "Employer Payroll", "amount": 3000.0, "category": "Income"},
        {"id": 5, "date": "2024-03-20", "description": "Whole Foods", "amount": -487.0, "category": "Groceries"},  # anomaly
        {"id": 6, "date": "2024-04-01", "description": "Netflix", "amount": -15.99, "category": "Bills & Utilities"},
        {"id": 7, "date": "2024-04-01", "description": "Netflix", "amount": -15.99, "category": "Bills & Utilities"},  # duplicate
    ])
    df.to_sql("transactions", con, if_exists="replace", index=False)
    con.close()
    return db_path


# ── Test: get_transactions ────────────────────────────────────────────────────
def test_get_transactions_date_filter(tmp_db):
    from agent.tools import get_transactions
    result = get_transactions.invoke({"start_date": "2024-03-01", "end_date": "2024-03-31"})
    assert len(result) == 5
    assert all(r["date"].startswith("2024-03") for r in result)


def test_get_transactions_category_filter(tmp_db):
    from agent.tools import get_transactions
    result = get_transactions.invoke({
        "start_date": "2024-01-01", "end_date": "2024-12-31", "category": "Groceries"
    })
    assert len(result) == 2
    assert all(r["category"] == "Groceries" for r in result)


# ── Test: calculate_total ─────────────────────────────────────────────────────
def test_calculate_total(tmp_db):
    from agent.tools import calculate_total
    total = calculate_total.invoke({"category": "Groceries", "month": "2024-03"})
    assert total == pytest.approx(572.0)  # 85 + 487


def test_calculate_total_empty(tmp_db):
    from agent.tools import calculate_total
    total = calculate_total.invoke({"category": "Healthcare", "month": "2024-03"})
    assert total == 0.0


# ── Test: ingest_csv ──────────────────────────────────────────────────────────
def test_ingest_csv(tmp_path, monkeypatch):
    db_path = str(tmp_path / "ingest_test.db")
    monkeypatch.setattr("agent.tools.DB_PATH", db_path)

    csv_path = tmp_path / "test.csv"
    csv_path.write_text("id,date,description,amount,category\n1,2024-01-15,Amazon,-45.0,\n")

    from agent.tools import ingest_csv
    n = ingest_csv(str(csv_path))
    assert n == 1

    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM transactions", con)
    con.close()
    assert len(df) == 1
    assert df.iloc[0]["description"] == "Amazon"


# ── Test: Pydantic schema validation ─────────────────────────────────────────
def test_category_result_valid():
    from agent.tools import CategoryResult
    r = CategoryResult(category="Groceries", merchant="Whole Foods", confidence=0.95, reasoning="food store")
    assert r.category == "Groceries"


def test_category_result_invalid_category():
    from agent.tools import CategoryResult
    with pytest.raises(Exception):
        CategoryResult(category="Fake Category", merchant="X", confidence=0.5, reasoning="")


def test_category_result_confidence_clamped():
    from agent.tools import CategoryResult
    r = CategoryResult(category="Other", merchant="X", confidence=1.5, reasoning="")
    assert r.confidence == 1.0


def test_parse_llm_json_fallback():
    from agent.tools import CategoryResult, _parse_llm_json
    result = _parse_llm_json(
        "not valid json at all!!!",
        CategoryResult,
        fallback={"category": "Other", "merchant": "unknown", "confidence": 0.0, "reasoning": "parse failure"},
    )
    assert result.category == "Other"
    assert result.confidence == 0.0


# ── Test: categorize_tx (mocked LLM) ─────────────────────────────────────────
def test_categorize_tx_uses_rag(tmp_db):
    """When RAG returns high-similarity correction, it should appear in prompt context."""
    mock_response = MagicMock()
    mock_response.content = '{"category": "Groceries", "merchant": "Whole Foods", "confidence": 0.97, "reasoning": "grocery store"}'

    with patch("agent.tools._get_llm") as mock_llm, \
         patch("agent.tools.find_similar_correction") as mock_rag:
        mock_rag.return_value = [{"corrected_category": "Groceries", "merchant": "Whole Foods", "similarity": 0.92}]
        mock_llm.return_value.return_value = mock_response
        # Patch the chain invoke
        with patch("agent.prompts.categorize_prompt.__or__") as mock_chain:
            mock_chain.return_value.invoke = MagicMock(return_value=mock_response)
            from agent.tools import categorize_tx
            result = categorize_tx.invoke({"transaction": {"id": 1, "description": "Whole Foods", "amount": -85.0, "date": "2024-03-01"}})
        assert result["category"] == "Groceries"


# ── Test: detect_anomalies (rule-based checks) ────────────────────────────────
def test_detect_anomalies_finds_high_amount(tmp_db):
    """$487 Whole Foods should be flagged (avg ~$85, threshold 2.5x = $212.5)."""
    mock_response = MagicMock()
    mock_response.content = '{"is_anomaly": true, "severity": "high", "reason": "exceeds 2.5x average"}'

    with patch("agent.tools._get_llm") as mock_llm:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        mock_llm.return_value.__or__ = MagicMock(return_value=mock_chain)

        with patch("agent.prompts.anomaly_prompt.__or__", return_value=mock_chain):
            from agent.tools import detect_anomalies
            anomalies = detect_anomalies.invoke({"month": "2024-03"})

    # At least the $487 Whole Foods should be flagged
    assert len(anomalies) >= 1


def test_detect_anomalies_finds_duplicate(tmp_db):
    """Two Netflix charges on same day should be flagged."""
    mock_response = MagicMock()
    mock_response.content = '{"is_anomaly": true, "severity": "medium", "reason": "duplicate charge"}'

    with patch("agent.prompts.anomaly_prompt.__or__") as mock_chain_factory:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        mock_chain_factory.return_value = mock_chain

        from agent.tools import detect_anomalies
        anomalies = detect_anomalies.invoke({"month": "2024-04"})

    assert len(anomalies) >= 1
