"""
All prompt templates in one place.
Using LangChain's ChatPromptTemplate so they're composable with any LLM.
"""
from langchain_core.prompts import ChatPromptTemplate

# ── Categorization ────────────────────────────────────────────────────────────
CATEGORIZE_SYSTEM = """You are a financial transaction categorization assistant.
Given a transaction, classify it into exactly ONE category from this list:
Food & Dining, Groceries, Transportation, Shopping, Bills & Utilities, \
Entertainment, Healthcare, Rent/Housing, Income, Transfer, Other.
Negative amounts are expenses; positive are income/credits.
If similar past user corrections are provided, prioritize matching that pattern.
Respond with ONLY valid JSON: \
{{"category": "", "merchant": "", "confidence": 0.0, "reasoning": ""}}"""

categorize_prompt = ChatPromptTemplate.from_messages([
    ("system", CATEGORIZE_SYSTEM),
    ("human", (
        "Transaction:\n"
        "  Description: {description}\n"
        "  Amount: {amount}\n"
        "  Date: {date}\n\n"
        "{correction_context}"
        "Classify this transaction."
    )),
])

# ── Anomaly detection ─────────────────────────────────────────────────────────
ANOMALY_SYSTEM = """You are a financial anomaly detection assistant. Given a \
transaction and the user's historical category average, flag it as anomalous if: \
amount exceeds 2.5x the category average, OR it's a new merchant with amount > $100, \
OR it's a same-day duplicate charge from the same merchant.
Respond with ONLY valid JSON: \
{{"is_anomaly": false, "severity": "", "reason": ""}}"""

anomaly_prompt = ChatPromptTemplate.from_messages([
    ("system", ANOMALY_SYSTEM),
    ("human", (
        "Transaction:\n"
        "  Description: {description}\n"
        "  Amount: {amount}\n"
        "  Date: {date}\n"
        "  Category: {category}\n\n"
        "Category average for this category: ${category_avg}\n"
        "Is this merchant new (never seen before)? {is_new_merchant}\n"
        "Is there a duplicate same-day charge from this merchant? {is_duplicate}\n\n"
        "Flag this transaction if any anomaly condition is met."
    )),
])

# ── Query / conversational node ───────────────────────────────────────────────
QUERY_SYSTEM = """You are a personal finance assistant with access to tools: \
get_transactions, calculate_total, detect_anomalies, categorize_tx.
CRITICAL: Never state a dollar amount or statistic unless retrieved via a tool \
call — do not estimate from memory. Always cite the time period/category you're \
referring to. Keep answers to 1-3 sentences unless asked for detail. If a query \
is ambiguous (e.g. no timeframe given), ask ONE clarifying question before \
calling any tool."""

query_prompt = ChatPromptTemplate.from_messages([
    ("system", QUERY_SYSTEM),
    ("placeholder", "{messages}"),
])
