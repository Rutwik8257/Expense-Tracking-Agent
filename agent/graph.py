"""
LangGraph StateGraph for the expense agent.

Flow:
  ingest_node → categorize_node → anomaly_node → END
                                                ↑
  query_node ←──────────────────────────────────┘  (invoked on user questions)

State is a TypedDict — LangGraph passes it between nodes and merges updates.
The query_node uses LangChain's tool-calling loop (ReAct pattern):
  LLM decides which tool to call → tool runs → result fed back → LLM answers.
"""
import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
# ToolNode implemented manually — langgraph 0.2.x doesn't include it in prebuilt
from langchain_core.messages import ToolMessage


class ToolNode:
    """Executes tool calls requested by the LLM and returns ToolMessages."""
    def __init__(self, tools: list):
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, state: dict) -> dict:
        last = state["messages"][-1]
        results = []
        for tc in last.tool_calls:
            tool = self.tools_by_name[tc["name"]]
            output = tool.invoke(tc["args"])
            results.append(ToolMessage(
                content=str(output),
                tool_call_id=tc["id"],
            ))
        return {"messages": results}
from typing_extensions import TypedDict

from agent.tools import (
    ALL_TOOLS,
    get_uncategorized,
    ingest_csv,
    update_category,
)
from agent.tools import categorize_tx as _categorize_tx_tool
from agent.tools import detect_anomalies as _detect_anomalies_tool


# ── State definition ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Conversation history — Annotated with operator.add means new messages
    # are APPENDED rather than replacing the list (LangGraph merge strategy)
    messages: Annotated[list[BaseMessage], operator.add]
    csv_path: str
    rows_ingested: int
    categorized_count: int
    anomalies: list[dict]
    status: str  # "idle" | "ingesting" | "categorizing" | "ready"


# ── Node 1: ingest ────────────────────────────────────────────────────────────
def ingest_node(state: AgentState) -> dict:
    """Read CSV → SQLite. Sets rows_ingested in state."""
    csv_path = state.get("csv_path", "data/sample_transactions.csv")
    n = ingest_csv(csv_path)
    return {"rows_ingested": n, "status": "ingesting"}


# ── Node 2: categorize ────────────────────────────────────────────────────────
def categorize_node(state: AgentState) -> dict:
    """
    Run categorize_tx on every uncategorized transaction.
    Calls the tool function directly (not via LLM routing) since this is a
    batch operation — we know exactly what to do.
    """
    uncategorized = get_uncategorized()
    count = 0
    for tx in uncategorized:
        # .invoke() calls the underlying function through the @tool wrapper
        result = _categorize_tx_tool.invoke({"transaction": tx})
        update_category(tx["id"], result["category"])
        count += 1
    return {"categorized_count": count, "status": "categorizing"}


# ── Node 3: anomaly detection ─────────────────────────────────────────────────
def anomaly_node(state: AgentState) -> dict:
    """Run detect_anomalies for every month present in the data."""
    from agent.tools import _df
    months_df = _df("SELECT DISTINCT strftime('%Y-%m', date) as month FROM transactions")
    all_anomalies = []
    for month in months_df["month"].tolist():
        found = _detect_anomalies_tool.invoke({"month": month})
        all_anomalies.extend(found)
    return {"anomalies": all_anomalies, "status": "ready"}


# ── Node 4: query (conversational ReAct loop) ─────────────────────────────────
def query_node(state: AgentState) -> dict:
    """
    The conversational node. Uses LLM with tool-calling (ReAct pattern).
    The LLM sees the conversation history + available tools, decides what to
    call, gets results, and formulates a grounded answer.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from agent.prompts import query_prompt

    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")

    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, temperature=0)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, temperature=0)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=0)

    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    chain = query_prompt | llm_with_tools
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# ── Routing logic ─────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """After query_node: if LLM requested tool calls, run them; else end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Build the graph ───────────────────────────────────────────────────────────
def build_pipeline_graph() -> Any:
    """
    The ingestion pipeline: ingest → categorize → anomaly → END.
    Run once when a CSV is uploaded.
    """
    g = StateGraph(AgentState)
    g.add_node("ingest", ingest_node)
    g.add_node("categorize", categorize_node)
    g.add_node("anomaly", anomaly_node)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "categorize")
    g.add_edge("categorize", "anomaly")
    g.add_edge("anomaly", END)
    return g.compile()


def build_query_graph() -> Any:
    """
    The conversational query graph: query_node ↔ tools (ReAct loop).
    Invoked on every user chat message.
    """
    g = StateGraph(AgentState)
    g.add_node("query", query_node)
    g.add_node("tools", ToolNode(ALL_TOOLS))

    g.set_entry_point("query")
    g.add_conditional_edges("query", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "query")  # tool results feed back to LLM
    return g.compile()


# Lazy singletons — built on first access to avoid import-time side effects
_pipeline_graph = None
_query_graph = None


def get_pipeline_graph():
    global _pipeline_graph
    if _pipeline_graph is None:
        _pipeline_graph = build_pipeline_graph()
    return _pipeline_graph


def get_query_graph():
    global _query_graph
    if _query_graph is None:
        _query_graph = build_query_graph()
    return _query_graph


# Keep these names for backward compatibility
pipeline_graph = property(get_pipeline_graph)
query_graph = property(get_query_graph)
