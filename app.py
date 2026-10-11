"""
Streamlit frontend for the Expense Tracking Agent.

Layout:
  Sidebar  → file upload + pipeline status
  Tab 1    → Chat Q&A
  Tab 2    → Spending charts (bar + line)
  Tab 3    → Anomalies panel
  Tab 4    → Correct a transaction
"""
import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

st.set_page_config(page_title="Expense Agent", page_icon="💳", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("pipeline_done", False),
    ("anomalies", []),
    ("rows_ingested", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar: upload + pipeline ────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Expense Agent")
    st.caption("Powered by LangGraph + Claude")

    uploaded = st.file_uploader("Upload transactions CSV", type="csv")

    if uploaded and not st.session_state.pipeline_done:
        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        progress = st.progress(0, text="Step 1/3: Ingesting CSV...")

        from agent.tools import ingest_csv, get_uncategorized, update_category
        from agent.tools import categorize_tx as _cat_tx
        from agent.tools import detect_anomalies as _det_anom
        from agent.tools import _df

        n = ingest_csv(tmp_path)
        os.unlink(tmp_path)
        progress.progress(10, text=f"Step 2/3: Categorizing {n} transactions...")

        uncategorized = get_uncategorized()
        total = len(uncategorized)
        for i, tx in enumerate(uncategorized):
            result = _cat_tx.invoke({"transaction": tx})
            update_category(tx["id"], result["category"])
            pct = 10 + int(70 * (i + 1) / max(total, 1))
            progress.progress(pct, text=f"Step 2/3: Categorized {i+1}/{total}...")

        progress.progress(80, text="Step 3/3: Detecting anomalies...")
        months_df = _df("SELECT DISTINCT strftime('%Y-%m', date) as month FROM transactions")
        all_anomalies = []
        for month in months_df["month"].tolist():
            all_anomalies.extend(_det_anom.invoke({"month": month}))

        progress.progress(100, text="Done!")
        progress.empty()

        st.session_state.pipeline_done = True
        st.session_state.rows_ingested = n
        st.session_state.anomalies = all_anomalies
        st.success(f"✅ {n} transactions loaded")
        st.info(f"🔍 {len(all_anomalies)} anomalies detected")

    elif st.session_state.pipeline_done:
        st.success(f"✅ {st.session_state.rows_ingested} transactions loaded")
        if st.button("Re-upload / Reset"):
            for k in ["pipeline_done", "anomalies", "rows_ingested", "messages"]:
                st.session_state[k] = [] if k in ("messages", "anomalies") else (False if k == "pipeline_done" else 0)
            st.rerun()
    else:
        st.info("Upload a CSV to get started, or use the default sample data below.")
        if st.button("Load sample data"):
            from agent.tools import ingest_csv, get_uncategorized, update_category
            from agent.tools import categorize_tx as _cat_tx
            from agent.tools import detect_anomalies as _det_anom
            from agent.tools import _df

            progress = st.progress(0, text="Step 1/3: Ingesting CSV...")
            n = ingest_csv("data/sample_transactions.csv")
            progress.progress(10, text=f"Step 2/3: Categorizing {n} transactions...")

            uncategorized = get_uncategorized()
            total = len(uncategorized)
            for i, tx in enumerate(uncategorized):
                result = _cat_tx.invoke({"transaction": tx})
                update_category(tx["id"], result["category"])
                pct = 10 + int(70 * (i + 1) / max(total, 1))
                progress.progress(pct, text=f"Step 2/3: Categorized {i+1}/{total}...")

            progress.progress(80, text="Step 3/3: Detecting anomalies...")
            months_df = _df("SELECT DISTINCT strftime('%Y-%m', date) as month FROM transactions")
            all_anomalies = []
            for month in months_df["month"].tolist():
                all_anomalies.extend(_det_anom.invoke({"month": month}))

            progress.progress(100, text="Done!")
            progress.empty()

            st.session_state.pipeline_done = True
            st.session_state.rows_ingested = n
            st.session_state.anomalies = all_anomalies
            st.rerun()


# ── Helper: load transactions from DB ─────────────────────────────────────────
@st.cache_data(ttl=30)
def load_df() -> pd.DataFrame:
    import sqlite3
    try:
        con = sqlite3.connect("db/transactions.db")
        df = pd.read_sql("SELECT * FROM transactions", con)
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_charts, tab_anomalies, tab_correct = st.tabs(
    ["💬 Chat", "📊 Charts", "⚠️ Anomalies", "✏️ Correct"]
)

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask about your spending")

    if not st.session_state.pipeline_done:
        st.warning("Load transaction data first (sidebar).")
    else:
        # Render history
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(msg.content)

        if prompt := st.chat_input("e.g. How much did I spend on groceries in March?"):
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    from agent.graph import get_query_graph
                    result = get_query_graph().invoke({
                        "messages": st.session_state.messages,
                        "csv_path": "",
                        "rows_ingested": st.session_state.rows_ingested,
                        "categorized_count": 0,
                        "anomalies": [],
                        "status": "ready",
                    })
                    # Last message is the AI response
                    ai_msg = result["messages"][-1]
                    st.write(ai_msg.content)
                    st.session_state.messages.append(ai_msg)


# ── Tab 2: Charts ─────────────────────────────────────────────────────────────
with tab_charts:
    if not st.session_state.pipeline_done:
        st.warning("Load transaction data first.")
    else:
        df = load_df()
        if df.empty:
            st.info("No data yet.")
        else:
            expenses = df[df["amount"] < 0].copy()
            expenses["amount_abs"] = expenses["amount"].abs()
            expenses["month"] = pd.to_datetime(expenses["date"]).dt.to_period("M").astype(str)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Spending by Category")
                cat_totals = expenses.groupby("category")["amount_abs"].sum().reset_index()
                cat_totals.columns = ["Category", "Total ($)"]
                fig = px.bar(
                    cat_totals.sort_values("Total ($)", ascending=True),
                    x="Total ($)", y="Category", orientation="h",
                    color="Category", color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Monthly Spending Trend")
                monthly = expenses.groupby("month")["amount_abs"].sum().reset_index()
                monthly.columns = ["Month", "Total ($)"]
                fig2 = px.line(
                    monthly.sort_values("Month"),
                    x="Month", y="Total ($)", markers=True,
                    color_discrete_sequence=["#636EFA"],
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

            # Summary metrics
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Transactions", len(df))
            m2.metric("Total Expenses", f"${expenses['amount_abs'].sum():,.2f}")
            m3.metric("Avg Transaction", f"${expenses['amount_abs'].mean():,.2f}")


# ── Tab 3: Anomalies ──────────────────────────────────────────────────────────
with tab_anomalies:
    if not st.session_state.pipeline_done:
        st.warning("Load transaction data first.")
    else:
        anomalies = st.session_state.anomalies
        if not anomalies:
            st.success("No anomalies detected! 🎉")
        else:
            st.subheader(f"{len(anomalies)} Flagged Transactions")
            for a in anomalies:
                tx = a["transaction"]
                severity = a.get("severity", "medium").lower()
                badge_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "🟡")
                with st.expander(
                    f"{badge_color} {tx.get('description', 'Unknown')} — "
                    f"${abs(tx.get('amount', 0)):.2f} on {tx.get('date', '')}"
                ):
                    st.write(f"**Severity:** {severity.capitalize()}")
                    st.write(f"**Reason:** {a.get('reason', 'N/A')}")
                    st.write(f"**Category:** {tx.get('category', 'N/A')}")


# ── Tab 4: Correct a transaction ──────────────────────────────────────────────
with tab_correct:
    if not st.session_state.pipeline_done:
        st.warning("Load transaction data first.")
    else:
        st.subheader("Correct a Transaction Category")
        st.caption("Corrections are saved to memory and improve future categorizations.")

        df = load_df()
        if not df.empty:
            # Let user pick a transaction
            df_display = df[["id", "date", "description", "amount", "category"]].copy()
            df_display["label"] = df_display.apply(
                lambda r: f"#{r['id']} | {r['date']} | {r['description']} | ${abs(r['amount']):.2f} | {r['category']}",
                axis=1,
            )
            selected_label = st.selectbox("Select transaction", df_display["label"].tolist())
            selected_id = int(selected_label.split("|")[0].replace("#", "").strip())
            tx_row = df[df["id"] == selected_id].iloc[0].to_dict()

            CATEGORIES = [
                "Food & Dining", "Groceries", "Transportation", "Shopping",
                "Bills & Utilities", "Entertainment", "Healthcare",
                "Rent/Housing", "Income", "Transfer", "Other",
            ]
            current_idx = CATEGORIES.index(tx_row["category"]) if tx_row["category"] in CATEGORIES else 0
            new_cat = st.selectbox("Correct category", CATEGORIES, index=current_idx)

            if st.button("Save Correction"):
                from agent.memory import save_correction
                from agent.tools import update_category
                save_correction(tx_row, new_cat)
                update_category(selected_id, new_cat)
                st.cache_data.clear()
                st.success(f"✅ Saved: '{tx_row['description']}' → {new_cat}")
