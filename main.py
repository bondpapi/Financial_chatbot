# financial_bot/main.py

import json
import csv
import io
import streamlit as st
from chatbot import run_agent
from finance_tools import (
    get_stock_price,
    get_trend_summary,
    get_sector_insight,
    format_trend_summary,
    format_sector_insight,
)
from validators import normalize_ticker, validate_query
from config import get_setting

st.set_page_config(page_title="Financial Advisor Bot", page_icon="📊")
st.title("📊 Financial Advisor Bot")
st.info("This app provides educational market information only and is **not financial advice**.")

# ---------- Dev-only visibility (uses st.query_params) ----------
try:
    dev_val = st.query_params.get("dev", "0")
    DEV_QUERY = dev_val[0] if isinstance(dev_val, list) else str(dev_val)
except Exception:
    DEV_QUERY = "0"

DEV_MODE = (
    get_setting("SHOW_AGENT_SETTINGS", "false").lower() in {"1", "true", "yes"}
    or DEV_QUERY.lower() in {"1", "true", "yes"}
)

# ---------- Conversation state ----------
if "chat" not in st.session_state:
    st.session_state.chat = []   # list[{"role": "user"/"assistant", "content": str}]

# ---------- Popular symbol quick picks ----------
POPULAR_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "BRK.B", "AVGO", "NFLX"]

def _quick_pick(sym: str):
    st.session_state["price_symbol"]  = sym
    st.session_state["trend_symbol"]  = sym
    st.session_state["sector_symbol"] = sym
    st.session_state["main_query"] = f"What's the latest on {sym} earnings today?"
    st.toast(f"Selected {sym}")
    st.rerun()

st.caption("Top Picks:")
for i in range(0, len(POPULAR_SYMBOLS), 6):
    row = POPULAR_SYMBOLS[i:i+6]
    cols = st.columns(len(row))
    for j, sym in enumerate(row):
        if cols[j].button(sym, key=f"pick_{sym}"):
            _quick_pick(sym)

# ---------- Main chat input ----------
query = st.text_input(
    "Ask about a stock, sector, or trend:",
    key="main_query",
    value=st.session_state.get("main_query", ""),
)


extra_system = None
use_fc = True
model_override = None
max_iterations = 3
if DEV_MODE:
    with st.expander("🪄 Debug / Agent settings (dev only)", expanded=False):
        extra_system = st.text_area("Additional developer instructions (optional)")
        use_fc = st.toggle("Use OpenAI function-calling", value=True)
        model_override = st.selectbox("Force model (optional)", [None, "gpt-4o", "gpt-4o-mini"], index=0)
        max_iterations = st.slider("Max tool iterations", 1, 5, 3, 1)

# Ask button
if st.button("Ask") and query:
    try:
        _ = validate_query(query)
        st.session_state.chat.append({"role": "user", "content": query})
        with st.spinner("Analyzing..."):
            result = run_agent(
                query,
                extra_system=extra_system,
                use_openai_functions=use_fc,
                model=model_override,
                max_iterations=max_iterations,
                return_steps=DEV_MODE,   # <— return tool trace for dev
            )
        if isinstance(result, dict):
            st.session_state.chat.append({"role": "assistant", "content": result["text"]})
            # visualize steps (dev only)
            with st.expander("🔍 Tool trace (dev)", expanded=False):
                for idx, (action, observation) in enumerate(result.get("steps", []), 1):
                    try:
                        tool_name = getattr(action, "tool", "tool")
                        tool_in   = getattr(action, "tool_input", "")
                        st.markdown(f"**{idx}. {tool_name}**")
                        st.code(str(tool_in))
                        if observation:
                            st.caption("Result:")
                            st.write(observation if len(str(observation)) < 1000 else str(observation)[:1000] + " …")
                        st.divider()
                    except Exception:
                        st.write(f"{idx}. (step parse error)")
        else:
            st.session_state.chat.append({"role": "assistant", "content": result})
    except Exception as e:
        st.error(f"Validation error: {e}")

# ---------- Render conversation ----------
if st.session_state.chat:
    st.subheader("💬 Conversation")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- Export buttons ----
    st.write("")
    colj, colc, colm = st.columns(3)

    # JSON export
    with colj:
        json_data = json.dumps(st.session_state.chat, ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ Export JSON",
            data=json_data,
            file_name="conversation.json",
            mime="application/json",
        )

    # CSV export (role, content)
    with colc:
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["role", "content"])
        for m in st.session_state.chat:
            writer.writerow([m["role"], m["content"].replace("\n", "\\n")])
        st.download_button(
            "⬇️ Export CSV",
            data=csv_buf.getvalue(),
            file_name="conversation.csv",
            mime="text/csv",
        )

    # Markdown export
    with colm:
        md_lines = []
        for m in st.session_state.chat:
            tag = "User" if m["role"] == "user" else "Assistant"
            md_lines.append(f"**{tag}:** {m['content']}")
        st.download_button(
            "⬇️ Export Markdown",
            data="\n\n".join(md_lines),
            file_name="conversation.md",
            mime="text/markdown",
        )

# ---------- Quick tools ----------
st.subheader("⚡ Quick tools")
col1, col2, col3 = st.columns(3)

with col1:
    sym = st.text_input("Stock Symbol for price (e.g., AAPL)", key="price_symbol",
                        value=st.session_state.get("price_symbol", "AAPL"))
    if st.button("Get Price"):
        try:
            st.markdown(get_stock_price(normalize_ticker(sym)))
        except Exception as e:
            st.error(f"Invalid symbol: {e}")

with col2:
    sym2 = st.text_input("Stock Symbol for 1-month trend", key="trend_symbol",
                         value=st.session_state.get("trend_symbol", "AAPL"))
    if st.button("Get 1-mo Trend"):
        try:
            trend = get_trend_summary(normalize_ticker(sym2))
            st.markdown(format_trend_summary(trend))
        except Exception as e:
            st.error(f"Invalid symbol: {e}")

with col3:
    sym3 = st.text_input("Stock Symbol for sector insight", key="sector_symbol",
                         value=st.session_state.get("sector_symbol", "AAPL"))
    if st.button("Sector Insight"):
        try:
            insight = get_sector_insight(normalize_ticker(sym3))
            st.markdown(format_sector_insight(insight))
        except Exception as e:
            st.error(f"Invalid symbol: {e}")
