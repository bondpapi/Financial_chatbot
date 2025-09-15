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
from config import get_setting, set_setting

st.set_page_config(page_title="Financial Advisor Bot", page_icon="📊")
st.title("📊 Financial Advisor Bot")

# --- API key management: allow user to paste key at runtime (never persisted) ---
with st.sidebar:
    st.subheader("🔐 API Keys")
    if not get_setting("OPENAI_API_KEY"):
        key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if key:
            set_setting("OPENAI_API_KEY", key)
            st.success("OpenAI key set for this session.")
    # Optional: NewsAPI, Tavily keys etc.
    # news_key = st.text_input("NewsAPI Key", type="password")

with st.expander("⚙️ Agent settings", expanded=False):
    sys_instr = st.text_area(
        "Optional system instructions (treated as system prompt)",
        placeholder="Paste special instructions (e.g., risk profile, tone, constraints, or a job description)...",
    )
    colA, colB, colC = st.columns(3)
    with colA:
        use_fc = st.toggle("Use OpenAI function-calling", value=False,
                           help="ON = OpenAI function-calling agent; OFF = ReAct agent")
    with colB:
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=1)
    with colC:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    max_iterations = st.slider("Max tool iterations", 1, 5, 3, 1)

# --- Main query box ---
query = st.text_input("Ask a question about a stock, sector, or trend:")

if st.button("Ask") and query:
    try:
        _ = validate_query(query)  # pre-validate for friendly UX
        with st.spinner("Analyzing..."):
            result = run_agent(
                query,
                system_instructions=sys_instr,
                use_openai_functions=use_fc,
                model=model,
                temperature=temperature,
                max_iterations=max_iterations,
            )
        st.markdown(result)
    except Exception as e:
        st.error(f"Validation error: {e}")

# --- Quick tools (manual) ---
st.subheader("⚡ Quick tools (manual)")
col1, col2, col3 = st.columns(3)

with col1:
    t = st.text_input("Ticker for price", key="price_ticker", value="AAPL")
    if st.button("Get Price"):
        try:
            st.markdown(get_stock_price(normalize_ticker(t)))
        except Exception as e:
            st.error(f"Invalid ticker: {e}")

with col2:
    t2 = st.text_input("Ticker for 1-mo trend", key="trend_ticker", value="AAPL")
    if st.button("Get 1-mo Trend"):
        try:
            trend = get_trend_summary(normalize_ticker(t2))
            st.markdown(format_trend_summary(trend))
        except Exception as e:
            st.error(f"Invalid ticker: {e}")

with col3:
    t3 = st.text_input("Ticker for sector insight", key="sector_ticker", value="AAPL")
    if st.button("Sector Insight"):
        try:
            insight = get_sector_insight(normalize_ticker(t3))
            st.markdown(format_sector_insight(insight))
        except Exception as e:
            st.error(f"Invalid ticker: {e}")

