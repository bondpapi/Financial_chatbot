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
st.title("Financial Advisor Bot")
st.info("This app provides educational market information only and is **not financial advice**.")

# --------- Dev-only settings visibility ---------
qp = st.experimental_get_query_params()
DEV_MODE = (
    (get_setting("SHOW_AGENT_SETTINGS", "false").lower() in {"1", "true", "yes"})
    or (qp.get("dev", ["0"])[0].lower() in {"1", "true", "yes"})
)

# --------- Main query box ---------
query = st.text_input("Ask about a stock, sector, or trend:")

# In prod, we keep settings hidden; in Dev, expose a small expander
extra_system = None
use_fc = True
model_override = None
max_iterations = 3

if DEV_MODE:
    with st.expander("⚙️ Agent settings (dev only)", expanded=False):
        extra_system = st.text_area(
            "Additional developer instructions (optional)",
            placeholder="Only for testing/tuning; end users cannot see this."
        )
        use_fc = st.toggle("Use OpenAI function-calling", value=True)
        model_override = st.selectbox(
            "Force model (optional)", [None, "gpt-4o", "gpt-4o-mini"], index=0,
            help="Leave None for automatic routing."
        )
        max_iterations = st.slider("Max tool iterations", 1, 5, 3, 1)


if st.button("Ask") and query:
    try:
        _ = validate_query(query)  # pre-validate for friendlier UX
        with st.spinner("Analyzing..."):
            # Mirrors user input into system prompt internally
            result = run_agent(
                query,
                extra_system=extra_system,
                use_openai_functions=use_fc,
                model=model_override,     # None = auto route
                max_iterations=max_iterations,
            )
        st.markdown(result)
    except Exception as e:
        st.error(f"Validation error: {e}")

# --------- Quick tools (manual) ---------
st.subheader("⚡ Quick tools")
col1, col2, col3 = st.columns(3)

with col1:
    sym = st.text_input("Stock symbol for price (e.g., AAPL)", key="price_symbol", value="AAPL")
    if st.button("Get Price"):
        try:
            st.markdown(get_stock_price(normalize_ticker(sym)))
        except Exception as e:
            st.error(f"Invalid symbol: {e}")

with col2:
    sym2 = st.text_input("Stock symbol for 1-month trend", key="trend_symbol", value="AAPL")
    if st.button("Get 1-mo Trend"):
        try:
            trend = get_trend_summary(normalize_ticker(sym2))
            st.markdown(format_trend_summary(trend))
        except Exception as e:
            st.error(f"Invalid symbol: {e}")

with col3:
    sym3 = st.text_input("Stock symbol for sector insight", key="sector_symbol", value="AAPL")
    if st.button("Sector Insight"):
        try:
            insight = get_sector_insight(normalize_ticker(sym3))
            st.markdown(format_sector_insight(insight))
        except Exception as e:
            st.error(f"Invalid symbol: {e}")