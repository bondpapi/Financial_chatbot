import yfinance as yf
import pandas as pd

def get_stock_price(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    if not data.empty:
        return f"The latest closing price of {ticker} is ${data['Close'].iloc[-1]:.2f}."
    return f"Could not fetch price for {ticker}."

def get_trend_summary(ticker: str) -> dict:
    """Return a 1-month trend description + simple volatility %."""
    hist = yf.Ticker(ticker).history(period="1mo")
    if hist.empty:
        return {"ticker": ticker, "trend_description": "No data", "volatility": "N/A"}
    pct = hist["Close"].pct_change().dropna()
    change_30d = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
    vol = pct.std() * (len(pct) ** 0.5) * 100  # rough monthly vol proxy
    direction = "📈 upward" if change_30d >= 0 else "📉 downward"
    return {"ticker": ticker, "trend_description": f"{direction} (~{change_30d:.1f}%)", "volatility": f"{vol:.1f}"}

def get_sector_insight(ticker: str) -> dict:
    """Fetch basic sector + simple recent performance narrative."""
    t = yf.Ticker(ticker)
    try:
        info = t.get_info()  # works on yfinance>=0.2.x
    except Exception:
        info = {}
    sector = info.get("sector", "Unknown sector")
    # Use same 1mo history as a crude 'performance' line
    hist = t.history(period="1mo")
    if hist.empty:
        perf = "No recent performance data."
        drivers, risks = [], []
    else:
        change = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
        perf = f"{sector} shows {('gains' if change>=0 else 'losses')} of ~{change:.1f}% over 1 month."
        drivers = ["Earnings/news flow", "Macro rates", "Market sentiment"]
        risks = ["Volatility", "Macro uncertainty"]
    return {"sector": sector, "performance": perf, "key_drivers": drivers, "risks": risks}

def format_stock_summary(summary: dict) -> str:
    ticker = summary.get("ticker")
    company = summary.get("company")
    price = summary.get("price")
    change = summary.get("change")
    trend = "📈 upward" if change > 0 else "📉 downward"
    return f"""
### {company} ({ticker}) Market Summary

**Latest Closing Price:** ${price:.2f}  
**30-Day Trend:** {trend}  
**Change Over Past Month:** {change:.2f}%
""".strip()

def format_trend_summary(trend: dict) -> str:
    return f"""
### {trend.get('ticker')} 1-Month Trend Summary

**Trend Description:** {trend.get('trend_description')}  
**Volatility:** {trend.get('volatility')}%
""".strip()

def format_sector_insight(insight: dict) -> str:
    drivers_formatted = "\n".join(f"- {d}" for d in insight.get("key_drivers", []))
    risks_formatted = "\n".join(f"- {r}" for r in insight.get("risks", []))
    return f"""
### {insight.get('sector')} Sector Insight

**Performance:** {insight.get('performance')}  
**Key Drivers:**  
{drivers_formatted}

**Risks:**  
{risks_formatted}
""".strip()
