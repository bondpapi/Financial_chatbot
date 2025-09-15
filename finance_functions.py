import os
import json
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# -----------------------------
# 1) API integration: NewsAPI
# -----------------------------
def fetch_news_headlines(query: str, limit: int = 5) -> Dict:
    """
    Fetch top news headlines for a query using NewsAPI (if NEWS_API_KEY is set).
    Falls back to empty list if key is missing or request fails.
    """
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        return {"query": query, "headlines": [], "note": "NEWS_API_KEY not set; returning empty headlines."}

    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": max(1, min(limit, 10)), "sortBy": "publishedAt", "language": "en"}
    headers = {"X-Api-Key": api_key}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])[:limit]
        headlines = [
            {
                "title": a.get("title"),
                "source": a.get("source", {}).get("name"),
                "url": a.get("url"),
                "publishedAt": a.get("publishedAt"),
            }
            for a in articles
            if a
        ]
        return {"query": query, "headlines": headlines}
    except Exception as e:
        return {"query": query, "headlines": [], "error": str(e)}

# -----------------------------------
# 2) Calculations: Position sizing
# -----------------------------------
def calculate_position_size(
    account_size: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> Dict:
    """
    Risk-based fixed-fractional position sizing.
    position_size = (account_size * risk_pct) / (entry - stop)  for longs
    Uses absolute risk per share (handle long/short).
    """
    if account_size <= 0 or risk_pct <= 0:
        return {"error": "account_size and risk_pct must be > 0"}
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return {"error": "entry_price and stop_price must differ"}
    cash_at_risk = account_size * risk_pct
    shares = math.floor(cash_at_risk / risk_per_share)
    return {
        "account_size": account_size,
        "risk_pct": risk_pct,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "risk_per_share": risk_per_share,
        "max_shares": int(shares),
        "cash_at_risk": cash_at_risk,
    }

# -------------------------------------------------
# 3) Data analysis: Price, technicals, portfolio
# -------------------------------------------------
def get_realtime_price(ticker: str) -> Dict:
    """
    Fetch last close (or last available) using yfinance.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="5d", interval="1d")
    if hist.empty:
        return {"ticker": ticker, "error": "No data returned"}
    last_close = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else None
    change_pct = None if prev_close in (None, 0) else (last_close / prev_close - 1.0) * 100
    return {
        "ticker": ticker.upper(),
        "last_close": round(last_close, 4),
        "prev_close": None if prev_close is None else round(prev_close, 4),
        "change_pct": None if change_pct is None else round(change_pct, 3),
        "as_of_days": int(hist.index[-1].to_pydatetime().day),
    }

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def technical_snapshot(ticker: str, window_short: int = 20, window_long: int = 50) -> Dict:
    """
    Compute simple technicals: SMA20, SMA50, 14-day RSI, 1M change, realized vol.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="6mo", interval="1d")
    if hist.empty or len(hist) < max(window_long, 20):
        return {"ticker": ticker, "error": "Insufficient data"}
    close = hist["Close"].dropna()
    sma_short = close.rolling(window_short).mean().iloc[-1]
    sma_long = close.rolling(window_long).mean().iloc[-1]
    rsi14 = _rsi(close, 14).iloc[-1]
    # 1-month (21 trading days) return & vol
    if len(close) >= 21:
        m_ret = (close.iloc[-1] / close.iloc[-21] - 1.0) * 100
        daily_pct = close.pct_change().dropna()
        vol_21 = float(daily_pct.tail(21).std() * np.sqrt(252) * 100)  # annualized %
    else:
        m_ret, vol_21 = None, None
    trend = "bullish" if sma_short > sma_long else "bearish"
    return {
        "ticker": ticker.upper(),
        "sma_short": round(float(sma_short), 4),
        "sma_long": round(float(sma_long), 4),
        "trend": trend,
        "rsi14": round(float(rsi14), 2),
        "one_month_change_pct": None if m_ret is None else round(float(m_ret), 2),
        "realized_vol_pct": None if vol_21 is None else round(vol_21, 2),
    }

def portfolio_metrics(allocations: Dict[str, float], lookback: str = "1y") -> Dict:
    """
    Compute expected annualized return and volatility for a simple long-only portfolio
    using historical daily returns (close-to-close).
    allocations: {"AAPL": 0.4, "MSFT": 0.6}
    """
    tickers = list(allocations.keys())
    weights = np.array([allocations[t] for t in tickers], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        return {"error": "Allocations must sum to 1.0"}

    prices = yf.download(tickers=tickers, period=lookback, interval="1d", group_by="ticker", auto_adjust=True, progress=False)
    # Normalize to a 2D frame of closes
    if isinstance(prices.columns, pd.MultiIndex):
        close = pd.concat({t: prices[t]["Close"] for t in tickers}, axis=1).dropna()
    else:
        # Single ticker case
        close = prices["Close"].to_frame(tickers[0]).dropna()

    if close.empty or close.shape[0] < 40:
        return {"error": "Insufficient price history"}

    rets = close.pct_change().dropna()
    mu = rets.mean().values * 252.0                      # annualized mean
    sigma = rets.cov().values * 252.0                    # annualized covariance
    exp_return = float(weights @ mu)
    exp_vol = float(np.sqrt(weights @ sigma @ weights.T))
    # Simple risk metrics
    port_series = (rets @ weights)
    sharpe = exp_return / exp_vol if exp_vol > 0 else None

    return {
        "tickers": tickers,
        "weights": weights.round(4).tolist(),
        "annual_return_pct": round(exp_return * 100, 2),
        "annual_volatility_pct": round(exp_vol * 100, 2),
        "sharpe_ratio": None if sharpe is None else round(float(sharpe), 2),
        "n_days": int(rets.shape[0]),
        "lookback": lookback,
    }
