import yfinance as yf
import requests
from dotenv import load_dotenv
import os


NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def fetch_company_summary(ticker="AAPL"):
    try:
        stock = yf.Ticker(ticker)
        info = stock.get_info()
        return info.get("longBusinessSummary", f"No summary available for {ticker}.")
    except Exception as e:
        return f"Error fetching summary for {ticker}: {e}"



def fetch_news(topic="stock market", page_size=5, api_key=NEWS_API_KEY):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "relevancy",
        "apiKey": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [f"{a['title']}\n\n{a.get('content') or a.get('description')}" for a in articles]
    except Exception as e:
        return [f"Error fetching news for topic '{topic}': {e}"]
    
def build_live_knowledge_base():
    docs = []
    # Fetch summaries for major companies
    major_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    for ticker in major_tickers:
        summary = fetch_company_summary(ticker)
        docs.append(f"Company Summary for {ticker}:\n{summary}")

    # Fetch recent news on stock market
    news_articles = fetch_news("stock market", page_size=5)
    for i, article in enumerate(news_articles, 1):
        docs.append(f"News Article {i}:\n{article}")

    return docs

