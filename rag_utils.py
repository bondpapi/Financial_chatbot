from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from fetch_sources import fetch_company_summary, fetch_news

def build_vectorstore_from_texts(texts, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents(texts)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def build_live_knowledge_base():
    texts = []
    for ticker in ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "META", "NFLX", "NVDA", "JPM", "V", "MA", "DIS", "PYPL", "ADBE", "CRM", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "NASDAQ100"]:
        texts.append(fetch_company_summary(ticker))
    texts.extend(fetch_news("stock market"))
    return texts


