import os
from urllib.parse import urlparse
from dotenv import load_dotenv

from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# optional rate limit decorator (if you included rate_limit.py)
try:
    from rate_limit import rate_limited
except Exception:
    def rate_limited(*a, **k):
        def wrap(fn): return fn
        return wrap

load_dotenv()

# Keep this tight & safe
ALLOWLIST = {
    "www.sec.gov", "sec.gov",
    "www.reuters.com", "reuters.com",
    "www.cnbc.com", "cnbc.com",
    "www.federalreserve.gov", "federalreserve.gov",
    "www.investopedia.com", "investopedia.com",
}

def _filter_urls(urls):
    safe = []
    for u in urls:
        try:
            p = urlparse(u)
            if p.scheme != "https":
                continue
            host = (p.netloc or "").lower()
            if host in ALLOWLIST:
                safe.append(u)
        except Exception:
            continue
    return safe

def _format_citations(answer: str, sources):
    seen, lines = set(), []
    for i, d in enumerate(sources or [], 1):
        src = d.metadata.get("source") or d.metadata.get("url") or "Unknown"
        if src in seen:
            continue
        seen.add(src)
        title = d.metadata.get("title") or "Source"
        lines.append(f"[{i}] {title} — {src}")
    cite_block = "\n\n**Sources:**\n" + "\n".join(lines) if lines else ""
    return (answer or "").strip() + cite_block

def _make_loader(urls):
    """
    Compatibility shim across langchain_community versions.
    Some versions accept timeout only, some accept max_concurrency, some neither.
    """
    # Try preferred signature (timeout)
    try:
        return AsyncHtmlLoader(urls, timeout=12)
    except TypeError:
        pass
    # Try older signature (max_concurrency)
    try:
        return AsyncHtmlLoader(urls)  # fall back – older versions accept only urls
    except TypeError:
        return AsyncHtmlLoader(urls)

def _online_retriever(query: str, k_search=5, k_retrieve=6):
    # Tavily search restricted to allowlist first
    retr = TavilySearchAPIRetriever(
        k=k_search,
        include_answer=False,
        include_raw_content=False,
        include_domains=list(ALLOWLIST),
    )
    hits = retr.get_relevant_documents(query)
    urls = [d.metadata.get("source") for d in hits if d.metadata.get("source")]
    urls = _filter_urls(urls)

    # If nothing on allowlist, do a broader search but still filter afterwards
    if not urls:
        retr = TavilySearchAPIRetriever(k=k_search)
        hits = retr.get_relevant_documents(query)
        urls = _filter_urls([d.metadata.get("source") for d in hits if d.metadata.get("source")])

    if not urls:
        return None  # caller can decide what to do

    loader = _make_loader(urls)
    docs = loader.load()  # fetch HTML concurrently under the hood (version-dependent)
    docs = Html2TextTransformer().transform_documents(docs)

    # Cap extreme pages to avoid blowing memory
    for d in docs:
        if d.page_content and len(d.page_content) > 200_000:
            d.page_content = d.page_content[:200_000]

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120).split_documents(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vs.as_retriever(search_kwargs={"k": k_retrieve})

@rate_limited(calls=6, period=60)
def web_rag(query: str) -> str:
    # Friendly message if Tavily isn't configured
    if not os.getenv("TAVILY_API_KEY"):
        return ("🔎 Live web search is disabled (no TAVILY_API_KEY). "
                "Try a non-time-sensitive question, or add TAVILY_API_KEY in Streamlit Secrets.")

    try:
        retriever = _online_retriever(query)
        if retriever is None:
            return "No trusted sources found for this query."
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
        )
        out = qa.invoke(query)
        return _format_citations(out.get("result", ""), out.get("source_documents", []))
    except Exception as e:
        # Keep user-friendly, avoid stack traces in UI
        return (f"⚠️ Live web search failed: {e}. "
                "Please try again later or ask a non-time-sensitive question.")
