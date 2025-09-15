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

# Rate limiting decorator 
try:
    from rate_limit import rate_limited
except Exception:
    def rate_limited(*a, **k):
        def wrap(fn): return fn
        return wrap

load_dotenv()

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
    for i, d in enumerate(sources, 1):
        src = d.metadata.get("source") or d.metadata.get("url") or "Unknown"
        if src in seen: 
            continue
        seen.add(src)
        title = d.metadata.get("title") or "Source"
        lines.append(f"[{i}] {title} — {src}")
    cite_block = "\n\n**Sources:**\n" + "\n".join(lines) if lines else ""
    return answer.strip() + cite_block

def _online_retriever(query: str, k_search=5, k_retrieve=6):
    retr = TavilySearchAPIRetriever(
        k=k_search, include_answer=False, include_raw_content=False,
        include_domains=list(ALLOWLIST)  # tighter search
    )
    hits = retr.get_relevant_documents(query)
    urls = [d.metadata.get("source") for d in hits if d.metadata.get("source")]
    urls = _filter_urls(urls)

    if not urls:
        # One more pass without include_domains (still filtered later)
        retr = TavilySearchAPIRetriever(k=k_search)
        hits = retr.get_relevant_documents(query)
        urls = _filter_urls([d.metadata.get("source") for d in hits if d.metadata.get("source")])

    loader = AsyncHtmlLoader(urls, max_concurrency=4, timeout=12)
    docs = loader.load()

    # HTML -> text
    docs = Html2TextTransformer().transform_documents(docs)

    # Cap size to avoid overloading
    for d in docs:
        if d.page_content and len(d.page_content) > 200_000:
            d.page_content = d.page_content[:200_000]

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120).split_documents(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vs.as_retriever(search_kwargs={"k": k_retrieve})

@rate_limited(calls=6, period=60)   
def web_rag(query: str) -> str:
    retriever = _online_retriever(query)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )
    out = qa.invoke(query)
    return _format_citations(out["result"], out.get("source_documents", []))

