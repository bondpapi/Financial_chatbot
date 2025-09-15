from guardrails import apply_guardrails, redact_sensitive, needs_disclaimer, DISCLAIMER
from logging_setup import get_logger
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQA
from web_rag import web_rag
from rag_utils import build_vectorstore_from_texts, build_live_knowledge_base
from finance_tools import get_stock_price, get_trend_summary, get_sector_insight
from logging_setup import get_logger
from validators import validate_query
from config import enable_langsmith_if_configured

logger = get_logger(__name__)
enable_langsmith_if_configured(logger)

# reusable local KB at import (in-memory only)
try:
    _docs = build_live_knowledge_base()
    _vectorstore = build_vectorstore_from_texts(_docs)
    _retriever = _vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        retriever=_retriever,
    )
    logger.info("Local RAG initialized.")
except Exception as e:
    logger.exception("Failed to initialize local RAG: %s", e)
    qa_chain = None

BASE_POLICY = """
You are Financial Advisor Bot.

Tool policy & cost control:
1) Prefer built-in tools first: GetStockPrice → GetTrendSummary → GetSectorInsight → RAG.
2) Use RAG (local KB) before any web search.
3) Use WebRAG ONLY when the user asks for very recent info (“today”, “latest”, “breaking”, “this week”),
   or when local tools/RAG can’t answer. WebRAG must return a Sources section.
4) Keep answers concise, well-punctuated, and in Markdown. Use bullets and short paragraphs.
"""

# Tool registry (skip RAG tool if init failed)
tools = [
    Tool(
        name="GetStockPrice",
        func=get_stock_price,
        description="Fetch latest stock price for a ticker. Input: ticker symbol (e.g., AAPL).",
    ),
    Tool(
        name="GetTrendSummary",
        func=get_trend_summary,
        description="Summarize 1-month trend & simple volatility for a stock. Input: ticker symbol.",
    ),
    Tool(
        name="GetSectorInsight",
        func=get_sector_insight,
        description="Provide basic sector info & recent performance drivers for a ticker.",
    ),
]
if qa_chain:
    tools.append(
        Tool(
            name="RAG",
            func=qa_chain.run,
            description="Answer finance questions using the in-memory knowledge base (faster/cheaper than WebRAG).",
        )
    )

tools.append(
    Tool(
        name="WebRAG",
        func=web_rag,
        description="Search the live web and answer with citations. Use for fresh/external info.",
    )
)

def _build_system_message(user_system_instructions: Optional[str] = None) -> str:
    if user_system_instructions and user_system_instructions.strip():
        return BASE_POLICY + "\n\n# User System Instructions\n" + user_system_instructions.strip()
    return BASE_POLICY

def _create_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)

def _make_agent(
    system_instructions: Optional[str] = None,
    use_openai_functions: bool = False,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_iterations: int = 3,
):
    system_message = _build_system_message(system_instructions)
    llm = _create_llm(model=model, temperature=temperature)
    agent_type = AgentType.OPENAI_FUNCTIONS if use_openai_functions else AgentType.ZERO_SHOT_REACT_DESCRIPTION
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=agent_type,
        agent_kwargs={"system_message": system_message},
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        verbose=False,
    )

def run_agent(
    query: str,
    system_instructions: Optional[str] = None,
    *,
    use_openai_functions: bool = False,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_iterations: int = 3,
) -> str:
    try:
        q = validate_query(query)
    except Exception as e:
        logger.warning("Validation failed: %s", e)
        return f"⚠️ {e}"

    ok, msg = apply_guardrails(q)
    if not ok:
        return msg
    q = msg 

    
    try:
        agent = _make_agent(
            system_instructions=system_instructions,
            use_openai_functions=use_openai_functions,
            model=model,
            temperature=temperature,
            max_iterations=max_iterations,
        )
        logger.info("Running agent | functions=%s | model=%s | temp=%.2f",
                    use_openai_functions, model, temperature)
        resp = agent.run(q)
    except Exception as e:
        logger.exception("Agent run failed: %s", redact_sensitive(str(e)))
        return "❌ Sorry, something went wrong while processing your request. Please try again."

    
    if needs_disclaimer(q):
        resp = f"{resp}\n\n{DISCLAIMER}"
    return resp


