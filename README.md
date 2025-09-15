# 📊 Financial Assistant Bot

> **Educational demo** that answers finance questions with tools, RAG, and live web search — built with **LangChain** + **OpenAI** + **Streamlit**.  
> **Not financial advice.**

---

## Table of Contents
- [Overview](#overview)
- [Requirements Coverage](#requirements-coverage)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment (.env)](#environment-env)
- [Quickstart (Local)](#quickstart-local)
- [Docker](#docker)
- [LangChain & Tools](#langchain--tools)
- [Security & Guardrails](#security--guardrails)
- [Validation & Rate Limiting](#validation--rate-limiting)
- [Logging & Monitoring](#logging--monitoring)
- [API Key Management](#api-key-management)
- [Example Prompts](#example-prompts)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Roadmap](#roadmap)
- [Credits](#credits)

---

## Overview
The **Financial Advisor Bot** is a Streamlit app that uses a LangChain Agent to answer finance questions. It prioritizes fast local tools (price, trend, sector), falls back to a small in-memory RAG, and only hits a **live web RAG** (with citations and a domain allowlist) when necessary. Users can supply **dynamic system instructions** that are appended to a safe base policy. You can toggle between **ReAct** and **OpenAI Function-Calling** agent styles.

---

## Requirements Coverage
**Domain Specialization**
- ✅ Specific domain: retail **market insights** (stocks/sector/trends).
- ✅ Focused knowledge base: small in-memory KB from company summaries + curated news.
- ✅ Domain prompts: base policy + user-supplied system instructions.
- ✅ Security for finance: injection filtering, PII redaction, domain allowlist, auto “not financial advice” disclaimer.

**Technical Implementation**
- ✅ **LangChain** for OpenAI integration.
- ✅ Proper error handling with user-friendly fallbacks.
- ✅ Logging & optional **LangSmith** tracing.
- ✅ User input validation (query, ticker).
- ✅ Lightweight rate limiting + environment-based API-key management.

---

## Features
- **Agent tool preference**:  
  `GetStockPrice` → `GetTrendSummary` → `GetSectorInsight` → **Local RAG** → **WebRAG (citations)**
- **Dynamic system prompt**: user instructions appended to a safe base policy
- **Toggle**: ReAct vs **OpenAI Function-Calling**
- **Security**: prompt-injection soft filter, PII redaction, domain allowlist, auto disclaimer
- **Validation**: ticker & query checks
- **Rate limiting**: decorators for heavy functions
- **Logging**: console + rotating file; optional **LangSmith** tracing
- **Dockerized** for reproducible runs

---

## Architecture

- **Streamlit UI**
    - calls `chatbot.run_agent(query, system_instructions)`
        - **LangChain Agent** (ReAct or OPENAI_FUNCTIONS)
            - `GetStockPrice` / `GetTrendSummary` / `GetSectorInsight`
            - **RAG** (FAISS over small in-memory KB from `fetch_sources.py`)
            - **WebRAG** (Tavily → HTML loader → chunk → FAISS) + citations
        - **Guardrails** (injection & PII)
        - **Validators** (query, ticker)
        - **Logging + Rate limiting**

## Project Structure

| File/Folder           | Description                                               |
|-----------------------|----------------------------------------------------------|
| `chatbot.py`          | Agent factory and `run_agent` (dynamic system prompt)    |
| `main.py`             | Streamlit app                                            |
| `finance_tools.py`    | Stock/Trend/Sector tools and formatters                  |
| `fetch_sources.py`    | Seed texts (company summaries and news)                  |
| `rag_utils.py`        | Build in-memory FAISS from texts                         |
| `web_rag.py`          | Live web RAG with domain allowlist and citations         |
| `finance_functions.py`| Extra function-calling examples (optional)               |
| `validators.py`       | Prompt/ticker validation and cleaning                    |
| `guardrails.py`       | Injection/PII filters and disclaimers                    |
| `rate_limit.py`       | Lightweight token-bucket limiter                         |
| `logging_setup.py`    | Structured logging (console and rotating file)           |
| `config.py`           | Env handling and optional LangSmith toggle               |
| `requirements.txt`    | Python dependencies                                      |
| `Dockerfile`          | Docker build recipe                                      |
| `.dockerignore`       | Docker ignore rules                                      |
| `.gitignore`          | Git ignore rules                                         |

---

## Prerequisites
- Python **3.11**
- An **OpenAI API key**
- (Optional) **Tavily** and/or **NewsAPI** keys

---
## Environment (.env)
Create `financial_bot/.env` (do **not** commit):
```env
OPENAI_API_KEY=sk-...

# Optional:
TAVILY_API_KEY=tvly-...
NEWS_API_KEY=...

# Monitoring (optional):
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=financial-advisor-bot
# LANGCHAIN_ENDPOINT=https://api.langchain.com
``` 

## Quickstart (Local)
```bash 
cd financial_bot
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run Streamlit
streamlit run main.py
# Visit http://localhost:8501
```
## 🐳 Docker
Build & run:
```bash
# from repo root (the folder that contains financial_bot/)
docker build -t financial-bot:latest .
docker run --rm -p 8501:8501 --env-file financial_bot/.env financial-bot:latest
```
Docker Compose (optional):
```yaml
services:
  app:
    build: .
    ports: ["8501:8501"]
    env_file: financial_bot/.env
    restart: unless-stopped
```
## LangChain & Tools
- **LangChain**: agent framework, tools, chains, memory, prompt templates
- **Tools**:
    - `GetStockPrice`: fetches current stock price (mocked or via Tavily)
    - `GetTrendSummary`: summarizes market trends (mocked or via NewsAPI)
    - `GetSectorInsight`: provides sector insights (mocked or via NewsAPI)
    - **Local RAG**: FAISS over small in-memory KB (company summaries + curated news)
    - **Web RAG**: live web search with citations (Tavily → HTML loader → chunk → FAISS)
- **Agent Types**: ReAct and OpenAI Function-Calling

## Security & Guardrails
- **Prompt Injection Filter**: soft filter to detect/remove harmful patterns
- **PII Redaction**: removes sensitive info (emails, phones, SSNs)
- **Domain Allowlist**: restricts web RAG to finance-related domains
- **Auto Disclaimer**: appends “not financial advice” to responses
- **Guardrails**: integrated with LangChain responses

## Validation & Rate Limiting
- **Input Validation**: checks for empty/malicious queries; validates stock tickers
- **Rate Limiting**: lightweight token-bucket limiter on heavy functions 

## Logging & Monitoring
- **Structured Logging**: console + rotating file logs with timestamps, levels, messages
- **LangSmith Tracing**: optional, enabled via env vars

## API Key Management
- All API keys managed via environment variables

## Example Prompts
- "What's the current price of AAPL and how is the tech sector performing?"
- "Summarize recent trends in renewable energy stocks."
- "Provide insights on the healthcare sector and any notable companies."
- "What are the latest news and trends affecting the financial markets?"
- "How is the automotive sector doing this quarter?"
- "Can you give me an overview of the stock market performance this week?"
- "What are the top-performing stocks in the technology sector?"
- "Are there any significant trends in the energy sector right now?"
- "What impact is the current economic climate having on retail stocks?"
- "Give me insights on the pharmaceutical sector and recent developments."

## Troubleshooting
- Ensure all required API keys are set in the `.env` file.
- Check logs in `financial_bot/logs/` for errors.
- Verify Python version is 3.11.
- For Docker issues, ensure ports are correctly mapped and env vars are passed.
- If LangSmith tracing isn't working, verify `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY`.
- For rate limiting issues, adjust limits in `rate_limit.py` as needed.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Roadmap
- [x] Core functionality with LangChain agent and tools
- [x] Dynamic system prompt support
- [x] Input validation and rate limiting
- [x] Logging and optional LangSmith tracing
- [x] Dockerization
- [ ] User authentication and API key management
- [ ] More robust error handling and fallbacks
- [ ] Expand knowledge base with more sources
- [ ] UI/UX improvements in Streamlit app
- [ ] Additional tools and agent capabilities
- [ ] Deployment scripts for cloud hosting
- [ ] User authentication and API key management
- [ ] More robust error handling and fallbacks
- [ ] Expand knowledge base with more sources
- [ ] UI/UX improvements in Streamlit app
- [ ] Additional tools and agent capabilities
- [ ] Deployment scripts for cloud hosting

## Credits
[Michael Bond](https://github.com/bondpapi)
- Built with [LangChain](https://langchain.com/), [OpenAI](https://openai.com/), and [Streamlit](https://streamlit.io/)
- Inspired by various LangChain examples and templates
- Logo by [Streamlit](https://streamlit.io/)
- Guardrails by [Guardrails AI](https://guardrails.io/)
- Rate limiting inspired by [Flask-Limiter](https://flask-limiter.readthedocs.io/en/stable/)
- Logging setup inspired by [Real Python](https://realpython.com/python-logging/)
- Web RAG integration inspired by [Tavily](https://tavily.com/)


