# Mini Project 3 — Agentic AI in FinTech

## Overview

You will build and compare three AI architectures that answer real financial market questions using live data from Yahoo Finance and Alpha Vantage. The experiment is entirely yours to design within the structure provided. We are doing this step by step, so only build what I instruct you to build, do not go ahead and build the entire model 0 to 1 style. The following is just an overview to give you an idea what the big picture should look like and here enables you to build the respective sub parts whenever I tell you to do so. 

| Architecture | Description |
|---|---|
| **Baseline** | Plain LLM call — no tools, answers from training knowledge only |
| **Single Agent** | One LLM with access to all 7 tools — you design the system prompt |
| **Multi-Agent** | Multiple agents working together — you choose the architecture |

Results are written to an Excel file with scores, timings, hallucination flags, and confidence metrics for every question across all three architectures.
You are expected to submit a document with  insight of your learning from the assignment and the choices you made while completing it

---

## Setup (just for your information, I will get the API/Alpha key and provide  it to you)

### 1 — Python dependencies

```bash
pip install openai requests pandas yfinance python-dotenv openpyxl
```

Python 3.9 or higher required.

### 2 — API keys

**OpenAI**
→ https://platform.openai.com/api-keys
You will use two models: `gpt-4o-mini` and `gpt-4o`.
Estimated total cost for both full evaluation runs: $2–4.

**Alpha Vantage (free tier)**
→ https://www.alphavantage.co/support/#api-key
Free tier: 25 requests/day, ~5/minute. No credit card required.
If you hit the daily limit, split your evaluation across two sessions.

Create a `.env` file in the same folder as the notebook — never commit this file:

```
OPENAI_API_KEY=sk-proj-...
ALPHAVANTAGE_API_KEY=...
```

### 3 — S&P 500 data (I downloaded and put in the project folder) 

1. Go to https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
2. Download and unzip
3. Copy `sp500_companies.csv` into the same folder as the notebook
4. Run `create_local_database()` in Step 1 of the notebook

---

## What You Implement

### Task 1 — Two tool functions (20 pts)

**`get_company_overview(ticker)`**
Calls the Alpha Vantage OVERVIEW endpoint. Returns P/E ratio, EPS, market cap, 52-week high/low.
Docs: https://www.alphavantage.co/documentation/#company-overview

**`get_tickers_by_sector(sector)`**
Queries `stocks.db` for companies in a sector or industry.
Critical: the DB stores broad sectors (`"Information Technology"`) in the `sector` column and specific sub-sectors (`"Semiconductors"`) in the `industry` column. You must implement a two-step fallback — exact match on `sector`, then `LIKE` match on `industry`.
Run `create_local_database()` first to see the actual sector values stored.

Both tools have automated assertion tests in the notebook that must all pass before you continue.

---

