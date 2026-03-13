import os, json, time, sqlite3, requests, textwrap
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

try:
    from google.colab import userdata
    OPENAI_API_KEY = userdata.get('OpenAI_proj.3')
    ALPHAVANTAGE_API_KEY = userdata.get('ALPHA_proj3')
    print("Loaded keys from Google Colab Secrets")
except ImportError:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OpenAI_proj.3", "YOUR_KEY")
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHA_proj3", "YOUR_KEY")
    print("Loaded keys from local .env file")

MODEL_SMALL  = "gpt-4o-mini"
MODEL_LARGE  = "gpt-4o"
ACTIVE_MODEL = MODEL_SMALL          # switch to MODEL_LARGE for the second run

client = OpenAI(api_key=OPENAI_API_KEY)
print(f"✅ Ready  |  active model: {ACTIVE_MODEL}")

DB_PATH = "stocks.db"

def create_local_database(csv_path: str = "sp500_companies.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found.\n"
            "Download from: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks"
        )
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "symbol":"ticker", "shortname":"company",
        "sector":"sector",  "industry":"industry",
        "exchange":"exchange", "marketcap":"market_cap_raw"
    })
    def cap_bucket(v):
        try:
            v = float(v)
            return "Large" if v >= 10_000_000_000 else "Mid" if v >= 2_000_000_000 else "Small"
        except: return "Unknown"
    df["market_cap"] = df["market_cap_raw"].apply(cap_bucket)
    df = (df.dropna(subset=["ticker","company"])
            .drop_duplicates(subset=["ticker"])
            [["ticker","company","sector","industry","market_cap","exchange"]])
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("stocks", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker ON stocks(ticker)")
    conn.commit()
    n = pd.read_sql_query("SELECT COUNT(*) AS n FROM stocks", conn).iloc[0]["n"]
    print(f"✅ {n} companies loaded into stocks.db")
    print("\nDistinct sector values stored in DB:")
    print(pd.read_sql_query("SELECT DISTINCT sector FROM stocks ORDER BY sector", conn).to_string(index=False))
    conn.close()

if __name__ == "__main__":
    create_local_database()

# ── Tool 1 ── Provided ────────────────────────────────────────
def get_price_performance(tickers: list, period: str = "1y") -> dict:
    """
    % price change for a list of tickers over a period.
    Valid periods: '1mo', '3mo', '6mo', 'ytd', '1y'
    Returns: { TICKER: {start_price, end_price, pct_change, period} }
    """
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end   = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price"  : round(end,   2),
                "pct_change" : round((end - start) / start * 100, 2),
                "period"     : period,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results

# ── Tool 2 ── Provided ────────────────────────────────────────
def get_market_status() -> dict:
    """Open / closed status for global stock exchanges."""
    return requests.get(
        f"https://www.alphavantage.co/query?function=MARKET_STATUS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()

# ── Tool 3 ── Provided ────────────────────────────────────────
def get_top_gainers_losers() -> dict:
    """Today's top gaining, top losing, and most active tickers."""
    return requests.get(
        f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()

# ── Tool 4 ── Provided ────────────────────────────────────────
def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    """
    Latest headlines + Bullish / Bearish / Neutral sentiment for a ticker.
    Returns: { ticker, articles: [{title, source, sentiment, score}] }
    """
    data = requests.get(
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
        f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title"    : a.get("title"),
                "source"   : a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score"    : a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }

# ── Tool 5 ── Provided ────────────────────────────────────────
def query_local_db(sql: str) -> dict:
    """
    Run any SQL SELECT on stocks.db.
    Table 'stocks' columns: ticker, company, sector, industry, market_cap, exchange
    market_cap values: 'Large' | 'Mid' | 'Small'
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("✅ 5 provided tools ready")

# ── Tool 6 — YOUR IMPLEMENTATION ─────────────────────────────
def get_company_overview(ticker: str) -> dict:
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    try:
        data = requests.get(url, timeout=10).json()
        if "Name" not in data:
            return {"error": f"No overview data for {ticker}"}
        return {
            "ticker": ticker,
            "name": data.get("Name", ""),
            "sector": data.get("Sector", ""),
            "pe_ratio": data.get("PERatio", ""),
            "eps": data.get("EPS", ""),
            "market_cap": data.get("MarketCapitalization", ""),
            "52w_high": data.get("52WeekHigh", ""),
            "52w_low": data.get("52WeekLow", "")
        }
    except Exception as e:
        return {"error": str(e)}


# ── Tool 7 — YOUR IMPLEMENTATION ─────────────────────────────
def get_tickers_by_sector(sector: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT ticker, company, industry FROM stocks WHERE LOWER(sector) = LOWER(?)", conn, params=(sector,))
    if df.empty:
        pattern = f"%{sector}%"
        df = pd.read_sql_query("SELECT ticker, company, industry FROM stocks WHERE industry LIKE ?", conn, params=(pattern,))
    conn.close()
    return {
        "sector": sector,
        "stocks": df.to_dict(orient="records")
    }


# ── Automated tests — all assertions must pass ────────────────
if __name__ == "__main__":
    print("=== Tool 6 tests ===")
    aapl = get_company_overview("AAPL")
    assert "pe_ratio" in aapl,             "Missing pe_ratio key"
    assert aapl.get("ticker") == "AAPL",   "ticker key wrong"
    assert aapl.get("name"),               "name should not be empty"
    print(f"  AAPL P/E: {aapl['pe_ratio']} ✅")
    
    bad = get_company_overview("INVALIDTICKER999")
    assert "error" in bad,                 "Invalid ticker should return error key"
    print(f"  Invalid ticker handled correctly ✅")
    
    print("\n=== Tool 7 tests ===")
    tech = get_tickers_by_sector("Information Technology")
    assert len(tech["stocks"]) > 0,        "Should find IT stocks (exact sector match)"
    print(f"  'Information Technology' → {len(tech['stocks'])} stocks ✅")
    
    semi = get_tickers_by_sector("semiconductor")
    assert len(semi["stocks"]) > 0,        "Should find via industry fallback (LIKE match)"
    print(f"  'semiconductor' (industry fallback) → {len(semi['stocks'])} stocks ✅")
    
    print("\n✅ All tool tests passed")

def _s(name, desc, props, req):
    return {"type":"function","function":{
        "name":name,"description":desc,
        "parameters":{"type":"object","properties":props,"required":req}}}

SCHEMA_TICKERS  = _s("get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Information Technology', 'Energy') or sub-sectors ('semiconductor', 'insurance').",
    {"sector":{"type":"string","description":"Sector or industry name"}}, ["sector"])

SCHEMA_PRICE    = _s("get_price_performance",
    "Get % price change for a list of tickers over a time period. "
    "Periods: '1mo','3mo','6mo','ytd','1y'.",
    {"tickers":{"type":"array","items":{"type":"string"}},
     "period":{"type":"string","default":"1y"}}, ["tickers"])

SCHEMA_OVERVIEW = _s("get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker":{"type":"string","description":"Ticker symbol e.g. 'AAPL'"}}, ["ticker"])

SCHEMA_STATUS   = _s("get_market_status",
    "Check whether global stock exchanges are currently open or closed.", {}, [])

SCHEMA_MOVERS   = _s("get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.", {}, [])

SCHEMA_NEWS     = _s("get_news_sentiment",
    "Get latest news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {"ticker":{"type":"string"},"limit":{"type":"integer","default":5}}, ["ticker"])

SCHEMA_SQL      = _s("query_local_db",
    "Run a SQL SELECT on stocks.db. "
    "Table 'stocks': ticker, company, sector, industry, market_cap (Large/Mid/Small), exchange.",
    {"sql":{"type":"string","description":"A valid SQL SELECT statement"}}, ["sql"])

# All 7 schemas in one list — used by single agent
ALL_SCHEMAS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW,
               SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL]

# Dispatch map — maps the tool name string the LLM returns → the Python function to call
ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector" : get_tickers_by_sector,
    "get_price_performance"  : get_price_performance,
    "get_company_overview"   : get_company_overview,
    "get_market_status"      : get_market_status,
    "get_top_gainers_losers" : get_top_gainers_losers,
    "get_news_sentiment"     : get_news_sentiment,
    "query_local_db"         : query_local_db,
}

if __name__ == "__main__":
    print("✅ Schemas ready")
    print(f"   Tools available: {list(ALL_TOOL_FUNCTIONS.keys())}")

@dataclass
class AgentResult:
    agent_name   : str
    answer       : str
    tools_called : list  = field(default_factory=list)   # tool names in order called
    raw_data     : dict  = field(default_factory=dict)   # tool name → raw tool output
    confidence   : float = 0.0                           # set by evaluator / critic
    issues_found : list  = field(default_factory=list)   # set by evaluator / critic
    reasoning    : str   = ""

    def summary(self):
        print(f"\n{'─'*54}")
        print(f"Agent      : {self.agent_name}")
        print(f"Tools used : {', '.join(self.tools_called) or 'none'}")
        print(f"Confidence : {self.confidence:.0%}")
        if self.issues_found:
            print(f"Issues     : {'; '.join(self.issues_found)}")
        print(f"Answer     :\n{textwrap.indent(self.answer[:500], '  ')}")

if __name__ == "__main__":
    print("✅ AgentResult defined")

def run_specialist_agent(
    agent_name   : str,
    system_prompt: str,
    task         : str,
    tool_schemas : list,
    max_iters    : int  = 8,
    verbose      : bool = True,
) -> AgentResult:
    """
    Core agentic loop used by every agent in this project.

    How it works:
      1. Sends system_prompt + task to the LLM
      2. If the LLM requests a tool call → looks up the function in ALL_TOOL_FUNCTIONS,
         executes it, appends the result to the message history, loops back to step 1
      3. When the LLM produces a response with no tool calls → returns an AgentResult

    Parameters
    ----------
    agent_name    : display name for logging
    system_prompt : the agent's persona, rules, and focus area
    task          : the specific question or sub-task for this agent
    tool_schemas  : list of schema dicts this agent is allowed to use
                    (pass [] for no tools — used by baseline)
    max_iters     : hard cap on iterations to prevent infinite loops
    verbose       : print each tool call as it happens
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]
    tools_called = []
    raw_data = {}

    for _ in range(max_iters):
        response = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            temperature=0.0
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return AgentResult(
                agent_name=agent_name,
                answer=msg.content or "",
                tools_called=tools_called,
                raw_data=raw_data
            )

        messages.append(msg)
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if verbose:
                print(f"[{agent_name}] Calling {func_name} with {args}")

            tools_called.append(func_name)

            if func_name in ALL_TOOL_FUNCTIONS:
                try:
                    res = ALL_TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    res = {"error": str(e)}
            else:
                res = {"error": f"Unknown function: {func_name}"}

            raw_data[func_name] = res
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(res)
            })

    return AgentResult(
        agent_name=agent_name,
        answer="Error: max_iters reached",
        tools_called=tools_called,
        raw_data=raw_data
    )
if __name__ == "__main__":
    print("✅ run_specialist_agent ready")

def run_baseline(question: str, verbose: bool = True) -> AgentResult:
    response = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful financial AI assistant. Answer the user's question directly based on your training knowledge. Do not use tools. If you are uncertain about exact numbers, state that you are making an estimate."},
            {"role": "user", "content": question}
        ],
        temperature=0.0
    )
    answer = response.choices[0].message.content
    return AgentResult(
        agent_name="Baseline",
        answer=answer,
        tools_called=[]
    )
    # Implement a single LLM call with no tools.
    # Use run_specialist_agent() with an empty tool_schemas list — or make the call directly.
    # Return an AgentResult with agent_name="Baseline" and tools_called=[].

if __name__ == "__main__":
    # Quick test
    bl = run_baseline("What is Apple's approximate P/E ratio?", verbose=True)
    assert bl.tools_called == [], "Baseline must not call any tools"
    assert bl.answer, "Answer must not be empty"
    bl.summary()

# ── YOUR SINGLE AGENT IMPLEMENTATION ─────────────────────────
# Write your system prompt and run_single_agent() function here.
# Comments above explain what to think about — the implementation is yours.

SINGLE_AGENT_PROMPT = """
You are an expert financial AI assistant equipped with tools to fetch live stock data and interact with a local database.
Your goal is to answer the user's questions truthfully and accurately using ONLY the data provided by your tools.

Follow these STRICT RULES:
1. NEVER fabricate or guess any data (prices, P/E ratios, tickers, news sentiment, etc.). If a tool returns an error or empty data, state that clearly instead of inventing a number.
2. If the user asks about a SECTOR or INDUSTRY, you MUST ALWAYS use the `get_tickers_by_sector` tool FIRST to find the relevant tickers in the database. DO NOT guess which tickers belong to a sector.
3. If the user asks a multi-step question (e.g. "Which energy stocks grew the most?"), you MUST:
   a. Use `get_tickers_by_sector` to get the tickers.
   b. Extract the ticker symbols from the result.
   c. Pass those ticker symbols into `get_price_performance` or `get_company_overview` as required by the question.
4. For filtering by market cap or executing generic DB filters, you can use `query_local_db` if needed.
5. Provide your final answer clearly, confirming the data used to reach your conclusion.
"""

def run_single_agent(question: str, verbose: bool = True) -> AgentResult:
    return run_specialist_agent(
        agent_name="Single Agent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=question,
        tool_schemas=ALL_SCHEMAS,
        max_iters=15,
        verbose=verbose
    )



BENCHMARK_QUESTIONS = [
    # ── EASY ──────────────────────────────────────────────────────────────
    {"id":"Q01","complexity":"easy","category":"sector_lookup",
     "question":"List all semiconductor companies in the database.",
     "expected":"Should return company names and tickers for semiconductor stocks from the local DB. "
                "Tickers include NVDA, AMD, INTC, QCOM, AVGO, TXN, ADI, MU and others."},
    {"id":"Q02","complexity":"easy","category":"market_status",
     "question":"Are the US stock markets open right now?",
     "expected":"Should return the current open/closed status for NYSE and NASDAQ "
                "with their trading hours."},
    {"id":"Q03","complexity":"easy","category":"fundamentals",
     "question":"What is the P/E ratio of Apple (AAPL)?",
     "expected":"Should return AAPL P/E ratio as a single numeric value fetched from Alpha Vantage."},
    {"id":"Q04","complexity":"easy","category":"sentiment",
     "question":"What is the latest news sentiment for Microsoft (MSFT)?",
     "expected":"Should return 3–5 recent MSFT headlines with Bullish/Bearish/Neutral labels and scores."},
    {"id":"Q05","complexity":"easy","category":"price",
     "question":"What is NVIDIA's stock price performance over the last month?",
     "expected":"Should return NVDA start price, end price, and % change for the 1-month period."},

    # ── MEDIUM ─────────────────────────────────────────────────────────────
    {"id":"Q06","complexity":"medium","category":"price_comparison",
     "question":"Compare the 1-year price performance of AAPL, MSFT, and GOOGL. Which grew the most?",
     "expected":"Should fetch 1y performance for all 3 tickers, return % change for each, "
                "and identify the highest performer."},
    {"id":"Q07","complexity":"medium","category":"fundamentals",
     "question":"Compare the P/E ratios of AAPL, MSFT, and NVDA. Which looks most expensive?",
     "expected":"Should return P/E ratios for all 3 tickers and identify which has the highest P/E."},
    {"id":"Q08","complexity":"medium","category":"sector_price",
     "question":"Which energy stocks in the database had the best 6-month performance?",
     "expected":"Should query the DB for energy sector tickers, fetch 6-month price performance "
                "for each, and return them ranked by % change."},
    {"id":"Q09","complexity":"medium","category":"sentiment",
     "question":"What is the news sentiment for Tesla (TSLA) and how has its stock moved this month?",
     "expected":"Should return TSLA news sentiment (label + score) AND 1-month price % change "
                "from two separate tool calls."},
    {"id":"Q10","complexity":"medium","category":"fundamentals",
     "question":"What are the 52-week high and low for JPMorgan (JPM) and Goldman Sachs (GS)?",
     "expected":"Should return 52-week high and low for both JPM and GS fetched from Alpha Vantage."},

    # ── HARD ───────────────────────────────────────────────────────────────
    {"id":"Q11","complexity":"hard","category":"multi_condition",
     "question":"Which tech stocks dropped this month but grew this year? Return the top 3.",
     "expected":"Should get tech tickers from DB, fetch both 1-month and year-to-date performance, "
                "filter for negative 1-month AND positive YTD, return top 3 by yearly growth with "
                "exact percentages. Results must satisfy both conditions simultaneously."},
    {"id":"Q12","complexity":"hard","category":"multi_condition",
     "question":"Which large-cap technology stocks on NASDAQ have grown more than 20% this year?",
     "expected":"Should query DB for large-cap NASDAQ tech stocks, fetch YTD performance, "
                "filter for >20% growth, and return matching tickers with exact % change."},
    {"id":"Q13","complexity":"hard","category":"cross_domain",
     "question":"For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios "
                "and current news sentiment?",
     "expected":"Should find semiconductor tickers in DB, rank by 1-year return to find top 3, "
                "then fetch P/E ratio AND news sentiment for each — requiring three separate "
                "data domains (price, fundamentals, sentiment)."},
    {"id":"Q14","complexity":"hard","category":"cross_domain",
     "question":"Compare the market cap, P/E ratio, and 1-year stock performance of JPM, GS, and BAC.",
     "expected":"Should return market cap, P/E, and 1-year % change for all 3 tickers, "
                "combining Alpha Vantage fundamentals and yfinance price data."},
    {"id":"Q15","complexity":"hard","category":"multi_condition",
     "question":"Which finance sector stocks are trading closer to their 52-week low than their "
                "52-week high? Return the news sentiment for each.",
     "expected":"Should get finance sector tickers from DB, fetch 52-week high and low for each, "
                "compute proximity to the low, then fetch news sentiment for qualifying stocks."},
]

if __name__ == "__main__":
    print("--- Running Single Agent on 5 Benchmark Questions ---")
    for q in BENCHMARK_QUESTIONS[:5]:
        print(f"\nQ: {q['question']}")
        res = run_single_agent(q['question'])
        if res:
            res.summary()


# ── YOUR MULTI-AGENT IMPLEMENTATION ──────────────────────────
#
from helper_agents import run_multi_agent

if __name__ == "__main__":
    # Test 1 — check return contract
    out1 = run_multi_agent("What is the P/E ratio of Apple (AAPL)?")
    assert "final_answer"   in out1, "Missing final_answer key"
    assert "agent_results"  in out1, "Missing agent_results key"
    assert "elapsed_sec"    in out1, "Missing elapsed_sec key"
    assert "architecture"   in out1, "Missing architecture key"
    print(f"Architecture : {out1['architecture']}")
    print(f"Agents ran   : {[r.agent_name for r in out1['agent_results']]}")
    print(f"Answer       : {out1['final_answer'][:200]}")
    
    # Test 2 — cross-domain hard question
    out2 = run_multi_agent("For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios and current news sentiment?")
    print(f"Architecture : {out2['architecture']}")
    print(f"Agents ran   : {[r.agent_name for r in out2['agent_results']]}")
    print(f"Tools used   : {[t for r in out2['agent_results'] for t in r.tools_called]}")
    print(f"\nAnswer:\n{out2['final_answer'][:400]}")

# ── YOUR EVALUATOR IMPLEMENTATION ────────────────────────────
#
from helper_agents import run_evaluator

if __name__ == "__main__":
    # ── Calibration tests — all three must behave as expected ─────
    print("=== Calibration Test 1 — correct answer (expect score=3) ===")
    t1 = run_evaluator(
        question        = "What is the P/E ratio of Apple (AAPL)?",
        expected_answer = "Should return AAPL P/E ratio as a single numeric value from Alpha Vantage.",
        agent_answer    = "The current P/E ratio of Apple Inc. (AAPL) is 33.45 according to Alpha Vantage Data.",
    )
    print(f"  Score: {t1['score']}/3 | Hallucination: {t1['hallucination_detected']}")
    print(f"  Reasoning: {t1['reasoning']}")
    
    print("\n=== Calibration Test 2 — fabricated number (expect hallucination=True, score≤1) ===")
    t2 = run_evaluator(
        question        = "What is the P/E ratio of Apple (AAPL)?",
        expected_answer = "Should return AAPL P/E ratio as a single numeric value from Alpha Vantage.",
        agent_answer    = "Apple's P/E ratio is approximately 28.5 based on current market conditions.",
    )
    print(f"  Score: {t2['score']}/3 | Hallucination: {t2['hallucination_detected']}")
    print(f"  Reasoning: {t2['reasoning']}")
    assert t2["hallucination_detected"] == True, "Should detect fabricated P/E as hallucination"
    
    print("\n=== Calibration Test 3 — refusal (expect score=0) ===")
    t3 = run_evaluator(
        question        = "What is the P/E ratio of Apple (AAPL)?",
        expected_answer = "Should return AAPL P/E ratio as a single numeric value from Alpha Vantage.",
        agent_answer    = "I cannot retrieve real-time financial data. Please check Yahoo Finance.",
    )
    print(f"  Score: {t3['score']}/3 | Hallucination: {t3['hallucination_detected']}")
    print(f"  Reasoning: {t3['reasoning']}")
    assert t3["score"] == 0, "Refusal should score 0"
    
    print("\n✅ Evaluator calibration complete")

BENCHMARK_QUESTIONS = [
    # ── EASY ──────────────────────────────────────────────────────────────
    {"id":"Q01","complexity":"easy","category":"sector_lookup",
     "question":"List all semiconductor companies in the database.",
     "expected":"Should return company names and tickers for semiconductor stocks from the local DB. "
                "Tickers include NVDA, AMD, INTC, QCOM, AVGO, TXN, ADI, MU and others."},
    {"id":"Q02","complexity":"easy","category":"market_status",
     "question":"Are the US stock markets open right now?",
     "expected":"Should return the current open/closed status for NYSE and NASDAQ "
                "with their trading hours."},
    {"id":"Q03","complexity":"easy","category":"fundamentals",
     "question":"What is the P/E ratio of Apple (AAPL)?",
     "expected":"Should return AAPL P/E ratio as a single numeric value fetched from Alpha Vantage."},
    {"id":"Q04","complexity":"easy","category":"sentiment",
     "question":"What is the latest news sentiment for Microsoft (MSFT)?",
     "expected":"Should return 3–5 recent MSFT headlines with Bullish/Bearish/Neutral labels and scores."},
    {"id":"Q05","complexity":"easy","category":"price",
     "question":"What is NVIDIA's stock price performance over the last month?",
     "expected":"Should return NVDA start price, end price, and % change for the 1-month period."},

    # ── MEDIUM ─────────────────────────────────────────────────────────────
    {"id":"Q06","complexity":"medium","category":"price_comparison",
     "question":"Compare the 1-year price performance of AAPL, MSFT, and GOOGL. Which grew the most?",
     "expected":"Should fetch 1y performance for all 3 tickers, return % change for each, "
                "and identify the highest performer."},
    {"id":"Q07","complexity":"medium","category":"fundamentals",
     "question":"Compare the P/E ratios of AAPL, MSFT, and NVDA. Which looks most expensive?",
     "expected":"Should return P/E ratios for all 3 tickers and identify which has the highest P/E."},
    {"id":"Q08","complexity":"medium","category":"sector_price",
     "question":"Which energy stocks in the database had the best 6-month performance?",
     "expected":"Should query the DB for energy sector tickers, fetch 6-month price performance "
                "for each, and return them ranked by % change."},
    {"id":"Q09","complexity":"medium","category":"sentiment",
     "question":"What is the news sentiment for Tesla (TSLA) and how has its stock moved this month?",
     "expected":"Should return TSLA news sentiment (label + score) AND 1-month price % change "
                "from two separate tool calls."},
    {"id":"Q10","complexity":"medium","category":"fundamentals",
     "question":"What are the 52-week high and low for JPMorgan (JPM) and Goldman Sachs (GS)?",
     "expected":"Should return 52-week high and low for both JPM and GS fetched from Alpha Vantage."},

    # ── HARD ───────────────────────────────────────────────────────────────
    {"id":"Q11","complexity":"hard","category":"multi_condition",
     "question":"Which tech stocks dropped this month but grew this year? Return the top 3.",
     "expected":"Should get tech tickers from DB, fetch both 1-month and year-to-date performance, "
                "filter for negative 1-month AND positive YTD, return top 3 by yearly growth with "
                "exact percentages. Results must satisfy both conditions simultaneously."},
    {"id":"Q12","complexity":"hard","category":"multi_condition",
     "question":"Which large-cap technology stocks on NASDAQ have grown more than 20% this year?",
     "expected":"Should query DB for large-cap NASDAQ tech stocks, fetch YTD performance, "
                "filter for >20% growth, and return matching tickers with exact % change."},
    {"id":"Q13","complexity":"hard","category":"cross_domain",
     "question":"For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios "
                "and current news sentiment?",
     "expected":"Should find semiconductor tickers in DB, rank by 1-year return to find top 3, "
                "then fetch P/E ratio AND news sentiment for each — requiring three separate "
                "data domains (price, fundamentals, sentiment)."},
    {"id":"Q14","complexity":"hard","category":"cross_domain",
     "question":"Compare the market cap, P/E ratio, and 1-year stock performance of JPM, GS, and BAC.",
     "expected":"Should return market cap, P/E, and 1-year % change for all 3 tickers, "
                "combining Alpha Vantage fundamentals and yfinance price data."},
    {"id":"Q15","complexity":"hard","category":"multi_condition",
     "question":"Which finance sector stocks are trading closer to their 52-week low than their "
                "52-week high? Return the news sentiment for each.",
     "expected":"Should get finance sector tickers from DB, fetch 52-week high and low for each, "
                "compute proximity to the low, then fetch news sentiment for qualifying stocks."},
]

if __name__ == "__main__":
    print(f"✅ {len(BENCHMARK_QUESTIONS)} questions loaded")
    for tier in ["easy","medium","hard"]:
        count = sum(1 for q in BENCHMARK_QUESTIONS if q["complexity"]==tier)
        print(f"   {tier:8s}: {count} questions")

@dataclass
class EvalRecord:
    # Question
    question_id : str;  question    : str;  complexity : str
    category    : str;  expected    : str
    # Baseline
    bl_answer       : str   = "";  bl_time         : float = 0.0
    bl_score        : int   = -1;  bl_reasoning    : str   = ""
    bl_hallucination: str   = "";  bl_issues       : str   = ""
    # Single agent
    sa_answer       : str   = "";  sa_tools        : str   = ""
    sa_tool_count   : int   = 0;   sa_iters        : int   = 0
    sa_time         : float = 0.0; sa_score        : int   = -1
    sa_reasoning    : str   = "";  sa_hallucination: str   = ""
    sa_issues       : str   = ""
    # Multi agent
    ma_answer       : str   = "";  ma_tools        : str   = ""
    ma_tool_count   : int   = 0;   ma_time         : float = 0.0
    ma_confidence   : str   = "";  ma_critic_issues: int   = 0
    ma_agents       : str   = "";  ma_architecture : str   = ""
    ma_score        : int   = -1;  ma_reasoning    : str   = ""
    ma_hallucination: str   = "";  ma_issues       : str   = ""


# ── Column rename map (internal name → Excel header) ─────────
_COL_NAMES = {
    "question_id":"Question ID","question":"Question","complexity":"Difficulty",
    "category":"Category","expected":"Expected Answer",
    "bl_answer":"Baseline Answer","bl_time":"Baseline Time (s)",
    "bl_score":"Baseline Score /3","bl_reasoning":"Baseline Eval Reasoning",
    "bl_hallucination":"Baseline Hallucination","bl_issues":"Baseline Issues",
    "sa_answer":"SA Answer","sa_tools":"SA Tools Used","sa_tool_count":"SA Tool Count",
    "sa_iters":"SA Iterations","sa_time":"SA Time (s)",
    "sa_score":"SA Score /3","sa_reasoning":"SA Eval Reasoning",
    "sa_hallucination":"SA Hallucination","sa_issues":"SA Issues",
    "ma_answer":"MA Answer","ma_tools":"MA Tools Used","ma_tool_count":"MA Tool Count",
    "ma_time":"MA Time (s)","ma_confidence":"MA Avg Confidence",
    "ma_critic_issues":"MA Critic Issue Count","ma_agents":"MA Agents Activated",
    "ma_architecture":"MA Architecture",
    "ma_score":"MA Score /3","ma_reasoning":"MA Eval Reasoning",
    "ma_hallucination":"MA Hallucination","ma_issues":"MA Issues",
}


def _save_excel(records: list, path: str):
    df = pd.DataFrame([r.__dict__ for r in records]).rename(columns=_COL_NAMES)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # ── Sheet 1: full results ──────────────────────────────
        df.to_excel(writer, index=False, sheet_name="Results")

        # ── Sheet 2: summary by architecture × difficulty ──────
        rows = []
        for arch, sc, tc, hc in [
            ("Baseline",     "Baseline Score /3", "Baseline Time (s)", "Baseline Hallucination"),
            ("Single Agent", "SA Score /3",       "SA Time (s)",       "SA Hallucination"),
            ("Multi Agent",  "MA Score /3",       "MA Time (s)",       "MA Hallucination"),
        ]:
            for tier in ["easy", "medium", "hard", "all"]:
                subset = df if tier == "all" else df[df["Difficulty"] == tier]
                valid  = subset[subset[sc] >= 0]
                avg_s  = valid[sc].mean() if len(valid) else 0
                rows.append({
                    "Architecture"   : arch,
                    "Difficulty"     : tier,
                    "Questions Scored": len(valid),
                    "Avg Score /3"   : round(avg_s, 2),
                    "Accuracy %"     : round(avg_s / 3 * 100, 1),
                    "Avg Time (s)"   : round(df[tc].mean(), 1),
                    "Hallucinations" : (df[hc] == "True").sum(),
                })
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Summary")


def run_full_evaluation(output_xlsx: str = "results.xlsx", delay_sec: float = 3.0):
    """
    Run all 15 questions through baseline, single agent, and multi agent.
    Score each answer. Write results to Excel after every question.
    """
    records = []
    total   = len(BENCHMARK_QUESTIONS)
    print(f"\n{'='*62}")
    print(f"  FULL EVALUATION  |  {total} questions × 3 architectures")
    print(f"  Model: {ACTIVE_MODEL}  |  Output: {output_xlsx}")
    print(f"{'='*62}\n")

    for i, q in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"[{i:02d}/{total}] {q['id']} ({q['complexity']:6s}) {q['question'][:52]}...")
        rec = EvalRecord(question_id=q["id"], question=q["question"],
                         complexity=q["complexity"], category=q["category"],
                         expected=q["expected"])

        # ── Baseline ───────────────────────────────────────────
        print("         baseline  ...", end=" ", flush=True)
        try:
            t0 = time.time()
            bl = run_baseline(q["question"], verbose=False)
            rec.bl_answer = bl.answer.replace("\n", " ")
            rec.bl_time   = round(time.time() - t0, 2)
            ev = run_evaluator(q["question"], q["expected"], bl.answer)
            rec.bl_score        = ev.get("score", -1)
            rec.bl_reasoning    = ev.get("reasoning", "")
            rec.bl_hallucination= str(ev.get("hallucination_detected", False))
            rec.bl_issues       = " | ".join(ev.get("key_issues", []))
            print(f"✅  {rec.bl_time:5.1f}s  score {rec.bl_score}/3")
        except Exception as e:
            print(f"❌  {e}")

        # ── Single Agent ───────────────────────────────────────
        print("         single    ...", end=" ", flush=True)
        try:
            t0 = time.time()
            sa = run_single_agent(q["question"], verbose=False)
            rec.sa_answer    = sa.answer.replace("\n", " ")
            rec.sa_tools     = ", ".join(sa.tools_called)
            rec.sa_tool_count= len(sa.tools_called)
            rec.sa_iters     = len(sa.tools_called) + 1   # approx
            rec.sa_time      = round(time.time() - t0, 2)
            ev = run_evaluator(q["question"], q["expected"], sa.answer)
            rec.sa_score        = ev.get("score", -1)
            rec.sa_reasoning    = ev.get("reasoning", "")
            rec.sa_hallucination= str(ev.get("hallucination_detected", False))
            rec.sa_issues       = " | ".join(ev.get("key_issues", []))
            print(f"✅  {rec.sa_time:5.1f}s  score {rec.sa_score}/3"
                  f"  tools [{rec.sa_tools or 'none'}]")
        except Exception as e:
            print(f"❌  {e}")

        # ── Multi Agent ────────────────────────────────────────
        print("         multi     ...", end=" ", flush=True)
        try:
            t0  = time.time()
            ma  = run_multi_agent(q["question"], verbose=False)
            res = ma.get("agent_results", [])
            all_tools  = [t for r in res for t in r.tools_called]
            all_issues = [iss for r in res for iss in r.issues_found]
            avg_conf   = sum(r.confidence for r in res) / len(res) if res else 0.0
            rec.ma_answer        = ma["final_answer"].replace("\n", " ")
            rec.ma_tools         = ", ".join(dict.fromkeys(all_tools))
            rec.ma_tool_count    = len(all_tools)
            rec.ma_time          = round(time.time() - t0, 2)
            rec.ma_confidence    = f"{avg_conf:.0%}"
            rec.ma_critic_issues = len(all_issues)
            rec.ma_agents        = ", ".join(r.agent_name for r in res)
            rec.ma_architecture  = ma.get("architecture", "")
            ev = run_evaluator(q["question"], q["expected"], ma["final_answer"])
            rec.ma_score        = ev.get("score", -1)
            rec.ma_reasoning    = ev.get("reasoning", "")
            rec.ma_hallucination= str(ev.get("hallucination_detected", False))
            rec.ma_issues       = " | ".join(ev.get("key_issues", []))
            print(f"✅  {rec.ma_time:5.1f}s  score {rec.ma_score}/3"
                  f"  conf {rec.ma_confidence}  issues {rec.ma_critic_issues}")
        except Exception as e:
            print(f"❌  {e}")

        records.append(rec)
        _save_excel(records, output_xlsx)       # save progress after every question

        if i < total:
            print(f"         ⏳ waiting {delay_sec}s ...\n")
            time.sleep(delay_sec)

    # ── Print summary table ────────────────────────────────────
    print(f"\n{'='*62}  RESULTS")
    print(f"{'Architecture':<18} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>8}")
    print("─" * 52)
    for arch, sk in [("Baseline","bl_score"),("Single Agent","sa_score"),("Multi Agent","ma_score")]:
        def pct(tier):
            s = [getattr(r, sk) for r in records
                 if getattr(r, sk) >= 0 and (tier=="all" or r.complexity==tier)]
            return f"{sum(s)/len(s)/3*100:.0f}%" if s else "—"
        print(f"{arch:<18} {pct('easy'):>8} {pct('medium'):>8} {pct('hard'):>8} {pct('all'):>8}")

    print(f"\n✅ Saved → {output_xlsx}")
    return output_xlsx

if __name__ == "__main__":
    print("✅ Evaluation runner ready")
    
    # ── Sanity check — one question, all three architectures ──────
    q_test = BENCHMARK_QUESTIONS[2]        # Q03 — easy fundamentals
    print(f"Test question: {q_test['question']}\n")
    
    bl_t = run_baseline(q_test["question"], verbose=False)
    sa_t = run_single_agent(q_test["question"], verbose=False)
    ma_t = run_multi_agent(q_test["question"], verbose=False)
    
    print(f"Baseline     : {bl_t.answer[:120]}")
    print(f"Single Agent : {sa_t.answer[:120]}  |  tools: {sa_t.tools_called}")
    print(f"Multi Agent  : {ma_t['final_answer'][:120]}  |  arch: {ma_t['architecture']}")
    
    ev_bl = run_evaluator(q_test["question"], q_test["expected"], bl_t.answer)
    ev_sa = run_evaluator(q_test["question"], q_test["expected"], sa_t.answer)
    ev_ma = run_evaluator(q_test["question"], q_test["expected"], ma_t["final_answer"])
    print(f"\nScores — Baseline: {ev_bl['score']}/3  |  Single: {ev_sa['score']}/3  |  Multi: {ev_ma['score']}/3")

# ── Full evaluation — gpt-4o-mini ─────────────────────────────
#ACTIVE_MODEL = MODEL_SMALL
#run_full_evaluation(output_xlsx="results_gpt4o_mini.xlsx", delay_sec=3.0)

# ── Full evaluation — gpt-4o (required for reflection Q4) ─────
#ACTIVE_MODEL = MODEL_LARGE
#run_full_evaluation(output_xlsx="results_gpt4o.xlsx", delay_sec=3.0)