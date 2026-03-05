# Mini Project 3: Finance Agent 📈🤖

So far, we've successfully implemented the foundational components and several single-agent variants for this mini-project! Here's a quick rundown of what we've accomplished:

## What We've Built 🛠️

1. **Local Database & API Tools (Task 1)**
   - We set up our `stocks.db` from the S&P 500 company list.
   - We wrote the `get_company_overview` function to hit the Alpha Vantage API and retrieve live P/E ratios, EPS, market cap, and 52-week highs/lows.
   - We built the `get_tickers_by_sector` function to query our local SQLite database. It even has a smart fallback: if an exact sector match fails, it falls back to a fuzzier `LIKE` match on the industry column!

2. **The Baseline Agent (Task 2)**
   - We put together a `Baseline` LLM function using `gpt-4o-mini`. This baseline simply relies on its raw training data to answer standard financial questions directly, completely blind to our external tools.

3. **The Single Agent (Task 3)**
   - We built an advanced `Single Agent` that has full access to all 7 tools provided in the notebook.
   - We wrote a custom `SINGLE_AGENT_PROMPT` containing strict rules. We explicitly told the agent never to guess or fabricate numbers, and to thoughtfully chain its tools (e.g., if asked about a sector, always find the tickers first using the database before iterating through them to fetch price data).
   - We tested its iterative looping, wrangled some timeout bugs, and proved that the agent correctly chains multiple tool calls to arrive at the perfect answer on harder queries!

## Next Steps 🚀

- Moving on to **Task 4: The Multi-Agent System**, where we'll set up specialized AI agents (like a Manager, a Researcher, etc.) to collaborate, instead of trusting a single heavily-loaded prompt to do everything.

_Note: If you plan to test this project yourself, you'll need to create your own `.env` file with an `OPENAI_API_KEY` and an `ALPHAVANTAGE_API_KEY`!_
