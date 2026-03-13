# Mini Project 3: Finance Agent 📈🤖

For this project, we built a fully functional Financial Intelligence Assistant! It started as a set of python scripts but we've now upgraded it into a sleek, dark-themed Streamlit application.

Here is a breakdown of what we accomplished:

## What We've Built 🛠️

1. **Local Database & API Tools**
   - We set up our `stocks.db` from the S&P 500 company list.
   - We wrote the `get_company_overview` function to hit the Alpha Vantage API and retrieve live P/E ratios, EPS, market cap, and 52-week highs/lows.
   - We built the `get_tickers_by_sector` function to query our local SQLite database. It even has a smart fallback: if an exact sector match fails, it falls back to a fuzzier `LIKE` match on the industry column!

2. **The Baseline Agent**
   - We put together a `Baseline` LLM function using `gpt-4o-mini`. This baseline simply relies on its raw training data to answer standard financial questions directly, completely blind to our external tools.

3. **The Single Agent**
   - We built an advanced `Single Agent` that has full access to all 7 tools provided.
   - We wrote a custom `SINGLE_AGENT_PROMPT` containing strict rules. We explicitly told the agent never to guess or fabricate numbers, and to thoughtfully chain its tools.

4. **The Multi-Agent System**
   - We set up a collaborative environment where a Database Agent, Fundamentals Agent, Technical/Price Agent, and Sentiment Agent all work together, orchestrated by a central planner and evaluated by a Critic Agent.

5. **The Streamlit Application**
   - Finally, we wrapped everything into a stunning black, white, and neon green Streamlit app!
   - We created a sidebar to let users swap between the Baseline, Single Agent, and Multi-Agent architectures, pick either `gpt-4o-mini` or `gpt-4o`, and directly chat with our AI agents to get live financial insights.

## How to Run 🚀

If you want to run this project yourself:

1. Make sure you have your own `.env` file with `OPENAI_API_KEY` and `ALPHAVANTAGE_API_KEY`.
2. Install the necessary packages.
3. Run `python -m streamlit run app.py` and interact with the AI in your browser!
