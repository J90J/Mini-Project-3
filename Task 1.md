
### Task 1 — Two tool functions (20 pts)

**`get_company_overview(ticker)`**
Calls the Alpha Vantage OVERVIEW endpoint. Returns P/E ratio, EPS, market cap, 52-week high/low.
Docs: https://www.alphavantage.co/documentation/#company-overview

**`get_tickers_by_sector(sector)`**
Queries `stocks.db` for companies in a sector or industry.
Critical: the DB stores broad sectors (`"Information Technology"`) in the `sector` column and specific sub-sectors (`"Semiconductors"`) in the `industry` column. You must implement a two-step fallback — exact match on `sector`, then `LIKE` match on `industry`.
Run `create_local_database()` first to see the actual sector values stored.

Both tools have automated assertion tests in the notebook that must all pass before you continue.


