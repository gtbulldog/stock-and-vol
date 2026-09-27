# Stock and Vol

Streamlit dashboard for equity + options analytics: price, moving averages, IV curves, IV term structure, gamma exposure, realized/implied vol history, fundamentals, income statement (with analyst-consensus projections), quarterly balance sheet and cash-burn, and filtered news.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Sign in at [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**, pick this repo, main branch, entry point `app.py`.
4. Under **Advanced settings → Secrets**, paste:

   ```toml
   FMP_API_KEY = "your-key-here"
   ```

5. Click **Deploy**. Streamlit will install `requirements.txt` and start the app.

## Run locally

```bash
pip install -r requirements.txt
# Add your FMP key to .streamlit/secrets.toml (see .streamlit/secrets.toml.example)
streamlit run app.py
```

## Data sources

- **Prices, options, fundamentals, news**: Yahoo Finance via `yfinance`
- **Analyst-consensus projections**: Financial Modeling Prep (FMP)
- Fallbacks: Yahoo `revenue_estimate` / `earnings_estimate` → growth-rate extrapolation

## Secrets

The FMP API key never lives in this repo. Locally, put it in `.streamlit/secrets.toml`
(which is gitignored). On Streamlit Cloud, set it in the app's Secrets UI.
