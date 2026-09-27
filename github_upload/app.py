"""
Stock Volatility Dashboard
--------------------------
Streamlit app: ticker -> price/volume chart, 30d IV, 30d HV, and
Yang-Zhang realized-volatility cones.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# Persistent IV history cache (so we can compute D/W/M changes across sessions)
IV_CACHE_FILE = Path(__file__).resolve().parent / "_iv_history.csv"

# ---------------------------------------------------------------------------
# Volatility math
# ---------------------------------------------------------------------------

TRADING_DAYS = 252


def yang_zhang_vol(df: pd.DataFrame, window: int, trading_days: int = TRADING_DAYS) -> pd.Series:
    """
    Annualized rolling Yang-Zhang volatility on an OHLC dataframe.

    σ²_YZ = σ²_overnight + k · σ²_open-close + (1 - k) · σ²_RS
    where k = 0.34 / (1.34 + (n+1)/(n-1))

    Yang & Zhang (2000) showed this is the minimum-variance unbiased
    estimator that handles both overnight jumps and intraday drift.
    """
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    c_prev = c.shift(1)

    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_co = np.log(c / o)
    log_oc_prev = np.log(o / c_prev)  # overnight return

    # Rogers-Satchell (drift independent intraday vol)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    overnight_var = log_oc_prev.rolling(window).var(ddof=0)
    open_close_var = log_co.rolling(window).var(ddof=0)
    rs_var = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
    yz_vol = np.sqrt(yz_var.clip(lower=0) * trading_days)
    return yz_vol


def close_to_close_vol(df: pd.DataFrame, window: int, trading_days: int = TRADING_DAYS) -> pd.Series:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    return log_ret.rolling(window).std(ddof=0) * np.sqrt(trading_days)


# ---------------------------------------------------------------------------
# Black-Scholes pricing + implied vol inversion
# ---------------------------------------------------------------------------

import math

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes-Merton price for a European option with continuous dividend yield."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = (S - K) if is_call else (K - S)
        return max(intrinsic, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * math.exp(-q * T) * _norm_pdf(d1) * sqrtT


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """BSM gamma (same for calls and puts under continuous dividend yield)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * sqrtT)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    tol: float = 1e-6,
    max_iter: int = 80,
) -> float | None:
    """Invert Black-Scholes for IV. Newton-Raphson with bisection fallback."""
    if price is None or price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    # No-arb bounds
    if is_call:
        intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        upper = S * math.exp(-q * T)
    else:
        intrinsic = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
        upper = K * math.exp(-r * T)
    if price < intrinsic - 1e-4 or price > upper + 1e-4 or price <= intrinsic + 1e-7:
        return None

    # Newton-Raphson
    sigma = 0.3
    for _ in range(max_iter):
        p = bs_price(S, K, T, r, q, sigma, is_call)
        v = bs_vega(S, K, T, r, q, sigma)
        diff = p - price
        if abs(diff) < tol:
            return sigma
        if v < 1e-10:
            break
        sigma_new = sigma - diff / v
        if sigma_new <= 1e-5 or sigma_new > 10.0:
            break
        sigma = sigma_new

    # Bisection fallback
    lo, hi = 1e-4, 5.0
    p_lo = bs_price(S, K, T, r, q, lo, is_call) - price
    p_hi = bs_price(S, K, T, r, q, hi, is_call) - price
    if p_lo * p_hi > 0:
        return None
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(S, K, T, r, q, mid, is_call) - price
        if abs(p_mid) < tol:
            return mid
        if p_lo * p_mid < 0:
            hi, p_hi = mid, p_mid
        else:
            lo, p_lo = mid, p_mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Option expiration helpers
# ---------------------------------------------------------------------------


def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Date of the 3rd Friday of the given month."""
    first = pd.Timestamp(year=year, month=month, day=1)
    days_to_first_fri = (4 - first.weekday()) % 7  # 4 = Friday
    return first + pd.Timedelta(days=days_to_first_fri + 14)


def classify_expirations(expirations, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    For a list of expiration date strings, return a DataFrame with flags
    is_third_friday (the monthly expiration) and is_quarterly (Mar/Jun/Sep/Dec).

    The "monthly" expiration is normally the 3rd Friday. When that Friday is a
    market holiday (Juneteenth — first hits 2026-06-19 — or Good Friday), the
    contract rolls to the prior trading day (Thursday). We detect this
    dynamically: a Thursday counts as the monthly when the corresponding 3rd
    Friday is absent from Yahoo's expiration list. No hardcoded holiday
    calendar required.
    """
    if today is None:
        today = pd.Timestamp.today().normalize()
    exp_set = {str(e) for e in expirations}
    rows = []
    for e in expirations:
        d = pd.Timestamp(e)
        days = (d - today).days
        if days < 0:
            continue
        third_fri = _third_friday(d.year, d.month)
        # Case 1: this IS the 3rd Friday
        is_third_friday_exact = (
            d.weekday() == 4 and 15 <= d.day <= 21
        )
        # Case 2: this is the Thursday immediately before a 3rd Friday that
        # itself isn't a tradeable expiration (holiday roll)
        is_thursday_roll = (
            d.weekday() == 3
            and d == third_fri - pd.Timedelta(days=1)
            and third_fri.strftime("%Y-%m-%d") not in exp_set
        )
        is_monthly = is_third_friday_exact or is_thursday_roll
        is_quarterly = is_monthly and d.month in (3, 6, 9, 12)
        rows.append(
            dict(
                expiration=e,
                date=d,
                days_to_exp=days,
                is_third_friday=is_monthly,  # column kept for downstream compat; semantics: "is monthly expiry"
                is_quarterly=is_quarterly,
            )
        )
    if not rows:
        return pd.DataFrame(columns=["expiration", "date", "days_to_exp", "is_third_friday", "is_quarterly"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def select_curve_expirations(exp_info: pd.DataFrame) -> pd.DataFrame:
    """
    Pick: nearest 2 Friday expirations + the front 3 *months* (each
    represented by its monthly expiration when available, falling back to
    the listed expiration nearest the 3rd Friday of that month) + next 4
    quarterlies (Mar/Jun/Sep/Dec) that aren't already covered.

    The "front 3 months" guarantee is the important one — for tickers like
    APP where Yahoo may not list a clean 3rd-Friday monthly every month, we
    still surface the nearest expiration in each of the next 3 distinct
    months so the front of the term structure is always visible.

    Dedup priority when an expiration qualifies for more than one bucket:
    Monthly > Quarterly > Weekly. So June's 3rd Friday (also a quarterly)
    is labeled Monthly, while the more distant quarterlies stand alone.
    """
    if exp_info.empty:
        return exp_info.assign(category=pd.Series(dtype=str))

    # Bucket 1: nearest 2 Fridays of any kind
    fridays = exp_info[exp_info["date"].dt.weekday == 4].head(2)

    # Bucket 2: front 3 months. For each of the next 3 distinct year-months
    # in the chain, take the 3rd-Friday/Thursday-roll monthly when present,
    # else the listed expiration closest to that month's 3rd Friday.
    monthly_rows: list[pd.Series] = []
    seen_months: set[tuple[int, int]] = set()
    for _, candidate_row in exp_info.iterrows():
        if len(monthly_rows) >= 3:
            break
        ym = (candidate_row["date"].year, candidate_row["date"].month)
        if ym in seen_months:
            continue
        seen_months.add(ym)
        in_month = exp_info[
            (exp_info["date"].dt.year == ym[0])
            & (exp_info["date"].dt.month == ym[1])
        ]
        monthly_in_month = in_month[in_month["is_third_friday"]]
        if not monthly_in_month.empty:
            monthly_rows.append(monthly_in_month.iloc[0])
        else:
            # No 3rd-Friday/Thursday-roll listed for this month — fall back
            # to whatever Yahoo *does* list, picking nearest to the 3rd Friday
            third_fri = _third_friday(ym[0], ym[1])
            nearest_idx = (in_month["date"] - third_fri).abs().idxmin()
            monthly_rows.append(in_month.loc[nearest_idx])
    monthlies = (
        pd.DataFrame(monthly_rows).reset_index(drop=True)
        if monthly_rows
        else exp_info.iloc[0:0]
    )
    monthly_dates = set(monthlies["expiration"]) if not monthlies.empty else set()

    # Bucket 3: next 4 quarterlies that aren't already covered by monthlies.
    # No date cap — we want Jun-of-next-year to show up.
    quarterlies = exp_info[
        exp_info["is_quarterly"] & ~exp_info["expiration"].isin(monthly_dates)
    ].head(4)

    seen: set[str] = set()
    rows: list[dict] = []
    for src, label in (
        (monthlies, "Monthly"),
        (quarterlies, "Quarterly"),
        (fridays, "Weekly"),
    ):
        for _, row in src.iterrows():
            exp = row["expiration"]
            if exp in seen:
                continue
            seen.add(exp)
            rec = row.to_dict()
            rec["category"] = label
            rows.append(rec)

    if not rows:
        return exp_info.iloc[0:0].assign(category=pd.Series(dtype=str))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def get_expiration_data(ticker: str, expiration: str, spot: float) -> dict | None:
    """
    Pull all option data for one expiration. Prices use bid/ask mid when
    two-sided quotes are available, otherwise fall back to lastPrice.

    Returns dict with:
      curve         : DataFrame (OTM puts + OTM calls, with strike, IV, etc.)
      atm_strike    : float — strike closest to spot
      atm_iv        : float — average of ATM call & put IV
      atm_call_iv   : float
      atm_put_iv    : float
      atm_straddle  : float — ATM call price + ATM put price (mid or last)
      atm_call_px   : float
      atm_put_px    : float
      n_mid         : strikes priced from bid/ask mid
      n_last        : strikes priced from last (fallback)
      n_dropped     : strikes with no usable price
      days_to_exp   : int
    """
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiration)
    except Exception:
        return None

    calls = chain.calls
    puts = chain.puts
    if (calls is None or calls.empty) and (puts is None or puts.empty):
        return None

    today = pd.Timestamp.today().normalize()
    days_to_exp = max((pd.Timestamp(expiration) - today).days, 1)
    T = days_to_exp / 365.0
    r = get_risk_free_rate()
    q = get_dividend_yield(ticker, spot)

    # --- ATM details (closest strike to spot) ---
    atm_strike = None
    atm_call_iv = None
    atm_put_iv = None
    atm_call_px = None
    atm_put_px = None

    if calls is not None and not calls.empty:
        atm_call_row = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
        atm_call_px, _src = _option_price(atm_call_row)
        atm_strike_call = float(atm_call_row["strike"])
        if atm_call_px is not None:
            atm_call_iv = implied_vol(atm_call_px, spot, atm_strike_call, T, r, q, True)
        atm_strike = atm_strike_call

    if puts is not None and not puts.empty:
        atm_put_row = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]].iloc[0]
        atm_put_px, _src = _option_price(atm_put_row)
        atm_strike_put = float(atm_put_row["strike"])
        if atm_put_px is not None:
            atm_put_iv = implied_vol(atm_put_px, spot, atm_strike_put, T, r, q, False)
        if atm_strike is None:
            atm_strike = atm_strike_put

    atm_iv_components = [
        iv for iv in (atm_call_iv, atm_put_iv) if iv is not None and 0 < iv < 5.0
    ]
    atm_iv = sum(atm_iv_components) / len(atm_iv_components) if atm_iv_components else None

    if atm_call_px is not None and atm_put_px is not None:
        atm_straddle = float(atm_call_px) + float(atm_put_px)
    else:
        atm_straddle = None

    # --- OTM curve for the IV plot ---
    cols = ["strike", "bid", "ask", "lastPrice", "openInterest", "volume"]
    parts = []
    if puts is not None and not puts.empty:
        p = puts[puts["strike"] < spot][[c for c in cols if c in puts.columns]].copy()
        p["side"] = "Put"
        p["is_call"] = False
        parts.append(p)
    if calls is not None and not calls.empty:
        c = calls[calls["strike"] >= spot][[c for c in cols if c in calls.columns]].copy()
        c["side"] = "Call"
        c["is_call"] = True
        parts.append(c)

    curve_df: pd.DataFrame | None = None
    n_mid = n_last = n_drop = 0

    if parts:
        out = pd.concat(parts, ignore_index=True)
        prices, sources, ivs = [], [], []
        for _, row in out.iterrows():
            px, src = _option_price(row)
            prices.append(px)
            sources.append(src)
            if src == "mid":
                n_mid += 1
            elif src == "last":
                n_last += 1
            else:
                n_drop += 1
            if px is None:
                ivs.append(np.nan)
                continue
            iv = implied_vol(px, spot, float(row["strike"]), T, r, q, bool(row["is_call"]))
            ivs.append(iv if iv is not None else np.nan)

        out["price"] = prices
        out["price_source"] = sources
        out["iv"] = ivs
        out = out[(out["iv"] > 0.01) & (out["iv"] < 5.0)]
        out = out.dropna(subset=["iv"]).sort_values("strike").reset_index(drop=True)
        if not out.empty:
            out["moneyness"] = out["strike"] / spot
            out["impliedVolatility"] = out["iv"]  # alias retained for downstream
            curve_df = out

    return {
        "curve": curve_df,
        "atm_strike": atm_strike,
        "atm_iv": atm_iv,
        "atm_call_iv": atm_call_iv,
        "atm_put_iv": atm_put_iv,
        "atm_call_px": atm_call_px,
        "atm_put_px": atm_put_px,
        "atm_straddle": atm_straddle,
        "n_mid": n_mid,
        "n_last": n_last,
        "n_dropped": n_drop,
        "days_to_exp": days_to_exp,
    }


# --- NYSE holiday list for trading-day counts -------------------------------
# Hardcoded because pandas_market_calendars isn't a required dep. Covers major
# NYSE closures for the current calendar and the following year — enough for
# any option expiration inside a ~1yr horizon. Update annually.
_NYSE_HOLIDAYS: list[str] = [
    # 2025
    "2025-01-01", "2025-01-09",  # New Year's, Carter Day of Mourning
    "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03",  # July 4 falls Saturday → observed Fri
    "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18",  # Juneteenth falls Saturday → observed Fri
    "2027-07-05",  # July 4 falls Sunday → observed Mon
    "2027-09-06", "2027-11-25", "2027-12-24",  # Christmas Sat → observed Fri
]


def _trading_days_between(start_date, end_date) -> int:
    """
    Count NYSE trading days between two dates (exclusive of `start_date`,
    inclusive of `end_date` — i.e. how many trading sessions until the option
    expires, counting expiration day itself).

    Uses np.busday_count with weekends excluded plus the hardcoded NYSE
    holiday list above.
    """
    try:
        holidays = np.array(_NYSE_HOLIDAYS, dtype="datetime64[D]")
        # busday_count end date is exclusive; add 1 day to include expiration.
        end_inclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        return int(np.busday_count(
            np.datetime64(pd.Timestamp(start_date).date()),
            np.datetime64(end_inclusive.date()),
            holidays=holidays,
        ))
    except Exception:
        # Fallback: raw weekday count without holidays.
        try:
            return int(np.busday_count(
                np.datetime64(pd.Timestamp(start_date).date()),
                np.datetime64((pd.Timestamp(end_date) + pd.Timedelta(days=1)).date()),
            ))
        except Exception:
            return 0


@st.cache_data(ttl=600, show_spinner=False)
def get_gamma_hedge_be(ticker: str, spot: float, target_days: int = 30) -> dict | None:
    """
    Estimated daily breakeven on a long-straddle gamma hedge.

    Formula: **straddle offer ÷ √(trading days to expiration)**

    Uses the monthly expiration (3rd-Friday cycle, or the Thursday roll on
    holiday-adjusted months) closest to `target_days` DTE, the OFFER-side
    straddle (ATM call ask + ATM put ask, with lastPrice fallback on either
    side if the ask is missing or zero), and trading days counted with the
    hardcoded NYSE holiday calendar.

    Reading: the underlying needs to move at least this many dollars each
    trading day for a long-gamma hedge to break even on daily theta bleed —
    the classic "daily move" number.
    """
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
    except Exception:
        return None
    if not expirations:
        return None

    exp_info = classify_expirations(expirations)
    # Only monthlies (3rd Friday / holiday-adjusted Thursday) with time left.
    monthlies = exp_info[(exp_info["is_third_friday"]) & (exp_info["days_to_exp"] > 0)]
    if monthlies.empty:
        return None

    idx = (monthlies["days_to_exp"] - target_days).abs().idxmin()
    row = monthlies.loc[idx]
    exp_str = str(row["expiration"])
    exp_date = pd.Timestamp(row["date"])

    try:
        chain = tk.option_chain(exp_str)
    except Exception:
        return None
    calls, puts = chain.calls, chain.puts
    if calls is None or calls.empty or puts is None or puts.empty:
        return None

    def _offer_px(opt_row) -> float | None:
        # Prefer live ask; fall back to lastPrice when ask is missing or zero
        # (common on illiquid strikes even inside the front month).
        for col in ("ask", "lastPrice"):
            v = opt_row.get(col)
            if pd.notna(v):
                try:
                    f = float(v)
                    if f > 0:
                        return f
                except Exception:
                    pass
        return None

    atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
    atm_put = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]].iloc[0]
    atm_strike = float(atm_call["strike"])
    call_offer = _offer_px(atm_call)
    put_offer = _offer_px(atm_put)
    if call_offer is None or put_offer is None:
        return None
    straddle_offer = call_offer + put_offer

    trading_days = _trading_days_between(pd.Timestamp.today().normalize(), exp_date)
    if trading_days <= 0:
        return None

    be_daily = straddle_offer / math.sqrt(trading_days)
    return {
        "be_daily": be_daily,
        "expiration": exp_str,
        "trading_days": trading_days,
        "straddle_offer": straddle_offer,
        "atm_strike": atm_strike,
        "call_offer": call_offer,
        "put_offer": put_offer,
    }


# ---------------------------------------------------------------------------
# Gamma Exposure (GEX)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def get_expiration_gex(ticker: str, expiration: str, spot: float) -> pd.DataFrame | None:
    """
    Per-strike Gamma Exposure for one expiration.

    Convention (matches optioncharts.io / SqueezeMetrics-style visualization):
      Call GEX$ per strike  = +Γ_call × call_OI × 100 × S² × 0.01
      Put GEX$ per strike   = −Γ_put  × put_OI  × 100 × S² × 0.01
      Net GEX$ per strike   = Call GEX + Put GEX

    Units: dollars of dealer gamma per 1% move in spot. Positive = stabilizing
    (dealers presumed long gamma here), negative = destabilizing.

    Returns a DataFrame with columns:
      strike, call_oi, put_oi, call_iv, put_iv, call_gex, put_gex, net_gex
    or None if the chain is unusable.
    """
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiration)
    except Exception:
        return None

    calls = chain.calls if chain.calls is not None else pd.DataFrame()
    puts = chain.puts if chain.puts is not None else pd.DataFrame()
    if calls.empty and puts.empty:
        return None

    today = pd.Timestamp.today().normalize()
    days_to_exp = max((pd.Timestamp(expiration) - today).days, 1)
    T = days_to_exp / 365.0
    r = get_risk_free_rate()
    q = get_dividend_yield(ticker, spot)
    contract_mult = 100.0  # standard equity option

    def _per_strike(df: pd.DataFrame, is_call: bool) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["strike", "oi", "iv", "gex"])
        keep = [c for c in ("strike", "bid", "ask", "lastPrice", "openInterest") if c in df.columns]
        d = df[keep].copy()
        d["openInterest"] = pd.to_numeric(d.get("openInterest"), errors="coerce").fillna(0.0)
        ivs, gexs = [], []
        for _, row in d.iterrows():
            K = float(row["strike"])
            oi = float(row["openInterest"])
            if oi <= 0 or K <= 0:
                ivs.append(np.nan)
                gexs.append(0.0)
                continue
            px, _src = _option_price(row)
            iv = implied_vol(px, spot, K, T, r, q, is_call) if px is not None else None
            if iv is None or iv <= 0.01 or iv >= 5.0:
                ivs.append(np.nan)
                gexs.append(0.0)
                continue
            gamma = bs_gamma(spot, K, T, r, q, iv)
            # dollar gamma per 1% move in spot
            dollar_gex = gamma * oi * contract_mult * spot * spot * 0.01
            if not is_call:
                dollar_gex = -dollar_gex
            ivs.append(iv)
            gexs.append(dollar_gex)
        d["iv"] = ivs
        d["gex"] = gexs
        return d[["strike", "openInterest", "iv", "gex"]].rename(
            columns={"openInterest": "oi"}
        )

    call_df = _per_strike(calls, True)
    put_df = _per_strike(puts, False)

    merged = pd.merge(
        call_df.rename(columns={"oi": "call_oi", "iv": "call_iv", "gex": "call_gex"}),
        put_df.rename(columns={"oi": "put_oi", "iv": "put_iv", "gex": "put_gex"}),
        on="strike",
        how="outer",
    ).fillna({"call_oi": 0, "put_oi": 0, "call_gex": 0.0, "put_gex": 0.0})
    merged = merged.sort_values("strike").reset_index(drop=True)
    merged["net_gex"] = merged["call_gex"] + merged["put_gex"]
    return merged


# ---------------------------------------------------------------------------
# Fundamentals (market cap, P/E, growth, ROE, ROIC)
# ---------------------------------------------------------------------------


def _safe(d, key, default=None):
    """Pull a value from a yfinance row/dict, returning default on missing/NaN."""
    if d is None:
        return default
    try:
        v = d.get(key) if hasattr(d, "get") else d[key]
    except Exception:
        return default
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return v


def _first_match(index_obj, candidates):
    """Return first candidate name present in a pandas Index, else None."""
    for c in candidates:
        if c in index_obj:
            return c
    return None


def _compute_roic(tk) -> float | None:
    """
    ROIC ≈ NOPAT / (Total Debt + Total Equity)
    NOPAT = EBIT × (1 − effective tax rate)
    Uses most recent annual income statement + balance sheet.
    """
    try:
        inc = tk.income_stmt
        bal = tk.balance_sheet
        if inc is None or inc.empty or bal is None or bal.empty:
            return None
        latest_inc = inc.iloc[:, 0]
        latest_bal = bal.iloc[:, 0]

        ebit_key = _first_match(inc.index, ["EBIT", "Operating Income"])
        if ebit_key is None:
            return None
        ebit = _safe(latest_inc, ebit_key)
        if ebit is None:
            return None

        tax_key = _first_match(inc.index, ["Tax Provision", "Income Tax Expense"])
        pretax_key = _first_match(inc.index, ["Pretax Income", "Income Before Tax"])
        tax = _safe(latest_inc, tax_key) if tax_key else None
        pretax = _safe(latest_inc, pretax_key) if pretax_key else None
        if tax is not None and pretax is not None and float(pretax) > 0:
            tax_rate = max(min(float(tax) / float(pretax), 0.5), 0.0)
        else:
            tax_rate = 0.21  # reasonable US default

        nopat = float(ebit) * (1.0 - tax_rate)

        debt_key = _first_match(
            bal.index, ["Total Debt", "Long Term Debt And Capital Lease Obligation"]
        )
        eq_key = _first_match(
            bal.index,
            [
                "Stockholders Equity",
                "Total Equity Gross Minority Interest",
                "Common Stock Equity",
            ],
        )
        if eq_key is None:
            return None
        total_debt = float(_safe(latest_bal, debt_key, 0.0) or 0.0) if debt_key else 0.0
        equity = _safe(latest_bal, eq_key)
        if equity is None:
            return None
        invested_capital = total_debt + float(equity)
        if invested_capital <= 0:
            return None
        return nopat / invested_capital
    except Exception:
        return None


def _compute_growth_yoy(inc_df, candidates):
    """latest / prior - 1 from the income statement, given candidate row names."""
    if inc_df is None or inc_df.empty or inc_df.shape[1] < 2:
        return None
    key = _first_match(inc_df.index, candidates)
    if key is None:
        return None
    latest = inc_df.iloc[:, 0].get(key)
    prior = inc_df.iloc[:, 1].get(key)
    try:
        if latest is None or prior is None or pd.isna(latest) or pd.isna(prior):
            return None
        prior = float(prior)
        if prior == 0 or prior < 0:
            # Skip when prior is non-positive (growth rate undefined / misleading)
            return None
        return float(latest) / prior - 1.0
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_income_history(ticker: str, max_years: int = 5) -> pd.DataFrame | None:
    """
    Annual Revenue / Gross Profit / Operating Income / Net Income for up to
    the last `max_years` fiscal years, oldest-first.

    Robustness notes:
      * yfinance does not guarantee `income_stmt` columns are date-sorted, so
        we sort period-ends descending before slicing — otherwise the wrong
        year can be dropped.
      * The oldest column Yahoo returns is often sparse. Each metric has
        fallbacks (e.g. Gross Profit = Revenue − Cost of Revenue) so a year
        isn't left blank just because one exact row label is missing.
      * `tk.financials` is merged in as a secondary source — it sometimes
        carries an older year that `income_stmt` omits.

    Returns a DataFrame indexed by fiscal-year label with columns:
      Revenue, Gross Profit, Operating Income, Net Income
    Values are in raw dollars. Returns None if no income statement is available.
    """
    try:
        tk = yf.Ticker(ticker)
        inc = tk.income_stmt
    except Exception:
        inc = None

    # Secondary source — occasionally has a year income_stmt drops.
    try:
        fin = tk.financials
    except Exception:
        fin = None

    frames = [f for f in (inc, fin) if f is not None and not f.empty]
    if not frames:
        return None

    # Merge sources column-wise, keeping the first non-null per (row, period).
    merged = frames[0].copy()
    for extra in frames[1:]:
        for col in extra.columns:
            if col not in merged.columns:
                merged[col] = extra[col]
            else:
                merged[col] = merged[col].combine_first(extra[col])
        for idx in extra.index:
            if idx not in merged.index:
                merged.loc[idx] = extra.loc[idx].reindex(merged.columns)

    def _row(candidates):
        key = _first_match(merged.index, candidates)
        return merged.loc[key] if key is not None else None

    rev_row = _row(["Total Revenue", "Operating Revenue"])
    gp_row = _row(["Gross Profit"])
    cogs_row = _row(["Cost Of Revenue", "Cost Of Goods Sold", "Reconciled Cost Of Revenue"])
    oi_row = _row(["Operating Income", "EBIT", "Operating Income Or Loss"])
    opex_row = _row(["Operating Expense", "Total Operating Expenses"])
    ni_row = _row([
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Continuous Operations",
        "Net Income Including Noncontrolling Interests",
    ])

    def _val(row, col):
        if row is None:
            return np.nan
        try:
            v = row.get(col)
            return float(v) if v is not None and not pd.isna(v) else np.nan
        except Exception:
            return np.nan

    # Sort period-ends newest-first, then take the most recent max_years.
    try:
        cols_sorted = sorted(merged.columns, key=lambda c: pd.Timestamp(c), reverse=True)
    except Exception:
        cols_sorted = list(merged.columns)

    records: list[tuple[str, dict]] = []
    for c in cols_sorted:
        revenue = _val(rev_row, c)
        gross = _val(gp_row, c)
        if np.isnan(gross):  # fallback: Revenue − COGS
            cogs = _val(cogs_row, c)
            if not np.isnan(revenue) and not np.isnan(cogs):
                gross = revenue - cogs
        operating = _val(oi_row, c)
        if np.isnan(operating):  # fallback: Gross Profit − Operating Expense
            opex = _val(opex_row, c)
            if not np.isnan(gross) and not np.isnan(opex):
                operating = gross - opex
        net = _val(ni_row, c)
        row_data = {
            "Revenue": revenue,
            "Gross Profit": gross,
            "Operating Income": operating,
            "Net Income": net,
        }
        # Skip columns with no usable data at all (TTM placeholders, junk cols).
        if all(pd.isna(v) for v in row_data.values()):
            continue
        records.append((pd.Timestamp(c).strftime("%Y"), row_data))
        if len(records) >= max_years:
            break

    if not records:
        return None

    # Oldest-first for left-to-right chronological bars.
    records.reverse()
    df = pd.DataFrame(
        [r[1] for r in records],
        index=[r[0] for r in records],
    )
    return df


def _fmp_api_key() -> str | None:
    """Read the Financial Modeling Prep API key from Streamlit secrets first,
    then FMP_API_KEY env var. Returns None if neither is set."""
    try:
        v = st.secrets.get("FMP_API_KEY")
        if v:
            return str(v)
    except Exception:
        pass
    import os
    return os.environ.get("FMP_API_KEY")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fmp_analyst_estimates(ticker: str) -> list[dict] | None:
    """
    Financial Modeling Prep — annual analyst-consensus estimates.

    Endpoint returns a list of dicts with fiscal-year period-end dates and
    consensus Revenue / EBIT / Net Income / EPS estimates. Much better
    coverage than Yahoo's `revenue_estimate`/`earnings_estimate`, and gives
    us EBIT (real operating income) rather than a margin-derived proxy.

    Response shape (one item per year, newest last):
      {
        "date": "2027-12-31",
        "symbol": "MU",
        "estimatedRevenueLow/High/Avg": <float>,
        "estimatedEbitLow/High/Avg": <float>,
        "estimatedEbitdaLow/High/Avg": <float>,
        "estimatedNetIncomeLow/High/Avg": <float>,
        "estimatedSgaExpenseLow/High/Avg": <float>,
        "estimatedEpsLow/High/Avg": <float>,
        "numberAnalystEstimatedRevenue": <int>,
        "numberAnalystsEstimatedEps": <int>,
      }
    """
    key = _fmp_api_key()
    if not key:
        return None
    import urllib.request
    import json
    # Try modern /stable/ endpoint first, fall back to /api/v3/ legacy path.
    urls = [
        f"https://financialmodelingprep.com/stable/analyst-estimates"
        f"?symbol={ticker}&period=annual&page=0&limit=10&apikey={key}",
        f"https://financialmodelingprep.com/api/v3/analyst-estimates/"
        f"{ticker}?apikey={key}",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read())
            if isinstance(data, list) and data:
                return data
        except Exception:
            continue
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_income_ttm_and_projection(ticker: str) -> dict:
    """
    TTM income figures (sum of last 4 quarters) plus CURRENT-fiscal-year (0y)
    and NEXT-fiscal-year (+1y) analyst-consensus projections.

    Returns:
      {
        "ttm": {Revenue, Gross Profit, Operating Income, Net Income} or None,
        "ttm_end": "YYYY-MM-DD" or None,           # most recent quarter end
        "proj_cy": {...same metrics...} or None,   # current fiscal year est
        "proj_cy_label": "YYYY (est)" or None,
        "proj_ny": {...same metrics...} or None,   # next fiscal year est
        "proj_ny_label": "YYYY (est)" or None,
      }

    Projection sources per period:
      Revenue     : revenue_estimate['avg']
      Net Income  : earnings_estimate['avg'] (EPS) × sharesOutstanding
      Gross / Op  : derived from TTM margins applied to projected revenue
                    (analyst consensus doesn't publish separate GP/OI lines)
    """
    result: dict = {
        "ttm": None, "ttm_end": None,
        "proj_cy": None, "proj_cy_label": None,
        "proj_ny": None, "proj_ny_label": None,
    }

    try:
        tk = yf.Ticker(ticker)
    except Exception:
        return result

    # --- TTM: sum last 4 quarters ---------------------------------------
    qinc = None
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        try:
            df_ = getattr(tk, attr, None)
            if df_ is not None and not df_.empty:
                qinc = df_
                break
        except Exception:
            pass

    def _row(df, candidates):
        if df is None or df.empty:
            return None
        key = _first_match(df.index, candidates)
        return df.loc[key] if key is not None else None

    def _sum4(row):
        if row is None:
            return None
        vals = row.dropna().iloc[:4]
        return float(vals.sum()) if len(vals) > 0 else None

    if qinc is not None and not qinc.empty:
        # Ensure quarter columns are newest-first for iloc[:4].
        try:
            qinc = qinc[sorted(qinc.columns, key=lambda c: pd.Timestamp(c), reverse=True)]
        except Exception:
            pass

        rev = _sum4(_row(qinc, ["Total Revenue", "Operating Revenue"]))
        gp = _sum4(_row(qinc, ["Gross Profit"]))
        if gp is None:
            cogs = _sum4(_row(qinc, ["Cost Of Revenue", "Cost Of Goods Sold",
                                     "Reconciled Cost Of Revenue"]))
            if rev is not None and cogs is not None:
                gp = rev - cogs
        oi = _sum4(_row(qinc, ["Operating Income", "EBIT",
                               "Operating Income Or Loss"]))
        if oi is None:
            opex = _sum4(_row(qinc, ["Operating Expense", "Total Operating Expenses"]))
            if gp is not None and opex is not None:
                oi = gp - opex
        ni = _sum4(_row(qinc, [
            "Net Income", "Net Income Common Stockholders",
            "Net Income Continuous Operations",
            "Net Income Including Noncontrolling Interests",
        ]))

        if any(v is not None for v in (rev, gp, oi, ni)):
            result["ttm"] = {
                "Revenue": rev,
                "Gross Profit": gp,
                "Operating Income": oi,
                "Net Income": ni,
            }
            try:
                result["ttm_end"] = pd.Timestamp(qinc.columns[0]).strftime("%Y-%m-%d")
            except Exception:
                pass

    # --- Projections: current fiscal year (0y) and next fiscal year (+1y)
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    shares = _safe(info, "sharesOutstanding") or _safe(info, "impliedSharesOutstanding")

    def _extract_period(df, period_key: str) -> float | None:
        """Pull the `avg` cell for a given yfinance period label from an
        estimate DataFrame, handling both index-based and column-based layouts."""
        if df is None or not hasattr(df, "empty") or df.empty:
            return None
        target_row = None
        try:
            if period_key in df.index:
                target_row = df.loc[period_key]
            elif "period" in getattr(df, "columns", []):
                match = df[df["period"] == period_key]
                if not match.empty:
                    target_row = match.iloc[0]
        except Exception:
            return None
        if target_row is None:
            return None
        for col in ("avg", "average", "estimate"):
            v = target_row.get(col) if hasattr(target_row, "get") else None
            if v is not None:
                try:
                    if not pd.isna(v):
                        return float(v)
                except Exception:
                    pass
        return None

    # Load estimate DataFrames once each.
    re_df = None
    for attr in ("revenue_estimate", "get_revenue_estimate"):
        try:
            obj = getattr(tk, attr, None)
            candidate = obj() if callable(obj) else obj
            if candidate is not None and hasattr(candidate, "empty") and not candidate.empty:
                re_df = candidate
                break
        except Exception:
            pass
    ee_df = None
    for attr in ("earnings_estimate", "get_earnings_estimate"):
        try:
            obj = getattr(tk, attr, None)
            candidate = obj() if callable(obj) else obj
            if candidate is not None and hasattr(candidate, "empty") and not candidate.empty:
                ee_df = candidate
                break
        except Exception:
            pass

    # Fiscal-year labels: bump the most-recent Yahoo actual by +1 / +2.
    latest_fy = None
    try:
        inc_annual = tk.income_stmt
        if inc_annual is not None and not inc_annual.empty:
            latest_fy = pd.Timestamp(inc_annual.columns[0]).year
    except Exception:
        pass
    if latest_fy is None:
        latest_fy = pd.Timestamp.today().year - 1

    # Growth-rate extrapolation fallbacks (used only when analyst-consensus
    # DataFrames don't return data for this ticker).
    rev_growth_fallback = _safe(info, "revenueGrowth")   # quarterly YoY
    earn_growth_fallback = _safe(info, "earningsGrowth")  # quarterly YoY
    # Latest annual actuals — for anchoring the extrapolation.
    latest_annual_rev = None
    latest_annual_ni = None
    try:
        # Re-fetch to be resilient — the earlier `inc_annual` var may not have
        # loaded (silent try/except above).
        _inc_annual = tk.income_stmt
        if _inc_annual is not None and not _inc_annual.empty:
            first_col = _inc_annual.columns[0]
            rev_key = _first_match(_inc_annual.index,
                                   ["Total Revenue", "Operating Revenue"])
            ni_key = _first_match(_inc_annual.index,
                                  ["Net Income", "Net Income Common Stockholders"])
            if rev_key is not None:
                v = _inc_annual.loc[rev_key, first_col]
                if pd.notna(v):
                    latest_annual_rev = float(v)
            if ni_key is not None:
                v = _inc_annual.loc[ni_key, first_col]
                if pd.notna(v):
                    latest_annual_ni = float(v)
    except Exception:
        pass

    # --- FMP analyst estimates (preferred source when available) --------
    fmp_by_year: dict[int, dict] = {}
    fmp_list = fetch_fmp_analyst_estimates(ticker)
    if fmp_list:
        for item in fmp_list:
            try:
                yr = pd.Timestamp(item.get("date")).year
                fmp_by_year[yr] = item
            except Exception:
                pass

    # Track which source produced each projection so we can label it honestly.
    source_flags: list[str] = []

    def _fnum(v):
        """Coerce an FMP field to float or None."""
        if v is None:
            return None
        try:
            f = float(v)
            if pd.isna(f):
                return None
            return f
        except Exception:
            return None

    def _build_projection(period_key: str, year_label: int, exponent: int) -> tuple[dict | None, str | None]:
        """
        exponent: 1 for current FY (single year of growth from latest annual),
                  2 for next FY (compounded twice).

        Sources tried, in order:
          1. FMP `analyst-estimates` — matched by fiscal-year END year.
             Provides Revenue, EBIT (real Operating Income), and Net Income.
          2. Yahoo `revenue_estimate` / `earnings_estimate` at period_key.
          3. `forwardEps × shares` (next-FY NI only).
          4. Growth-rate extrapolation from latest annual actual.
        """
        proj_rev = None
        proj_oi = None
        proj_ni = None
        source = None

        # 1. FMP
        fmp_row = fmp_by_year.get(year_label)
        if fmp_row:
            proj_rev = _fnum(fmp_row.get("estimatedRevenueAvg"))
            proj_oi = _fnum(fmp_row.get("estimatedEbitAvg"))
            proj_ni = _fnum(fmp_row.get("estimatedNetIncomeAvg"))
            if proj_rev is not None or proj_ni is not None:
                source = "FMP"

        # 2. Yahoo estimates
        if proj_rev is None:
            v = _extract_period(re_df, period_key)
            if v is not None:
                proj_rev = v
                source = source or "Yahoo"
        if proj_ni is None:
            eps_avg = _extract_period(ee_df, period_key)
            if eps_avg is not None and shares:
                try:
                    proj_ni = eps_avg * float(shares)
                    source = source or "Yahoo"
                except Exception:
                    pass

        # 3. forwardEps × shares (next-FY only)
        if proj_ni is None and period_key == "+1y":
            fwd_eps = _safe(info, "forwardEps")
            if fwd_eps and shares:
                try:
                    proj_ni = float(fwd_eps) * float(shares)
                    source = source or "Yahoo (forwardEps)"
                except Exception:
                    pass

        # 4. Growth-rate extrapolation
        if proj_rev is None and latest_annual_rev is not None and rev_growth_fallback is not None:
            try:
                proj_rev = float(latest_annual_rev) * (1.0 + float(rev_growth_fallback)) ** exponent
                source = source or "growth-extrapolated"
            except Exception:
                pass
        if proj_ni is None and latest_annual_ni is not None and earn_growth_fallback is not None:
            try:
                proj_ni = float(latest_annual_ni) * (1.0 + float(earn_growth_fallback)) ** exponent
                source = source or "growth-extrapolated"
            except Exception:
                pass

        if proj_rev is None and proj_ni is None:
            return None, None

        source_flags.append(f"{year_label}={source or 'unknown'}")

        # GP: always TTM-margin-derived (no consensus GP available anywhere).
        # OI: prefer FMP EBIT (already assigned); otherwise TTM-margin-derived.
        proj_gp = None
        ttm = result["ttm"]
        if proj_rev is not None and ttm is not None and ttm.get("Revenue"):
            if ttm.get("Gross Profit") is not None:
                proj_gp = proj_rev * (ttm["Gross Profit"] / ttm["Revenue"])
            if proj_oi is None and ttm.get("Operating Income") is not None:
                proj_oi = proj_rev * (ttm["Operating Income"] / ttm["Revenue"])
        return (
            {
                "Revenue": proj_rev,
                "Gross Profit": proj_gp,
                "Operating Income": proj_oi,
                "Net Income": proj_ni,
            },
            f"{year_label} (est)",
        )

    cy_data, cy_label = _build_projection("0y", latest_fy + 1, exponent=1)
    ny_data, ny_label = _build_projection("+1y", latest_fy + 2, exponent=2)
    result["proj_sources"] = source_flags
    result["proj_cy"] = cy_data
    result["proj_cy_label"] = cy_label
    result["proj_ny"] = ny_data
    result["proj_ny_label"] = ny_label

    return result


@st.cache_data(ttl=600, show_spinner=False)
def get_company_news(ticker: str, n: int = 3) -> list[dict]:
    """
    Most recent news items for `ticker`, filtered to remove things a trader
    wouldn't care about: listicle clickbait ("5 Stocks to Buy"), videos,
    sponsored junk, and headlines that don't mention the company.

    Handles both the newer Yahoo response shape ({"content": {...}}) and the
    older flat dict shape. Returns up to `n` items sorted newest-first.
    """
    try:
        tk = yf.Ticker(ticker)
        news_raw = tk.news or []
    except Exception:
        return []

    # Phrases that mark a headline as low-signal listicle/promotional content.
    JUNK_PATTERNS = [
        "stocks to buy", "stocks to watch", "stocks to own", "stocks to consider",
        "stocks to sell", "best stocks", "top stocks", "hottest stocks",
        "stocks under $", "stocks to load up", "stocks for the next",
        "should you buy", "should you sell", "is now the time to buy",
        "magnificent 7", "magnificent seven", "fab 5",
        "3 stocks", "5 stocks", "7 stocks", "10 stocks",
        "top 3 ", "top 5 ", "top 7 ", "top 10 ",
        "best ai stocks", "best tech stocks", "best dividend",
        "stocks that could", "stocks set to", "stocks poised",
        "ai stocks", "growth stocks", "value stocks",  # only when generic
        "wall street's favorite", "wall street's top",
    ]

    items: list[dict] = []
    for raw in news_raw:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content") if isinstance(raw.get("content"), dict) else raw

        title = (content.get("title") or "").strip()
        if not title:
            continue

        # Publisher — provider.displayName (new) or publisher (old)
        publisher = None
        prov = content.get("provider")
        if isinstance(prov, dict):
            publisher = prov.get("displayName")
        publisher = publisher or content.get("publisher") or "—"

        # Timestamp — try several shapes
        ts_obj = None
        for k in ("pubDate", "displayTime"):
            v = content.get(k)
            if v:
                try:
                    ts_obj = pd.Timestamp(v)
                    break
                except Exception:
                    pass
        if ts_obj is None:
            v = content.get("providerPublishTime")
            if isinstance(v, (int, float)) and v > 0:
                try:
                    ts_obj = pd.Timestamp.fromtimestamp(float(v), tz="UTC")
                except Exception:
                    pass
        # Normalize to tz-aware UTC for comparison
        if ts_obj is not None and ts_obj.tzinfo is None:
            ts_obj = ts_obj.tz_localize("UTC")

        # URL — canonicalUrl.url (new), clickThroughUrl.url, or link (old)
        link = None
        for k in ("canonicalUrl", "clickThroughUrl"):
            v = content.get(k)
            if isinstance(v, dict) and v.get("url"):
                link = v["url"]
                break
        link = link or content.get("link")

        ntype = str(content.get("contentType") or content.get("type") or "").upper()

        items.append({
            "title": title,
            "publisher": publisher,
            "timestamp": ts_obj,
            "link": link,
            "type": ntype,
        })

    # --- Filter ---------------------------------------------------------
    def _relevant(it: dict) -> bool:
        if it["type"] in ("VIDEO",):
            return False
        title_lc = it["title"].lower()
        # Drop listicle/clickbait
        for pat in JUNK_PATTERNS:
            if pat in title_lc:
                return False
        # The headline should at least mention the company by ticker or name.
        if ticker.lower() not in title_lc:
            # accept anyway if Yahoo flagged it as the primary subject — but
            # since the JUNK filter already cleaned listicles, we only require
            # a hard ticker mention when the headline reads like an aggregator
            # post. Most company-specific stories don't include the ticker in
            # the title (e.g. "Apple unveils ..."), so don't be aggressive here.
            pass
        return True

    items = [it for it in items if _relevant(it)]

    # Newest first, items without a timestamp pushed to the bottom.
    items.sort(
        key=lambda it: (it["timestamp"] is None,
                        -(it["timestamp"].value if it["timestamp"] is not None else 0)),
    )
    return items[:n]


@st.cache_data(ttl=3600, show_spinner=False)
def get_quarterly_balance_history(ticker: str, n: int = 4) -> pd.DataFrame | None:
    """
    Last `n` quarters of cash-related balance-sheet lines, newest-first across
    columns.

    Returns a DataFrame indexed by metric (Cash & Equivalents, Short-Term
    Investments, Cash + ST Investments, Total Debt, Net Cash) with one column
    per quarter-end. Values are raw dollars. Returns None if no quarterly
    balance sheet is available.
    """
    try:
        tk = yf.Ticker(ticker)
        qbs = getattr(tk, "quarterly_balance_sheet", None)
    except Exception:
        qbs = None
    if qbs is None or qbs.empty:
        return None

    def _row(candidates):
        key = _first_match(qbs.index, candidates)
        return qbs.loc[key] if key is not None else None

    cash_row = _row([
        "Cash And Cash Equivalents",
        "Cash",
        "Cash Financial",
    ])
    sti_row = _row([
        "Short Term Investments",
        "Other Short Term Investments",
    ])
    cash_sti_row = _row([
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Short Term Investments",
    ])
    total_debt_row = _row([
        "Total Debt",
        "Net Debt",  # not ideal but better than nothing
    ])
    long_debt_row = _row(["Long Term Debt"])
    short_debt_row = _row(["Current Debt", "Short Term Debt", "Short Long Term Debt"])

    def _val(row, col):
        if row is None:
            return np.nan
        try:
            v = row.get(col)
            return float(v) if v is not None and not pd.isna(v) else np.nan
        except Exception:
            return np.nan

    try:
        cols_sorted = sorted(qbs.columns, key=lambda c: pd.Timestamp(c), reverse=True)
    except Exception:
        cols_sorted = list(qbs.columns)
    cols_sorted = cols_sorted[:n]
    if not cols_sorted:
        return None

    data = {}
    for c in cols_sorted:
        cash = _val(cash_row, c)
        sti = _val(sti_row, c)
        cash_plus = _val(cash_sti_row, c)
        if np.isnan(cash_plus):
            cash_plus = (0 if np.isnan(cash) else cash) + (0 if np.isnan(sti) else sti)
            if np.isnan(cash) and np.isnan(sti):
                cash_plus = np.nan
        total_debt = _val(total_debt_row, c)
        if np.isnan(total_debt):
            ld = _val(long_debt_row, c)
            sd = _val(short_debt_row, c)
            if not (np.isnan(ld) and np.isnan(sd)):
                total_debt = (0 if np.isnan(ld) else ld) + (0 if np.isnan(sd) else sd)
        net_cash = np.nan
        if not np.isnan(cash_plus) and not np.isnan(total_debt):
            net_cash = cash_plus - total_debt
        data[pd.Timestamp(c).strftime("%Y-%m-%d")] = {
            "Cash & Equivalents": cash,
            "Short-Term Investments": sti,
            "Cash + ST Investments": cash_plus,
            "Total Debt": total_debt,
            "Net Cash": net_cash,
        }

    df = pd.DataFrame(data)  # columns = quarter dates (newest-first), rows = metrics
    df = df.reindex(
        ["Cash & Equivalents", "Short-Term Investments", "Cash + ST Investments",
         "Total Debt", "Net Cash"]
    )
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_quarterly_cash_burn(ticker: str, n: int = 4) -> pd.DataFrame | None:
    """
    Last `n` quarters of cash-flow lines, newest-first across columns.

    Returns a DataFrame indexed by metric (Operating Cash Flow, Capex,
    Free Cash Flow, Burn Rate) with one column per quarter-end. "Burn Rate"
    is reported as positive dollars when FCF is negative, and 0 otherwise —
    so it reads as "cash going out the door per quarter".
    """
    try:
        tk = yf.Ticker(ticker)
        qcf = getattr(tk, "quarterly_cashflow", None)
    except Exception:
        qcf = None
    if qcf is None or qcf.empty:
        return None

    def _row(candidates):
        key = _first_match(qcf.index, candidates)
        return qcf.loc[key] if key is not None else None

    ocf_row = _row([
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    ])
    capex_row = _row([
        "Capital Expenditure",
        "Capital Expenditures",
    ])
    fcf_row = _row(["Free Cash Flow", "FreeCashFlow"])

    def _val(row, col):
        if row is None:
            return np.nan
        try:
            v = row.get(col)
            return float(v) if v is not None and not pd.isna(v) else np.nan
        except Exception:
            return np.nan

    try:
        cols_sorted = sorted(qcf.columns, key=lambda c: pd.Timestamp(c), reverse=True)
    except Exception:
        cols_sorted = list(qcf.columns)
    cols_sorted = cols_sorted[:n]
    if not cols_sorted:
        return None

    data = {}
    for c in cols_sorted:
        ocf = _val(ocf_row, c)
        capex = _val(capex_row, c)
        fcf = _val(fcf_row, c)
        if np.isnan(fcf):
            if not (np.isnan(ocf) or np.isnan(capex)):
                fcf = ocf + capex  # capex is signed negative in Yahoo
        burn = np.nan
        if not np.isnan(fcf):
            burn = -fcf if fcf < 0 else 0.0
        data[pd.Timestamp(c).strftime("%Y-%m-%d")] = {
            "Operating Cash Flow": ocf,
            "Capex": capex,
            "Free Cash Flow": fcf,
            "Burn Rate (= −FCF when FCF<0)": burn,
        }

    df = pd.DataFrame(data)
    df = df.reindex(
        ["Operating Cash Flow", "Capex", "Free Cash Flow",
         "Burn Rate (= −FCF when FCF<0)"]
    )
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_calendar(ticker: str) -> dict:
    """
    Past and upcoming earnings dates.

    Returns {"past": [Timestamp, ...], "next": Timestamp | None} with all
    timestamps tz-naive and normalized to midnight. Past dates are sorted
    ascending; "next" is the soonest future earnings date.
    """
    out: dict = {"past": [], "next": None, "source": None, "approximate": False}
    try:
        tk = yf.Ticker(ticker)
    except Exception:
        return out

    today = pd.Timestamp.today().normalize()

    def _naive(ts) -> pd.Timestamp | None:
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is not None:
                t = t.tz_localize(None)
            return t.normalize()
        except Exception:
            return None

    # Gather earnings dates from every source yfinance exposes — versions
    # differ on which of these works, so we try them all and merge.
    dates: list[pd.Timestamp] = []
    sources_hit: list[str] = []

    def _harvest(df, name: str):
        if df is None:
            return
        try:
            if hasattr(df, "empty") and df.empty:
                return
        except Exception:
            return
        idx = getattr(df, "index", None)
        if idx is None:
            return
        hit = False
        for i in idx:
            t = _naive(i)
            if t is not None:
                dates.append(t)
                hit = True
        if hit:
            sources_hit.append(name)

    # 1. get_earnings_dates(limit=...) — method form
    try:
        _harvest(tk.get_earnings_dates(limit=24), "get_earnings_dates(24)")
    except Exception:
        pass
    # 2. earnings_dates — property form (different code path in some versions)
    try:
        _harvest(tk.earnings_dates, "earnings_dates")
    except Exception:
        pass
    # 3. get_earnings_dates() — no limit
    if not dates:
        try:
            _harvest(tk.get_earnings_dates(), "get_earnings_dates()")
        except Exception:
            pass
    # 4. earnings_history / get_earnings_history — past EPS actuals, often
    #    indexed by the report date even when get_earnings_dates is empty.
    for getter in ("earnings_history", "get_earnings_history"):
        try:
            obj = getattr(tk, getter, None)
            if obj is None:
                continue
            df_eh = obj() if callable(obj) else obj
            _harvest(df_eh, getter)
        except Exception:
            pass

    dates = sorted(set(dates))

    # 5. Last-resort fallback: quarterly income-statement period-end dates.
    #    These are fiscal QUARTER ENDS, not announcement dates — earnings are
    #    typically reported a few weeks later — but they reliably mark roughly
    #    when each quarter's results landed when nothing else is available.
    quarter_end_fallback = False
    if not [d for d in dates if d <= today]:
        for attr in ("quarterly_income_stmt", "quarterly_financials"):
            try:
                qdf = getattr(tk, attr, None)
                if qdf is not None and not qdf.empty:
                    for c in qdf.columns:
                        t = _naive(c)
                        if t is not None:
                            dates.append(t)
                    quarter_end_fallback = True
                    sources_hit.append(f"{attr} (quarter-end approx)")
                    break
            except Exception:
                pass
        dates = sorted(set(dates))

    out["approximate"] = quarter_end_fallback
    out["source"] = ", ".join(dict.fromkeys(sources_hit)) or None
    out["past"] = [d for d in dates if d <= today]
    future = [d for d in dates if d > today]
    if future:
        out["next"] = future[0]

    # Fallback for the next date: the calendar endpoint
    if out["next"] is None:
        try:
            cal = tk.calendar
            ed = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
            elif cal is not None and hasattr(cal, "index") and "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"].tolist()
            if ed is not None:
                if isinstance(ed, (list, tuple)):
                    ed = ed[0] if ed else None
                t = _naive(ed) if ed is not None else None
                if t is not None and t > today:
                    out["next"] = t
        except Exception:
            pass

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(ticker: str) -> dict:
    """
    Pull headline fundamentals: market cap, P/E (trailing & forward),
    YoY revenue & earnings growth (annual), ROE, ROIC.
    """
    out = {
        "long_name": None,
        "sector": None,
        "industry": None,
        "business_summary": None,
        "website": None,
        "market_cap": None,
        "trailing_pe": None,
        "forward_pe": None,
        "peg": None,
        "rev_growth_yoy": None,
        "earnings_growth_yoy": None,
        "roe": None,
        "roic": None,
        "short_pct_float": None,
        "beta": None,
        "fcf_ttm": None,
        "p_fcf": None,
        "fcf_source": None,
        "days_to_cover": None,
        "revenue_ttm": None,
        "fcf_margin": None,
        "rule_of_40": None,
    }
    try:
        tk = yf.Ticker(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        # Prefer longName ("Apple Inc.") and fall back to shortName ("Apple")
        out["long_name"] = _safe(info, "longName") or _safe(info, "shortName")
        # Company description fields — sector ("Technology") is the GICS-style
        # broad bucket; industry ("Software—Application") narrows it. The
        # longBusinessSummary is Yahoo's prose blurb covering business model.
        out["sector"] = _safe(info, "sector")
        out["industry"] = _safe(info, "industry")
        out["business_summary"] = _safe(info, "longBusinessSummary")
        out["website"] = _safe(info, "website")
        out["market_cap"] = _safe(info, "marketCap")
        out["trailing_pe"] = _safe(info, "trailingPE")
        out["forward_pe"] = _safe(info, "forwardPE")
        # PEG: Yahoo's pegRatio is the 5-year expected (forward) PEG, which is
        # the version most people quote. Fall back to trailingPegRatio when
        # the forward field is missing.
        out["peg"] = _safe(info, "pegRatio") or _safe(info, "trailingPegRatio")
        out["roe"] = _safe(info, "returnOnEquity")
        # Yahoo's `beta` is the 5-year monthly beta vs S&P 500.
        out["beta"] = _safe(info, "beta")
        # Yahoo returns shortPercentOfFloat as a decimal (e.g. 0.0432 = 4.32%).
        # Some less-followed names only carry the raw shortRatio (days-to-cover);
        # leave None in that case so we render N/A rather than mislabel.
        out["short_pct_float"] = _safe(info, "shortPercentOfFloat")
        if out["short_pct_float"] is None:
            shares_short = _safe(info, "sharesShort")
            float_shares = _safe(info, "floatShares")
            if shares_short and float_shares:
                try:
                    out["short_pct_float"] = float(shares_short) / float(float_shares)
                except Exception:
                    pass
        # Days to cover = sharesShort / average daily volume. Yahoo publishes
        # this directly as `shortRatio`; fall back to computing it from the
        # underlying components if the field is missing.
        out["days_to_cover"] = _safe(info, "shortRatio")
        if out["days_to_cover"] is None:
            shares_short = _safe(info, "sharesShort")
            adv = (
                _safe(info, "averageDailyVolume10Day")
                or _safe(info, "averageVolume10days")
                or _safe(info, "averageVolume")
            )
            if shares_short and adv:
                try:
                    out["days_to_cover"] = float(shares_short) / float(adv)
                except Exception:
                    pass

        # Annual income statement → YoY growth
        try:
            inc = tk.income_stmt
        except Exception:
            inc = None
        out["rev_growth_yoy"] = _compute_growth_yoy(
            inc, ["Total Revenue", "Operating Revenue"]
        )
        out["earnings_growth_yoy"] = _compute_growth_yoy(
            inc, ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operations"]
        )

        # Fall back to info-provided growth (typically quarterly YoY) if annual missing
        if out["rev_growth_yoy"] is None:
            out["rev_growth_yoy"] = _safe(info, "revenueGrowth")
        if out["earnings_growth_yoy"] is None:
            out["earnings_growth_yoy"] = _safe(info, "earningsGrowth")

        out["roic"] = _compute_roic(tk)

        # --- Price / Free Cash Flow (TTM) --------------------------------
        # Preferred source: sum the last 4 quarters of "Free Cash Flow" from
        # the quarterly cash-flow statement — this is the actual TTM. Yahoo's
        # `info["freeCashflow"]` is also TTM but is sometimes Levered FCF
        # (post-debt-service), so we treat it as a fallback.
        fcf_ttm = None
        fcf_src = None
        try:
            qcf = getattr(tk, "quarterly_cashflow", None)
            if qcf is not None and not qcf.empty:
                fcf_row = None
                for label in ("Free Cash Flow", "FreeCashFlow"):
                    if label in qcf.index:
                        fcf_row = qcf.loc[label]
                        break
                if fcf_row is None:
                    # Compute FCF = Operating Cash Flow − Capex
                    ocf_label = next(
                        (l for l in (
                            "Operating Cash Flow",
                            "Total Cash From Operating Activities",
                            "Cash Flow From Continuing Operating Activities",
                        ) if l in qcf.index),
                        None,
                    )
                    capex_label = next(
                        (l for l in (
                            "Capital Expenditure",
                            "Capital Expenditures",
                        ) if l in qcf.index),
                        None,
                    )
                    if ocf_label and capex_label:
                        fcf_row = qcf.loc[ocf_label] + qcf.loc[capex_label]  # capex is negative
                if fcf_row is not None:
                    last4 = fcf_row.dropna().iloc[:4]
                    if len(last4) >= 1:
                        fcf_ttm = float(last4.sum())
                        fcf_src = f"sum of last {len(last4)} quarters · quarterly_cashflow"
        except Exception:
            pass
        if fcf_ttm is None:
            v = _safe(info, "freeCashflow")
            if v is not None:
                try:
                    fcf_ttm = float(v)
                    fcf_src = "info.freeCashflow (Yahoo TTM)"
                except Exception:
                    pass
        out["fcf_ttm"] = fcf_ttm
        out["fcf_source"] = fcf_src
        if fcf_ttm and out["market_cap"]:
            try:
                if fcf_ttm > 0:
                    out["p_fcf"] = float(out["market_cap"]) / fcf_ttm
            except Exception:
                pass

        # --- Rule of 40 -------------------------------------------------
        # Revenue growth (%) + FCF margin (%). SaaS rule of thumb: ≥ 40
        # signals a healthy balance of growth and profitability. Prefer
        # Yahoo's TTM totalRevenue; fall back to summing the last 4
        # quarters of the quarterly income statement.
        rev_ttm = _safe(info, "totalRevenue")
        if rev_ttm is None:
            try:
                qinc = getattr(tk, "quarterly_income_stmt", None)
                if qinc is not None and not qinc.empty:
                    rev_label = next(
                        (l for l in ("Total Revenue", "Operating Revenue")
                         if l in qinc.index),
                        None,
                    )
                    if rev_label is not None:
                        last4 = qinc.loc[rev_label].dropna().iloc[:4]
                        if len(last4) >= 1:
                            rev_ttm = float(last4.sum())
            except Exception:
                pass
        try:
            out["revenue_ttm"] = float(rev_ttm) if rev_ttm is not None else None
        except Exception:
            out["revenue_ttm"] = None

        if out["revenue_ttm"] and fcf_ttm is not None and out["revenue_ttm"] > 0:
            try:
                out["fcf_margin"] = float(fcf_ttm) / float(out["revenue_ttm"])
            except Exception:
                pass
        if out["rev_growth_yoy"] is not None and out["fcf_margin"] is not None:
            try:
                out["rule_of_40"] = (
                    float(out["rev_growth_yoy"]) + float(out["fcf_margin"])
                )
            except Exception:
                pass
    except Exception:
        pass
    return out


def fmt_market_cap(v) -> str:
    if v is None:
        return "N/A"
    try:
        v = float(v)
    except Exception:
        return "N/A"
    if v >= 1e12:
        return f"${v / 1e12:,.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:,.2f}M"
    return f"${v:,.0f}"


def fmt_pct(v, decimals=1) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v) * 100:.{decimals}f}%"
    except Exception:
        return "N/A"


def fmt_multiple(v, decimals=1) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}x"
    except Exception:
        return "N/A"


def fmt_number(v, decimals=2) -> str:
    """Plain decimal (no suffix). For beta, ratios, etc."""
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return "N/A"


import re as _re


def _brief_summary(text: str | None, max_sentences: int = 2, max_chars: int = 320) -> str | None:
    """Return the first 1–2 sentences of a business summary, capped at ~320 chars."""
    if not text:
        return None
    sentences = _re.split(r"(?<=[.!?])\s+", text.strip())
    brief = " ".join(sentences[:max_sentences]).strip()
    if len(brief) > max_chars:
        brief = brief[: max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return brief or None


# ---------------------------------------------------------------------------
# IV history persistence (for D/W/M change metrics)
# ---------------------------------------------------------------------------


def append_iv_snapshot(ticker: str, target_days: int, iv: float) -> None:
    today = pd.Timestamp.today().normalize()
    new_row = pd.DataFrame(
        [{"date": today, "ticker": ticker, "target_days": target_days, "iv": float(iv)}]
    )
    if IV_CACHE_FILE.exists():
        try:
            existing = pd.read_csv(IV_CACHE_FILE, parse_dates=["date"])
            mask = (
                (existing["date"] == today)
                & (existing["ticker"] == ticker)
                & (existing["target_days"] == target_days)
            )
            existing = existing[~mask]
            out = pd.concat([existing, new_row], ignore_index=True)
        except Exception:
            out = new_row
    else:
        out = new_row
    try:
        out.sort_values(["ticker", "target_days", "date"]).to_csv(IV_CACHE_FILE, index=False)
    except Exception:
        pass


def iv_history(ticker: str, target_days: int) -> pd.DataFrame | None:
    """
    Return the full cached IV snapshot series for (ticker, target_days),
    sorted by date. Used to overlay historical IV on the rolling-vol chart.

    The cache only carries dates the app was actually loaded on, so the
    series may have gaps if you don't run the app every weekday. For a
    backfilled IV history you'd need a paid options vendor — yfinance only
    exposes the live chain.
    """
    if not IV_CACHE_FILE.exists():
        return None
    try:
        df = pd.read_csv(IV_CACHE_FILE, parse_dates=["date"])
    except Exception:
        return None
    df = df[(df["ticker"] == ticker) & (df["target_days"] == target_days)].sort_values("date")
    return df if not df.empty else None


def iv_changes(ticker: str, target_days: int) -> dict | None:
    """Return {'1D': {...}, '1W': {...}, '1M': {...}} based on cached history."""
    if not IV_CACHE_FILE.exists():
        return None
    try:
        df = pd.read_csv(IV_CACHE_FILE, parse_dates=["date"])
    except Exception:
        return None
    df = df[(df["ticker"] == ticker) & (df["target_days"] == target_days)].sort_values("date")
    if df.empty:
        return None
    today = pd.Timestamp.today().normalize()
    current_iv = float(df["iv"].iloc[-1])
    out = {}
    for label, days in (("1D", 1), ("1W", 7), ("1M", 30)):
        target = today - pd.Timedelta(days=days)
        prior = df[df["date"] <= target]
        if prior.empty:
            out[label] = None
            continue
        prev_iv = float(prior["iv"].iloc[-1])
        prev_date = prior["date"].iloc[-1]
        out[label] = {
            "current": current_iv,
            "prev": prev_iv,
            "abs_change_pp": (current_iv - prev_iv) * 100,  # vol points
            "pct_change": ((current_iv - prev_iv) / prev_iv * 100) if prev_iv else None,
            "days_back": int((today - prev_date).days),
            "prev_date": prev_date.strftime("%Y-%m-%d"),
        }
    return out


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def load_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, auto_adjust=False)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_risk_free_rate() -> float:
    """Return current 13-week T-bill yield as a decimal (e.g. 0.045). Fallback to 4.5%."""
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if not irx.empty:
            return float(irx.iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.045


@st.cache_data(ttl=3600, show_spinner=False)
def get_dividend_yield(ticker: str, spot: float) -> float:
    """Trailing 12-month dividend yield as a decimal."""
    try:
        divs = yf.Ticker(ticker).dividends
        if divs is None or divs.empty:
            return 0.0
        cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
        last_year = divs[divs.index >= cutoff]
        if last_year.empty:
            return 0.0
        return float(last_year.sum() / spot)
    except Exception:
        return 0.0


def _mid_price(row: pd.Series) -> float | None:
    """Return bid/ask mid; require both > 0 and ask >= bid."""
    bid = row.get("bid")
    ask = row.get("ask")
    if pd.isna(bid) or pd.isna(ask):
        return None
    bid = float(bid)
    ask = float(ask)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    return 0.5 * (bid + ask)


def _option_price(row: pd.Series) -> tuple[float | None, str | None]:
    """
    Best-available option price.
      1) bid/ask mid if both sides quoted
      2) lastPrice if positive
    Returns (price, source) where source is 'mid' or 'last'.
    """
    mid = _mid_price(row)
    if mid is not None:
        return mid, "mid"
    last = row.get("lastPrice")
    if pd.notna(last):
        try:
            last_f = float(last)
            if last_f > 0:
                return last_f, "last"
        except Exception:
            pass
    return None, None


@st.cache_data(ttl=600, show_spinner=False)
def load_atm_iv(ticker: str, spot: float, target_days: int = 30) -> dict | None:
    """
    Pull ATM call & put implied vol from the option expiry closest to
    ``target_days``. IV is computed from bid/ask mid via Black-Scholes
    inversion (not Yahoo's last-trade IV).
    """
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None

        today = pd.Timestamp.today().normalize()
        diffs = [
            (exp, abs((pd.Timestamp(exp) - today).days - target_days))
            for exp in expirations
        ]
        chosen = min(diffs, key=lambda x: x[1])[0]
        days_to_exp = (pd.Timestamp(chosen) - today).days
        T = max(days_to_exp, 1) / 365.0

        chain = tk.option_chain(chosen)
        calls = chain.calls
        puts = chain.puts
        if (calls is None or calls.empty) and (puts is None or puts.empty):
            return None

        r = get_risk_free_rate()
        q = get_dividend_yield(ticker, spot)

        ivs: list[float] = []
        atm_strike = None
        debug = {"r": r, "q": q, "T": T}

        for df_chain, is_call in ((calls, True), (puts, False)):
            if df_chain is None or df_chain.empty:
                continue
            row = df_chain.iloc[(df_chain["strike"] - spot).abs().argsort()[:1]].iloc[0]
            px, _src = _option_price(row)
            if px is None:
                continue
            K = float(row["strike"])
            iv = implied_vol(px, spot, K, T, r, q, is_call)
            if iv is None or iv <= 0:
                continue
            ivs.append(iv)
            if atm_strike is None:
                atm_strike = K

        if not ivs:
            return None

        return {
            "iv": float(np.mean(ivs)),
            "expiry": chosen,
            "days_to_expiry": int(days_to_exp),
            "strike": atm_strike,
            "n_legs": len(ivs),
            **debug,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Stock and Vol.", layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      [data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    chart_period = st.selectbox(
        "Price chart window",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
    )
    cone_lookback = st.selectbox(
        "Cone history",
        ["2y", "3y", "5y", "10y", "max"],
        index=2,
        help="How much history to use to build the percentile cone.",
    )
    iv_target_days = st.slider(
        "IV target (days to expiry)", min_value=7, max_value=90, value=30, step=1
    )
    hv_window = st.slider("HV window (days)", min_value=5, max_value=120, value=30, step=1)

if not ticker:
    st.title("Stock and Vol.")
    st.caption("Price · Volume · Fundamentals · Implied Vol · Realized Vol · Yang-Zhang Vol Cones")
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

# Pull fundamentals first so we can put the company name in the title.
# get_fundamentals is cached (ttl=3600) so calling it here is essentially free.
fundamentals = get_fundamentals(ticker)
company_name = fundamentals.get("long_name")
title_text = f"{ticker} — {company_name}" if company_name else ticker
st.title(title_text)
st.caption("Price · Volume · Fundamentals · Implied Vol · Realized Vol · Yang-Zhang Vol Cones")

# --- Load data --------------------------------------------------------------
with st.spinner(f"Loading {ticker}…"):
    hist_full = load_history(ticker, period=cone_lookback)

if hist_full is None or hist_full.empty:
    st.error(f"No price data found for `{ticker}`. Check the symbol.")
    st.stop()

last_price = float(hist_full["Close"].iloc[-1])
prev_close = float(hist_full["Close"].iloc[-2]) if len(hist_full) > 1 else last_price
chg = last_price - prev_close
pct = (chg / prev_close * 100) if prev_close else 0.0

# --- Fundamentals row -------------------------------------------------------

# Row 1: size / risk / valuation
r1c1, r1c2, r1c3, r1c4, r1c5, r1c6, r1c7 = st.columns(7)
r1c1.metric("Market Cap", fmt_market_cap(fundamentals.get("market_cap")))
r1c2.metric(
    "Beta",
    fmt_number(fundamentals.get("beta"), decimals=2),
    help=(
        "5-year monthly beta vs S&P 500, from Yahoo. Reading: 1.00 = moves "
        "in line with the market; >1 amplifies market moves; <1 dampens them; "
        "negative is rare and means inverse correlation."
    ),
)
r1c3.metric("P/E (TTM)", fmt_multiple(fundamentals.get("trailing_pe")))
r1c4.metric("Forward P/E", fmt_multiple(fundamentals.get("forward_pe")))
r1c5.metric(
    "PEG",
    fmt_number(fundamentals.get("peg"), decimals=2),
    help=(
        "PEG = P/E ÷ expected EPS growth. Uses Yahoo's pegRatio (5-yr "
        "expected, forward-looking) with fallback to trailingPegRatio. "
        "Rule of thumb: ~1 is fair, <1 looks cheap-for-growth, >2 looks "
        "rich. Sensitive to estimate quality — treat with healthy skepticism "
        "for names with thin analyst coverage."
    ),
)
r1c6.metric(
    "P/FCF (TTM)",
    fmt_multiple(fundamentals.get("p_fcf")),
    help=(
        "Market cap ÷ trailing-twelve-month free cash flow. FCF is summed "
        "from the last 4 quarters of Yahoo's cash-flow statement (Free Cash "
        "Flow row, or Operating Cash Flow + Capex if that row is missing); "
        "falls back to Yahoo's `info.freeCashflow` TTM field. Shows **N/A** "
        "when TTM FCF is negative. Source for the current value is shown in "
        "the diagnostic expander further down."
    ),
)
# Gamma Hedge Daily BE — straddle offer ÷ √(trading days to exp) on the
# ~30d monthly. Fetched here so the caller uses the freshly loaded spot.
_gh = get_gamma_hedge_be(ticker, last_price, target_days=30)
if _gh is not None:
    _gh_display = f"${_gh['be_daily']:.2f}"
    _gh_help = (
        f"Estimated daily breakeven on a long-straddle gamma hedge. "
        f"**Straddle offer ÷ √(trading days to expiration).**\n\n"
        f"Using **{_gh['expiration']}** monthly expiration · "
        f"ATM strike ${_gh['atm_strike']:.2f} · "
        f"call ask ${_gh['call_offer']:.2f} + put ask ${_gh['put_offer']:.2f} "
        f"= straddle offer **${_gh['straddle_offer']:.2f}** · "
        f"**{_gh['trading_days']} NYSE trading days** to expiration.\n\n"
        f"Reading: the underlying needs to move roughly this many dollars "
        f"per trading day for a long-gamma hedge to break even on daily "
        f"theta bleed."
    )
else:
    _gh_display = "N/A"
    _gh_help = (
        "Estimated daily breakeven on a long-straddle gamma hedge = "
        "straddle offer ÷ √(trading days to expiration). Currently N/A: no "
        "monthly (3rd-Friday) expiration with valid ATM ask quotes was "
        "found for this ticker."
    )
r1c7.metric("Gamma Hedge Daily BE", _gh_display, help=_gh_help)
if _gh is not None:
    # Subtitle: which expiration + trading-day count fed the calculation.
    r1c7.caption(
        f"<span style='font-size:0.7rem;color:#666;'>{_gh['expiration']} · "
        f"{_gh['trading_days']}td</span>",
        unsafe_allow_html=True,
    )

# Row 2: growth / returns / sentiment
r2c1, r2c2, r2c3, r2c4, r2c5, r2c6, r2c7 = st.columns(7)
r2c1.metric(
    "Rev Growth YoY",
    fmt_pct(fundamentals.get("rev_growth_yoy")),
    help="Most recent fiscal year vs prior fiscal year. Falls back to TTM/quarterly YoY if annual unavailable.",
)
r2c2.metric(
    "Earnings Growth YoY",
    fmt_pct(fundamentals.get("earnings_growth_yoy")),
    help="Net Income, most recent fiscal year vs prior. Returns N/A when prior-year earnings were negative or zero.",
)
# Rule of 40: Revenue Growth % + FCF Margin %. The classic SaaS yardstick —
# the score is in percentage POINTS, not a percent of anything, so it's
# rendered as a plain decimal (e.g. "47.3").
_rule_val = fundamentals.get("rule_of_40")
_rule_display = (
    f"{_rule_val * 100:.1f}" if _rule_val is not None else "N/A"
)
_rev_g = fundamentals.get("rev_growth_yoy")
_fcf_m = fundamentals.get("fcf_margin")
_rule_help = (
    "Rule of 40 = Revenue Growth YoY (%) + FCF Margin (%). "
    "Classic SaaS heuristic: ≥ 40 signals a healthy balance of growth "
    "and profitability. Components used for this number: "
    f"Rev growth = {fmt_pct(_rev_g)}, FCF margin = {fmt_pct(_fcf_m)}."
)
r2c3.metric("Rule of 40", _rule_display, help=_rule_help)
r2c4.metric("ROE", fmt_pct(fundamentals.get("roe")))
r2c5.metric(
    "ROIC",
    fmt_pct(fundamentals.get("roic")),
    help="NOPAT / (Total Debt + Total Equity), using EBIT × (1 − tax rate) from the latest annual statements.",
)
r2c6.metric(
    "Days to Cover",
    fmt_number(fundamentals.get("days_to_cover"), decimals=2),
    help=(
        "Short interest ÷ average daily trading volume — how many trading "
        "days it would take all shorts to buy back their positions at recent "
        "average volume. Higher = more vulnerable to a short squeeze. Uses "
        "Yahoo's `shortRatio`; falls back to sharesShort ÷ 10-day average "
        "volume when the direct field is missing."
    ),
)
r2c7.metric(
    "Short % Float",
    fmt_pct(fundamentals.get("short_pct_float"), decimals=2),
    help=(
        "Shares sold short as a percentage of float, from Yahoo's "
        "`shortPercentOfFloat`. Falls back to sharesShort / floatShares if "
        "the direct field is missing. Snapshot — typically updated twice "
        "monthly by exchanges with a settlement lag."
    ),
)

st.divider()

# --- Snapshot metrics -------------------------------------------------------
yz_now = yang_zhang_vol(hist_full, hv_window).dropna()
cc_now = close_to_close_vol(hist_full, hv_window).dropna()

yz_val = float(yz_now.iloc[-1]) if len(yz_now) else float("nan")
cc_val = float(cc_now.iloc[-1]) if len(cc_now) else float("nan")

iv_info = load_atm_iv(ticker, last_price, target_days=iv_target_days)

# Snapshot today's IV for the D/W/M change tracker
if iv_info:
    append_iv_snapshot(ticker, iv_target_days, iv_info["iv"])

# Compute change history once so both the sidebar and the snapshot row reuse it
changes = iv_changes(ticker, iv_target_days) if iv_info else None

# --- Sidebar: IV change tracker -------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader(f"IV change · ~{iv_target_days}d ATM")
    if not iv_info:
        st.caption("No options chain — change tracking unavailable.")
    elif changes is None:
        st.caption("Snapshot saved. Run again tomorrow to start tracking changes.")
    else:
        any_data = False
        for label in ("1D", "1W", "1M"):
            entry = changes.get(label)
            if entry is None:
                st.metric(label, "N/A", help="Not enough cached history yet")
                continue
            any_data = True
            pp = entry["abs_change_pp"]
            pct = entry["pct_change"]
            delta_str = f"{pp:+.2f}pp" + (f"  ({pct:+.1f}%)" if pct is not None else "")
            st.metric(
                label,
                f"{entry['current'] * 100:.2f}%",
                delta_str,
                help=(
                    f"vs {entry['prev'] * 100:.2f}% on {entry['prev_date']} "
                    f"({entry['days_back']}d ago)"
                ),
            )
        if not any_data:
            st.caption("Cache too short — keep running the app daily to build history.")
        st.caption(f"Cache: `{IV_CACHE_FILE.name}` · pp = vol points")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Last price", f"${last_price:,.2f}", f"{chg:+.2f}  ({pct:+.2f}%)")
c2.metric(f"{hv_window}d HV — Yang-Zhang", f"{yz_val * 100:.2f}%")
c3.metric(f"{hv_window}d HV — Close-to-Close", f"{cc_val * 100:.2f}%")

if iv_info:
    iv_pct = iv_info["iv"] * 100
    spread = (iv_info["iv"] - yz_val) * 100
    c4.metric(
        f"~{iv_target_days}d IV (ATM)",
        f"{iv_pct:.2f}%",
        f"{spread:+.2f}% vs YZ HV",
        help=(
            f"Expiry {iv_info['expiry']} ({iv_info['days_to_expiry']}d), "
            f"strike ${iv_info['strike']:,.2f}, {iv_info['n_legs']} leg avg"
        ),
    )
    vrp = iv_info["iv"] - yz_val
    c5.metric("Vol risk premium", f"{vrp * 100:+.2f}%", help="IV − realized YZ vol")

    # 24h IV change — relative % change in ~30d ATM IV vs the last cached snapshot
    one_day = changes.get("1D") if changes else None
    if one_day is not None and one_day.get("pct_change") is not None:
        pct_change = one_day["pct_change"]
        pp_change = one_day["abs_change_pp"]
        c6.metric(
            f"24h IV Δ (~{iv_target_days}d)",
            f"{pct_change:+.2f}%",
            f"{pp_change:+.2f}pp",
            delta_color="inverse",  # vol up = red (typically risk-off), vol down = green
            help=(
                f"% change in ~{iv_target_days}d ATM IV vs the snapshot taken "
                f"{one_day['days_back']}d ago ({one_day['prev_date']}). "
                f"Prior IV: {one_day['prev'] * 100:.2f}%. "
                "pp = vol points (absolute change)."
            ),
        )
    else:
        c6.metric(
            f"24h IV Δ (~{iv_target_days}d)",
            "N/A",
            help=(
                "Need at least one prior snapshot ≥1 day old. The cache "
                "populates each time you load the app — come back tomorrow."
            ),
        )
else:
    c4.metric(f"~{iv_target_days}d IV (ATM)", "N/A", help="No options chain available")
    c5.metric("Vol risk premium", "—")
    c6.metric(f"24h IV Δ (~{iv_target_days}d)", "—")

# --- Company description (sector, industry, business summary) ---------------
sector = fundamentals.get("sector")
industry = fundamentals.get("industry")
summary = fundamentals.get("business_summary")
website = fundamentals.get("website")
brief = _brief_summary(summary, max_sentences=2, max_chars=320)

if sector or industry or brief or website:
    meta_bits: list[str] = []
    if sector:
        meta_bits.append(f"**Sector:** {sector}")
    if industry:
        meta_bits.append(f"**Industry:** {industry}")
    if website:
        href = website if website.startswith(("http://", "https://")) else f"https://{website}"
        display = website.replace("https://", "").replace("http://", "").rstrip("/")
        meta_bits.append(f"**Web:** [{display}]({href})")
    if brief or summary:
        with st.expander("About the company", expanded=False):
            if meta_bits:
                st.markdown(" · ".join(meta_bits))
            if brief:
                st.write(brief)
            # Offer the full summary inside a nested expander for users who
            # want the long form, without making it the default view.
            if summary and brief and len(summary) > len(brief):
                with st.expander("Full description", expanded=False):
                    st.write(summary)

st.divider()

# --- Price + Volume chart ---------------------------------------------------
chart_df = load_history(ticker, period=chart_period)
if chart_df.empty:
    st.warning("No data for the selected chart window.")
else:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=("", "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Overlay moving averages — computed off the full 5y history so that
    # 200d MA is correct even on a short chart window.
    ma_specs = [
        (10, "#ffd54f"),   # amber
        (20, "#ff7043"),   # deep orange
        (50, "#1565c0"),   # darker blue (so 200d baby blue stays distinct)
        (100, "#ab47bc"),  # purple
        (200, "#89cff0"),  # baby blue
    ]
    for period, color in ma_specs:
        ma_full = hist_full["Close"].rolling(period).mean()
        ma_aligned = ma_full.reindex(chart_df.index)
        fig.add_trace(
            go.Scatter(
                x=ma_aligned.index,
                y=ma_aligned.values,
                name=f"{period}d MA",
                line=dict(color=color, width=1.5),
                mode="lines",
                hovertemplate=f"{period}d MA: $%{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    vol_colors = np.where(
        chart_df["Close"] >= chart_df["Open"], "#26a69a", "#ef5350"
    )
    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # --- Earnings markers ---------------------------------------------------
    # Small "e" markers near the bottom of the price panel for past earnings
    # dates in view, plus the next scheduled date. The x-axis is extended so
    # the upcoming earnings date is visible even though it's past the last bar.
    earnings = get_earnings_calendar(ticker)
    idx_tz = chart_df.index.tz  # match the candlestick axis timezone

    def _to_axis(ts):
        """Coerce a tz-naive earnings Timestamp onto the chart's x-axis tz."""
        t = pd.Timestamp(ts)
        if idx_tz is not None:
            t = t.tz_localize(idx_tz) if t.tzinfo is None else t.tz_convert(idx_tz)
        elif t.tzinfo is not None:
            t = t.tz_localize(None)
        return t

    win_start = chart_df.index[0]
    win_end = chart_df.index[-1]
    price_low = float(chart_df["Low"].min())
    price_high = float(chart_df["High"].max())
    # Place the "e" markers just below the visible lows.
    e_y = price_low - (price_high - price_low) * 0.03

    win_start_naive = win_start.tz_localize(None) if win_start.tzinfo else win_start
    is_approx = earnings.get("approximate", False)
    past_e = [d for d in earnings["past"] if d >= win_start_naive]
    if past_e:
        fig.add_trace(
            go.Scatter(
                x=[_to_axis(d) for d in past_e],
                y=[e_y] * len(past_e),
                mode="markers+text",
                text=["e"] * len(past_e),
                textposition="middle center",
                textfont=dict(size=9, color="white"),
                marker=dict(size=15, color="#5c6bc0", symbol="circle"),
                name=("Quarter end (approx)" if is_approx else "Earnings (reported)"),
                hovertemplate=(
                    ("≈ Quarter end" if is_approx else "Earnings reported")
                    + "<br>%{x|%Y-%m-%d}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    next_e = earnings["next"]
    x_end = win_end
    if next_e is not None:
        next_e_axis = _to_axis(next_e)
        fig.add_trace(
            go.Scatter(
                x=[next_e_axis],
                y=[e_y],
                mode="markers+text",
                text=["e"],
                textposition="middle center",
                textfont=dict(size=9, color="white"),
                marker=dict(
                    size=17,
                    color="#fb8c00",
                    symbol="circle",
                    line=dict(color="#e65100", width=1.5),
                ),
                name="Next earnings",
                hovertemplate="Next earnings (scheduled)<br>%{x|%Y-%m-%d}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # Extend the x-axis so the upcoming date is in view, with a week of pad.
        win_end_naive = win_end.tz_localize(None) if win_end.tzinfo else win_end
        if next_e > win_end_naive:
            x_end = _to_axis(next_e + pd.Timedelta(days=7))

    fig.update_layout(
        height=620,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(range=[win_start, x_end])
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.subheader(f"{ticker} — Price & Volume ({chart_period})")
    st.plotly_chart(fig, use_container_width=True)
    next_e_caption = (
        f"📅 Next earnings: **{next_e.strftime('%b %d, %Y')}**"
        if next_e is not None
        else "Next earnings date not available from Yahoo."
    )
    is_approx = earnings.get("approximate", False)
    if past_e:
        past_label = (
            f"{len(past_e)} past quarter-ends in view (approx.)"
            if is_approx
            else f"{len(past_e)} past earnings in view"
        )
        marker_legend = (
            "indigo **e** = quarter end (approx), orange **e** = next scheduled."
            if is_approx
            else "indigo **e** = past reported, orange **e** = next scheduled."
        )
    else:
        past_label = "no past earnings markers in view"
        marker_legend = "orange **e** = next scheduled."
    st.caption(next_e_caption + f"  ·  {past_label}  ·  " + marker_legend)
    if not earnings.get("past"):
        st.warning(
            "Past earnings dates could not be retrieved from Yahoo right now. "
            "This feed is intermittently rate-limited — try the 🔄 Refresh "
            "chain button or reload in a minute.",
            icon="⚠️",
        )
    elif is_approx:
        st.info(
            "Past earnings markers are **approximate** — Yahoo's reported "
            "earnings-date feed was unavailable, so fiscal quarter-end dates "
            "are shown instead (actual report dates are typically a few weeks "
            "later).",
            icon="ℹ️",
        )

    with st.expander("🔍 Earnings dates (diagnostic)"):
        st.write(
            f"**yfinance source(s) that returned data:** "
            f"{earnings.get('source') or '_none — all earnings_dates calls failed_'}"
        )
        st.write(
            f"**Total dates found:** {len(earnings['past']) + (1 if earnings['next'] else 0)}  ·  "
            f"**Past:** {len(earnings['past'])}  ·  "
            f"**In current chart window:** {len(past_e)}"
        )
        st.write(
            f"**Dates are approximate (quarter-end fallback):** "
            f"{earnings.get('approximate', False)}"
        )
        if earnings["past"]:
            st.write(
                "**All past earnings dates:** "
                + ", ".join(d.strftime("%Y-%m-%d") for d in earnings["past"])
            )
        else:
            st.write(
                "_No past earnings dates returned. Yahoo's earnings history "
                "feed is intermittently unavailable — try the 🔄 Refresh "
                "chain button or reload in a minute._"
            )

# --- IV curves across selected expirations ----------------------------------
st.subheader("Implied Volatility Curves")
st.caption(
    "Nearest 2 Friday expirations · next 3 monthlies (3rd Friday — or the "
    "Thursday before, when the 3rd Friday is a market holiday like Juneteenth "
    "or Good Friday) · next 4 quarterlies (Mar/Jun/Sep/Dec). "
    "OTM puts on the left, OTM calls on the right."
)

cache_cols = st.columns([1, 5])
with cache_cols[0]:
    if st.button("🔄 Refresh chain", help="Clear cached option data and refetch from Yahoo"):
        st.cache_data.clear()
        st.rerun()

try:
    expirations_all = yf.Ticker(ticker).options or []
except Exception:
    expirations_all = []

if not expirations_all:
    st.info("No listed options for this ticker.")
else:
    exp_info = classify_expirations(expirations_all)
    selected = select_curve_expirations(exp_info)

    # Always show what's being attempted so missing expirations aren't a mystery.
    st.caption(
        "**Selected expirations:** "
        + (", ".join(
            f"{e} ({c}, {int(d)}d)"
            for e, c, d in zip(selected["expiration"], selected["category"], selected["days_to_exp"])
        ) if not selected.empty else "_none_")
    )

    if selected.empty:
        st.info("No future expirations available.")
    else:
        # Filter strikes inside a sensible moneyness band so wings don't dominate
        moneyness_band = st.slider(
            "Strike filter (as fraction of spot)",
            min_value=0.50,
            max_value=1.50,
            value=(0.75, 1.25),
            step=0.05,
            key="iv_curve_moneyness",
            help=(
                "Filter strikes shown on the chart by their distance from spot. "
                "0.75–1.25 means strikes from 75% to 125% of the current price."
            ),
        )

        # Build a color scale by days-to-expiry (short = warm red, long = cool blue)
        max_days = float(selected["days_to_exp"].max() or 1)
        colors_scale = [
            "#e53935", "#fb8c00", "#fdd835", "#43a047",
            "#00897b", "#1e88e5", "#3949ab", "#8e24aa",
        ]

        curves_fig = go.Figure()
        rows_for_table = []
        any_curve = False
        dropped: list[str] = []   # expirations that returned no curve at all
        sparse_curve: list[str] = []  # rendered, but no strikes inside the moneyness band

        for idx, row in selected.iterrows():
            data = get_expiration_data(ticker, row["expiration"], last_price)
            if data is None or data.get("curve") is None or data["curve"].empty:
                dropped.append(f"{row['expiration']} ({row['category']})")
                continue
            curve = data["curve"]
            mask = (
                (curve["moneyness"] >= moneyness_band[0])
                & (curve["moneyness"] <= moneyness_band[1])
            )
            curve_filt = curve[mask]
            if curve_filt.empty:
                sparse_curve.append(f"{row['expiration']} ({row['category']})")
                continue
            any_curve = True
            color = colors_scale[idx % len(colors_scale)]

            # Legend label: header line + ATM IV + straddle
            atm_iv = data.get("atm_iv")
            atm_iv_str = f"{atm_iv * 100:.1f}%" if atm_iv is not None else "n/a"
            straddle = data.get("atm_straddle")
            straddle_str = f"${straddle:.2f}" if straddle is not None else "n/a"
            atm_strike_disp = data.get("atm_strike")
            label = (
                f"{row['expiration']} · {int(row['days_to_exp'])}d · {row['category']}"
                f"<br><sub>ATM IV {atm_iv_str} · Straddle {straddle_str}</sub>"
            )

            curves_fig.add_trace(
                go.Scatter(
                    x=curve_filt["strike"],
                    y=curve_filt["impliedVolatility"] * 100,
                    name=label,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    hovertemplate=(
                        "Strike $%{x:.2f}<br>"
                        "Moneyness %{customdata[0]:.3f}<br>"
                        "IV %{y:.2f}%<br>"
                        f"<extra>{row['expiration']} · {row['category']}</extra>"
                    ),
                    customdata=curve_filt[["moneyness"]].values,
                )
            )

            rows_for_table.append(
                {
                    "Expiration": row["expiration"],
                    "DTE": int(row["days_to_exp"]),
                    "Category": row["category"],
                    "ATM strike": float(atm_strike_disp) if atm_strike_disp else float("nan"),
                    "ATM IV": float(atm_iv) if atm_iv is not None else float("nan"),
                    "Straddle": float(straddle) if straddle is not None else float("nan"),
                    "n strikes (chart)": int(len(curve_filt)),
                    "px source mid/last": f"{data.get('n_mid', 0)}/{data.get('n_last', 0)}",
                }
            )

        if any_curve:
            curves_fig.add_vline(
                x=last_price,
                line=dict(color="gray", dash="dot", width=1),
                annotation_text=f"Spot ${last_price:,.2f}",
                annotation_position="top",
            )
            curves_fig.update_layout(
                height=520,
                xaxis_title="Strike ($)",
                yaxis_title="Implied volatility (%)",
                xaxis=dict(tickprefix="$", tickformat=",.2f"),
                hovermode="closest",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(
                    orientation="v",
                    traceorder="normal",  # nearest expiration on top (traces are added in date order)
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255,255,255,0.6)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1,
                    title=dict(text="Expiration", font=dict(size=12)),
                ),
            )
            st.plotly_chart(curves_fig, use_container_width=True)

            # --- ATM IV term structure -------------------------------------
            # ATM IV (y) vs days-to-expiration (x) across EVERY listed
            # expiration on Yahoo (not just the curated set used for the curves
            # chart). Each call is cached by (ticker, expiration), so this is
            # cheap on reload.
            # Upward slope = contango (longer-dated vol richer, the normal
            # state); downward slope = backwardation (front-month vol bid,
            # typical around earnings or stress).
            ts_pts: list[dict] = []
            ts_missing: list[str] = []
            for _, row in exp_info.iterrows():
                exp_str = str(row["expiration"])
                dte = int(row["days_to_exp"])
                if dte < 0:
                    continue
                data = get_expiration_data(ticker, exp_str, last_price)
                atm_iv_val = data.get("atm_iv") if data else None
                if atm_iv_val is None or pd.isna(atm_iv_val):
                    ts_missing.append(f"{exp_str} ({dte}d)")
                    continue
                ts_pts.append(
                    {"Expiration": exp_str, "DTE": dte, "ATM IV": float(atm_iv_val)}
                )
            ts_pts.sort(key=lambda r: r["DTE"])
            if len(ts_pts) >= 2:
                st.subheader("ATM IV Term Structure")
                ts_x = [r["DTE"] for r in ts_pts]
                ts_y = [r["ATM IV"] * 100 for r in ts_pts]
                slope = "contango ↑" if ts_y[-1] >= ts_y[0] else "backwardation ↓"
                term_fig = go.Figure()
                term_fig.add_trace(
                    go.Scatter(
                        x=ts_x,
                        y=ts_y,
                        mode="lines+markers",
                        line=dict(color="#1e88e5", width=2),
                        marker=dict(size=7, color="#1e88e5"),
                        text=[r["Expiration"] for r in ts_pts],
                        hovertemplate=(
                            "%{text}<br>%{x}d to expiration<br>"
                            "ATM IV %{y:.2f}%<extra></extra>"
                        ),
                    )
                )
                term_fig.update_layout(
                    height=380,
                    xaxis_title="Days to expiration",
                    yaxis_title="ATM implied volatility (%)",
                    xaxis=dict(ticksuffix="d"),
                    yaxis=dict(ticksuffix="%"),
                    hovermode="closest",
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                st.plotly_chart(term_fig, use_container_width=True)
                st.caption(
                    f"ATM IV across all **{len(ts_pts)}** listed expirations · "
                    f"curve is currently in **{slope}**. Upward slope (contango) "
                    "is the normal state; a downward slope (backwardation) means "
                    "front-month vol is bid — common into earnings or market stress."
                )
                if ts_missing:
                    with st.expander(
                        f"⚠ {len(ts_missing)} expiration(s) with no ATM IV (excluded)"
                    ):
                        for label in ts_missing:
                            st.markdown(f"- {label}")

            # ATM IV term-structure summary table
            if rows_for_table:
                tbl = pd.DataFrame(rows_for_table)
                tbl_display = tbl.copy()
                tbl_display["ATM strike"] = tbl_display["ATM strike"].map(
                    lambda x: f"${x:,.2f}" if pd.notna(x) else "n/a"
                )
                tbl_display["ATM IV"] = tbl_display["ATM IV"].map(
                    lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "n/a"
                )
                tbl_display["Straddle"] = tbl_display["Straddle"].map(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "n/a"
                )
                with st.expander("ATM IV by expiration"):
                    st.dataframe(tbl_display, hide_index=True, use_container_width=True)
                    st.caption(
                        "**px source mid/last** — count of strikes priced from bid/ask "
                        "mid vs. lastPrice fallback. Higher mid is healthier; high last "
                        "means the chain is illiquid and IVs may lag."
                    )
        else:
            st.warning("Couldn't fetch IV data for any of the selected expirations.")

        # Surface any expirations that didn't make it into the chart so missing
        # contracts (e.g. an illiquid June with no two-sided quotes) aren't silent.
        if dropped or sparse_curve:
            with st.expander(
                f"⚠ {len(dropped) + len(sparse_curve)} expiration(s) not plotted",
                expanded=True,
            ):
                if dropped:
                    st.markdown(
                        "**No usable option prices** (chain returned no two-sided "
                        "quotes and no positive lastPrice):"
                    )
                    for label in dropped:
                        st.markdown(f"- {label}")
                if sparse_curve:
                    st.markdown(
                        "**No strikes inside the moneyness band** "
                        f"({moneyness_band[0]:.2f} – {moneyness_band[1]:.2f}). "
                        "Try widening the slider:"
                    )
                    for label in sparse_curve:
                        st.markdown(f"- {label}")

        # Diagnostic: every expiration Yahoo returned, with classification flags.
        # Use this to confirm whether June is in Yahoo's data and which bucket
        # the picker assigned it to.
        with st.expander("🔍 All Yahoo expirations (diagnostic)"):
            diag = exp_info.copy()
            diag["selected_as"] = diag["expiration"].map(
                dict(zip(selected["expiration"], selected["category"]))
            ).fillna("—")
            diag["expiration"] = diag["expiration"].astype(str)
            diag["date"] = diag["date"].dt.strftime("%Y-%m-%d (%a)")
            st.dataframe(
                diag[
                    [
                        "expiration",
                        "date",
                        "days_to_exp",
                        "is_third_friday",
                        "is_quarterly",
                        "selected_as",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"Yahoo returned **{len(expirations_all)}** expirations. "
                f"Picker selected **{len(selected)}**."
            )


# --- Gamma Exposure (GEX) ---------------------------------------------------
st.subheader("Gamma Exposure by Strike")

if not expirations_all:
    st.info("No listed options for this ticker, so no GEX to show.")
else:
    gex_exp_choices = exp_info[exp_info["days_to_exp"] >= 0].copy()
    gex_exp_choices["label"] = gex_exp_choices.apply(
        lambda r: (
            f"{r['expiration']}"
            + ("(m)" if r["is_third_friday"] and not r["is_quarterly"] else "")
            + ("(q)" if r["is_quarterly"] else "")
            + f" · {int(r['days_to_exp'])}d"
        ),
        axis=1,
    )

    if gex_exp_choices.empty:
        st.info("No upcoming expirations.")
    else:
        default_choices = gex_exp_choices["label"].head(2).tolist()
        picked_labels = st.multiselect(
            "Expirations",
            options=gex_exp_choices["label"].tolist(),
            default=default_choices,
            key="gex_expirations",
            help="Toggle one or more expirations. GEX aggregates across all selected.",
        )
        picked_rows = gex_exp_choices[gex_exp_choices["label"].isin(picked_labels)]

        if picked_rows.empty:
            st.info("Pick at least one expiration.")
        else:
            per_exp_frames: list[pd.DataFrame] = []
            gex_dropped: list[str] = []
            for _, exp_row in picked_rows.iterrows():
                gdf = get_expiration_gex(ticker, exp_row["expiration"], last_price)
                if gdf is None or gdf.empty:
                    gex_dropped.append(exp_row["expiration"])
                    continue
                per_exp_frames.append(gdf)

            if not per_exp_frames:
                st.warning("No usable GEX data for the selected expirations.")
            else:
                # Aggregate net GEX per strike across selected expirations
                agg = (
                    pd.concat(per_exp_frames, ignore_index=True)
                    .groupby("strike", as_index=False)[["call_gex", "put_gex", "net_gex"]]
                    .sum()
                    .sort_values("strike")
                    .reset_index(drop=True)
                )

                # Color each bar by sign — green for positive net, red for negative
                colors = [
                    "#8fd9a8" if v >= 0 else "#f4a1a1" for v in agg["net_gex"]
                ]

                # Pretty caption — ticker + which expirations are aggregated
                exp_labels_str = ", ".join(
                    [
                        f"{r['expiration']}"
                        + ("(m)" if r["is_third_friday"] and not r["is_quarterly"] else "")
                        + ("(q)" if r["is_quarterly"] else "")
                        for _, r in picked_rows.iterrows()
                    ]
                )
                st.caption(f"Showing results for **{ticker}**, {exp_labels_str}, calls & puts")

                gex_fig = go.Figure()
                gex_fig.add_trace(
                    go.Bar(
                        x=agg["strike"],
                        y=agg["net_gex"] / 1e6,
                        name="Net GEX",
                        marker_color=colors,
                        hovertemplate=(
                            "Strike $%{x:.2f}<br>"
                            "Net GEX %{y:.2f}M $/1%<extra></extra>"
                        ),
                    )
                )
                gex_fig.add_vline(
                    x=last_price,
                    line=dict(color="rgba(0,0,0,0.55)", dash="dot", width=1),
                    annotation_text=f"<b>{last_price:,.2f}</b>",
                    annotation_position="bottom",
                    annotation=dict(
                        bgcolor="#1f3a68",
                        bordercolor="#1f3a68",
                        font=dict(color="white", size=12),
                    ),
                )

                gex_fig.update_layout(
                    height=480,
                    yaxis=dict(title="Gamma Exposure ($ / 1% move)", zeroline=True, zerolinecolor="rgba(0,0,0,0.2)", ticksuffix="M"),
                    xaxis=dict(title="Strikes"),
                    margin=dict(l=10, r=10, t=10, b=40),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="center",
                        x=0.5,
                    ),
                    bargap=0.15,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(gex_fig, use_container_width=True)

                if gex_dropped:
                    st.caption(
                        "⚠ No usable GEX (no OI or unpriced strikes): "
                        + ", ".join(gex_dropped)
                    )


# --- Yang-Zhang volatility cones --------------------------------------------
st.subheader("Yang-Zhang Volatility Cones")
st.caption(
    f"Built from {cone_lookback} of OHLC history. Bands show min/25th/median/75th/max "
    "of rolling Yang-Zhang realized vol at each window. The blue diamond is the *current* "
    "rolling YZ vol for that window."
)

cone_windows = [10, 20, 30, 60, 90, 120, 180, 252]
rows = []
for w in cone_windows:
    if len(hist_full) <= w + 1:
        continue
    series = yang_zhang_vol(hist_full, w).dropna()
    if series.empty:
        continue
    rows.append(
        {
            "Window (d)": w,
            "Min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "Median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "Max": float(series.max()),
            "Current": float(series.iloc[-1]),
        }
    )
cone_df = pd.DataFrame(rows)

if cone_df.empty:
    st.warning("Not enough history to build cones.")
else:
    cone_fig = go.Figure()
    x = cone_df["Window (d)"]
    # Shaded band between min and max
    cone_fig.add_trace(
        go.Scatter(
            x=pd.concat([x, x[::-1]]),
            y=pd.concat([cone_df["Max"] * 100, cone_df["Min"][::-1] * 100]),
            fill="toself",
            fillcolor="rgba(120,144,156,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Shaded band between p25 and p75
    cone_fig.add_trace(
        go.Scatter(
            x=pd.concat([x, x[::-1]]),
            y=pd.concat([cone_df["p75"] * 100, cone_df["p25"][::-1] * 100]),
            fill="toself",
            fillcolor="rgba(66,133,244,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    cone_fig.add_trace(
        go.Scatter(x=x, y=cone_df["Max"] * 100, name="Max", line=dict(color="#ef5350", width=1, dash="dot"))
    )
    cone_fig.add_trace(
        go.Scatter(x=x, y=cone_df["p75"] * 100, name="75th pct", line=dict(color="#fb8c00", width=1.5))
    )
    cone_fig.add_trace(
        go.Scatter(x=x, y=cone_df["Median"] * 100, name="Median", line=dict(color="#546e7a", width=2))
    )
    cone_fig.add_trace(
        go.Scatter(x=x, y=cone_df["p25"] * 100, name="25th pct", line=dict(color="#42a5f5", width=1.5))
    )
    cone_fig.add_trace(
        go.Scatter(x=x, y=cone_df["Min"] * 100, name="Min", line=dict(color="#26a69a", width=1, dash="dot"))
    )
    cone_fig.add_trace(
        go.Scatter(
            x=x,
            y=cone_df["Current"] * 100,
            name="Current",
            mode="markers+lines",
            marker=dict(size=11, color="#1e88e5", symbol="diamond", line=dict(color="white", width=1)),
            line=dict(color="#1e88e5", width=2),
        )
    )
    if iv_info:
        cone_fig.add_hline(
            y=iv_info["iv"] * 100,
            line=dict(color="#8e24aa", dash="dash"),
            annotation_text=f"~{iv_target_days}d ATM IV: {iv_info['iv']*100:.1f}%",
            annotation_position="top left",
        )
    cone_fig.update_layout(
        height=520,
        xaxis_title="Rolling window (trading days)",
        yaxis_title="Annualized volatility (%)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(cone_fig, use_container_width=True)

    with st.expander("Cone data table"):
        display = cone_df.copy()
        for col in ["Min", "p25", "Median", "p75", "Max", "Current"]:
            display[col] = (display[col] * 100).round(2).astype(str) + "%"
        st.dataframe(display, hide_index=True, use_container_width=True)

# --- Rolling realized vol time series ---------------------------------------
st.subheader(f"Rolling {hv_window}-day Realized Vol vs ~{iv_target_days}d ATM IV")
yz_series = yang_zhang_vol(hist_full, hv_window) * 100
cc_series = close_to_close_vol(hist_full, hv_window) * 100

# Show last 2y or whatever fits the chart window choice
ts_period_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
days = ts_period_map.get(chart_period, 730)
cutoff = hist_full.index[-1] - pd.Timedelta(days=days)
yz_recent = yz_series[yz_series.index >= cutoff].dropna()
cc_recent = cc_series[cc_series.index >= cutoff].dropna()

ts_fig = go.Figure()
ts_fig.add_trace(go.Scatter(x=yz_recent.index, y=yz_recent.values, name="Yang-Zhang HV", line=dict(color="#1e88e5", width=2)))
ts_fig.add_trace(go.Scatter(x=cc_recent.index, y=cc_recent.values, name="Close-to-Close HV", line=dict(color="#fb8c00", width=1.5, dash="dot")))

# Overlay historical IV from the local snapshot cache. Falls back to a
# horizontal line at today's IV when no history is available.
iv_hist_df = iv_history(ticker, iv_target_days)
iv_overlay_caption = None
if iv_hist_df is not None and not iv_hist_df.empty:
    iv_hist_df = iv_hist_df.copy()
    # yfinance returns a tz-aware index, but the cache CSV is tz-naive.
    # Normalize both to tz-naive for the cutoff comparison.
    iv_hist_df["date"] = pd.to_datetime(iv_hist_df["date"])
    if iv_hist_df["date"].dt.tz is not None:
        iv_hist_df["date"] = iv_hist_df["date"].dt.tz_convert(None)
    cutoff_naive = (
        cutoff.tz_convert(None) if getattr(cutoff, "tzinfo", None) is not None else cutoff
    )
    iv_recent = iv_hist_df[iv_hist_df["date"] >= cutoff_naive]
    if not iv_recent.empty:
        ts_fig.add_trace(
            go.Scatter(
                x=iv_recent["date"],
                y=iv_recent["iv"] * 100,
                name=f"~{iv_target_days}d ATM IV",
                mode="lines+markers" if len(iv_recent) < 60 else "lines",
                line=dict(color="#8e24aa", width=2),
                marker=dict(size=5, color="#8e24aa"),
            )
        )
        iv_overlay_caption = (
            f"IV line built from local snapshot cache "
            f"(`{IV_CACHE_FILE.name}`) — {len(iv_hist_df)} total snapshot(s) for "
            f"{ticker} ~{iv_target_days}d, first on "
            f"{iv_hist_df['date'].iloc[0].strftime('%Y-%m-%d')}. yfinance does "
            "not expose historical option chains, so the line only covers "
            "dates you've actually opened the app."
        )
if iv_overlay_caption is None and iv_info:
    # No usable cache history yet — keep the horizontal line so today's level is visible.
    ts_fig.add_hline(
        y=iv_info["iv"] * 100,
        line=dict(color="#8e24aa", dash="dash"),
        annotation_text=f"~{iv_target_days}d ATM IV (today)",
        annotation_position="top left",
    )
    iv_overlay_caption = (
        "No cached IV history yet — only today's IV is shown as a dashed line. "
        "Each time you load the app, today's snapshot is appended to "
        f"`{IV_CACHE_FILE.name}`, so this will fill in as a real time series "
        "as you keep using the app."
    )

ts_fig.update_layout(
    height=350,
    yaxis_title="Annualized vol (%)",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(ts_fig, use_container_width=True)
if iv_overlay_caption:
    st.caption(iv_overlay_caption)


# --- Annual income statement (last 5 fiscal years) --------------------------
st.divider()
st.subheader("Income Statement — Last 4 Years")

income_df = get_income_history(ticker, max_years=5)
income_extra = get_income_ttm_and_projection(ticker)
if income_df is None or income_df.empty:
    st.info("No annual income-statement data available for this ticker.")
else:
    metric_cols = ["Revenue", "Gross Profit", "Operating Income", "Net Income"]

    # Append current-FY (0y) and next-FY (+1y) projections after historicals.
    cy_label = income_extra.get("proj_cy_label") if income_extra else None
    ny_label = income_extra.get("proj_ny_label") if income_extra else None
    projection_labels: set[str] = set()

    extra_rows: list[tuple[str, dict]] = []
    if income_extra and income_extra.get("proj_cy") and cy_label:
        extra_rows.append((cy_label, income_extra["proj_cy"]))
        projection_labels.add(cy_label)
    if income_extra and income_extra.get("proj_ny") and ny_label:
        extra_rows.append((ny_label, income_extra["proj_ny"]))
        projection_labels.add(ny_label)

    if extra_rows:
        extra_df = pd.DataFrame(
            [{m: row.get(m) for m in metric_cols} for _, row in extra_rows],
            index=[label for label, _ in extra_rows],
        )
        income_df = pd.concat([income_df[metric_cols], extra_df])

    n_hist = len([i for i in income_df.index if i not in projection_labels])
    proj_note = ""
    if projection_labels:
        proj_note = (
            f"  ·  {' & '.join(sorted(projection_labels))} are **analyst "
            "consensus projections** (rightmost bars, transparent + hatched)."
        )
    st.caption(
        f"Annual Revenue, Gross Profit, Operating Income and Net Income — "
        f"{n_hist} fiscal year{'s' if n_hist != 1 else ''} from Yahoo Finance"
        f"{proj_note}"
    )

    metric_colors = {
        "Revenue": "#1565c0",
        "Gross Profit": "#26a69a",
        "Operating Income": "#ffb300",
        "Net Income": "#8e24aa",
    }
    # Faded / desaturated versions for projection bars.
    metric_colors_proj = {
        "Revenue": "rgba(21,101,192,0.35)",
        "Gross Profit": "rgba(38,166,154,0.35)",
        "Operating Income": "rgba(255,179,0,0.35)",
        "Net Income": "rgba(142,36,170,0.35)",
    }

    def _fmt_usd(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        a = abs(v)
        if a >= 1e12:
            return f"${v / 1e12:.2f}T"
        if a >= 1e9:
            return f"${v / 1e9:.2f}B"
        if a >= 1e6:
            return f"${v / 1e6:.1f}M"
        return f"${v:,.0f}"

    x_labels = list(income_df.index)
    # Per-bar styling: solid for historicals, faded/hatched for projections.
    def _per_bar_style(metric: str):
        colors, patterns = [], []
        base = metric_colors[metric]
        faded = metric_colors_proj[metric]
        for lbl in x_labels:
            if lbl in projection_labels:
                colors.append(faded)
                patterns.append("/")
            else:
                colors.append(base)
                patterns.append("")
        return colors, patterns

    inc_fig = go.Figure()
    for metric in metric_cols:
        if metric not in income_df.columns:
            continue
        vals = income_df[metric]
        colors, patterns = _per_bar_style(metric)
        inc_fig.add_trace(
            go.Bar(
                x=x_labels,
                y=vals.values / 1e9,
                name=metric,
                marker=dict(
                    color=colors,
                    pattern=dict(shape=patterns),
                    line=dict(width=0),
                ),
                text=[_fmt_usd(v) for v in vals.values],
                textposition="outside",
                hovertemplate="%{x}<br>" + metric + " %{text}<extra></extra>",
            )
        )

    inc_fig.update_layout(
        height=440,
        barmode="group",
        yaxis=dict(title="USD (billions)", zeroline=True, zerolinecolor="rgba(0,0,0,0.25)"),
        xaxis=dict(title="Fiscal year"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(inc_fig, use_container_width=True)

    if projection_labels:
        src_flags = income_extra.get("proj_sources") if income_extra else []
        src_str = " · ".join(src_flags) if src_flags else ""
        st.caption(
            "**Projection sourcing** (per bar): "
            + (f"`{src_str}`. " if src_str else "")
            + "Priority is **FMP analyst consensus** (Revenue, EBIT = Operating "
            "Income, Net Income directly from the analyst-estimates endpoint), "
            "then Yahoo `revenue_estimate` / `earnings_estimate × "
            "sharesOutstanding`, then `forwardEps × shares` for next-year NI, "
            "then growth-rate extrapolation from the latest actual using "
            "Yahoo's `revenueGrowth` / `earningsGrowth`. Gross Profit is "
            "always derived from **TTM margins** applied to projected revenue "
            "since no source publishes a consensus GP line. Treat projections "
            "as directional, not gospel."
        )

    # Diagnostic — surface everything the projection builder saw so we can
    # tell WHY a bar is missing.
    with st.expander("🔍 Projection sources (diagnostic)"):
        try:
            _fmp_test = fetch_fmp_analyst_estimates(ticker)
            st.write(f"**FMP key present:** {_fmp_api_key() is not None}")
            if _fmp_test is None:
                st.write("**FMP response:** _None_ — endpoint returned empty "
                         "or errored. Check API key or plan limits.")
            else:
                st.write(f"**FMP response:** {len(_fmp_test)} item(s) returned")
                for _item in _fmp_test[:6]:
                    st.write({
                        "date": _item.get("date"),
                        "revenueAvg": _item.get("estimatedRevenueAvg"),
                        "ebitAvg": _item.get("estimatedEbitAvg"),
                        "netIncomeAvg": _item.get("estimatedNetIncomeAvg"),
                        "epsAvg": _item.get("estimatedEpsAvg"),
                        "n_analysts_rev": _item.get("numberAnalystEstimatedRevenue"),
                    })
        except Exception as _e:
            st.write(f"FMP diag error: {_e}")
        try:
            _tk = yf.Ticker(ticker)
            st.write("**Yahoo revenue_estimate:**")
            try:
                st.write(_tk.revenue_estimate)
            except Exception as _e:
                st.write(f"_error: {_e}_")
            st.write("**Yahoo earnings_estimate:**")
            try:
                st.write(_tk.earnings_estimate)
            except Exception as _e:
                st.write(f"_error: {_e}_")
            _info = getattr(_tk, "info", {}) or {}
            st.write({
                "revenueGrowth (Yahoo info)": _info.get("revenueGrowth"),
                "earningsGrowth (Yahoo info)": _info.get("earningsGrowth"),
                "forwardEps (Yahoo info)": _info.get("forwardEps"),
                "sharesOutstanding (Yahoo info)": _info.get("sharesOutstanding"),
            })
        except Exception as _e:
            st.write(f"Yahoo diag error: {_e}")
        st.write("**Final projection payload from get_income_ttm_and_projection:**")
        st.write(income_extra)

    with st.expander("🔍 Raw income statement (diagnostic)"):
        st.caption(
            "Every fiscal-year column Yahoo returned, before filtering. If a "
            "year is missing here, Yahoo's free feed simply doesn't carry it "
            "for this ticker — it can't be recovered from this data source."
        )
        try:
            raw = yf.Ticker(ticker).income_stmt
            if raw is not None and not raw.empty:
                raw_disp = raw.copy()
                raw_disp.columns = [
                    pd.Timestamp(c).strftime("%Y-%m-%d") for c in raw_disp.columns
                ]
                st.dataframe(raw_disp, use_container_width=True)
            else:
                st.write("yfinance returned an empty income statement.")
        except Exception as e:
            st.write(f"Could not load raw income statement: {e}")


# --- Quarterly balance sheet (last 4 quarters) ------------------------------
st.divider()
st.subheader("Quarterly Balance Sheet — Last 4 Quarters")

bs_df = get_quarterly_balance_history(ticker, n=4)
if bs_df is None or bs_df.empty:
    st.info("No quarterly balance sheet data available for this ticker.")
else:
    def _fmt_usd_bs(v) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v))):
            return "n/a"
        a = abs(v)
        sign = "-" if v < 0 else ""
        if a >= 1e12:
            return f"{sign}${a / 1e12:.2f}T"
        if a >= 1e9:
            return f"{sign}${a / 1e9:.2f}B"
        if a >= 1e6:
            return f"{sign}${a / 1e6:.1f}M"
        return f"{sign}${a:,.0f}"

    # Cash trend bar chart — the headline number on this section.
    if "Cash + ST Investments" in bs_df.index:
        cash_series = bs_df.loc["Cash + ST Investments"]
        # Columns are newest-first; reverse for chronological left-to-right.
        cash_chart_x = list(reversed(cash_series.index.tolist()))
        cash_chart_y = [float(v) if not pd.isna(v) else None for v in reversed(cash_series.tolist())]
        cash_fig = go.Figure()
        cash_fig.add_trace(
            go.Bar(
                x=cash_chart_x,
                y=[v / 1e9 if v is not None else None for v in cash_chart_y],
                marker_color="#2e7d32",
                text=[_fmt_usd_bs(v) for v in cash_chart_y],
                textposition="outside",
                hovertemplate="Quarter end %{x}<br>Cash + ST inv. %{text}<extra></extra>",
                name="Cash + ST Investments",
            )
        )
        cash_fig.update_layout(
            height=340,
            yaxis=dict(title="USD (billions)", zeroline=True, zerolinecolor="rgba(0,0,0,0.25)"),
            xaxis=dict(title="Quarter end"),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(cash_fig, use_container_width=True)

    st.markdown("**Balance sheet detail**")
    # Reverse column order so chronologically oldest is on the left and the
    # most recent quarter sits on the far right.
    bs_display = bs_df.iloc[:, ::-1].map(_fmt_usd_bs)
    st.dataframe(bs_display, use_container_width=True)
    st.caption(
        "Cash + ST Investments is the most defensible \"cash on hand\" figure "
        "for runway math. Net Cash = (Cash + ST Investments) − Total Debt; "
        "positive Net Cash = the company could pay off all debt with cash on "
        "the balance sheet today. Columns are quarter-end dates, oldest on the "
        "left and the most recent quarter on the far right."
    )


# --- Quarterly cash burn ----------------------------------------------------
st.subheader("Quarterly Cash Burn")

burn_df = get_quarterly_cash_burn(ticker, n=4)
if burn_df is None or burn_df.empty:
    st.info("No quarterly cash-flow data available for this ticker.")
else:
    def _fmt_usd_burn(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        a = abs(v)
        sign = "-" if v < 0 else ""
        if a >= 1e12:
            return f"{sign}${a / 1e12:.2f}T"
        if a >= 1e9:
            return f"{sign}${a / 1e9:.2f}B"
        if a >= 1e6:
            return f"{sign}${a / 1e6:.1f}M"
        return f"{sign}${a:,.0f}"

    # Reverse column order: oldest quarter on the left, most recent on the right.
    burn_display = burn_df.iloc[:, ::-1].map(_fmt_usd_burn)
    st.dataframe(burn_display, use_container_width=True)

    # Headline: average quarterly burn (only over quarters where FCF<0) and
    # implied runway against the most recent Cash + ST Investments.
    try:
        fcf_row = burn_df.loc["Free Cash Flow"].dropna().astype(float)
        burn_qtrs = fcf_row[fcf_row < 0]
        if len(burn_qtrs) > 0:
            avg_burn_q = float(-burn_qtrs.mean())  # positive number
            avg_burn_str = _fmt_usd_burn(avg_burn_q)
            runway_str = "n/a"
            if bs_df is not None and "Cash + ST Investments" in bs_df.index:
                latest_cash = bs_df.loc["Cash + ST Investments"].dropna()
                if len(latest_cash) > 0:
                    latest_cash_val = float(latest_cash.iloc[0])  # newest is first column
                    if avg_burn_q > 0:
                        qtrs_runway = latest_cash_val / avg_burn_q
                        runway_str = f"~{qtrs_runway:.1f} quarters ({qtrs_runway * 3:.0f} months)"
            st.caption(
                f"📉 **Average burn** over the {len(burn_qtrs)} cash-burning "
                f"quarter(s) of the last 4: **{avg_burn_str} / quarter**. "
                f"Implied runway at current cash on hand: **{runway_str}**. "
                "Burn = −Free Cash Flow when FCF is negative; runway uses the "
                "most recent Cash + ST Investments balance."
            )
        else:
            st.caption(
                "✅ Free cash flow was positive in every quarter shown — no "
                "burn (the company is generating cash, not consuming it)."
            )
    except Exception:
        pass


# --- Latest news ------------------------------------------------------------
st.divider()
st.subheader("Latest News")

news_items = get_company_news(ticker, n=3)
if not news_items:
    st.info(
        "No trader-relevant headlines found. Yahoo's news feed may be empty "
        "or only returned listicle/aggregator items, which are filtered out."
    )
else:
    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    for it in news_items:
        title = it["title"]
        link = it["link"]
        pub = it["publisher"] or "—"
        t = it["timestamp"]
        if t is not None:
            try:
                age = now_utc - t
                hrs = age.total_seconds() / 3600.0
                if hrs < 1:
                    age_str = f"{int(age.total_seconds() / 60)}m ago"
                elif hrs < 24:
                    age_str = f"{int(hrs)}h ago"
                else:
                    age_str = f"{int(hrs / 24)}d ago"
            except Exception:
                age_str = ""
            when_str = t.strftime("%b %d, %Y")
            meta = f"*{pub}* · {when_str}" + (f" · {age_str}" if age_str else "")
        else:
            meta = f"*{pub}*"
        if link:
            st.markdown(f"**[{title}]({link})**  \n{meta}")
        else:
            st.markdown(f"**{title}**  \n{meta}")
st.caption(
    "Filtered to remove listicles ('5 Stocks to Buy'), videos, and "
    "aggregator clickbait. Source: Yahoo Finance news feed."
)

st.caption(
    f"Data: Yahoo Finance via yfinance · Last bar: "
    f"{hist_full.index[-1].strftime('%Y-%m-%d')} · "
    f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
