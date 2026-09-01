import math
import re
import time
import random
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="RSI Bullish Divergence — NSE F&O",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "RSI Bullish + Bearish Divergence — NSE F&O Stocks"
NSE_FO_LOTS = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
FALLBACK_URL = "https://optionperks.com/lot_size"

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

TF_MAP = {
    "Monthly": {"interval":"1mo", "period":"10y"},
    "Weekly":  {"interval":"1wk", "period":"10y"},
    "Daily":   {"interval":"1d",  "period":"10y"},
    "4 Hour":  {"interval":"1h",  "period":"730d"},
}

def clean_symbol(x):
    return str(x).strip().upper()

@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stocks():
    """Current NSE individual F&O stock universe.
    Primary: official NSE permitted lot-size CSV.
    Fallback: public F&O lot-size table.
    Indices are always excluded.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }

    # 1) Official NSE
    try:
        r = requests.get(NSE_FO_LOTS, headers=headers, timeout=25)
        r.raise_for_status()
        text = r.text
        df = pd.read_csv(StringIO(text), skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]

        # NSE file commonly has "SYMBOL" or "Symbol"
        sym_col = next((c for c in df.columns if c.strip().upper() == "SYMBOL"), None)
        if sym_col is None:
            # Some versions have first column text such as "UNDERLYING"
            for c in df.columns:
                vals = df[c].astype(str)
                if vals.str.fullmatch(r"[A-Z0-9&\-]+", na=False).mean() > 0.6:
                    sym_col = c
                    break
        if sym_col is None:
            raise ValueError("Symbol column not found in NSE F&O lot-size file.")

        syms = sorted({
            clean_symbol(x)
            for x in df[sym_col].dropna()
            if clean_symbol(x) and clean_symbol(x) not in INDEX_SYMBOLS
        })
        # sanity check
        if len(syms) >= 150:
            return syms, "Official NSE permitted lot-size file"
    except Exception:
        pass

    # 2) Fallback table
    try:
        html = requests.get(
            FALLBACK_URL,
            headers={"User-Agent":"Mozilla/5.0"},
            timeout=25
        ).text
        tables = pd.read_html(StringIO(html))
        best = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("symbol" in c for c in cols):
                best = t
                break
        if best is None:
            raise ValueError("Fallback F&O table not found.")
        sym_col = next(c for c in best.columns if "symbol" in str(c).lower())
        syms = sorted({
            clean_symbol(x)
            for x in best[sym_col].dropna()
            if clean_symbol(x) and clean_symbol(x) not in INDEX_SYMBOLS
        })
        if len(syms) < 150:
            raise ValueError("Fallback returned too few symbols.")
        return syms, "Fallback F&O lot-size table"
    except Exception as e:
        raise RuntimeError(
            "F&O universe load nahi ho paya. Thodi der baad Refresh Universe try karein."
        ) from e

def rsi_wilder(close, period=14):
    s = pd.to_numeric(close, errors="coerce")
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100/(1+rs))
    out = out.where(al != 0, 100.0)
    return out

def pivot_lows(series, left=2, right=2):
    a = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = []
    for i in range(left, len(a)-right):
        if not np.isfinite(a[i]):
            continue
        if np.all(a[i] < a[i-left:i]) and np.all(a[i] <= a[i+1:i+right+1]):
            out.append(i)
    return out

def pivot_highs(series, left=2, right=2):
    a = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = []
    for i in range(left, len(a)-right):
        if not np.isfinite(a[i]):
            continue
        if np.all(a[i] > a[i-left:i]) and np.all(a[i] >= a[i+1:i+right+1]):
            out.append(i)
    return out

def flatten(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def to_4h(df):
    if df.empty:
        return df
    x = df.copy()
    idx = pd.to_datetime(x.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata")
    x.index = idx
    return x.resample(
        "4h", origin="start_day", offset="9h15min"
    ).agg({
        "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"
    }).dropna(subset=["Close"])

def remove_incomplete(df, timeframe):
    if df.empty:
        return df
    x = df.copy()
    idx = pd.to_datetime(x.index)
    now = pd.Timestamp.now(tz="Asia/Kolkata")

    if timeframe == "Monthly":
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        x.index = idx
        x = x[
            (x.index.year < now.year) |
            ((x.index.year == now.year) & (x.index.month < now.month))
        ]

    elif timeframe == "Weekly":
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        x.index = idx
        if len(x) and x.index[-1].to_period("W") == now.tz_localize(None).to_period("W"):
            x = x.iloc[:-1]

    elif timeframe == "Daily":
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        x.index = idx
        x = x[x.index.date < now.date()]

    elif timeframe == "4 Hour":
        if idx.tz is None:
            idx = idx.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            idx = idx.tz_convert("Asia/Kolkata")
        x.index = idx
        if len(x) and x.index[-1] + pd.Timedelta(hours=4) > now:
            x = x.iloc[:-1]

    return x.dropna(subset=["Close"])

def _current_trading_day():
    """IST calendar date string, used only as a cache-busting key so the
    cache naturally refreshes once per day (new key each day)."""
    return pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d")

def _yf_download_with_retry(ticker, period, interval, attempts=4):
    """Yahoo Finance ka unofficial API concurrent/high-frequency requests par
    randomly rate-limit/empty-response deta hai. Retry + exponential backoff
    + jitter se transient failures handle hote hain, taaki wahi symbol
    baar-baar 'no data' na ban jaaye."""
    last_exc = None
    for i in range(attempts):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=20,
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last_exc = e
        # backoff before retrying (skip sleep after the last attempt)
        if i < attempts - 1:
            time.sleep((1.5 ** i) + random.uniform(0, 0.75))
    return pd.DataFrame()

@st.cache_data(ttl=6*3600, show_spinner=False)
def download_tf(symbol, timeframe, years, trading_day):
    """trading_day is only used as a cache key (IST date string) so results
    for a given symbol+timeframe+years are fetched once per day and then
    served from cache — same-day reruns give identical, stable results
    instead of hitting Yahoo again and risking a different rate-limit
    outcome each time.

    IMPORTANT: on failure we RAISE instead of returning an empty
    DataFrame. Streamlit's cache_data does not cache a call that raises,
    so a symbol that fails (rate-limited / no data) is retried fresh on
    the next scan instead of being stuck as "no data" for the rest of
    the cache TTL.
    """
    info = TF_MAP[timeframe]
    period = info["period"]
    if timeframe in ("Monthly","Weekly","Daily"):
        req_years = max(5, int(math.ceil(years))+5)
        period = f"{min(req_years,10)}y"

    df = _yf_download_with_retry(symbol + ".NS", period, info["interval"])
    if df is None or df.empty:
        raise RuntimeError(f"no data for {symbol}")

    df = flatten(df)
    need = ["Open","High","Low","Close","Volume"]
    if not all(c in df.columns for c in need):
        raise RuntimeError(f"missing OHLCV columns for {symbol}")

    df = df[need].dropna(subset=["Close"])
    if timeframe == "4 Hour":
        df = to_4h(df)
    result = remove_incomplete(df, timeframe)
    if result.empty:
        raise RuntimeError(f"empty after cleanup for {symbol}")
    return result

def signal_cutoff(years):
    return pd.Timestamp.now().tz_localize(None) - pd.DateOffset(
        years=max(1, int(math.ceil(years)))
    )


def detect_signals(
    df, timeframe, divergence_mode, rsi_period, left, right, search_bars, min_gap,
    min_rsi_diff, min_price_diff_pct, strict_extreme, years
):
    if len(df) < rsi_period + left + right + 5:
        return []

    x = df.copy()
    x["RSI"] = rsi_wilder(x["Close"], rsi_period)
    cutoff = signal_cutoff(years)
    out = []

    def fmt(ts):
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("Asia/Kolkata")
        return (
            t.strftime("%Y-%m-%d %H:%M")
            if timeframe == "4 Hour"
            else t.strftime("%Y-%m-%d")
        )

    def add_bullish():
        piv = pivot_lows(x["Close"], left, right)
        if len(piv) < 2:
            return
        for k in range(1, len(piv)):
            cur = piv[k]
            signal_i = cur + right
            if signal_i >= len(x):
                continue

            prev_candidates = [
                p for p in piv[:k]
                if min_gap <= (cur-p) <= search_bars
            ]
            if not prev_candidates:
                continue
            prev = prev_candidates[-1]

            cc = float(x["Close"].iloc[cur])
            pc = float(x["Close"].iloc[prev])
            cr = float(x["RSI"].iloc[cur])
            pr = float(x["RSI"].iloc[prev])

            if not (np.isfinite(cr) and np.isfinite(pr) and pc != 0):
                continue

            price_diff = (pc-cc)/pc*100
            rsi_diff = cr-pr

            ok = (
                cc < pc and
                cr > pr and
                price_diff >= min_price_diff_pct and
                rsi_diff >= min_rsi_diff
            )

            if strict_extreme:
                ok = ok and cr < 30 and pr < 30

            sig_ts = pd.Timestamp(x.index[signal_i])
            sig_cmp = sig_ts.tz_localize(None) if sig_ts.tzinfo is not None else sig_ts

            if ok and sig_cmp >= cutoff:
                out.append({
                    "Type": "BULLISH",
                    "Timeframe": timeframe,
                    "Signal_Date": fmt(x.index[signal_i]),
                    "Current_Pivot_Date": fmt(x.index[cur]),
                    "Current_Close": round(cc,2),
                    "Current_RSI": round(cr,2),
                    "Previous_Pivot_Date": fmt(x.index[prev]),
                    "Previous_Close": round(pc,2),
                    "Previous_RSI": round(pr,2),
                    "Price_Difference_%": round(price_diff,2),
                    "RSI_Difference": round(rsi_diff,2),
                    "Extreme_Check": "BOTH <30" if cr < 30 and pr < 30 else "NO"
                })

    def add_bearish():
        piv = pivot_highs(x["Close"], left, right)
        if len(piv) < 2:
            return
        for k in range(1, len(piv)):
            cur = piv[k]
            signal_i = cur + right
            if signal_i >= len(x):
                continue

            prev_candidates = [
                p for p in piv[:k]
                if min_gap <= (cur-p) <= search_bars
            ]
            if not prev_candidates:
                continue
            prev = prev_candidates[-1]

            cc = float(x["Close"].iloc[cur])
            pc = float(x["Close"].iloc[prev])
            cr = float(x["RSI"].iloc[cur])
            pr = float(x["RSI"].iloc[prev])

            if not (np.isfinite(cr) and np.isfinite(pr) and pc != 0):
                continue

            price_diff = (cc-pc)/pc*100
            rsi_diff = pr-cr

            ok = (
                cc > pc and
                cr < pr and
                price_diff >= min_price_diff_pct and
                rsi_diff >= min_rsi_diff
            )

            if strict_extreme:
                ok = ok and cr > 70 and pr > 70

            sig_ts = pd.Timestamp(x.index[signal_i])
            sig_cmp = sig_ts.tz_localize(None) if sig_ts.tzinfo is not None else sig_ts

            if ok and sig_cmp >= cutoff:
                out.append({
                    "Type": "BEARISH",
                    "Timeframe": timeframe,
                    "Signal_Date": fmt(x.index[signal_i]),
                    "Current_Pivot_Date": fmt(x.index[cur]),
                    "Current_Close": round(cc,2),
                    "Current_RSI": round(cr,2),
                    "Previous_Pivot_Date": fmt(x.index[prev]),
                    "Previous_Close": round(pc,2),
                    "Previous_RSI": round(pr,2),
                    "Price_Difference_%": round(price_diff,2),
                    "RSI_Difference": round(rsi_diff,2),
                    "Extreme_Check": "BOTH >70" if cr > 70 and pr > 70 else "NO"
                })

    if divergence_mode in ("Bullish", "Both"):
        add_bullish()
    if divergence_mode in ("Bearish", "Both"):
        add_bearish()

    return out

def scan_symbol(symbol, params):
    (
        timeframe, divergence_mode, years, rsi_period, left, right, search_bars, min_gap,
        min_rsi_diff, min_price_diff_pct, strict_extreme
    ) = params

    # Chhota random stagger — sab threads ek hi instant pe Yahoo ko hit
    # nahi karte, jisse rate-limit / empty-response chance kam ho jaata hai.
    time.sleep(random.uniform(0, 0.4))

    try:
        d = download_tf(symbol, timeframe, years, _current_trading_day())
    except Exception as e:
        return symbol, [], f"no data ({e})"

    if d.empty:
        return symbol, [], "no data"

    sigs = detect_signals(
        d, timeframe, divergence_mode, rsi_period, left, right, search_bars, min_gap,
        min_rsi_diff, min_price_diff_pct, strict_extreme, years
    )

    rows = [{"Symbol": symbol, **s} for s in sigs]
    return symbol, rows, None

st.markdown("""
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-size: 1.4rem; }
.stButton button { width: 100%; min-height: 3rem; font-weight: 700; }
@media (max-width: 700px) {
    .block-container { padding-left: .7rem; padding-right: .7rem; }
}
</style>
""", unsafe_allow_html=True)

st.title("📈 RSI Bullish + Bearish Divergence — NSE F&O Stocks")
st.caption(
    "Mobile Web Scanner • Individual NSE F&O stocks only • "
    "Bullish + Bearish • Monthly / Weekly / Daily / 4 Hour • closed-candle confirmation"
)

try:
    fno_stocks, universe_source = get_fno_stocks()
except Exception as e:
    st.error(str(e))
    fno_stocks = []
    universe_source = "Unavailable"

m1, m2, m3 = st.columns(3)
m1.metric("F&O stocks loaded", len(fno_stocks))
m2.metric("Indices included", "0")
m3.metric("Universe", "NSE F&O only")

st.caption(f"Universe source: {universe_source}")

with st.sidebar:
    st.header("Scanner Settings")

    timeframe = st.selectbox(
        "Timeframe",
        ["Monthly","Weekly","Daily","4 Hour"],
        index=0
    )

    divergence_mode = st.selectbox(
        "Divergence",
        ["Both", "Bullish", "Bearish"],
        index=0
    )

    years = st.number_input(
        "Backtest / signal history years",
        min_value=0.1,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    if timeframe == "4 Hour" and years > 2:
        st.warning("4 Hour data provider usually ~2 years tak hi reliable history deta hai.")

    st.subheader("RSI / Pivot")
    rsi_period = st.number_input("RSI period", 2, 50, 14)
    left = st.number_input("Pivot left", 1, 10, 2)
    right = st.number_input("Pivot right", 0, 10, 0)
    search_bars = st.number_input("Previous pivot search bars", 5, 200, 36)
    min_gap = st.number_input("Min bars between pivots", 1, 50, 2)
    min_rsi_diff = st.number_input("Min RSI difference", 0.0, 30.0, 2.0, step=0.1)
    min_price_diff_pct = st.number_input("Min price difference %", 0.0, 30.0, 0.1, step=0.1)

    strict_extreme = st.checkbox(
        "STRICT RSI EXTREME — Bullish both <30 / Bearish both >70",
        value=False
    )

    workers = st.slider("Parallel workers", 1, 8, 2)
    st.caption(
        "Kam workers = Yahoo Finance rate-limit hone ka chance kam, "
        "results zyada consistent aate hain."
    )

    if st.button("🔄 Refresh F&O Universe"):
        get_fno_stocks.clear()
        download_tf.clear()
        st.rerun()

run = st.button(
    f"▶ RUN SCANNER — {len(fno_stocks)} F&O STOCKS",
    type="primary",
    disabled=(len(fno_stocks) == 0)
)

if run:
    params = (
        timeframe, divergence_mode, float(years), int(rsi_period), int(left), int(right),
        int(search_bars), int(min_gap), float(min_rsi_diff),
        float(min_price_diff_pct), bool(strict_extreme)
    )

    progress = st.progress(0)
    status = st.empty()
    result_area = st.empty()

    rows = []
    errors = 0
    completed = 0
    error_detail = []

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futures = {
            ex.submit(scan_symbol, sym, params): sym
            for sym in fno_stocks
        }

        for fut in as_completed(futures):
            sym = futures[fut]
            completed += 1
            try:
                _, found, err = fut.result()
                rows.extend(found)
                if err:
                    errors += 1
                    error_detail.append((sym, err))
            except Exception as e:
                errors += 1
                error_detail.append((sym, f"exception: {e}"))

            progress.progress(completed / max(1, len(fno_stocks)))
            status.info(
                f"{divergence_mode} {timeframe} scan: {completed}/{len(fno_stocks)} | "
                f"signals: {len(rows)} | data errors: {errors}"
            )

    progress.empty()
    status.empty()

    if error_detail:
        with st.expander(f"⚠ Data fetch failed for {len(error_detail)} symbol(s) — click to see which"):
            st.dataframe(
                pd.DataFrame(error_detail, columns=["Symbol", "Reason"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Yeh symbols is scan mein Yahoo se data nahi le paaye (rate-limit/timeout ke baad "
                "bhi 4 retries fail), isliye inke liye koi signal check hi nahi ho paaya — 'no signal' "
                "aur 'fetch failed' alag cheezein hain. Dobara RUN SCANNER dabao, failed symbols agli "
                "baar fresh retry honge (cache nahi hote)."
            )

    if rows:
        df = pd.DataFrame(rows)
        # ISO-like dates sort correctly as strings.
        df = df.sort_values(
            ["Signal_Date","Symbol"],
            ascending=[False, True]
        ).reset_index(drop=True)

        st.success(
            f"Complete — {len(df)} signals | "
            f"{divergence_mode} | {timeframe} | {len(fno_stocks)} F&O stocks scanned"
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=520
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Results CSV",
            data=csv,
            file_name=f"rsi_divergence_{divergence_mode.lower()}_fno_{timeframe.replace(' ','_').lower()}.csv",
            mime="text/csv"
        )
    else:
        st.info(
            f"No signals found. {len(fno_stocks)} F&O stocks scan hue. "
            "Filters ko relax karke dobara try kar sakte hain."
        )

st.divider()
st.caption(
    "Note: Bullish = lower price low + higher RSI low; Bearish = higher price high + lower RSI high. F&O eligibility exchange ke saath change ho sakti hai. "
    "App current permitted-lot-size universe refresh karta hai, aur indices ko exclude karta hai."
)
