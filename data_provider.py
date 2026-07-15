"""Market data providers with a resilient Yahoo intraday fallback."""
import os
import time
from datetime import date
from functools import lru_cache

import pandas as pd
import requests
from requests.exceptions import SSLError

# Yahoo's public chart endpoint avoids the native curl-cffi transport used by
# yfinance, which is unnecessary here and can destabilize small cloud workers.
_HTTP_SESSION = requests.Session()


def to_chart_timestamp(index, interval_seconds: int) -> int:
    """Encode exchange-local wall time for chart libraries that display UTC."""
    stamp = pd.Timestamp(index)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    wall_clock_utc = stamp.tz_localize("UTC")
    return (int(wall_clock_utc.timestamp()) // interval_seconds) * interval_seconds


def get_intraday_data(ticker: str, period: str = "1d", interval: str = "5m", prepost: bool = True) -> pd.DataFrame:
    """Fetch intraday candles from Yahoo's public chart endpoint."""
    # The UI refreshes every 15 seconds, so a shorter network cache only creates
    # duplicate Yahoo requests without making the visible chart any fresher.
    bucket = int(time.time() // 15)
    return _get_intraday_cached(ticker.upper(), period, interval, prepost, bucket).copy()


def get_analysis_data(ticker: str, interval: str = "5m", prepost: bool = True) -> pd.DataFrame:
    """Return live candles plus enough free Yahoo history for time-adjusted RVOL.

    Yahoo limits one-minute history, so 1m/3m use five days. Higher intervals use
    one month (normally 20 sessions). The larger baseline is refreshed every five
    minutes while today's candles retain the normal five-second refresh cadence.
    """
    ticker = ticker.upper()
    history_period = "5d" if interval in {"1m", "3m"} else "1mo"
    history_bucket = int(time.time() // 300)
    history = _get_intraday_cached(ticker, history_period, interval, prepost, history_bucket).copy()
    current = get_intraday_data(ticker, period="1d", interval=interval, prepost=prepost)
    if history.empty:
        return current
    if current.empty:
        return history
    combined = pd.concat([history, current])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


@lru_cache(maxsize=256)
def _get_intraday_cached(ticker: str, period: str, interval: str, prepost: bool, _bucket: int) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    yahoo_interval = "1m" if interval == "3m" else interval
    request_options = {
        "params": {"range": period, "interval": yahoo_interval, "includePrePost": str(prepost).lower(), "events": "div,splits"},
        "headers": {"User-Agent": "Mozilla/5.0"}, "timeout": 12,
    }
    try:
        response = _HTTP_SESSION.get(url, **request_options)
    except SSLError:
        # Some fresh Windows Python installs cannot see the Windows root store.
        # This fallback is restricted to Yahoo's public, read-only chart endpoint.
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = _HTTP_SESSION.get(url, verify=False, **request_options)
    response.raise_for_status()
    payload = response.json().get("chart", {})
    if payload.get("error") or not payload.get("result"):
        return pd.DataFrame()
    result = payload["result"][0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps or not quotes:
        return pd.DataFrame()
    frame = pd.DataFrame({
        "Open": quotes.get("open", []), "High": quotes.get("high", []),
        "Low": quotes.get("low", []), "Close": quotes.get("close", []),
        "Volume": quotes.get("volume", []),
    }, index=pd.to_datetime(timestamps, unit="s", utc=True))
    timezone = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    frame.index = frame.index.tz_convert(timezone)
    frame.index.name = "Datetime"
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    if interval == "3m" and not frame.empty:
        frame = frame.resample("3min", origin="start_day", label="left", closed="left").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
        }).dropna(subset=["Open", "High", "Low", "Close"])
    return frame


def get_5min_data(ticker: str, trading_date: str | None = None) -> pd.DataFrame:
    from polygon import RESTClient
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY environment variable is not set.")

    selected_date = trading_date or date.today().isoformat()
    client = RESTClient(api_key)
    aggs = client.get_aggs(
        ticker=ticker,
        multiplier=5,
        timespan="minute",
        from_=selected_date,
        to=selected_date,
        adjusted=True,
        sort="asc",
        limit=5000,
    )

    rows = [
        {
            "Datetime": pd.to_datetime(bar.timestamp, unit="ms", utc=True),
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume,
        }
        for bar in aggs
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Datetime")
