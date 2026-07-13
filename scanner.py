from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from data_provider import get_analysis_data, get_intraday_data
from indicators import (build_trade_analysis, calculate_indicators, latest_trading_day,
                        move_strength, percentage_move)

WATCHLIST = ["TSLA", "NVDA", "BABA", "AAPL", "META", "AMD", "MSFT", "SPY", "QQQ", "RIVN"]


def _thresholds(ticker: str) -> tuple[float, float, float]:
    if ticker in {"SPY", "QQQ"}:
        return .25, .50, 1.00
    if ticker in {"NVDA", "AMD", "TSLA", "META"}:
        return .50, 1.00, 1.50
    return .75, 1.50, 2.00


def _scan_symbol(ticker: str, spy_change: float | None) -> dict | None:
    try:
        data = get_analysis_data(ticker)
        if data.empty or len(data) < 20:
            return None
        data = latest_trading_day(calculate_indicators(data))
        latest = data.iloc[-1]
        analysis = build_trade_analysis(data, spy_change_pct=spy_change)
        session_move = percentage_move(float(latest["Close"]), float(data["Open"].iloc[0]))
        pullback = percentage_move(float(latest["Close"]), float(data["High"].tail(20).max()))
        early, moderate, major = _thresholds(ticker)
        absolute_move = abs(session_move)
        alert = "Major move" if absolute_move >= major else "Significant move" if absolute_move >= moderate else "Movement alert" if absolute_move >= early else ""
        return {
            "Ticker": ticker, "Price": round(float(latest["Close"]), 2),
            "Move %": round(session_move, 2), "Pullback %": round(pullback, 2),
            "Move Strength": move_strength(absolute_move), "Score": analysis["score"],
            "Signal": analysis["signal"], "Pattern": analysis["pattern"],
            "RVOL": round(float(latest["RVOL"]), 2) if pd.notna(latest["RVOL"]) else None,
            "Alert": alert,
        }
    except Exception as exc:
        return {"Ticker": ticker, "Price": None, "Move %": None, "Pullback %": None,
                "Move Strength": "Error", "Score": 0, "Signal": "Error", "Pattern": "",
                "RVOL": None, "Alert": str(exc)[:70]}


def scan_watchlist() -> pd.DataFrame:
    spy_change = None
    try:
        spy = get_intraday_data("SPY")
        if not spy.empty:
            spy_change = percentage_move(float(spy["Close"].iloc[-1]), float(spy["Open"].iloc[0]))
    except Exception:
        pass

    # Network-bound symbol requests run together instead of serially.
    # Four workers keep the free cloud container responsive while still making
    # the network-bound scan substantially faster than a serial loop.
    with ThreadPoolExecutor(max_workers=min(4, len(WATCHLIST))) as pool:
        results = list(pool.map(lambda symbol: _scan_symbol(symbol, spy_change), WATCHLIST))
    results = [result for result in results if result is not None]
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(["Score", "Move %"], ascending=[False, False]).reset_index(drop=True)
