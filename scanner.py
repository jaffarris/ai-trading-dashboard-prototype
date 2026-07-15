import pandas as pd

from data_provider import get_analysis_data, get_intraday_data
from indicators import (assess_data_quality, calculate_indicators, latest_trading_day,
                        percentage_move)
from risk_engine import (analyze_levels, analyze_market_context, analyze_trend, analyze_volume,
                         build_conservative_decision, classify_move_strength)

WATCHLIST = ["TSLA", "NVDA", "BABA", "AAPL", "META", "AMD", "MSFT", "SPY", "QQQ", "RIVN"]


def _thresholds(ticker: str) -> tuple[float, float, float]:
    if ticker in {"SPY", "QQQ"}:
        return .25, .50, 1.00
    if ticker in {"NVDA", "AMD", "TSLA", "META"}:
        return .50, 1.00, 1.50
    return .75, 1.50, 2.00


def _scan_symbol(ticker: str, spy: pd.DataFrame, qqq: pd.DataFrame) -> dict | None:
    try:
        data = get_analysis_data(ticker)
        if data.empty or len(data) < 20:
            return None
        data = latest_trading_day(calculate_indicators(data))
        latest = data.iloc[-1]
        market = analyze_market_context(data, data if ticker == "SPY" else spy,
                                        data if ticker == "QQQ" else qqq, qqq, "QQQ")
        trend, volume = analyze_trend(data), analyze_volume(data)
        levels = analyze_levels(data, "5m", volume)
        quality = assess_data_quality(data, "5m")
        analysis = build_conservative_decision(ticker, data, "5m", market, trend, volume,
                                               levels, quality)
        session_move = percentage_move(float(latest["Close"]), float(data["Open"].iloc[0]))
        pullback = percentage_move(float(latest["Close"]), float(data["High"].tail(20).max()))
        early, moderate, major = _thresholds(ticker)
        absolute_move = abs(session_move)
        alert = "Major move" if absolute_move >= major else "Significant move" if absolute_move >= moderate else "Movement alert" if absolute_move >= early else ""
        return {
            "Ticker": ticker, "Price": round(float(latest["Close"]), 2),
            "Move %": round(session_move, 2), "Pullback %": round(pullback, 2),
            "Move Strength": classify_move_strength(ticker, data, volume)["label"],
            "Score": analysis["trade_score"], "Stock Score": analysis["stock_score"],
            "Trade Score": analysis["trade_score"], "Trade Readiness": analysis["trade_readiness"],
            "Signal": analysis["signal"], "Trade Quality": analysis["grade"],
            "Pattern": analysis.get("pattern", {}).get("name", "No clear pattern"),
            "Top Reasons": analysis.get("top_reasons", [])[:3],
            "Relative Strength": round(market["vs_spy"], 2),
            "Trend Strength": analysis["components"]["Trend Strength"],
            "Momentum": analysis["components"]["Momentum"],
            "Volume Score": analysis["components"]["Volume Intelligence"],
            "Risk Reward": round(analysis["expected"]["risk_reward"], 1),
            "Confidence": analysis["confidence"],
            "Expected Continuation": round(analysis["expected"]["continuation_move"], 2),
            "RVOL": round(volume["rvol"], 2) if volume["rvol"] else None,
            "Alert": alert,
        }
    except Exception as exc:
        return {"Ticker": ticker, "Price": None, "Move %": None, "Pullback %": None,
                "Move Strength": "Error", "Score": 0, "Signal": "Error", "Pattern": "",
                "Trade Quality": "F", "Stock Score": 0, "Trade Score": 0,
                "Trade Readiness": 0, "Top Reasons": [], "Relative Strength": None,
                "Trend Strength": 0, "Momentum": 0, "Volume Score": 0,
                "Risk Reward": 0.0, "Confidence": 0, "Expected Continuation": 0.0,
                "RVOL": None, "Alert": str(exc)[:70]}


def scan_watchlist() -> pd.DataFrame:
    spy = pd.DataFrame()
    qqq = pd.DataFrame()
    try:
        spy = latest_trading_day(calculate_indicators(get_analysis_data("SPY")))
        qqq = latest_trading_day(calculate_indicators(get_analysis_data("QQQ")))
    except Exception:
        pass

    # Keep requests serial in the free cloud container. Concurrent SSL/network
    # work inside Streamlit's script worker has caused native process crashes.
    results = [_scan_symbol(symbol, spy, qqq) for symbol in WATCHLIST]
    results = [result for result in results if result is not None]
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(
        ["Trade Readiness", "Trade Score", "Stock Score", "Move %"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
