"""Conservative, explainable trade-selection and risk-management engine.

All percentages and confidence values in this module are heuristic technical
estimates. They are not historical win probabilities unless a separate
validation result explicitly says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from indicators import (
    calculate_support_resistance,
    detect_candlestick_patterns,
    detect_candlestick_statuses,
    percentage_move,
)


SECTOR_ETFS = {
    "NVDA": "SOXX", "AMD": "SOXX", "MSFT": "XLK", "AAPL": "XLK",
    "META": "XLC", "TSLA": "XLY", "RIVN": "XLY", "BABA": "FXI",
    "SPY": "SPY", "QQQ": "QQQ",
}

VOLATILITY_GROUPS = {
    "INDEX": {"SPY", "QQQ"},
    "MOMENTUM": {"NVDA", "AMD", "TSLA", "META"},
    "HIGH_BETA": {"RIVN", "BABA"},
}


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _interval_minutes(data: pd.DataFrame, interval: str) -> int:
    configured = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30}.get(interval)
    if configured:
        return configured
    if len(data) < 2:
        return 5
    diffs = data.index.to_series().diff().dt.total_seconds().div(60).dropna()
    diffs = diffs[(diffs > 0) & (diffs <= 60)]
    return max(1, int(round(_finite(diffs.median(), 5))))


def candle_is_complete(data: pd.DataFrame, interval: str, now=None) -> bool:
    if data.empty:
        return False
    stamp = pd.Timestamp(data.index[-1])
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=stamp.tz)
    if stamp.tzinfo is not None and current.tzinfo is None:
        current = current.tz_localize(stamp.tzinfo)
    elif stamp.tzinfo is None and current.tzinfo is not None:
        current = current.tz_localize(None)
    return current >= stamp + pd.Timedelta(minutes=_interval_minutes(data, interval))


def session_label(data: pd.DataFrame) -> str:
    if data.empty:
        return "NO DATA"
    stamp = pd.Timestamp(data.index[-1])
    minute = stamp.hour * 60 + stamp.minute
    if minute < 570:
        return "PREMARKET"
    if minute < 960:
        return "REGULAR"
    return "AFTER-HOURS"


def session_move(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    return percentage_move(_finite(data["Close"].iloc[-1]), _finite(data["Open"].iloc[0]))


def analyze_market_context(stock_data: pd.DataFrame, spy_data: pd.DataFrame,
                           qqq_data: pd.DataFrame, sector_data: pd.DataFrame,
                           sector_symbol: str) -> dict:
    stock_move = session_move(stock_data)
    spy_move, qqq_move, sector_move = map(session_move, (spy_data, qqq_data, sector_data))

    def direction(move: float) -> int:
        return 1 if move >= .10 else -1 if move <= -.10 else 0

    votes = [direction(spy_move), direction(qqq_move)]
    if sector_symbol not in {"SPY", "QQQ"} and not sector_data.empty:
        votes.append(direction(sector_move))
    bullish, bearish = votes.count(1), votes.count(-1)
    if bullish >= 2 and bearish == 0:
        status = "Bullish"
    elif bearish >= 2 and bullish == 0:
        status = "Bearish"
    elif bullish and bearish:
        status = "Conflicted"
    else:
        status = "Neutral"

    vs_spy, vs_qqq = stock_move-spy_move, stock_move-qqq_move
    if vs_spy >= .25 and vs_qqq >= .20:
        leadership = "Market leader"
    elif vs_spy <= -.25 and vs_qqq <= -.20:
        leadership = "Market laggard"
    else:
        leadership = "In line with market"
    alignment_votes = []
    stock_direction = direction(stock_move)
    for benchmark_move in (spy_move, qqq_move, sector_move):
        benchmark_direction = direction(benchmark_move)
        alignment_votes.append(50 if not stock_direction or not benchmark_direction else 100 if stock_direction == benchmark_direction else 0)
    alignment_score = int(round(sum(alignment_votes) / max(len(alignment_votes), 1)))
    def condition(move: float) -> str:
        return "Bullish" if move >= .10 else "Bearish" if move <= -.10 else "Weak / neutral"
    return {
        "status": status, "stock_move": stock_move, "spy_move": spy_move,
        "qqq_move": qqq_move, "sector_move": sector_move,
        "sector_symbol": sector_symbol, "vs_spy": vs_spy, "vs_qqq": vs_qqq,
        "leadership": leadership,
        "alignment_score": alignment_score,
        "spy_status": condition(spy_move), "qqq_status": condition(qqq_move),
        "sector_status": condition(sector_move),
        "benchmarks_complete": not spy_data.empty and not qqq_data.empty and not sector_data.empty,
        "summary": (
            f"Stock {stock_move:+.2f}% | SPY {spy_move:+.2f}% | QQQ {qqq_move:+.2f}% | "
            f"{sector_symbol} {sector_move:+.2f}% | vs SPY {vs_spy:+.2f}% | "
            f"vs QQQ {vs_qqq:+.2f}% | {leadership} | Alignment {alignment_score}/100"
        ),
    }


def analyze_trend(data: pd.DataFrame) -> dict:
    if data.empty:
        return {"direction": "Neutral", "strength": 0, "ema_alignment": "Unavailable",
                "vwap_position": "Unavailable", "structure": "Unavailable",
                "range_location": 50.0, "votes": 0}
    last = data.iloc[-1]
    close = _finite(last["Close"])
    ema9, ema20, vwap = map(_finite, (last.get("EMA9"), last.get("EMA20"), last.get("VWAP")))
    ema9_slope, ema20_slope = map(_finite, (last.get("EMA9_SLOPE"), last.get("EMA20_SLOPE")))
    vwap_slope = _finite(data["VWAP"].diff(3).iloc[-1]) if "VWAP" in data and len(data) >= 4 else 0.0
    recent = data.tail(6)
    higher_highs = int((recent["High"].diff() > 0).sum())
    higher_lows = int((recent["Low"].diff() > 0).sum())
    lower_highs = int((recent["High"].diff() < 0).sum())
    lower_lows = int((recent["Low"].diff() < 0).sum())
    bullish_structure = higher_highs >= 3 and higher_lows >= 3
    bearish_structure = lower_highs >= 3 and lower_lows >= 3
    votes = sum((close > vwap, ema9 > ema20, close > ema9, ema9_slope > 0, ema20_slope > 0, vwap_slope > 0, bullish_structure))
    bearish_votes = sum((close < vwap, ema9 < ema20, close < ema9, ema9_slope < 0, ema20_slope < 0, vwap_slope < 0, bearish_structure))
    net = votes-bearish_votes
    conflicted = votes >= 3 and bearish_votes >= 3
    direction = "Conflicted" if conflicted else "Bullish" if net >= 3 else "Bearish" if net <= -3 else "Neutral"
    strength = int(np.clip(abs(net) / 7 * 100 + (12 if (bullish_structure or bearish_structure) else 0), 0, 100))
    if direction == "Bullish":
        trend_label = "Strong Bullish Trend" if strength >= 80 else "Bullish Trend" if strength >= 55 else "Weak Bullish Trend"
    elif direction == "Bearish":
        trend_label = "Strong Bearish Trend" if strength >= 80 else "Bearish Trend" if strength >= 55 else "Weak Bearish Trend"
    elif direction == "Conflicted":
        trend_label = "Trend conflict"
    else:
        trend_label = "Range-bound"
    session_low, session_high = _finite(data["Low"].min()), _finite(data["High"].max())
    range_location = (close-session_low) / max(session_high-session_low, 1e-9) * 100
    structure = "Higher highs / higher lows" if bullish_structure else "Lower highs / lower lows" if bearish_structure else "Mixed structure"
    return {
        "direction": direction, "label": trend_label, "strength": strength,
        "ema_alignment": "Bullish" if ema9 > ema20 else "Bearish" if ema9 < ema20 else "Flat",
        "vwap_position": "Above VWAP" if close > vwap else "Below VWAP" if close < vwap else "At VWAP",
        "vwap_slope": vwap_slope, "structure": structure,
        "range_location": float(np.clip(range_location, 0, 100)),
        "votes": net,
    }


def analyze_location(data: pd.DataFrame, levels: dict) -> dict[str, Any]:
    """Describe trade location without pretending levels are exact to the cent."""
    last = data.iloc[-1]
    close = _finite(last["Close"])
    support_distance = abs(percentage_move(close, levels["support"]))
    resistance_distance = abs(percentage_move(close, levels["resistance"]))
    if levels.get("breakout_confirmed"):
        primary = "Breaking out"
    elif levels.get("breakdown_confirmed"):
        primary = "Breaking down"
    elif resistance_distance <= .15:
        primary = "At resistance"
    elif resistance_distance <= .45:
        primary = "Near resistance"
    elif support_distance <= .15:
        primary = "At support"
    elif support_distance <= .45:
        primary = "Near support"
    else:
        recent = data.tail(8)
        compression = _finite((recent["High"].max()-recent["Low"].min()) / max(close, 1e-9) * 100)
        primary = "Inside consolidation" if compression <= .75 else "Between key levels"
    tags = [primary,
            "Above VWAP" if close > _finite(last.get("VWAP")) else "Below VWAP",
            "Above EMA9" if close > _finite(last.get("EMA9")) else "Below EMA9",
            "Above EMA20" if close > _finite(last.get("EMA20")) else "Below EMA20"]
    return {"primary": primary, "tags": tags,
            "support_distance": support_distance, "resistance_distance": resistance_distance}


def calculate_component_scores(data: pd.DataFrame, direction: str, market: dict,
                               trend: dict, volume: dict, expected: dict) -> dict[str, int]:
    """Return auditable 0-100 setup components; none is a win probability."""
    last = data.iloc[-1]
    trend_score = int(np.clip(trend.get("strength", 0), 0, 100))
    recent = data["Close"].tail(6)
    recent_change = percentage_move(_finite(recent.iloc[-1]), _finite(recent.iloc[0])) if len(recent) > 1 else 0.0
    ema_slope = _finite(last.get("EMA9_SLOPE"))
    directional_momentum = recent_change if direction != "PUT" else -recent_change
    directional_slope = ema_slope if direction != "PUT" else -ema_slope
    scale = max(_finite(last.get("ATR14")), _finite(last["Close"])*.001)
    momentum_score = int(np.clip(50 + directional_momentum*18 + directional_slope/scale*35, 0, 100))
    rvol = volume.get("rvol", 0.0)
    pressure = volume.get("buying_pressure", 50.0) if direction != "PUT" else volume.get("selling_pressure", 50.0)
    volume_score = int(np.clip(20 + min(rvol, 3)/3*55 + (pressure-50)*.5 + min(volume.get("acceleration", 0), 2)*7.5, 0, 100))
    market_score = int(np.clip(market.get("alignment_score", 0), 0, 100))
    if (direction == "CALL" and market.get("leadership") == "Market leader") or (direction == "PUT" and market.get("leadership") == "Market laggard"):
        market_score = min(100, market_score+10)
    elif direction in {"CALL", "PUT"} and market.get("leadership") not in {"In line with market"}:
        market_score = max(0, market_score-10)
    risk_reward_score = int(np.clip(expected.get("risk_reward", 0)/3*100, 0, 100))
    favorable = expected.get("favorable_move", 0.0)
    adverse = abs(expected.get("adverse_move", 0.0))
    expected_score = int(np.clip(50 + (favorable-adverse)*55, 0, 100))
    return {"Trend Strength": trend_score, "Momentum": momentum_score,
            "Volume Intelligence": volume_score, "Market Alignment": market_score,
            "Risk Reward": risk_reward_score, "Expected Move": expected_score}


def _weighted_score(values: dict[str, int], weights: dict[str, int]) -> tuple[int, list[tuple[str, int]]]:
    """Return a bounded score and integer contributions that add to that score."""
    contributions = [(name, int(round(weights[name] * np.clip(value, 0, 100) / 100)))
                     for name, value in values.items()]
    score = int(np.clip(sum(points for _, points in contributions), 0, 100))
    difference = score - sum(points for _, points in contributions)
    if difference:
        contributions.append(("Rounding adjustment", difference))
    return score, contributions


def calculate_separated_scores(data: pd.DataFrame, direction: str, market: dict,
                                trend: dict, volume: dict, expected: dict,
                                levels: dict, pattern: dict | None,
                                last_complete: bool, confirmation_missing: bool,
                                conflicts: list[str], data_quality: dict) -> dict:
    """Score the stock, setup, and immediate entry readiness independently."""
    base = calculate_component_scores(data, direction, market, trend, volume, expected)
    directional_sign = -1 if direction == "PUT" else 1
    relative_edge = directional_sign * ((market.get("vs_spy", 0.0) + market.get("vs_qqq", 0.0)) / 2)
    relative_strength = int(np.clip(50 + relative_edge * 25, 0, 100))
    leadership = market.get("leadership")
    leadership_score = 90 if ((direction == "CALL" and leadership == "Market leader") or
                              (direction == "PUT" and leadership == "Market laggard")) else 50
    if ((direction == "CALL" and leadership == "Market laggard") or
            (direction == "PUT" and leadership == "Market leader")):
        leadership_score = 10
    last = data.iloc[-1]
    atr = max(_finite(last.get("ATR14")), _finite(last.get("Close")) * .001)
    slow_slope = directional_sign * _finite(last.get("EMA20_SLOPE")) / atr
    slow_alignment = int(np.clip(50 + slow_slope * 45, 0, 100))
    if (trend.get("ema_alignment") == "Bullish") == (direction == "CALL"):
        slow_alignment = min(100, slow_alignment + 20)
    elif direction:
        slow_alignment = max(0, slow_alignment - 20)
    stock_components = {
        "Trend quality": base["Trend Strength"],
        "Relative strength": relative_strength,
        "Momentum": base["Momentum"],
        "Volume intelligence": base["Volume Intelligence"],
        "Market leadership": leadership_score,
        "Slow-trend alignment": slow_alignment,
    }
    stock_weights = {"Trend quality": 24, "Relative strength": 18, "Momentum": 18,
                     "Volume intelligence": 14, "Market leadership": 12,
                     "Slow-trend alignment": 14}
    stock_score, stock_breakdown = _weighted_score(stock_components, stock_weights)

    opposing_distance = abs(levels.get("distance_resistance", 0.0) if direction == "CALL"
                            else levels.get("distance_support", 0.0))
    level_confirmed = bool((direction == "CALL" and levels.get("breakout_confirmed")) or
                           (direction == "PUT" and levels.get("breakdown_confirmed")))
    entry_quality = 90 if level_confirmed else 25 if opposing_distance <= .35 else 70
    pattern_status = pattern.get("status") if pattern else None
    pattern_bias = pattern.get("bias") if pattern else None
    pattern_aligned = bool(pattern and ((direction == "CALL" and pattern_bias == "Bullish") or
                                        (direction == "PUT" and pattern_bias == "Bearish")))
    confirmation_quality = 100 if pattern_aligned and pattern_status == "CONFIRMED" else 55
    if confirmation_missing or not last_complete:
        confirmation_quality = min(confirmation_quality, 25)
    if pattern_bias == "Neutral":
        confirmation_quality = min(confirmation_quality, 35)
    remaining_ratio = abs(expected.get("remaining_move", 0.0)) / max(expected.get("favorable_move", .01), .01)
    remaining_score = int(np.clip(remaining_ratio * 100, 0, 100))
    pullback_quality = int(np.clip(100 - expected.get("pullback", 50), 0, 100))
    level_positioning = 95 if level_confirmed else 35 if opposing_distance <= .35 else 70
    trade_components = {
        "Risk / reward": base["Risk Reward"],
        "Entry quality": entry_quality,
        "Confirmation": confirmation_quality,
        "Expected move remaining": remaining_score,
        "Market alignment": base["Market Alignment"],
        "Level positioning": level_positioning,
        "Pullback quality": pullback_quality,
    }
    trade_weights = {"Risk / reward": 22, "Entry quality": 17, "Confirmation": 17,
                     "Expected move remaining": 13, "Market alignment": 13,
                     "Level positioning": 10, "Pullback quality": 8}
    trade_score, trade_breakdown = _weighted_score(trade_components, trade_weights)
    if data_quality.get("delayed"):
        trade_score = min(trade_score, 20)
    if not market.get("benchmarks_complete", False):
        trade_score = min(trade_score, 25)
    if conflicts:
        trade_score = min(trade_score, 55)
    if trade_score != sum(points for _, points in trade_breakdown):
        trade_breakdown.append(("Conservative setup cap", trade_score-sum(points for _, points in trade_breakdown)))

    volume_ready = base["Volume Intelligence"] if volume.get("rvol", 0) >= 1.2 else min(40, base["Volume Intelligence"])
    risk_ready = 100 if expected.get("risk_reward", 0) >= 2 else 55 if expected.get("risk_reward", 0) >= 1.5 else 0
    pattern_ready = 100 if pattern_aligned and pattern_status == "CONFIRMED" else 70 if not confirmation_missing else 15
    readiness_components = {
        "Candle completed": 100 if last_complete else 0,
        "Pattern confirmation": pattern_ready,
        "Market confirmation": base["Market Alignment"],
        "Volume confirmation": volume_ready,
        "Risk acceptable": risk_ready,
        "Momentum acceptable": base["Momentum"],
        "Entry trigger satisfied": 100 if level_confirmed and last_complete else 0,
    }
    readiness_weights = {"Candle completed": 15, "Pattern confirmation": 15,
                         "Market confirmation": 14, "Volume confirmation": 14,
                         "Risk acceptable": 16, "Momentum acceptable": 10,
                         "Entry trigger satisfied": 16}
    readiness, readiness_breakdown = _weighted_score(readiness_components, readiness_weights)
    if data_quality.get("delayed") or not market.get("benchmarks_complete", False):
        readiness = 0
    elif not last_complete:
        readiness = min(readiness, 35)
    elif confirmation_missing:
        readiness = min(readiness, 49)
    if conflicts:
        readiness = min(readiness, 45)
    if volume.get("rvol", 0) < 1.2 or expected.get("risk_reward", 0) < 2:
        readiness = min(readiness, 55)
    if not level_confirmed:
        readiness = min(readiness, 69)
    if readiness != sum(points for _, points in readiness_breakdown):
        readiness_breakdown.append(("Readiness gate", readiness-sum(points for _, points in readiness_breakdown)))
    readiness_label = ("READY" if readiness >= 90 else "ALMOST READY" if readiness >= 75 else
                       "WAIT FOR CONFIRMATION" if readiness >= 50 else
                       "NOT READY" if readiness >= 25 else "NO EDGE")
    return {
        "stock_score": stock_score, "trade_score": trade_score,
        "trade_readiness": readiness, "readiness_label": readiness_label,
        "stock_components": stock_components, "trade_components": trade_components,
        "readiness_components": readiness_components,
        "stock_breakdown": stock_breakdown, "trade_breakdown": trade_breakdown,
        "readiness_breakdown": readiness_breakdown,
    }


def analyze_volume(data: pd.DataFrame) -> dict:
    if data.empty:
        return {"quality": "Unavailable", "rvol": 0.0, "current": 0.0, "average": 0.0,
                "vs_prior5": 0.0, "acceleration": 0.0, "buying_pressure": 50.0,
                "selling_pressure": 50.0}
    last = data.iloc[-1]
    current = _finite(last.get("Volume"))
    average = _finite(last.get("AVG_VOLUME20"))
    rvol = _finite(last.get("RVOL"))
    prior5 = _finite(data["Volume"].iloc[-6:-1].mean()) if len(data) >= 6 else average
    recent3 = _finite(data["Volume"].tail(3).mean())
    previous3 = _finite(data["Volume"].iloc[-6:-3].mean()) if len(data) >= 6 else average
    vs_prior5 = current / prior5 if prior5 > 0 else 0.0
    acceleration = recent3 / previous3 if previous3 > 0 else 0.0
    recent = data.tail(5)
    spans = (recent["High"]-recent["Low"]).replace(0, np.nan)
    close_location = ((recent["Close"]-recent["Low"])/spans).fillna(.5)
    weights = pd.to_numeric(recent["Volume"], errors="coerce").fillna(0)
    buying = _finite((close_location*weights).sum() / max(weights.sum(), 1) * 100, 50)
    if rvol < .7:
        quality = "Weak volume"
    elif rvol < 1.25:
        quality = "Normal volume"
    elif rvol < 2:
        quality = "Elevated volume"
    elif rvol <= 4:
        quality = "Strong confirmation"
    else:
        quality = "Extreme/news-driven"
    return {
        "quality": quality, "rvol": rvol, "current": current, "average": average,
        "vs_prior5": vs_prior5, "acceleration": acceleration,
        "buying_pressure": buying, "selling_pressure": 100-buying,
        "breakout_quality": "Adequate" if rvol >= 1.2 and vs_prior5 >= 1.1 else "Weak",
    }


def analyze_levels(data: pd.DataFrame, interval: str, volume: dict,
                   manual_support: float | None = None,
                   manual_resistance: float | None = None) -> dict:
    close = _finite(data["Close"].iloc[-1])
    prior = data.iloc[:-1].tail(20) if len(data) > 1 else data
    swing_support, swing_resistance = calculate_support_resistance(prior)
    if manual_support and manual_support > 0:
        swing_support = max(swing_support, float(manual_support)) if manual_support < close else swing_support
    if manual_resistance and manual_resistance > 0:
        swing_resistance = min(swing_resistance, float(manual_resistance)) if manual_resistance > close else swing_resistance
    minutes = data.index.hour*60+data.index.minute
    pre = data[(minutes >= 240) & (minutes < 570)]
    regular = data[(minutes >= 570) & (minutes < 960)]
    pre_high = _finite(pre["High"].max(), np.nan) if not pre.empty else np.nan
    pre_low = _finite(pre["Low"].min(), np.nan) if not pre.empty else np.nan
    session_high, session_low = _finite(data["High"].max()), _finite(data["Low"].min())
    atr = _finite(data.iloc[-1].get("ATR14"), close*.003)
    zone_half_width = max(atr*.12, close*.0005)
    resistance_tests = int(((prior["High"]-swing_resistance).abs() <= zone_half_width).sum())
    support_tests = int(((prior["Low"]-swing_support).abs() <= zone_half_width).sum())
    resistance_mask = (prior["High"]-swing_resistance).abs() <= zone_half_width
    support_mask = (prior["Low"]-swing_support).abs() <= zone_half_width
    last_resistance_test = prior.index[resistance_mask][-1] if resistance_mask.any() else None
    last_support_test = prior.index[support_mask][-1] if support_mask.any() else None
    resistance_test_volume = _finite(prior.loc[resistance_mask, "Volume"].iloc[-1]) if resistance_mask.any() else 0.0
    support_test_volume = _finite(prior.loc[support_mask, "Volume"].iloc[-1]) if support_mask.any() else 0.0
    last_complete = candle_is_complete(data, interval)
    distance_ema = abs(percentage_move(close, _finite(data.iloc[-1].get("EMA9"), close)))
    breakout = last_complete and close > swing_resistance and volume["breakout_quality"] == "Adequate" and distance_ema <= 1.0
    breakdown = last_complete and close < swing_support and volume["breakout_quality"] == "Adequate" and distance_ema <= 1.0
    if breakout:
        level_status = "Confirmed breakout"
    elif breakdown:
        level_status = "Confirmed breakdown"
    elif close >= swing_resistance-zone_half_width:
        level_status = "Resistance test - not a breakout"
    elif close <= swing_support+zone_half_width:
        level_status = "Support test - not a breakdown"
    else:
        level_status = "Inside key levels"
    return {
        "support": swing_support, "resistance": swing_resistance,
        "support_zone": (swing_support-zone_half_width, swing_support+zone_half_width),
        "resistance_zone": (swing_resistance-zone_half_width, swing_resistance+zone_half_width),
        "premarket_high": pre_high, "premarket_low": pre_low,
        "session_high": session_high, "session_low": session_low,
        "recent_swing_high": swing_resistance, "recent_swing_low": swing_support,
        "support_tests": support_tests, "resistance_tests": resistance_tests,
        "last_support_test": last_support_test, "last_resistance_test": last_resistance_test,
        "support_test_volume": support_test_volume, "resistance_test_volume": resistance_test_volume,
        "status": level_status, "breakout_confirmed": breakout,
        "breakdown_confirmed": breakdown,
        "distance_resistance": percentage_move(close, swing_resistance),
        "distance_support": percentage_move(close, swing_support),
        "zone_half_width": zone_half_width,
    }


def classify_move_strength(ticker: str, data: pd.DataFrame, volume: dict) -> dict:
    move = abs(session_move(data))
    close = _finite(data["Close"].iloc[-1])
    atr_pct = _finite(data.iloc[-1].get("ATR14")) / max(close, 1e-9) * 100
    average_candle_pct = _finite(((data["High"]-data["Low"])/data["Open"].replace(0, np.nan)*100).tail(20).median())
    baseline = max(average_candle_pct, atr_pct, .10)
    normalized = move / baseline
    if ticker in VOLATILITY_GROUPS["INDEX"]:
        thresholds = (.25, .50, 1.00, 1.50, 2.00)
    elif ticker in VOLATILITY_GROUPS["MOMENTUM"]:
        thresholds = (.50, 1.00, 1.50, 2.25, 3.00)
    else:
        thresholds = (.75, 1.50, 2.00, 3.00, 4.00)
    labels = ("Normal noise", "Early movement", "Moderate move", "Significant move", "Strong move", "Major momentum/news move")
    bucket = sum(move >= threshold for threshold in thresholds)
    if normalized >= 3 and volume.get("rvol", 0) >= 2:
        bucket = max(bucket, 5)
    return {"label": labels[min(bucket, 5)], "score": int(np.clip(normalized/3*100, 0, 100)),
            "session_move": session_move(data), "atr_pct": atr_pct,
            "average_candle_pct": average_candle_pct}


def expected_move_engine(data: pd.DataFrame, direction: str, levels: dict,
                         trend: dict, volume: dict, interval: str = "5m") -> dict:
    last = data.iloc[-1]
    close = _finite(last["Close"])
    atr_pct = abs(percentage_move(close+_finite(last.get("ATR14")), close))
    realized = _finite(((data["High"]-data["Low"])/data["Open"].replace(0, np.nan)*100).tail(20).median())
    participation = np.clip(volume.get("rvol", 0), .5, 2.5)
    projected = max(realized*1.5, atr_pct*.75) * (0.8+participation*.12)
    if direction == "CALL":
        invalidation = max(levels["support"], _finite(last.get("EMA20"), levels["support"]))
        entry = max(close, levels["resistance"])
        adverse = max(.05, abs(percentage_move(entry, invalidation)))
        favorable = max(.05, projected)
        target = entry*(1+favorable/100)
    elif direction == "PUT":
        invalidation = min(levels["resistance"], _finite(last.get("EMA20"), levels["resistance"]))
        entry = min(close, levels["support"])
        adverse = max(.05, abs(percentage_move(entry, invalidation)))
        favorable = max(.05, projected)
        target = max(.01, entry*(1-favorable/100))
    else:
        invalidation, adverse, favorable, target, entry = close, max(.05, atr_pct*.6), max(.05, projected*.6), close, close
    risk_reward = favorable/adverse if adverse > 0 else 0.0
    continuation_raw = 30 + trend.get("strength", 0)*.35 + min(volume.get("rvol", 0), 2)*10
    pullback_raw = 28 + max(0, 50-trend.get("strength", 0))*.25
    reversal_raw = 20 + (12 if volume.get("rvol", 0) < .8 else 0) + (10 if trend.get("direction") == "Conflicted" else 0)
    raw = np.array([continuation_raw, pullback_raw, reversal_raw], dtype=float)
    scaled = raw/raw.sum()*100
    whole = np.floor(scaled).astype(int)
    for index in np.argsort(-(scaled-whole))[:100-int(whole.sum())]:
        whole[index] += 1
    sign = -1 if direction == "PUT" else 1
    current_move = session_move(data)
    continuation_move = sign*favorable
    pullback_move = -sign*max(adverse, favorable*.55)
    reversal_move = -sign*max(adverse*1.5, favorable*.9)
    overextension = min(.75, abs(current_move)/max(atr_pct*4, .20))
    remaining_move = continuation_move*max(.25, 1-overextension)
    interval_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30}.get(interval, 5)
    bars_needed = int(np.clip(np.ceil(abs(remaining_move)/max(realized, .05)), 1, 6))
    low_minutes = interval_minutes * max(1, bars_needed-1)
    high_minutes = interval_minutes * min(8, bars_needed+1)
    return {
        "upside_remaining": favorable if direction != "PUT" else adverse,
        "downside_risk": -adverse if direction != "PUT" else -favorable,
        "continuation": int(whole[0]), "pullback": int(whole[1]),
        "reversal": int(whole[2]), "favorable_move": favorable,
        "adverse_move": -adverse, "risk_reward": risk_reward,
        "entry": entry, "invalidation": invalidation, "target": target,
        "current_move": current_move, "continuation_move": continuation_move,
        "pullback_move": pullback_move, "reversal_move": reversal_move,
        "remaining_move": remaining_move,
        "estimated_time": f"{low_minutes}-{high_minutes} minutes",
        "label": "Heuristic estimate - not a historical win probability",
    }


def assess_risk_controls(settings: dict, trade_grade: str | None = None) -> dict:
    account = max(0.0, _finite(settings.get("account_size"), 25000.0))
    risk_pct = max(0.0, _finite(settings.get("max_risk_pct"), .5))
    daily_loss_pct = max(0.0, _finite(settings.get("daily_loss_limit_pct"), 1.5))
    profit_lock_pct = max(0.0, _finite(settings.get("daily_profit_lock_pct"), 2.0))
    pnl = _finite(settings.get("daily_pnl"), 0.0)
    trades = int(settings.get("trades_today", 0) or 0)
    losses = int(settings.get("consecutive_losses", 0) or 0)
    max_trades = int(settings.get("max_trades", 3) or 3)
    stop_losses = int(settings.get("stop_after_losses", 2) or 2)
    loss_limit = account*daily_loss_pct/100
    profit_lock = account*profit_lock_pct/100
    reasons = []
    if pnl <= -loss_limit and loss_limit > 0:
        reasons.append("TRADING LOCKED - DAILY LOSS LIMIT REACHED")
    if trades >= max_trades:
        reasons.append("Maximum trades per day reached")
    if losses >= stop_losses:
        reasons.append("Consecutive-loss stop reached")
    protect = pnl >= profit_lock > 0 and trade_grade not in {"A+", "A"}
    if protect:
        reasons.append("PROTECT PROFITS - NO NEW TRADE")
    return {"locked": bool(reasons), "reasons": reasons,
            "max_risk_dollars": account*risk_pct/100,
            "loss_limit_dollars": loss_limit, "profit_lock_dollars": profit_lock}


def build_conservative_decision(ticker: str, data: pd.DataFrame, interval: str,
                                market: dict, trend: dict, volume: dict,
                                levels: dict, data_quality: dict,
                                risk_settings: dict | None = None) -> dict:
    patterns = [row for row in detect_candlestick_statuses(data) if row.get("detected")]
    rank = {"CONFIRMED": 3, "COMPLETE": 2, "WATCH": 1, "FORMING": 0}
    pattern = max(patterns, key=lambda row: (rank.get(row.get("status"), 0), row.get("confidence", 0))) if patterns else None
    direction = "CALL" if trend["direction"] == "Bullish" else "PUT" if trend["direction"] == "Bearish" else ""
    expected = expected_move_engine(data, direction, levels, trend, volume, interval)
    last_complete = candle_is_complete(data, interval)
    breakdown: list[tuple[str, int]] = []

    def add(label: str, points: int) -> None:
        breakdown.append((label, int(points)))

    if direction:
        add(f"{trend['label']} aligned with {direction}", 12 if trend["strength"] >= 70 else 8)
        add(f"EMA alignment supports {direction}", 8 if (trend["ema_alignment"] == "Bullish") == (direction == "CALL") else -10)
        add(f"VWAP position supports {direction}", 8 if (trend["vwap_position"] == "Above VWAP") == (direction == "CALL") else -10)
        if (direction == "CALL" and "Higher highs" in trend["structure"]) or (direction == "PUT" and "Lower highs" in trend["structure"]):
            add(f"Structure supports {direction}", 8)
    else:
        add("No clear trend direction", -18)

    market_aligned = (direction == "CALL" and market["status"] == "Bullish") or (direction == "PUT" and market["status"] == "Bearish")
    market_opposed = (direction == "CALL" and market["status"] == "Bearish") or (direction == "PUT" and market["status"] == "Bullish")
    if market_aligned:
        add("SPY / QQQ / sector confirm direction", 10)
    elif market_opposed or market["status"] == "Conflicted":
        add("Market direction conflicts with stock", -15)
    else:
        add("Market direction is neutral", -5)
    if not market.get("benchmarks_complete", False):
        add("Market benchmark data is incomplete", -15)
    if market["leadership"] == "Market leader" and direction == "CALL":
        add("Outperforming SPY and QQQ", 8)
    elif market["leadership"] == "Market laggard" and direction == "PUT":
        add("Underperforming SPY and QQQ", 8)
    elif (market["leadership"] == "Market laggard" and direction == "CALL") or (market["leadership"] == "Market leader" and direction == "PUT"):
        add("Relative strength opposes trade direction", -10)

    volume_points = {"Weak volume": -15, "Normal volume": 0, "Elevated volume": 8,
                     "Strong confirmation": 12, "Extreme/news-driven": 6}.get(volume["quality"], -8)
    add(f"{volume['quality']} (RVOL {volume['rvol']:.2f}x)", volume_points)
    if volume["quality"] == "Extreme/news-driven":
        add("Extreme volume increases exhaustion risk", -6)

    confirmation_missing = False
    pattern_conflict = False
    if pattern:
        status, bias = pattern.get("status"), pattern.get("bias", "Neutral")
        aligned = (direction == "CALL" and bias == "Bullish") or (direction == "PUT" and bias == "Bearish")
        opposed = (direction == "CALL" and bias == "Bearish") or (direction == "PUT" and bias == "Bullish")
        if bias == "Neutral" and status == "FORMING":
            add(f"{pattern['name']} candle is still forming", 0)
        elif bias == "Neutral":
            add(f"Neutral {pattern['name']}", -6)
        elif aligned and status == "CONFIRMED":
            add(f"Confirmed {pattern['name']} supports {direction}", 10)
        elif aligned:
            add(f"{pattern['name']} detected but not confirmed", 2)
        elif opposed and status != "FORMING":
            add(f"{pattern['name']} conflicts with {direction}", -12)
            pattern_conflict = True
        confirmation_missing = bool(pattern.get("confirmation") and status != "CONFIRMED")
        if confirmation_missing:
            add("Required confirmation missing", -10)
    else:
        add("No confirming candlestick pattern", -4)

    if not last_complete:
        add("Current candle is incomplete", -15)
    if data_quality.get("delayed"):
        add("Market data appears delayed", -25)
    if data_quality.get("extended"):
        add("Extended-hours signal has lower reliability", -10)
    stamp = pd.Timestamp(data.index[-1])
    if 930 <= stamp.hour*60+stamp.minute < 960:
        add("Late-day liquidity risk", -6)
    opposing_distance = abs(levels["distance_resistance"] if direction == "CALL" else levels["distance_support"])
    if direction and opposing_distance <= .35 and not (levels["breakout_confirmed"] or levels["breakdown_confirmed"]):
        add("Poor trade location near opposing level", -10)
    if (direction == "CALL" and levels["breakout_confirmed"]) or (direction == "PUT" and levels["breakdown_confirmed"]):
        add(levels["status"], 10)
    elif "test" in levels["status"].lower():
        add(levels["status"], -5)

    distance_ema9 = abs(percentage_move(_finite(data["Close"].iloc[-1]), _finite(data.iloc[-1].get("EMA9"))))
    if distance_ema9 > max(1.0, expected["favorable_move"]*1.25):
        add("Price is extended from EMA9", -12)
    if expected["risk_reward"] >= 2:
        add(f"Favorable heuristic risk/reward {expected['risk_reward']:.1f}:1", 12)
    elif expected["risk_reward"] >= 1.5:
        add(f"Acceptable heuristic risk/reward {expected['risk_reward']:.1f}:1", 6)
    else:
        add(f"Poor heuristic risk/reward {expected['risk_reward']:.1f}:1", -18)

    factor_breakdown = breakdown
    components = calculate_component_scores(data, direction, market, trend, volume, expected)
    weights = {"Trend Strength": 22, "Momentum": 16, "Volume Intelligence": 15,
               "Market Alignment": 18, "Risk Reward": 17, "Expected Move": 12}
    breakdown = []
    for component, component_score in components.items():
        add(f"{component} {component_score}/100", round(weights[component]*component_score/100))
    if volume["quality"] == "Weak volume":
        add(f"Weak volume (RVOL {volume['rvol']:.2f}x)", -10)
    adjustment_rules = (
        ("Confirmed ", 5), ("Neutral ", -4), ("conflicts with", -8),
        ("Required confirmation missing", -8), ("Current candle is incomplete", -12),
        ("Market data appears delayed", -20), ("Extended-hours", -8),
        ("Late-day liquidity", -5), ("Poor trade location", -8),
        ("Confirmed breakout", 5), ("Confirmed breakdown", 5),
        ("Price is extended", -10), ("Market benchmark data is incomplete", -15),
        ("No confirming candlestick", -3), ("Extreme volume increases", -4),
        ("still forming", 0),
    )
    used_adjustments: set[str] = set()
    for label, _ in factor_breakdown:
        for marker, points in adjustment_rules:
            if marker.lower() in label.lower() and marker not in used_adjustments:
                add(label, points)
                used_adjustments.add(marker)
                break
    raw_score = sum(points for _, points in breakdown)
    if raw_score > 100:
        add("Score cap", 100-raw_score)
    elif raw_score < 0:
        add("Score floor", -raw_score)
    score = int(np.clip(sum(points for _, points in breakdown), 0, 100))
    classification = "HIGH CONVICTION" if score >= 90 else "GOOD SETUP" if score >= 80 else "WATCH" if score >= 70 else "WAIT" if score >= 60 else "NO TRADE"
    grade = "A+" if score >= 90 else "A" if score >= 80 else "B" if score >= 70 else "C" if score >= 60 else "NO TRADE"
    risk = assess_risk_controls(risk_settings or {}, grade)
    conflicts = []
    if market_opposed or market["status"] == "Conflicted":
        conflicts.append("Stock direction conflicts with SPY / QQQ / sector context")
    completed_neutral = pattern and pattern.get("bias") == "Neutral" and pattern.get("status") != "FORMING"
    if pattern_conflict or (completed_neutral and direction):
        conflicts.append(f"{direction} trend conflicts with {pattern.get('bias','neutral').lower()} {pattern.get('name','candle')}")
    if volume["quality"] == "Weak volume" and (levels["breakout_confirmed"] or levels["breakdown_confirmed"]):
        conflicts.append("Level break occurred on weak volume")
    entry_confirmed = last_complete and not confirmation_missing and volume["rvol"] >= 1.2 and not conflicts

    separated = calculate_separated_scores(
        data, direction, market, trend, volume, expected, levels, pattern,
        last_complete, confirmation_missing, conflicts, data_quality,
    )
    stock_score = separated["stock_score"]
    trade_score = separated["trade_score"]
    trade_readiness = separated["trade_readiness"]
    score = trade_score
    breakdown = separated["trade_breakdown"]
    classification = "A+" if trade_score >= 90 else "HIGH QUALITY" if trade_score >= 80 else "GOOD" if trade_score >= 70 else "FAIR" if trade_score >= 60 else "POOR"
    grade = "A+" if trade_score >= 90 else "A" if trade_score >= 80 else "B" if trade_score >= 70 else "C" if trade_score >= 60 else "D" if trade_score >= 50 else "F"
    risk = assess_risk_controls(risk_settings or {}, grade)

    no_trade_reason = ""
    if risk["locked"]:
        execution_action, edge_action, no_trade_reason = "NO TRADE", "NO TRADE", risk["reasons"][0]
    elif data_quality.get("delayed"):
        execution_action, edge_action, no_trade_reason = "NO TRADE", "NO TRADE", "Data is delayed or unreliable"
    elif not market.get("benchmarks_complete", False):
        execution_action, edge_action, no_trade_reason = "NO TRADE", "NO TRADE", "Market benchmark data is incomplete"
    elif not direction:
        execution_action, edge_action, no_trade_reason = "WAIT", "NO EDGE", "No directional edge"
    elif expected["risk_reward"] < 1.5:
        execution_action, edge_action, no_trade_reason = "WAIT", "NO EDGE", "Risk/reward is below 1.5:1"
    elif volume["rvol"] < .5:
        execution_action, edge_action, no_trade_reason = "WAIT", "NO EDGE", "RVOL is too weak"
    elif trade_score < 50 or stock_score < 45:
        execution_action, edge_action, no_trade_reason = "WAIT", "NO EDGE", "Stock or setup quality has no usable edge"
    elif trade_score < 75 or stock_score < 60 or trade_readiness < 75 or not entry_confirmed or expected["risk_reward"] < 2:
        execution_action, edge_action, no_trade_reason = "WAIT", "LOW EDGE", "Required alignment or confirmation is incomplete"
    else:
        execution_action = direction
        edge_action = (f"HIGH EDGE {direction}S" if trade_score >= 90 and stock_score >= 80 and trade_readiness >= 90
                       else f"MODERATE EDGE {direction}S")

    positives = sorted(((label, points) for label, points in factor_breakdown if points > 0), key=lambda item: item[1], reverse=True)
    negatives = sorted(((label, points) for label, points in factor_breakdown if points < 0), key=lambda item: item[1])
    if execution_action in {"CALL", "PUT"} and len(positives) < 3:
        execution_action, edge_action, no_trade_reason = "WAIT", "LOW EDGE", "Fewer than three independent positive reasons"
    key_reason = positives[0][0] if positives else "No strong positive factor"
    main_risk = negatives[0][0] if negatives else "No material technical risk detected"
    if direction == "CALL":
        entry_condition = f"Completed {interval.upper()} close above ${levels['resistance']:.2f} with RVOL at least 1.2x"
        invalidation = f"Close below ${expected['invalidation']:.2f}"
    elif direction == "PUT":
        entry_condition = f"Completed {interval.upper()} close below ${levels['support']:.2f} with RVOL at least 1.2x"
        invalidation = f"Close above ${expected['invalidation']:.2f}"
    else:
        entry_condition = f"Wait for completed {interval.upper()} confirmation beyond ${levels['resistance']:.2f} or ${levels['support']:.2f}"
        invalidation = "No position before trigger"
    options_technical_gate = bool(trade_score >= 80 and stock_score >= 70 and trade_readiness >= 80 and
                                  grade in {"A", "A+"} and not data_quality.get("delayed") and
                                  entry_confirmed and expected["risk_reward"] >= 2)
    signal = edge_action
    location = analyze_location(data, levels)
    pattern_context = {
        "preceding_trend": trend["label"], "trend_strength": trend["strength"],
        "location": location["primary"], "volume": volume["quality"],
        "context": pattern.get("context_note", "") if pattern else "No active pattern",
        "action": edge_action,
    }
    entry_price = expected["entry"]
    per_share_risk = abs(entry_price-expected["invalidation"])
    reward_sign = 1 if direction != "PUT" else -1
    stock_targets = {multiple: entry_price + reward_sign*per_share_risk*multiple for multiple in (1.5, 2.5, 4.0)}
    return {
        "action": edge_action, "execution_action": execution_action, "signal": signal,
        "direction": direction or "NEUTRAL", "score": trade_score,
        "classification": classification, "grade": grade,
        "confidence": trade_readiness, "confidence_label": "Trade readiness - not historical win probability",
        "stock_score": stock_score, "trade_score": trade_score,
        "trade_readiness": trade_readiness, "readiness_label": separated["readiness_label"],
        "technical_score": trade_score, "model_confidence": None, "historical_probability": None,
        "entry_condition": entry_condition, "invalidation": invalidation,
        "first_target": f"${expected['target']:.2f}", "key_reason": key_reason,
        "main_risk": no_trade_reason or main_risk, "breakdown": breakdown,
        "factor_breakdown": factor_breakdown,
        "top_reasons": [label for label, _ in positives[:3]],
        "conflicts": conflicts, "pattern": pattern or {}, "expected": expected,
        "components": components, "stock_components": separated["stock_components"],
        "trade_components": separated["trade_components"],
        "readiness_components": separated["readiness_components"],
        "stock_breakdown": separated["stock_breakdown"],
        "readiness_breakdown": separated["readiness_breakdown"],
        "location": location, "pattern_context": pattern_context,
        "risk": risk, "entry_confirmed": entry_confirmed,
        "targets": {"1.5R": stock_targets[1.5], "2.5R": stock_targets[2.5], "4R": stock_targets[4.0]},
        "options": {
            "eligible": False, "technical_gate": options_technical_gate,
            "status": "No option suggestion - live chain liquidity and spread data are unavailable",
            "delta": None, "expiration": None, "max_spread": None,
        },
    }


def historical_pattern_stats(data: pd.DataFrame, pattern_name: str, interval: str,
                             horizon: int = 3, min_sample: int = 20) -> dict[str, Any]:
    """Measure observed post-pattern behavior without inventing probabilities."""
    occurrences: list[dict[str, float]] = []
    if data.empty or len(data) < 25 + horizon:
        return {"pattern": pattern_name, "sample_size": 0, "sufficient": False,
                "label": "Insufficient historical sample", "timeframe": interval}
    for index in range(20, len(data) - horizon):
        match = next((row for row in detect_candlestick_patterns(data.iloc[:index + 1])
                      if row["name"] == pattern_name and row["detected"]), None)
        if not match:
            continue
        entry = _finite(data["Close"].iloc[index])
        future = data.iloc[index + 1:index + 1 + horizon]
        bias = match.get("bias", "Neutral")
        if bias == "Neutral":
            prior_move = _finite(data["Close"].iloc[index - 1]) - _finite(data["Close"].iloc[index - 5])
            bias = "Bearish" if prior_move > 0 else "Bullish" if prior_move < 0 else "Neutral"
        if not entry or bias == "Neutral":
            continue
        if bias == "Bullish":
            favorable = (_finite(future["High"].max()) / entry - 1) * 100
            adverse = (_finite(future["Low"].min()) / entry - 1) * 100
            final_move = (_finite(future["Close"].iloc[-1]) / entry - 1) * 100
        else:
            favorable = (1 - _finite(future["Low"].min()) / entry) * 100
            adverse = (1 - _finite(future["High"].max()) / entry) * 100
            final_move = (1 - _finite(future["Close"].iloc[-1]) / entry) * 100
        occurrences.append({"favorable": favorable, "adverse": adverse, "final": final_move})
    sample_size = len(occurrences)
    if sample_size < min_sample:
        return {"pattern": pattern_name, "sample_size": sample_size, "sufficient": False,
                "label": "Insufficient historical sample", "timeframe": interval}
    frame = pd.DataFrame(occurrences)
    continuation = round(float((frame["final"] > 0).mean()) * 100, 1)
    return {"pattern": pattern_name, "sample_size": sample_size, "sufficient": True,
            "label": "Observed historical sample - not a forecast", "timeframe": interval,
            "continuation_rate": continuation, "reversal_rate": round(100-continuation, 1),
            "average_favorable_move": round(float(frame["favorable"].mean()), 3),
            "average_adverse_move": round(float(frame["adverse"].mean()), 3),
            "median_move": round(float(frame["final"].median()), 3),
            "maximum_drawdown": round(float(frame["adverse"].min()), 3)}


def build_risk_alerts(ticker: str, interval: str, data: pd.DataFrame, decision: dict,
                      market: dict, volume: dict, levels: dict, data_quality: dict,
                      threshold: float) -> list[dict[str, Any]]:
    """Return fully explained alerts; confirmation language is strictly gated."""
    if data.empty:
        return []
    last = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else last
    current_price, move_pct = _finite(last["Close"]), session_move(data)
    expected, pattern = decision["expected"], decision.get("pattern", {})
    alerts: list[dict[str, Any]] = []

    def add(priority: str, reason: str, action: str, reference: float = 0.0,
            pattern_name: str = "-") -> None:
        alerts.append({"ticker": ticker, "timestamp": data.index[-1], "timeframe": interval,
                       "priority": priority, "current_price": round(current_price, 2),
                       "reference_price": round(reference, 2) if reference else None,
                       "percentage_move": round(move_pct, 2), "pattern": pattern_name,
                       "confidence": decision["confidence"],
                       "continuation_probability": expected["continuation"],
                       "pullback_risk": expected["pullback"],
                       "relevant_level": round(reference, 2) if reference else None,
                       "reason": reason, "action_required": action})

    if abs(move_pct) >= threshold:
        add("HIGH", f"Configurable move threshold reached: {move_pct:+.2f}%", "Review location before entry")
    atr = _finite(last.get("ATR14"))
    if atr and _finite(last["High"])-_finite(last["Low"]) >= atr*1.5:
        add("HIGH", "Candle range is at least 1.5 ATR", "Do not chase; wait for a completed retest")
    if volume["rvol"] >= 2:
        add("HIGH", f"Unusual RVOL {volume['rvol']:.2f}x", "Require technical confirmation")
    if volume["acceleration"] >= 1.8:
        add("MED", f"Volume accelerated to {volume['acceleration']:.2f}x the prior five candles", "Check buying/selling pressure")
    if expected["continuation"] > 70 and decision["direction"] in {"CALL", "PUT"}:
        add("HIGH", f"{decision['direction']} continuation likelihood is {expected['continuation']}% (heuristic)", "Require completed entry confirmation")
    if expected["pullback"] >= 50:
        add("MED", f"Pullback likelihood is elevated at {expected['pullback']}% (heuristic)", "Avoid chasing the current candle")
    if abs(expected["remaining_move"]) <= abs(expected["continuation_move"])*.3 and abs(move_pct) >= threshold:
        add("HIGH", "Expected move is mostly completed; momentum exhaustion risk is elevated", "Wait for consolidation or pullback")
    ema9 = _finite(last.get("EMA9"), current_price)
    if abs(percentage_move(current_price, ema9)) >= max(1.0, abs(expected["continuation_move"])*1.5):
        add("HIGH", "Price is overextended from EMA9", "Do not chase; wait for mean reversion or a base", ema9)
    if (_finite(last.get("EMA9"))-_finite(last.get("EMA20"))) * (_finite(previous.get("EMA9"))-_finite(previous.get("EMA20"))) < 0:
        add("MED", "EMA9 / EMA20 crossover", "Wait for slope and price confirmation", _finite(last.get("EMA20")))
    if (_finite(last["Close"])-_finite(last.get("VWAP"))) * (_finite(previous["Close"])-_finite(previous.get("VWAP"))) < 0:
        side = "reclaim" if current_price > _finite(last.get("VWAP")) else "rejection"
        add("MED", f"VWAP {side}", "Require a completed hold away from VWAP", _finite(last.get("VWAP")))
    if levels.get("breakout_confirmed"):
        add("HIGH", "Completed candle closed above resistance with adequate volume", "Watch for a hold or defended retest", levels["resistance"])
    if levels.get("breakdown_confirmed"):
        add("HIGH", "Completed candle closed below support with adequate volume", "Watch for a hold or rejected retest", levels["support"])
    bar_complete = candle_is_complete(data, interval)
    support_zone_high = levels["support_zone"][1]
    resistance_zone_low = levels["resistance_zone"][0]
    if bar_complete and _finite(last["Low"]) <= support_zone_high and current_price > support_zone_high:
        add("MED", "Support test closed back above the support zone", "Watch for a higher low before entry", levels["support"])
    if bar_complete and _finite(last["High"]) >= resistance_zone_low and current_price < resistance_zone_low:
        add("MED", "Resistance test closed back below the resistance zone", "Avoid calls until resistance is reclaimed", levels["resistance"])
    if bar_complete and _finite(previous["Close"]) > levels["resistance_zone"][1] and current_price < resistance_zone_low:
        add("HIGH", "Failed breakout: price closed back below the resistance zone", "Do not chase calls", levels["resistance"])
    if bar_complete and _finite(previous["Close"]) < levels["support_zone"][0] and current_price > support_zone_high:
        add("HIGH", "Failed breakdown: price closed back above the support zone", "Do not chase puts", levels["support"])
    market_direction = market.get("status", "Neutral")
    if decision["direction"] == "CALL" and market_direction in {"Bearish", "Conflicted"}:
        add("HIGH", "Market/stock divergence: bullish stock setup lacks broad-market confirmation", "Default to WAIT")
    if decision["direction"] == "PUT" and market_direction in {"Bullish", "Conflicted"}:
        add("HIGH", "Market/stock divergence: bearish stock setup lacks broad-market confirmation", "Default to WAIT")
    if pattern.get("status") == "CONFIRMED" and pattern.get("confirmed"):
        reference = levels["support"] if pattern.get("bias") == "Bullish" else levels["resistance"]
        add("HIGH", f"{pattern['name']} confirmation candle completed", "Use with trend and level context", reference, pattern["name"])
    elif pattern.get("detected"):
        add("MED", f"{pattern['name']} detected; confirmation is incomplete", "WAIT for the required candle to close", pattern_name=pattern["name"])
    for conflict in decision.get("conflicts", []):
        add("HIGH", f"SIGNAL CONFLICT: {conflict}", "Default to WAIT")
    if decision.get("execution_action") in {"CALL", "PUT"} and decision["trade_score"] >= 80 and decision["trade_readiness"] >= 80:
        add("HIGH", f"{decision['action']} passed technical and readiness gates", decision["entry_condition"])
    if decision["risk"]["locked"]:
        add("CRITICAL", decision["risk"]["reasons"][0], "TRADING LOCKED")
    if data_quality.get("delayed"):
        add("CRITICAL", "DATA MAY BE DELAYED - DECISION SUPPORT ONLY", "NO TRADE")
    if not alerts:
        add("LOW", "No actionable trigger", "WAIT")
    rank = {"CRITICAL": 0, "HIGH": 1, "MED": 2, "LOW": 3}
    return sorted(alerts, key=lambda item: rank[item["priority"]])
