"""Free, local trade-coaching helpers for the Streamlit dashboard."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd


POSITION_TYPES = ["Long shares", "Bought call", "Bought put"]


def chart_snapshot_svg(
    data: pd.DataFrame,
    ticker: str,
    support: float,
    resistance: float,
    analysis: dict,
    width: int = 700,
    height: int = 210,
) -> str:
    """Create a compact, dependency-free picture of the active chart."""
    recent = data.tail(42).copy()
    if recent.empty:
        return "<div>No chart data available.</div>"

    top, bottom, left, right = 22, 30, 44, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    low = float(min(recent["Low"].min(), support))
    high = float(max(recent["High"].max(), resistance))
    padding = max((high - low) * .08, max(abs(high), 1) * .001)
    low, high = low - padding, high + padding
    span = max(high - low, 1e-9)
    step = plot_w / max(len(recent), 1)
    body_w = max(2.2, min(8, step * .58))

    def x_at(index: int) -> float:
        return left + (index + .5) * step

    def y_at(value: float) -> float:
        return top + (high - float(value)) / span * plot_h

    elements = [
        f'<rect width="{width}" height="{height}" rx="8" fill="#0d141e"/>',
        f'<text x="12" y="16" fill="#e8eef7" font-size="11" font-weight="700">{escape(ticker)} · LIVE SETUP SNAPSHOT</text>',
    ]
    for grid in range(5):
        value = low + span * grid / 4
        y = y_at(value)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#243244" stroke-width="1"/>')
        elements.append(f'<text x="{width-right+5}" y="{y+3:.1f}" fill="#8492a6" font-size="8">{value:.2f}</text>')

    for level, label, color in ((resistance, "R", "#ff5c72"), (support, "S", "#25d695")):
        y = y_at(level)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{color}" stroke-width="1" stroke-dasharray="4 3"/>')
        elements.append(f'<text x="{left+3}" y="{y-3:.1f}" fill="{color}" font-size="8" font-weight="700">{label} {level:.2f}</text>')

    for idx, (_, candle) in enumerate(recent.iterrows()):
        x = x_at(idx)
        green = float(candle["Close"]) >= float(candle["Open"])
        color = "#25d695" if green else "#ff5c72"
        y_high, y_low = y_at(candle["High"]), y_at(candle["Low"])
        y_open, y_close = y_at(candle["Open"]), y_at(candle["Close"])
        body_y, body_h = min(y_open, y_close), max(abs(y_close-y_open), 1.2)
        elements.append(f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="{color}" stroke-width="1"/>')
        elements.append(f'<rect x="{x-body_w/2:.1f}" y="{body_y:.1f}" width="{body_w:.1f}" height="{body_h:.1f}" fill="{color}"/>')

    for column, color in (("EMA9", "#35c2ff"), ("EMA20", "#ffb547")):
        if column not in recent:
            continue
        points = " ".join(f"{x_at(i):.1f},{y_at(value):.1f}" for i, value in enumerate(recent[column]) if pd.notna(value))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.4"/>')

    first_time = recent.index[0].strftime("%H:%M")
    last_time = recent.index[-1].strftime("%H:%M")
    elements.append(f'<text x="{left}" y="{height-9}" fill="#8492a6" font-size="8">{first_time}</text>')
    elements.append(f'<text x="{width-right-28}" y="{height-9}" fill="#8492a6" font-size="8">{last_time}</text>')
    bias = escape(analysis.get("option_bias", "WAIT"))
    elements.append(f'<text x="{width-12}" y="16" fill="#35c2ff" font-size="10" font-weight="700" text-anchor="end">{bias}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(ticker)} chart snapshot" style="display:block;width:100%;height:auto">{"".join(elements)}</svg>'


def profit_target_scenarios(entry_price: float, position_type: str, analysis: dict, movement: dict, rvol: float) -> dict:
    """Return arithmetic profit targets and a conservative plausibility rank."""
    targets = {percent: entry_price * (1 + percent / 100) for percent in (10, 15, 20)}
    signal = analysis.get("signal", "WAIT")
    aligned = (
        (position_type in {"Long shares", "Bought call"} and signal == "CALL WATCH")
        or (position_type == "Bought put" and signal == "PUT WATCH")
    )
    probability = int(analysis.get("probability", 0))
    continuation = int(movement.get("Expected continuation", 0))
    healthy_rvol = 1.5 <= rvol <= 4

    if aligned and probability >= 84 and continuation >= 70 and healthy_rvol:
        plausible = 20
        reason = "Direction, continuation probability, and healthy RVOL are strongly aligned."
    elif aligned and probability >= 70 and continuation >= 55:
        plausible = 15
        reason = "The directional setup is aligned, but confirmation is not strong enough for the highest tier."
    else:
        plausible = 10
        if not aligned:
            reason = "The position is not aligned with the current AI direction, so only the lowest scenario is highlighted."
        elif rvol > 4:
            reason = "RVOL is extreme; participation is high, but exhaustion and reversal risk reduce target confidence."
        elif rvol < 1.5:
            reason = "Participation is not elevated enough to support an aggressive profit assumption."
        else:
            reason = "Current continuation evidence supports only the most conservative scenario."

    instrument_note = (
        "Targets use the option premium you paid. Actual premium also depends on volatility, time decay, spread, strike, and expiration."
        if position_type in {"Bought call", "Bought put"}
        else "Targets use the share purchase price and do not guarantee the market will trade there."
    )
    return {"targets": targets, "plausible": plausible, "reason": reason, "note": instrument_note}


def coach_response(
    prompt: str,
    ticker: str,
    latest_price: float,
    analysis: dict,
    movement: dict,
    support: float,
    resistance: float,
    rvol: float,
    position: dict | None,
) -> str:
    """Answer common trading questions from the dashboard's auditable signals."""
    question = prompt.lower().strip()
    bias = analysis.get("option_bias", "WAIT · NO CLEAR EDGE")
    reason = analysis.get("option_reason", "The signals are mixed.")
    pattern = analysis.get("pattern", "No clear pattern")
    continuation = movement.get("Expected continuation", 0)
    pullback = movement.get("Expected pullback", 0)

    if any(word in question for word in ("target", "sell", "profit", "bought", "entry")):
        if not position or float(position.get("entry_price", 0)) <= 0:
            return "Enter the price you paid and the position type above. I can then calculate 10%, 15%, and 20% exit-price scenarios and rank the most supportable tier."
        scenarios = profit_target_scenarios(float(position["entry_price"]), position["position_type"], analysis, movement, rvol)
        prices = scenarios["targets"]
        return (
            f"For your {position['position_type'].lower()} entered at ${position['entry_price']:.2f}, the scenarios are "
            f"10% = ${prices[10]:.2f}, 15% = ${prices[15]:.2f}, and 20% = ${prices[20]:.2f}. "
            f"The model currently highlights {scenarios['plausible']}% because {scenarios['reason'].lower()} "
            f"{scenarios['note']}"
        )

    if any(word in question for word in ("call", "put", "direction", "buy")):
        return f"Current setup bias: {bias}. {reason} This is a watch condition, not an automatic entry."

    if any(word in question for word in ("risk", "stop", "invalid", "loss")):
        return (
            f"Risk landmarks for {ticker}: support ${support:.2f}, resistance ${resistance:.2f}, and current price ${latest_price:.2f}. "
            f"Expected pullback is {pullback}%. A break through the level opposing your position, especially with rising RVOL, weakens the setup."
        )

    if any(word in question for word in ("rvol", "volume")):
        return f"RVOL-TOD is {rvol:.2f}×. It measures participation, not direction. Use it with the current {bias.lower()} bias and price confirmation."

    if any(word in question for word in ("chart", "pattern", "candle", "setup")):
        return (
            f"{ticker} is ${latest_price:.2f} with {pattern} detected. Continuation is {continuation}% and pullback is {pullback}%. "
            f"{reason} Support is ${support:.2f}; resistance is ${resistance:.2f}."
        )

    return (
        f"For {ticker}, the dashboard says {bias}. {reason} Price is ${latest_price:.2f}, RVOL-TOD is {rvol:.2f}×, "
        f"continuation is {continuation}%, and pullback is {pullback}%. Ask me about calls or puts, the chart, RVOL, risk, or a profit target."
    )
