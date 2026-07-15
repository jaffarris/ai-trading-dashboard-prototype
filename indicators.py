"""Provider-agnostic intraday technical analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

PATTERN_CATALOG = [
    ("Hammer", "Bullish", True, "↗"), ("Shooting Star", "Bearish", True, "↘"),
    ("Bullish Engulfing", "Bullish", True, "⇈"), ("Bearish Engulfing", "Bearish", True, "⇊"),
    ("Morning Star", "Bullish", True, "✦"), ("Evening Star", "Bearish", True, "✦"),
    ("Doji", "Neutral", True, "┼"), ("Dragonfly Doji", "Bullish", True, "┬"),
    ("Gravestone Doji", "Bearish", True, "┴"), ("Inside Bar", "Neutral", True, "▯"),
    ("Outside Bar", "Neutral", True, "▣"), ("Three White Soldiers", "Bullish", False, "▥"),
    ("Three Black Crows", "Bearish", False, "▥"), ("Marubozu", "Directional", False, "┃"),
    ("Spinning Top", "Neutral", True, "┿"),
]

PATTERN_CATALOG.extend([
    ("Inverted Hammer", "Bullish", True, "IH"),
    ("Hanging Man", "Bearish", True, "HM"),
    ("Piercing Line", "Bullish", True, "PL"),
    ("Dark Cloud Cover", "Bearish", True, "DC"),
    ("Tweezer Top", "Bearish", True, "TT"),
    ("Tweezer Bottom", "Bullish", True, "TB"),
])


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if df.empty:
        return df
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    trading_date = pd.Series(df.index.date, index=df.index)
    cumulative_pv = (typical * df["Volume"]).groupby(trading_date).cumsum()
    cumulative_volume = df["Volume"].groupby(trading_date).cumsum().replace(0, np.nan)
    df["VWAP"] = (cumulative_pv / cumulative_volume).ffill()
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14, min_periods=3).mean()
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df["AVG_VOLUME20"] = volume.rolling(20, min_periods=3).mean()
    fallback_rvol = volume / df["AVG_VOLUME20"].replace(0, np.nan)

    # RVOL-TOD compares cumulative volume with the same market session and
    # clock time on prior days. Premarket, regular hours, and after-hours are
    # separated so their very different volume profiles cannot distort the ratio.
    minute_of_day = pd.Series(df.index.hour * 60 + df.index.minute, index=df.index)
    session = pd.Series(
        np.select([minute_of_day < 570, minute_of_day < 960], ["PRE", "REG"], default="POST"),
        index=df.index,
    )
    session_cumulative = volume.groupby([trading_date, session]).cumsum()
    current_date = trading_date.iloc[-1]
    baseline_frame = pd.DataFrame({
        "date": trading_date,
        "session": session,
        "minute": minute_of_day,
        "cumulative": session_cumulative,
    })
    prior = baseline_frame[baseline_frame["date"] < current_date]
    if not prior.empty:
        expected = np.full(len(df), np.nan)
        samples = np.zeros(len(df), dtype=int)
        for session_name in ("PRE", "REG", "POST"):
            session_mask = session.eq(session_name)
            session_prior = prior[prior["session"] == session_name]
            if not session_mask.any() or session_prior.empty:
                continue
            minutes = sorted(minute_of_day[session_mask].unique())
            by_day = session_prior.pivot_table(
                index="date", columns="minute", values="cumulative", aggfunc="last"
            ).reindex(columns=minutes).ffill(axis=1)
            means = by_day.mean(axis=0)
            counts = by_day.count(axis=0)
            positions = np.flatnonzero(session_mask.to_numpy())
            target_minutes = minute_of_day[session_mask]
            expected[positions] = means.reindex(target_minutes.to_numpy()).to_numpy()
            samples[positions] = counts.reindex(target_minutes.to_numpy()).fillna(0).to_numpy(dtype=int)
        time_adjusted = session_cumulative / pd.Series(expected, index=df.index).replace(0, np.nan)
        current_mask = trading_date == current_date
        df["RVOL"] = fallback_rvol
        df.loc[current_mask, "RVOL"] = time_adjusted.loc[current_mask]
        df["RVOL_BASELINE_SESSIONS"] = 0
        df.loc[current_mask, "RVOL_BASELINE_SESSIONS"] = samples[current_mask.to_numpy()]
        df["RVOL_METHOD"] = "20-bar fallback"
        df.loc[current_mask & time_adjusted.notna(), "RVOL_METHOD"] = "time-adjusted"
    else:
        df["RVOL"] = fallback_rvol
        df["RVOL_BASELINE_SESSIONS"] = 0
        df["RVOL_METHOD"] = "20-bar fallback"

    # Yahoo commonly supplies after-hours price candles with zero volume. Keep
    # the most recent liquid reading from the same trading day instead of
    # replacing a valid regular-session RVOL with N/A.
    liquid_rvol = df["RVOL"].where(volume > 0).groupby(trading_date).ffill()
    liquid_samples = df["RVOL_BASELINE_SESSIONS"].where(volume > 0).groupby(trading_date).ffill()
    carried = df["RVOL"].isna() & liquid_rvol.notna()
    df.loc[carried, "RVOL"] = liquid_rvol.loc[carried]
    df.loc[carried, "RVOL_BASELINE_SESSIONS"] = liquid_samples.loc[carried].fillna(0).astype(int)
    df.loc[carried, "RVOL_METHOD"] = "last liquid candle"
    df["EMA9_SLOPE"] = df["EMA9"].diff(3)
    df["EMA20_SLOPE"] = df["EMA20"].diff(3)
    return df


def latest_trading_day(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest exchange-local date after indicators use prior sessions."""
    if data.empty:
        return data.copy()
    latest_date = data.index[-1].date()
    return data[pd.Index(data.index.date) == latest_date].copy()


def rvol_state(value: float) -> str:
    """Human-readable RVOL zones used by the dashboard and scanner."""
    if not np.isfinite(value): return "No baseline"
    if value < .5: return "Dead quiet"
    if value < 1: return "Below average"
    if value < 1.5: return "Normal"
    if value < 2: return "Getting interesting"
    if value <= 4: return "In play"
    return "Extreme · reversal risk"


def calculate_support_resistance(data: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    recent = data.tail(max(3, lookback))
    return float(recent["Low"].min()), float(recent["High"].max())


def percentage_move(current: float, reference: float) -> float:
    return ((current - reference) / reference) * 100 if reference else 0.0


def move_strength(value: float) -> str:
    if value >= 2: return "Major"
    if value >= 1.5: return "Strong"
    if value >= 1: return "Significant"
    if value >= .5: return "Moderate"
    if value >= .25: return "Early"
    return "Normal"


def _candle(row: pd.Series) -> dict:
    o, h, low, c = map(float, (row["Open"], row["High"], row["Low"], row["Close"]))
    span, body = max(h - low, 1e-9), abs(c - o)
    return {"o": o, "h": h, "l": low, "c": c, "range": span, "body": body,
            "upper": h - max(o, c), "lower": min(o, c) - low, "green": c > o}


def _infer_interval_minutes(data: pd.DataFrame) -> int:
    if len(data.index) < 2:
        return 5
    differences = data.index.to_series().diff().dt.total_seconds().div(60).dropna()
    differences = differences[(differences > 0) & (differences <= 60)]
    return max(1, int(round(float(differences.median())))) if not differences.empty else 5


def _bar_is_complete(data: pd.DataFrame, position: int = -1, now=None) -> bool:
    """Return True only after the selected candle's full interval has elapsed."""
    if data.empty:
        return False
    stamp = pd.Timestamp(data.index[position])
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=stamp.tz)
    if stamp.tzinfo is not None and current.tzinfo is None:
        current = current.tz_localize(stamp.tzinfo)
    elif stamp.tzinfo is None and current.tzinfo is not None:
        current = current.tz_localize(None)
    return current >= stamp + pd.Timedelta(minutes=_infer_interval_minutes(data))


def _neutral_pattern_context(data: pd.DataFrame) -> tuple[str, str]:
    """Evaluate context without converting an unconfirmed neutral candle to directional."""
    if data.empty:
        return "Neutral", "Insufficient context"
    pattern_bar = data.iloc[-1]
    prior = data.iloc[:-1].tail(20)
    if prior.empty:
        return "Neutral", "Insufficient prior candles"
    candle = _candle(pattern_bar)
    close = float(pattern_bar["Close"])
    support, resistance = calculate_support_resistance(prior)
    atr = float(pattern_bar.get("ATR14", candle["range"]))
    if not np.isfinite(atr) or atr <= 0:
        atr = candle["range"]
    tolerance = max(atr * .45, close * .0025)
    near_support = candle["l"] <= support + tolerance
    near_resistance = candle["h"] >= resistance - tolerance
    trend_up = len(prior) >= 4 and float(prior["Close"].iloc[-1]) > float(prior["Close"].iloc[-4])
    trend_down = len(prior) >= 4 and float(prior["Close"].iloc[-1]) < float(prior["Close"].iloc[-4])
    rvol = float(pattern_bar.get("RVOL", 0)) if pd.notna(pattern_bar.get("RVOL")) else 0.0
    volume_supports = rvol >= 1.0
    close_location = (close - candle["l"]) / candle["range"]
    bull_points = (2 if near_support and trend_down else 0) + (1 if close_location >= .58 else 0)
    bear_points = (2 if near_resistance and trend_up else 0) + (1 if close_location <= .42 else 0)
    if volume_supports:
        if bull_points > bear_points:
            bull_points += 1
        elif bear_points > bull_points:
            bear_points += 1
    context_bias = "Bullish" if bull_points >= 2 and bull_points > bear_points else "Bearish" if bear_points >= 2 and bear_points > bull_points else "Neutral"
    location = "support" if near_support else "resistance" if near_resistance else "mid-range"
    trend = "prior downtrend" if trend_down else "prior uptrend" if trend_up else "sideways trend"
    volume = f"RVOL {rvol:.1f}x" if rvol else "volume unavailable"
    return context_bias, f"{trend}; near {location}; {volume}; confirmation still required"


def detect_candlestick_patterns(data: pd.DataFrame) -> list[dict]:
    """Evaluate every supported pattern and return UI-ready rows."""
    if data.empty:
        return [{"name": n, "bias": b, "detected": False, "confidence": 0,
                 "confirmation": confirm, "icon": icon} for n, b, confirm, icon in PATTERN_CATALOG]
    candles = [_candle(row) for _, row in data.tail(6).iterrows()]
    cur, prev = candles[-1], candles[-2] if len(candles) > 1 else candles[-1]
    p2 = candles[-3] if len(candles) > 2 else prev
    trend_up = len(data) >= 6 and data["Close"].iloc[-2] > data["Close"].iloc[-6]
    trend_down = len(data) >= 6 and data["Close"].iloc[-2] < data["Close"].iloc[-6]
    tiny, small = cur["body"] <= cur["range"] * .10, cur["body"] <= cur["range"] * .30
    tweezer_tolerance = max(cur["range"], prev["range"]) * .08
    tweezer_tolerance = max(cur["range"], prev["range"]) * .08
    cond = {
        "Hammer": (small and cur["lower"] >= cur["range"]*.55 and cur["upper"] <= cur["range"]*.15 and trend_down, 88, None),
        "Shooting Star": (small and cur["upper"] >= cur["range"]*.55 and cur["lower"] <= cur["range"]*.15 and trend_up, 88, None),
        "Bullish Engulfing": (not prev["green"] and cur["green"] and cur["o"] <= prev["c"] and cur["c"] >= prev["o"], 92, None),
        "Bearish Engulfing": (prev["green"] and not cur["green"] and cur["o"] >= prev["c"] and cur["c"] <= prev["o"], 92, None),
        "Morning Star": (not p2["green"] and p2["body"] > p2["range"]*.5 and prev["body"] < prev["range"]*.35 and cur["green"] and cur["c"] > (p2["o"]+p2["c"])/2, 90, None),
        "Evening Star": (p2["green"] and p2["body"] > p2["range"]*.5 and prev["body"] < prev["range"]*.35 and not cur["green"] and cur["c"] < (p2["o"]+p2["c"])/2, 90, None),
        "Doji": (tiny, 76, None),
        "Dragonfly Doji": (tiny and cur["lower"] >= cur["range"]*.6 and cur["upper"] <= cur["range"]*.1, 86, None),
        "Gravestone Doji": (tiny and cur["upper"] >= cur["range"]*.6 and cur["lower"] <= cur["range"]*.1, 86, None),
        "Inside Bar": (cur["h"] < prev["h"] and cur["l"] > prev["l"], 78, "Neutral"),
        "Outside Bar": (cur["h"] > prev["h"] and cur["l"] < prev["l"], 82, "Bullish" if cur["green"] else "Bearish"),
        "Three White Soldiers": (len(candles) >= 3 and all(x["green"] and x["body"] >= x["range"]*.5 for x in candles[-3:]) and candles[-1]["c"] > candles[-2]["c"] > candles[-3]["c"], 94, None),
        "Three Black Crows": (len(candles) >= 3 and all(not x["green"] and x["body"] >= x["range"]*.5 for x in candles[-3:]) and candles[-1]["c"] < candles[-2]["c"] < candles[-3]["c"], 94, None),
        "Marubozu": (cur["body"] >= cur["range"]*.90, 89, "Bullish" if cur["green"] else "Bearish"),
        "Spinning Top": (small and cur["upper"] >= cur["body"] and cur["lower"] >= cur["body"], 72, None),
        "Inverted Hammer": (small and cur["upper"] >= cur["range"]*.55 and cur["lower"] <= cur["range"]*.15 and trend_down, 84, None),
        "Hanging Man": (small and cur["lower"] >= cur["range"]*.55 and cur["upper"] <= cur["range"]*.15 and trend_up, 84, None),
        "Piercing Line": (not prev["green"] and cur["green"] and cur["o"] <= prev["c"] and (prev["o"]+prev["c"])/2 < cur["c"] < prev["o"], 86, None),
        "Dark Cloud Cover": (prev["green"] and not cur["green"] and cur["o"] >= prev["c"] and prev["o"] < cur["c"] < (prev["o"]+prev["c"])/2, 86, None),
        "Tweezer Top": (trend_up and prev["green"] and not cur["green"] and abs(cur["h"]-prev["h"]) <= tweezer_tolerance, 82, None),
        "Tweezer Bottom": (trend_down and not prev["green"] and cur["green"] and abs(cur["l"]-prev["l"]) <= tweezer_tolerance, 82, None),
        "Inverted Hammer": (small and cur["upper"] >= cur["range"]*.55 and cur["lower"] <= cur["range"]*.15 and trend_down, 84, None),
        "Hanging Man": (small and cur["lower"] >= cur["range"]*.55 and cur["upper"] <= cur["range"]*.15 and trend_up, 84, None),
        "Piercing Line": (not prev["green"] and cur["green"] and cur["o"] <= prev["c"] and (prev["o"]+prev["c"])/2 < cur["c"] < prev["o"], 86, None),
        "Dark Cloud Cover": (prev["green"] and not cur["green"] and cur["o"] >= prev["c"] and prev["o"] < cur["c"] < (prev["o"]+prev["c"])/2, 86, None),
        "Tweezer Top": (trend_up and prev["green"] and not cur["green"] and abs(cur["h"]-prev["h"]) <= tweezer_tolerance, 82, None),
        "Tweezer Bottom": (trend_down and not prev["green"] and cur["green"] and abs(cur["l"]-prev["l"]) <= tweezer_tolerance, 82, None),
    }
    context_bias, context_note = _neutral_pattern_context(data)
    rows = []
    for name, bias, confirmation, icon in PATTERN_CATALOG:
        detected, confidence, dynamic_bias = cond[name]
        neutral_context = name in {"Doji", "Spinning Top"}
        rows.append({"name": name, "bias": dynamic_bias or bias, "detected": bool(detected),
                     "confidence": confidence if detected else 0, "confirmation": confirmation, "icon": icon,
                     "context_bias": context_bias if neutral_context else dynamic_bias or bias,
                     "context_note": context_note if neutral_context else ""})
    return rows


def detect_candlestick_pattern(data: pd.DataFrame) -> tuple[str, str]:
    hits = [p for p in detect_candlestick_statuses(data) if p["detected"]]
    if not hits: return "No clear pattern", "Neutral"
    status_rank = {"CONFIRMED": 3, "COMPLETE": 2, "WATCH": 1, "FORMING": 0}
    winner = max(hits, key=lambda p: (status_rank.get(p.get("status"), 0), p["confidence"]))
    return winner["name"], winner["bias"]


def detect_candlestick_statuses(data: pd.DataFrame) -> list[dict]:
    """Add detection time and next-candle confirmation state for the UI."""
    rows = detect_candlestick_patterns(data)
    if data.empty:
        return rows
    current_time = data.index[-1]
    current_complete = _bar_is_complete(data)
    for row in rows:
        row["detected_at"] = current_time if row["detected"] else None
        row["confirmed_at"] = current_time if row["detected"] and not row["confirmation"] else None
        row["confirmed"] = bool(row["detected"] and not row["confirmation"])
        row["status"] = "COMPLETE" if row["confirmed"] else "WATCH" if row["detected"] else "—"

    # COMPLETE means the pattern itself is finished. CONFIRMED is reserved for
    # a required follow-through candle that has also completed.
    for row in rows:
        row["confirmed_at"] = None
        row["confirmed"] = False
        if row["detected"]:
            row["status"] = "FORMING" if not current_complete else "WATCH" if row["confirmation"] else "COMPLETE"

    if len(data) < 3 or not current_complete:
        return rows
    previous_rows = detect_candlestick_patterns(data.iloc[:-1])
    pattern_bar = data.iloc[-2]
    confirmation_bar = data.iloc[-1]
    close = float(confirmation_bar["Close"])
    high, low = float(pattern_bar["High"]), float(pattern_bar["Low"])
    by_name = {row["name"]: row for row in rows}
    for previous in previous_rows:
        if not previous["detected"] or not previous["confirmation"]:
            continue
        bias = previous["bias"]
        bullish_break = close > high
        bearish_break = close < low
        confirmation_direction = "Bullish" if bullish_break else "Bearish" if bearish_break else "Neutral"
        if previous["name"] in {"Doji", "Spinning Top"}:
            context_bias = previous.get("context_bias", "Neutral")
            avg_volume = float(confirmation_bar.get("AVG_VOLUME20", 0))
            volume_ratio = float(confirmation_bar.get("Volume", 0)) / avg_volume if avg_volume > 0 else 0.0
            context_aligned = context_bias == confirmation_direction
            strong_confirmation = volume_ratio >= 1.2
            confirmed = confirmation_direction != "Neutral" and (context_aligned or (context_bias == "Neutral" and strong_confirmation))
        else:
            confirmed = bullish_break if "Bullish" in bias else bearish_break if "Bearish" in bias else bullish_break or bearish_break
        if not confirmed:
            continue
        row = by_name[previous["name"]]
        row.update({"detected": True, "confirmed": True, "status": "CONFIRMED",
                    "detected_at": data.index[-2], "confirmed_at": data.index[-1],
                    "confidence": min(99, int(previous["confidence"]) + 5)})
        if bias in {"Neutral", "Directional"} or previous["name"] in {"Doji", "Spinning Top"}:
            row["bias"] = confirmation_direction
        row["context_bias"] = previous.get("context_bias", row["bias"])
        row["context_note"] = previous.get("context_note", "")
    return rows


def _normalize_probabilities(values: list[float]) -> list[int]:
    """Use largest remainders so displayed integer outcomes total exactly 100."""
    clean = np.asarray([max(0.0, float(value)) for value in values], dtype=float)
    if clean.sum() <= 0:
        return [34, 33, 33]
    scaled = clean / clean.sum() * 100
    whole = np.floor(scaled).astype(int)
    for index in np.argsort(-(scaled - whole))[:100 - int(whole.sum())]:
        whole[index] += 1
    return whole.tolist()


def build_price_movement_analysis(data: pd.DataFrame, support: float, resistance: float) -> dict:
    last, price = data.iloc[-1], float(data["Close"].iloc[-1])
    move = percentage_move(price, float(data["Open"].iloc[0]))
    rvol = float(last["RVOL"]) if pd.notna(last.get("RVOL")) else 1
    trend_match = (last["EMA9"] > last["EMA20"]) == (move > 0)
    continuation_raw = 30 + 24*min(1, abs(move)/2) + 16*min(1, max(0, (rvol-.6)/1.8)) + 10*trend_match
    pullback_raw = 24 + min(22, abs(percentage_move(price, float(last["EMA9"]))) * 7)
    opposing_level = resistance if move >= 0 else support
    level_distance = abs(percentage_move(price, opposing_level))
    reversal_raw = 18 + (12 if rvol < 1 else 0) + (12 if level_distance <= .35 else 0)
    continuation, pullback, reversal = _normalize_probabilities([continuation_raw, pullback_raw, reversal_raw])
    return {"Current move": move, "From session low": percentage_move(price, float(data["Low"].min())),
            "From session high": percentage_move(price, float(data["High"].max())),
            "Distance from EMA9": percentage_move(price, float(last["EMA9"])),
            "Distance from EMA20": percentage_move(price, float(last["EMA20"])),
            "Distance from VWAP": percentage_move(price, float(last["VWAP"])),
            "Distance from resistance": percentage_move(price, resistance),
            "Distance from support": percentage_move(price, support),
            "Expected continuation": continuation, "Expected pullback": pullback, "Expected reversal": reversal}


def _build_trade_analysis_legacy(data: pd.DataFrame, spy_change_pct: float | None = None) -> dict:
    if data.empty or len(data) < 5:
        return {"score": 0, "probability": 0, "signal": "WAIT", "option_bias": "WAIT",
                "option_reason": "Not enough candles to establish a directional edge.",
                "pattern": "No data", "pattern_bias": "Neutral",
                "reasons": ["Not enough data"], "breakdown": []}
    last = data.iloc[-1]
    pattern, bias = detect_candlestick_pattern(data)
    breakdown = [
        ("Above VWAP" if last["Close"] > last["VWAP"] else "Below VWAP", 15 if last["Close"] > last["VWAP"] else -15),
        ("EMA9 above EMA20" if last["EMA9"] > last["EMA20"] else "EMA9 below EMA20", 10 if last["EMA9"] > last["EMA20"] else -10),
        ("Price above EMA9" if last["Close"] > last["EMA9"] else "Price below EMA9", 10 if last["Close"] > last["EMA9"] else -10),
    ]
    slopes_up = last.get("EMA9_SLOPE", 0) > 0 and last.get("EMA20_SLOPE", 0) > 0
    slopes_down = last.get("EMA9_SLOPE", 0) < 0 and last.get("EMA20_SLOPE", 0) < 0
    if slopes_up or slopes_down: breakdown.append(("Momentum rising" if slopes_up else "Momentum falling", 10 if slopes_up else -10))
    rvol = float(last["RVOL"]) if pd.notna(last.get("RVOL")) else 0
    if rvol >= 1.5:
        # 2x–4x is the useful participation zone. Extreme RVOL is still
        # meaningful, but receives fewer points because exhaustion risk rises.
        rvol_points = 20 if 2 <= rvol <= 4 else 10
        direction = 1 if last["Close"] >= last["Open"] else -1
        breakdown.append((f"RVOL-TOD {rvol:.1f}x · {rvol_state(rvol)}", rvol_points * direction))
    elif 0 < rvol < 1:
        breakdown.append((f"Low RVOL-TOD {rvol:.1f}x", -5 if last["Close"] >= last["Open"] else 5))
    if "Bullish" in bias: breakdown.append((pattern, 15))
    elif "Bearish" in bias: breakdown.append((pattern, -15))
    recent = data.tail(5)
    if recent["Low"].is_monotonic_increasing: breakdown.append(("Higher lows", 10))
    if recent["High"].is_monotonic_decreasing: breakdown.append(("Lower highs", -10))
    if spy_change_pct is not None:
        stock_move = percentage_move(float(last["Close"]), float(data["Open"].iloc[0]))
        if abs(stock_move-spy_change_pct) >= .2:
            better = stock_move > spy_change_pct
            breakdown.append(("Outperforming SPY" if better else "Underperforming SPY", 5 if better else -5))
    score = int(np.clip(50 + sum(points for _, points in breakdown), 0, 100))
    probability = int(np.clip(50 + abs(score-50)*.88 + min(rvol, 3)*3, 50, 96))
    signal = "CALL WATCH" if score >= 68 else "PUT WATCH" if score <= 32 else "WAIT"
    bullish = sorted(((label, points) for label, points in breakdown if points > 0), key=lambda item: item[1], reverse=True)
    bearish = sorted(((label, points) for label, points in breakdown if points < 0), key=lambda item: item[1])
    if signal == "CALL WATCH":
        option_bias = "LOOK FOR CALLS"
        evidence = ", ".join(label for label, _ in bullish[:3]) or "bullish factors outweigh bearish factors"
        option_reason = f"Bullish edge: {evidence}. Wait for price confirmation before entry."
    elif signal == "PUT WATCH":
        option_bias = "LOOK FOR PUTS"
        evidence = ", ".join(label for label, _ in bearish[:3]) or "bearish factors outweigh bullish factors"
        option_reason = f"Bearish edge: {evidence}. Wait for price confirmation before entry."
    else:
        option_bias = "WAIT · NO CLEAR EDGE"
        if bullish and bearish:
            option_reason = f"Mixed evidence: {bullish[0][0]}, but {bearish[0][0]}. Let direction confirm first."
        else:
            option_reason = "The score is neutral; wait for price, volume, and trend to align."
    return {"score": score, "probability": probability, "signal": signal, "pattern": pattern, "pattern_bias": bias,
            "option_bias": option_bias, "option_reason": option_reason,
            "reasons": [f"{label} {points:+d}" for label, points in breakdown], "breakdown": breakdown}


def build_trade_analysis(data: pd.DataFrame, spy_change_pct: float | None = None) -> dict:
    """Build a directional score whose conviction penalties always move toward WAIT."""
    if data.empty or len(data) < 5:
        return {"score": 50, "probability": 0, "signal": "WAIT", "option_bias": "WAIT",
                "option_reason": "Not enough candles to establish a directional edge.",
                "pattern": "No data", "pattern_bias": "Neutral", "pattern_status": "â€”",
                "top_reasons": ["Not enough data"], "signal_conflict": False,
                "conflict_message": "", "reasons": ["Not enough data"], "breakdown": []}

    last = data.iloc[-1]
    pattern_rows = [row for row in detect_candlestick_statuses(data) if row["detected"]]
    status_rank = {"CONFIRMED": 3, "COMPLETE": 2, "WATCH": 1, "FORMING": 0}
    active_pattern = max(
        pattern_rows,
        key=lambda row: (status_rank.get(row.get("status"), 0), row["confidence"]),
    ) if pattern_rows else None
    pattern = active_pattern["name"] if active_pattern else "No clear pattern"
    bias = active_pattern["bias"] if active_pattern else "Neutral"
    pattern_status = active_pattern.get("status", "â€”") if active_pattern else "â€”"

    breakdown = [
        ("Above VWAP" if last["Close"] > last["VWAP"] else "Below VWAP", 12 if last["Close"] > last["VWAP"] else -12),
        ("EMA9 above EMA20" if last["EMA9"] > last["EMA20"] else "EMA9 below EMA20", 10 if last["EMA9"] > last["EMA20"] else -10),
        ("Price above EMA9" if last["Close"] > last["EMA9"] else "Price below EMA9", 7 if last["Close"] > last["EMA9"] else -7),
    ]
    slopes_up = last.get("EMA9_SLOPE", 0) > 0 and last.get("EMA20_SLOPE", 0) > 0
    slopes_down = last.get("EMA9_SLOPE", 0) < 0 and last.get("EMA20_SLOPE", 0) < 0
    if slopes_up or slopes_down:
        breakdown.append(("Momentum rising" if slopes_up else "Momentum falling", 8 if slopes_up else -8))
    if active_pattern and "Bullish" in bias:
        points = 14 if pattern_status == "CONFIRMED" else 12 if pattern_status == "COMPLETE" else 4
        breakdown.append((f"{pattern} Â· {pattern_status.lower()}", points))
    elif active_pattern and "Bearish" in bias:
        points = 14 if pattern_status == "CONFIRMED" else 12 if pattern_status == "COMPLETE" else 4
        breakdown.append((f"{pattern} Â· {pattern_status.lower()}", -points))
    recent = data.tail(5)
    if recent["Low"].is_monotonic_increasing:
        breakdown.append(("Higher lows", 8))
    if recent["High"].is_monotonic_decreasing:
        breakdown.append(("Lower highs", -8))
    if spy_change_pct is not None:
        stock_move = percentage_move(float(last["Close"]), float(data["Open"].iloc[0]))
        if abs(stock_move-spy_change_pct) >= .2:
            better = stock_move > spy_change_pct
            breakdown.append(("Outperforming SPY" if better else "Underperforming SPY", 4 if better else -4))

    current_score = float(50 + sum(points for _, points in breakdown))
    if current_score > 100:
        breakdown.append(("Directional score cap", int(100-current_score)))
        current_score = 100
    elif current_score < 0:
        breakdown.append(("Directional score floor", int(-current_score)))
        current_score = 0

    def weaken_toward_wait(label: str, magnitude: int) -> None:
        nonlocal current_score
        adjustment = min(magnitude, int(abs(current_score - 50)))
        if adjustment <= 0:
            return
        points = -adjustment if current_score > 50 else adjustment
        breakdown.append((label, points))
        current_score += points

    rvol = float(last["RVOL"]) if pd.notna(last.get("RVOL")) else 0.0
    if rvol >= 1.5 and current_score != 50:
        participation = 12 if 2 <= rvol <= 4 else 6
        available = int(100-current_score) if current_score > 50 else int(current_score)
        participation = min(participation, max(0, available))
        points = participation if current_score > 50 else -participation
        breakdown.append((f"RVOL-TOD {rvol:.1f}x Â· {rvol_state(rvol)}", points))
        current_score += points
    elif rvol < 1:
        weaken_toward_wait(f"Low RVOL-TOD {rvol:.1f}x weakens conviction", 10)

    missing_confirmation = bool(
        active_pattern and active_pattern.get("confirmation") and pattern_status != "CONFIRMED"
    )
    if active_pattern and bias == "Neutral":
        weaken_toward_wait(f"{pattern} is neutral until context confirms", 8)
    if missing_confirmation:
        weaken_toward_wait("Required confirmation candle incomplete or missing", 10)

    reference_data = data.iloc[:-1] if len(data) > 5 else data
    support, resistance = calculate_support_resistance(reference_data)
    close = float(last["Close"])
    resistance_distance = percentage_move(resistance, close)
    support_distance = percentage_move(close, support)
    if current_score > 50 and 0 <= resistance_distance <= .40:
        weaken_toward_wait(f"Resistance only {resistance_distance:.2f}% overhead", 9)
    elif current_score < 50 and 0 <= support_distance <= .40:
        weaken_toward_wait(f"Support only {support_distance:.2f}% below", 9)

    trend_votes = sum((
        last["Close"] > last["VWAP"],
        last["EMA9"] > last["EMA20"],
        last["Close"] > last["EMA9"],
    ))
    signal_conflict = bool(
        active_pattern and (
            (trend_votes >= 2 and bias in {"Bearish", "Neutral"})
            or (trend_votes <= 1 and bias == "Bullish")
        )
    )
    conflict_message = ""
    if signal_conflict:
        trend_label = "bullish" if trend_votes >= 2 else "bearish"
        conflict_message = f"{trend_label.title()} trend indicators conflict with the {bias.lower()} {pattern} signal."
        weaken_toward_wait("Signal conflict reduces conviction", 8)

    score = int(round(np.clip(current_score, 0, 100)))
    signal = "CALL WATCH" if score >= 68 else "PUT WATCH" if score <= 32 else "WAIT"
    bullish = sorted(((label, points) for label, points in breakdown if points > 0), key=lambda item: item[1], reverse=True)
    bearish = sorted(((label, points) for label, points in breakdown if points < 0), key=lambda item: item[1])
    directional_reasons = bullish if signal == "CALL WATCH" else bearish if signal == "PUT WATCH" else sorted(bullish + bearish, key=lambda item: abs(item[1]), reverse=True)
    top_reasons = [label for label, _ in directional_reasons[:3]]
    confidence_penalty = (10 if signal_conflict else 0) + (8 if missing_confirmation else 0) + (8 if rvol < 1 else 0)
    probability = int(np.clip(48 + abs(score-50)*1.25 - confidence_penalty, 35, 95))

    if signal == "CALL WATCH":
        option_bias = "LOOK FOR CALLS"
        evidence = ", ".join(top_reasons) or "bullish factors outweigh bearish factors"
        option_reason = f"Bullish edge: {evidence}. Wait for the stated entry condition."
    elif signal == "PUT WATCH":
        option_bias = "LOOK FOR PUTS"
        evidence = ", ".join(top_reasons) or "bearish factors outweigh bullish factors"
        option_reason = f"Bearish edge: {evidence}. Wait for the stated entry condition."
    else:
        option_bias = "WAIT Â· NO CLEAR EDGE"
        if signal_conflict:
            option_reason = conflict_message + " Wait for price, volume, and candle confirmation to align."
        elif bullish and bearish:
            option_reason = f"Mixed evidence: {bullish[0][0]}, but {bearish[0][0]}. Let direction confirm first."
        else:
            option_reason = "The score is neutral; wait for price, volume, and trend to align."
    return {"score": score, "probability": probability, "signal": signal,
            "pattern": pattern, "pattern_bias": bias, "pattern_status": pattern_status,
            "top_reasons": top_reasons, "signal_conflict": signal_conflict,
            "conflict_message": conflict_message, "option_bias": option_bias,
            "option_reason": option_reason,
            "reasons": [f"{label} {points:+d}" for label, points in breakdown],
            "breakdown": breakdown}


def build_decision_card(data: pd.DataFrame, support: float, resistance: float,
                        analysis: dict, interval: str) -> dict:
    """Create a compact, risk-defined plan without presenting it as an order."""
    last = data.iloc[-1]
    action = analysis.get("signal", "WAIT").replace(" WATCH", "")
    atr = float(last.get("ATR14", 0)) if pd.notna(last.get("ATR14")) else 0.0
    price = float(last["Close"])
    atr = atr if atr > 0 else max(price * .003, .01)
    if action == "CALL":
        entry = max(resistance, price + atr * .10)
        invalidation = min(entry - atr * .35, max(support, float(last["EMA20"])))
        target = entry + max(entry - invalidation, atr) * 1.5
        condition = f"{interval.upper()} candle closes above ${entry:.2f} with RVOL at least 1.5x"
        invalidation_text, target_text = f"Close below ${invalidation:.2f}", f"${target:.2f} (about 1.5R)"
    elif action == "PUT":
        entry = min(support, price - atr * .10)
        invalidation = max(entry + atr * .35, min(resistance, float(last["EMA20"])))
        target = max(.01, entry - max(invalidation - entry, atr) * 1.5)
        condition = f"{interval.upper()} candle closes below ${entry:.2f} with RVOL at least 1.5x"
        invalidation_text, target_text = f"Close above ${invalidation:.2f}", f"${target:.2f} (about 1.5R)"
    else:
        condition = f"Wait for a completed {interval.upper()} close above ${resistance:.2f} or below ${support:.2f}"
        invalidation_text, target_text = "No position before trigger", "Set after confirmation (1.5R)"
    if analysis.get("pattern_status") in {"WATCH", "FORMING"}:
        condition += "; pattern confirmation must complete"
    directional_confidence = int(analysis.get("probability", 0))
    decision_confidence = 100-directional_confidence if action == "WAIT" else directional_confidence
    return {"action": action, "confidence": decision_confidence,
            "entry_condition": condition, "invalidation": invalidation_text,
            "first_target": target_text}


def assess_data_quality(data: pd.DataFrame, interval: str, now=None) -> dict:
    """Identify extended-hours candles and a stale Yahoo feed."""
    if data.empty:
        return {"extended": False, "delayed": True, "label": "No market data", "warning": "No market data is available."}
    latest = pd.Timestamp(data.index[-1])
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=latest.tz)
    if latest.tzinfo is not None and current.tzinfo is None:
        current = current.tz_localize(latest.tzinfo)
    elif latest.tzinfo is None and current.tzinfo is not None:
        current = current.tz_localize(None)
    latest_minute = latest.hour * 60 + latest.minute
    current_minute = current.hour * 60 + current.minute
    extended = latest_minute < 570 or latest_minute >= 960
    interval_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30}.get(interval, _infer_interval_minutes(data))
    feed_active = current.weekday() < 5 and 240 <= current_minute < 1200 and current.date() == latest.date()
    age_minutes = max(0.0, (current - (latest + pd.Timedelta(minutes=interval_minutes))).total_seconds() / 60)
    delayed = bool(feed_active and age_minutes > max(3, interval_minutes * 1.5))
    warnings = []
    if extended:
        warnings.append("Extended-hours data: spreads can be wider and volume can be thin")
    if delayed:
        warnings.append(f"Data appears delayed by about {int(round(age_minutes))} minutes")
    label = "Extended hours" if extended else "Delayed feed" if delayed else "Regular-hours feed"
    return {"extended": extended, "delayed": delayed, "label": label,
            "warning": " Â· ".join(warnings), "age_minutes": age_minutes}


def build_alerts(data: pd.DataFrame, analysis: dict, movement: dict, threshold: float) -> list[dict]:
    last, prev = data.iloc[-1], data.iloc[-2]
    patterns = {p["name"]: p for p in detect_candlestick_statuses(data) if p["detected"]}
    rvol = float(last["RVOL"]) if pd.notna(last.get("RVOL")) else 0
    atr = float(last["ATR14"]) if pd.notna(last.get("ATR14")) else 0
    items = []
    if abs(movement["Current move"]) >= threshold: items.append(("HIGH", f"Move threshold {movement['Current move']:+.2f}%", min(96, 70+int(abs(movement["Current move"])*8))))
    if atr and last["High"]-last["Low"] >= atr*1.5: items.append(("HIGH", "Large candle detected", 86))
    for name in ("Hammer", "Bullish Engulfing", "Bearish Engulfing"):
        if name in patterns and patterns[name].get("status") == "CONFIRMED":
            items.append(("HIGH", f"{name} confirmed", patterns[name]["confidence"]))
        elif name in patterns:
            items.append(("MED", f"{name} detected Â· confirmation pending", patterns[name]["confidence"]))
    if 2 <= rvol <= 4: items.append(("HIGH", f"RVOL-TOD {rvol:.2f}x · In play", 91))
    elif rvol > 4: items.append(("HIGH", f"RVOL-TOD {rvol:.2f}x · Extreme / reversal risk", 88))
    if last["Volume"] > last["AVG_VOLUME20"]*1.8: items.append(("MED", "Volume spike", 84))
    if (last["EMA9"]-last["EMA20"])*(prev["EMA9"]-prev["EMA20"]) < 0: items.append(("HIGH", "EMA crossover", 88))
    if (last["Close"]-last["VWAP"])*(prev["Close"]-prev["VWAP"]) < 0: items.append(("MED", "VWAP breakout", 82))
    prior = data.iloc[-21:-1] if len(data) > 21 else data.iloc[:-1]
    if not prior.empty and last["Close"] > prior["High"].max(): items.append(("HIGH", "Resistance breakout", 90))
    if not prior.empty and last["Close"] < prior["Low"].min(): items.append(("HIGH", "Support breakdown", 90))
    if analysis["signal"] != "WAIT" and ((analysis["score"] > 50) != (prev["Close"] > prev["VWAP"])): items.append(("MED", "Trend reversal", 78))
    if analysis.get("signal_conflict"):
        items.append(("HIGH", f"Signal conflict Â· {analysis['conflict_message']}", max(60, analysis.get("probability", 60))))
    if not items: items.append(("LOW", "No high-conviction trigger", 55))
    rank = {"HIGH": 0, "MED": 1, "LOW": 2}
    return [{"priority": p, "name": n, "confidence": c, "continuation": movement["Expected continuation"], "pullback": movement["Expected pullback"]}
            for p, n, c in sorted(items, key=lambda x: (rank[x[0]], -x[2]))]
