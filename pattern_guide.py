"""Compact candlestick formation diagrams and plain-language education."""
from __future__ import annotations

import pandas as pd

GUIDES = {
    "Hammer": ("Selling was rejected after a decline; buyers recovered most of the candle.", "A close above the hammer high."),
    "Shooting Star": ("Buyers pushed higher but sellers rejected the move near the top of an advance.", "A close below the star's low."),
    "Bullish Engulfing": ("A strong green body fully overwhelms the prior red body, showing a shift to buyers.", "Follow-through above the engulfing high."),
    "Bearish Engulfing": ("A strong red body fully overwhelms the prior green body, showing a shift to sellers.", "Follow-through below the engulfing low."),
    "Morning Star": ("A three-candle bottom: selling weakens, pauses, then buyers reclaim the prior decline.", "The third candle closes above the first candle's midpoint."),
    "Evening Star": ("A three-candle top: buying weakens, pauses, then sellers reclaim the prior advance.", "The third candle closes below the first candle's midpoint."),
    "Doji": ("Open and close are nearly equal, signaling balance and indecision between buyers and sellers.", "A break of the doji high or low in context."),
    "Dragonfly Doji": ("Sellers drove price lower but buyers recovered to the open, leaving a long lower wick.", "A close above the dragonfly high."),
    "Gravestone Doji": ("Buyers drove price higher but sellers forced it back to the open, leaving an upper wick.", "A close below the gravestone low."),
    "Inside Bar": ("The candle is contained inside the prior range, showing compression before expansion.", "A decisive break of the mother bar high or low."),
    "Outside Bar": ("The candle exceeds both sides of the prior range, showing aggressive two-way expansion.", "A close and follow-through in the outside bar's direction."),
    "Three White Soldiers": ("Three strong rising green candles show sustained, orderly buyer control.", "Usually complete; watch for continuation without exhaustion."),
    "Three Black Crows": ("Three strong falling red candles show sustained, orderly seller control.", "Usually complete; watch for continuation without exhaustion."),
    "Marubozu": ("A large body with almost no wicks shows one side controlled the candle from open to close.", "Continuation beyond the candle high or low."),
    "Spinning Top": ("A small body with wicks on both sides shows uncertainty after two-way rejection.", "A break of its range; direction depends on trend context."),
}

GUIDES.update({
    "Inverted Hammer": ("After a decline, buyers tested higher prices but did not yet control the close.", "A completed close above the inverted hammer high."),
    "Hanging Man": ("After an advance, a deep intrabar selloff warns that support may be weakening.", "A completed close below the hanging man's low."),
    "Piercing Line": ("Buyers recovered more than half of the prior bearish body after a decline.", "Follow-through above the piercing candle high."),
    "Dark Cloud Cover": ("Sellers erased more than half of the prior bullish body after an advance.", "Follow-through below the dark-cloud candle low."),
    "Tweezer Top": ("Two candles rejected approximately the same high after an advance.", "A completed close below the pair's low or nearby support."),
    "Tweezer Bottom": ("Two candles rejected approximately the same low after a decline.", "A completed close above the pair's high or nearby resistance."),
})

# Values are normalized prices: (open, close, high, low). The last few bars
# include surrounding context so the formation is understandable at a glance.
SCHEMATICS = {
    "Hammer": [(82,72,87,68),(73,62,77,58),(61,58,65,22),(57,72,76,53)],
    "Shooting Star": [(35,45,49,31),(44,55,59,40),(56,59,92,53),(59,45,63,41)],
    "Bullish Engulfing": [(68,58,72,54),(59,48,63,44),(50,72,76,46),(70,79,83,66)],
    "Bearish Engulfing": [(32,43,47,28),(42,54,58,38),(56,30,60,26),(32,23,36,19)],
    "Morning Star": [(76,48,80,44),(45,43,50,39),(42,69,73,38),(68,78,82,64)],
    "Evening Star": [(28,57,61,24),(60,62,66,56),(63,35,67,31),(36,26,40,22)],
    "Doji": [(64,52,68,48),(51,51,67,35),(52,61,65,48)],
    "Dragonfly Doji": [(64,52,68,48),(51,51,55,18),(52,64,68,48)],
    "Gravestone Doji": [(36,48,52,32),(49,49,83,45),(48,36,52,32)],
    "Inside Bar": [(46,70,76,40),(64,55,69,50),(56,72,76,52)],
    "Outside Bar": [(48,61,67,43),(64,38,72,31),(39,29,43,25)],
    "Three White Soldiers": [(72,58,76,54),(60,45,64,41),(47,31,51,27),(32,22,36,18)],
    "Three Black Crows": [(28,42,46,24),(40,56,60,36),(54,70,74,50),(69,79,83,65)],
    "Marubozu": [(62,52,68,48),(70,28,71,27),(29,20,33,16)],
    "Spinning Top": [(60,50,64,46),(53,49,72,31),(50,58,62,46)],
}

SCHEMATICS.update({
    "Inverted Hammer": [(72,62,76,58),(62,58,25,56),(58,70,74,54)],
    "Hanging Man": [(36,48,52,32),(49,53,57,84),(54,42,58,38)],
    "Piercing Line": [(42,68,72,38),(70,51,74,47),(52,40,56,36)],
    "Dark Cloud Cover": [(66,38,70,34),(36,55,59,32),(54,66,70,50)],
    "Tweezer Top": [(52,34,28,56),(35,53,28,57),(54,66,70,50)],
    "Tweezer Bottom": [(38,58,62,66),(59,41,63,66),(40,30,44,26)],
})

PATTERN_LENGTHS = {
    "Bullish Engulfing": 2, "Bearish Engulfing": 2, "Inside Bar": 2,
    "Outside Bar": 2, "Morning Star": 3, "Evening Star": 3,
    "Three White Soldiers": 3, "Three Black Crows": 3,
    "Piercing Line": 2, "Dark Cloud Cover": 2,
    "Tweezer Top": 2, "Tweezer Bottom": 2,
}


def snapshot_svg(name: str) -> str:
    candles = SCHEMATICS.get(name, SCHEMATICS["Doji"])
    width, height = 246, 82
    spacing = width / (len(candles) + 1)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="formation-svg" aria-label="{name} formation">',
             '<line x1="0" y1="41" x2="246" y2="41" stroke="#e6ebf2" stroke-dasharray="3 4"/>']
    for index, (open_, close, high, low) in enumerate(candles, 1):
        x = round(index * spacing, 1)
        y = lambda value: round(7 + value * .68, 1)
        bullish = close < open_  # SVG y coordinates run downward.
        color = "#20b985" if bullish else "#ef5366"
        top, bottom = min(y(open_), y(close)), max(y(open_), y(close))
        body_height = max(3.0, bottom - top)
        parts.append(f'<line x1="{x}" y1="{y(high)}" x2="{x}" y2="{y(low)}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="{x-7}" y="{top}" width="14" height="{body_height}" rx="1" fill="{color}"/>')
    parts.append('</svg>')
    return ''.join(parts)


def market_snapshot_svg(data: pd.DataFrame, name: str, pattern_time,
                        confirmation_time=None) -> str | None:
    """Render actual candles surrounding a detected pattern."""
    if data.empty or pattern_time is None:
        return None
    try:
        pattern_position = int(data.index.get_indexer([pattern_time], method="nearest")[0])
    except Exception:
        return None
    length = PATTERN_LENGTHS.get(name, 1)
    start = max(0, pattern_position - max(3, length))
    end = min(len(data), pattern_position + 3)
    sample = data.iloc[start:end]
    if sample.empty:
        return None
    low_price, high_price = float(sample["Low"].min()), float(sample["High"].max())
    price_span = max(high_price - low_price, 1e-9)
    width, height = 246, 98
    spacing = width / (len(sample) + 1)
    to_y = lambda price: round(8 + (high_price - float(price)) / price_span * 62, 1)
    confirmation_position = None
    if confirmation_time is not None:
        try: confirmation_position = int(data.index.get_indexer([confirmation_time], method="nearest")[0])
        except Exception: pass
    pattern_start = pattern_position - length + 1
    parts = [f'<svg viewBox="0 0 {width} {height}" class="formation-svg actual-svg" aria-label="Actual {name} candles">',
             '<line x1="0" y1="39" x2="246" y2="39" stroke="#e6ebf2" stroke-dasharray="3 4"/>']
    for local_index, (timestamp, row) in enumerate(sample.iterrows(), 1):
        global_index = start + local_index - 1
        x = round(local_index * spacing, 1)
        bullish = float(row["Close"]) >= float(row["Open"])
        color = "#20b985" if bullish else "#ef5366"
        top, bottom = sorted((to_y(row["Open"]), to_y(row["Close"])))
        body_height = max(3.0, bottom - top)
        if pattern_start <= global_index <= pattern_position:
            parts.append(f'<rect x="{x-10}" y="4" width="20" height="69" rx="3" fill="rgba(36,140,255,.07)" stroke="#248cff" stroke-width="1.4"/>')
        parts.append(f'<line x1="{x}" y1="{to_y(row["High"])}" x2="{x}" y2="{to_y(row["Low"])}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="{x-6}" y="{top}" width="12" height="{body_height}" rx="1" fill="{color}"/>')
        if global_index == confirmation_position:
            parts.append(f'<circle cx="{x}" cy="5" r="4" fill="#248cff"/><text x="{x}" y="7" text-anchor="middle" font-size="5" font-weight="900" fill="#fff">C</text>')
        parts.append(f'<text x="{x}" y="88" text-anchor="middle" font-size="6.5" fill="#718096">{timestamp.strftime("%H:%M")}</text>')
    parts.append('<text x="4" y="96" font-size="6.5" font-weight="800" fill="#248cff">BLUE OUTLINE = PATTERN · C = CONFIRMATION</text></svg>')
    return ''.join(parts)


def guide_html(name: str, bias: str, confirmation_needed: bool, lift: bool = False,
               actual_svg: str | None = None, snapshot_label: str = "FORMATION GUIDE",
               context_note: str = "") -> str:
    meaning, confirmation = GUIDES.get(name, ("A price-action formation requiring trend context.", "Wait for follow-through."))
    position_class = " tip-lift" if lift else ""
    bias_class = "bull" if "Bull" in bias else "bear" if "Bear" in bias else "neutral-tip"
    needed = "Needed" if confirmation_needed else "Pattern complete"
    diagram = actual_svg or snapshot_svg(name)
    if context_note:
        confirmation += f'<br><b>Context:</b> {context_note}'
    return (f'<div class="pattern-tip{position_class}"><div class="tip-head"><strong>{name}</strong>'
            f'<span class="{bias_class}">{bias.upper()}</span></div><div class="snapshot-label">{snapshot_label}</div>{diagram}'
            f'<div class="tip-meaning">{meaning}</div><div class="tip-confirm"><b>Confirmation · {needed}</b><br>{confirmation}</div></div>')
