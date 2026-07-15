import unittest

import numpy as np
import pandas as pd

from indicators import (
    build_alerts,
    build_decision_card,
    build_price_movement_analysis,
    build_trade_analysis,
    calculate_indicators,
    calculate_support_resistance,
    detect_candlestick_patterns,
    detect_candlestick_statuses,
)


def market_frame(last_volume: float = 1000, doji: bool = False) -> pd.DataFrame:
    index = pd.date_range("2026-07-14 10:00", periods=30, freq="5min", tz="America/New_York")
    close = np.linspace(100, 104, len(index))
    open_ = close - .12
    high = close + .28
    low = open_ - .25
    volume = np.full(len(index), 1000.0)
    volume[-1] = last_volume
    if doji:
        open_[-1] = close[-1]
        high[-1] = close[-1] + .5
        low[-1] = close[-1] - .5
    raw = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)
    return calculate_indicators(raw)


class TradingLogicTests(unittest.TestCase):
    def test_doji_defaults_neutral_until_confirmation(self):
        rows = detect_candlestick_patterns(market_frame(doji=True))
        doji = next(row for row in rows if row["name"] == "Doji")
        self.assertTrue(doji["detected"])
        self.assertEqual(doji["bias"], "Neutral")
        self.assertTrue(doji["confirmation"])

    def test_outcome_probabilities_total_100(self):
        data = market_frame()
        support, resistance = calculate_support_resistance(data)
        movement = build_price_movement_analysis(data, support, resistance)
        total = sum(movement[key] for key in (
            "Expected continuation", "Expected pullback", "Expected reversal"
        ))
        self.assertEqual(total, 100)

    def test_low_rvol_reduces_directional_conviction(self):
        high_volume = build_trade_analysis(market_frame(last_volume=3000))
        low_volume = build_trade_analysis(market_frame(last_volume=100))
        self.assertLessEqual(abs(low_volume["score"] - 50), abs(high_volume["score"] - 50))
        self.assertTrue(any("Low RVOL" in reason for reason in low_volume["reasons"]))

    def test_neutral_candle_can_raise_signal_conflict(self):
        analysis = build_trade_analysis(market_frame(doji=True))
        self.assertEqual(analysis["pattern_bias"], "Neutral")
        self.assertTrue(analysis["signal_conflict"])

    def test_score_matches_explainable_contributions(self):
        analysis = build_trade_analysis(market_frame(last_volume=3000))
        self.assertEqual(analysis["score"], 50 + sum(points for _, points in analysis["breakdown"]))

    def test_unconfirmed_pattern_alert_is_not_called_confirmed(self):
        data = market_frame(doji=True)
        support, resistance = calculate_support_resistance(data)
        movement = build_price_movement_analysis(data, support, resistance)
        analysis = build_trade_analysis(data)
        alerts = build_alerts(data, analysis, movement, .5)
        self.assertFalse(any("Doji confirmed" in alert["name"] for alert in alerts))
        statuses = detect_candlestick_statuses(data)
        doji = next(row for row in statuses if row["name"] == "Doji")
        self.assertNotEqual(doji["status"], "CONFIRMED")

    def test_decision_card_has_risk_defined_fields(self):
        data = market_frame()
        support, resistance = calculate_support_resistance(data)
        analysis = build_trade_analysis(data)
        card = build_decision_card(data, support, resistance, analysis, "5m")
        self.assertEqual(set(card), {"action", "confidence", "entry_condition", "invalidation", "first_target"})
        self.assertIn(card["action"], {"CALL", "PUT", "WAIT"})


if __name__ == "__main__":
    unittest.main()
