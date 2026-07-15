import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from indicators import PATTERN_CATALOG, calculate_indicators
from risk_engine import (
    analyze_levels,
    analyze_market_context,
    analyze_trend,
    analyze_volume,
    assess_risk_controls,
    build_conservative_decision,
    build_risk_alerts,
    expected_move_engine,
    historical_pattern_stats,
)


def frame(direction=1, last_volume=2500):
    index = pd.date_range("2026-07-14 10:00", periods=40, freq="5min", tz="America/New_York")
    close = np.linspace(100, 102 if direction > 0 else 98, len(index))
    open_ = close-direction*.08
    high = np.maximum(open_, close)+.16
    low = np.minimum(open_, close)-.16
    volume = np.full(len(index), 900.0)
    volume[-1] = last_volume
    return calculate_indicators(pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=index))


def components(direction=1, delayed=False, risk_settings=None):
    stock = frame(direction)
    spy = frame(direction)
    qqq = frame(direction)
    market = analyze_market_context(stock, spy, qqq, qqq, "QQQ")
    trend = analyze_trend(stock)
    volume = analyze_volume(stock)
    levels = analyze_levels(stock, "5m", volume)
    quality = {"delayed": delayed, "extended": False}
    decision = build_conservative_decision(
        "TEST", stock, "5m", market, trend, volume, levels, quality,
        risk_settings=risk_settings or {},
    )
    return stock, trend, volume, levels, decision


class RiskEngineTests(unittest.TestCase):
    def test_quality_score_is_auditable(self):
        *_, decision = components(1)
        self.assertEqual(decision["score"], sum(points for _, points in decision["breakdown"]))
        self.assertGreaterEqual(decision["score"], 0)
        self.assertLessEqual(decision["score"], 100)

    def test_bearish_direction_does_not_require_low_quality_score(self):
        *_, decision = components(-1)
        self.assertEqual(decision["direction"], "PUT")
        self.assertGreaterEqual(decision["score"], 40)

    def test_delayed_data_forces_no_trade(self):
        *_, decision = components(1, delayed=True)
        self.assertEqual(decision["action"], "NO TRADE")
        self.assertIn("delayed", decision["main_risk"].lower())

    def test_expected_outcomes_total_100(self):
        stock, trend, volume, levels, _ = components(1)
        expected = expected_move_engine(stock, "CALL", levels, trend, volume)
        self.assertEqual(expected["continuation"]+expected["pullback"]+expected["reversal"], 100)
        self.assertIn("Heuristic", expected["label"])
        self.assertIn("remaining_move", expected)
        self.assertGreater(expected["continuation_move"], 0)
        self.assertLess(expected["pullback_move"], 0)
        self.assertGreater(expected["target"], expected["entry"])

    def test_component_scores_are_bounded_and_complete(self):
        *_, decision = components(1)
        expected_names = {"Trend Strength", "Momentum", "Volume Intelligence",
                          "Market Alignment", "Risk Reward", "Expected Move"}
        self.assertEqual(set(decision["components"]), expected_names)
        self.assertTrue(all(0 <= value <= 100 for value in decision["components"].values()))

    def test_grade_mapping_matches_terminal_policy(self):
        *_, decision = components(1)
        score = decision["score"]
        expected_grade = "A+" if score >= 90 else "A" if score >= 80 else "B" if score >= 70 else "C" if score >= 60 else "D" if score >= 50 else "F"
        self.assertEqual(decision["grade"], expected_grade)

    def test_daily_loss_limit_locks_trading(self):
        risk = assess_risk_controls({
            "account_size": 10000, "daily_loss_limit_pct": 1.5,
            "daily_pnl": -151, "max_trades": 3, "trades_today": 0,
            "stop_after_losses": 2, "consecutive_losses": 0,
        })
        self.assertTrue(risk["locked"])
        self.assertIn("DAILY LOSS LIMIT", risk["reasons"][0])

    def test_weak_volume_cannot_create_directional_action(self):
        stock = frame(1, last_volume=50)
        market = analyze_market_context(stock, frame(1), frame(1), frame(1), "QQQ")
        trend, volume = analyze_trend(stock), analyze_volume(stock)
        levels = analyze_levels(stock, "5m", volume)
        decision = build_conservative_decision("TEST", stock, "5m", market, trend, volume, levels,
                                               {"delayed": False, "extended": False})
        self.assertIn(decision["action"], {"NO EDGE", "LOW EDGE", "NO TRADE"})
        self.assertNotIn(decision["execution_action"], {"CALL", "PUT"})
        self.assertTrue(any("Weak volume" in label for label, _ in decision["factor_breakdown"]))

    def test_full_pattern_catalog_is_available(self):
        names = {row[0] for row in PATTERN_CATALOG}
        self.assertTrue({"Inverted Hammer", "Hanging Man", "Piercing Line", "Dark Cloud Cover",
                         "Tweezer Top", "Tweezer Bottom"}.issubset(names))
        self.assertEqual(len(names), 21)

    def test_historical_statistics_do_not_fabricate_small_samples(self):
        result = historical_pattern_stats(frame(1), "Bullish Engulfing", "5m", min_sample=20)
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["label"], "Insufficient historical sample")
        self.assertGreaterEqual(result["sample_size"], 0)

    def test_alert_schema_and_confirmation_language(self):
        stock, trend, volume, levels, decision = components(1)
        decision["pattern"] = {"name": "Doji", "detected": True, "status": "WATCH", "confirmed": False}
        alerts = build_risk_alerts("TEST", "5m", stock, decision, {}, volume, levels,
                                   {"delayed": False}, .5)
        required = {"ticker", "timestamp", "timeframe", "priority", "current_price",
                    "reference_price", "percentage_move", "pattern", "confidence",
                    "continuation_probability", "pullback_risk", "relevant_level", "reason",
                    "action_required"}
        self.assertTrue(required.issubset(alerts[0]))
        pattern_alert = next(item for item in alerts if item["pattern"] == "Doji")
        self.assertIn("incomplete", pattern_alert["reason"])
        self.assertNotIn("confirmed", pattern_alert["reason"].lower())

    def test_options_are_withheld_without_live_chain_liquidity(self):
        *_, decision = components(1)
        self.assertFalse(decision["options"]["eligible"])
        self.assertIn("live chain", decision["options"]["status"].lower())

    def test_missing_market_benchmarks_force_no_trade(self):
        stock = frame(1)
        empty = stock.iloc[0:0]
        market = analyze_market_context(stock, empty, empty, empty, "QQQ")
        trend, volume = analyze_trend(stock), analyze_volume(stock)
        levels = analyze_levels(stock, "5m", volume)
        decision = build_conservative_decision("TEST", stock, "5m", market, trend, volume,
                                               levels, {"delayed": False, "extended": False})
        self.assertEqual(decision["action"], "NO TRADE")
        self.assertIn("benchmark", decision["main_risk"].lower())

    def test_forming_neutral_pattern_is_not_a_signal_conflict(self):
        stock = frame(-1)
        market = analyze_market_context(stock, frame(-1), frame(-1), frame(-1), "QQQ")
        trend, volume = analyze_trend(stock), analyze_volume(stock)
        levels = analyze_levels(stock, "5m", volume)
        forming_doji = [{"name": "Doji", "bias": "Neutral", "detected": True,
                         "confidence": 76, "confirmation": True, "confirmed": False,
                         "status": "FORMING", "context_note": "indecision"}]
        with patch("risk_engine.detect_candlestick_statuses", return_value=forming_doji):
            decision = build_conservative_decision("TEST", stock, "5m", market, trend, volume,
                                                   levels, {"delayed": False, "extended": False})
        self.assertFalse(any("neutral Doji" in conflict for conflict in decision["conflicts"]))
        self.assertTrue(any("still forming" in label for label, _ in decision["factor_breakdown"]))

    def test_stock_trade_and_readiness_scores_are_separate_and_auditable(self):
        *_, decision = components(1)
        for key in ("stock_score", "trade_score", "trade_readiness"):
            self.assertGreaterEqual(decision[key], 0)
            self.assertLessEqual(decision[key], 100)
        self.assertEqual(decision["trade_score"], sum(points for _, points in decision["breakdown"]))
        self.assertEqual(decision["stock_score"], sum(points for _, points in decision["stock_breakdown"]))
        self.assertEqual(decision["trade_readiness"], sum(points for _, points in decision["readiness_breakdown"]))

    def test_delayed_data_sets_readiness_to_zero(self):
        *_, decision = components(1, delayed=True)
        self.assertEqual(decision["trade_readiness"], 0)
        self.assertEqual(decision["action"], "NO TRADE")

    def test_user_facing_action_uses_edge_vocabulary(self):
        *_, decision = components(1)
        allowed = {"HIGH EDGE CALLS", "HIGH EDGE PUTS", "MODERATE EDGE CALLS",
                   "MODERATE EDGE PUTS", "LOW EDGE", "NO EDGE", "NO TRADE"}
        self.assertIn(decision["action"], allowed)


if __name__ == "__main__":
    unittest.main()
