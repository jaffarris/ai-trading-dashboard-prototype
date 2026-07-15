"""Local CORS bridge used by the persistent browser-side chart."""
from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from data_provider import get_intraday_data, to_chart_timestamp
from indicators import calculate_indicators, calculate_support_resistance

_server: ThreadingHTTPServer | None = None
INTERVAL_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800}


def _finite(value, fallback=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _history_payload(data, ticker: str, interval: str) -> dict:
    seconds = INTERVAL_SECONDS[interval]
    candle_map, volume_map = {}, {}
    volume_cap = max(1.0, float(data["Volume"].quantile(.95)))
    for index, row in data.iterrows():
        timestamp = to_chart_timestamp(index, seconds)
        bullish = float(row["Close"]) >= float(row["Open"])
        candle_map[timestamp] = {"time": timestamp, "open": float(row["Open"]), "high": float(row["High"]),
                                 "low": float(row["Low"]), "close": float(row["Close"])}
        volume_map[timestamp] = {"time": timestamp, "value": min(float(row["Volume"]), volume_cap),
                                 "actual": float(row["Volume"]),
                                 "color": "rgba(37,214,149,.42)" if bullish else "rgba(255,92,114,.42)"}
    def line(column):
        points = {}
        for index, value in data[column].dropna().items():
            timestamp = to_chart_timestamp(index, seconds)
            points[timestamp] = {"time": timestamp, "value": float(value)}
        return list(points.values())
    sessions = []
    local_index = data.index
    for session_date in sorted(set(local_index.date)):
        day = data[local_index.date == session_date]
        pre = day[(day.index.hour >= 4) & ((day.index.hour < 9) | ((day.index.hour == 9) & (day.index.minute < 30)))]
        after = day[(day.index.hour >= 16) & (day.index.hour < 20)]
        for kind, frame in (("premarket", pre), ("afterhours", after)):
            if len(frame) >= 2:
                sessions.append({"kind": kind,
                    "start": to_chart_timestamp(frame.index[0], seconds),
                    "end": to_chart_timestamp(frame.index[-1], seconds)})
    support, resistance = calculate_support_resistance(data)
    return {"ticker": ticker, "interval": interval, "candles": list(candle_map.values()),
            "ema9": line("EMA9"), "ema20": line("EMA20"), "vwap": line("VWAP"),
            "volume": list(volume_map.values()), "sessions": sessions,
            "support": support, "resistance": resistance, "volume_cap": volume_cap}


class _Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        try:
            query = parse_qs(urlparse(self.path).query)
            ticker = query.get("ticker", [""])[0].upper()
            interval = query.get("interval", ["5m"])[0]
            mode = query.get("mode", ["latest"])[0]
            if not ticker.isalnum() or len(ticker) > 8:
                self._send({"error": "Invalid symbol"}, 400)
                return
            if interval not in {"1m", "3m", "5m", "15m", "30m"}:
                self._send({"error": "Invalid interval"}, 400)
                return
            data = calculate_indicators(get_intraday_data(ticker, interval=interval))
            if data.empty:
                self._send({"error": "No data"}, 503)
                return
            if mode == "history":
                self._send(_history_payload(data, ticker, interval))
                return
            support, resistance = calculate_support_resistance(data)
            row = data.iloc[-1]
            interval_seconds = INTERVAL_SECONDS[interval]
            timestamp = to_chart_timestamp(data.index[-1], interval_seconds)
            bullish = float(row["Close"]) >= float(row["Open"])
            volume_cap = max(1.0, _finite(data["Volume"].quantile(.95), 1.0))
            volume = _finite(row["Volume"], 0.0)
            self._send({
                "candle": {"time": timestamp, "open": float(row["Open"]), "high": float(row["High"]),
                           "low": float(row["Low"]), "close": float(row["Close"])},
                "ema9": {"time": timestamp, "value": _finite(row["EMA9"])},
                "ema20": {"time": timestamp, "value": _finite(row["EMA20"])},
                "vwap": {"time": timestamp, "value": _finite(row["VWAP"])},
                "volume": {"time": timestamp, "value": min(volume, volume_cap),
                           "actual": volume,
                           "color": "rgba(37,214,149,.42)" if bullish else "rgba(255,92,114,.42)"},
                "support": support, "resistance": resistance,
            })
        except Exception as exc:
            self._send({"error": str(exc)[:160]}, 500)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8501")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, payload: dict, status: int = 200):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def ensure_live_proxy(port: int = 8502) -> None:
    global _server
    if _server is not None:
        return
    try:
        _server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    except OSError:
        return
    threading.Thread(target=_server.serve_forever, name="apex-live-chart", daemon=True).start()
