from datetime import datetime
from html import escape
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from data_provider import get_analysis_data, get_intraday_data, to_chart_timestamp
from live_chart import render_live_chart
from live_proxy import ensure_live_proxy
from pattern_guide import guide_html, market_snapshot_svg
from indicators import (assess_data_quality, build_alerts, build_decision_card,
    build_price_movement_analysis, build_trade_analysis,
    calculate_indicators, calculate_support_resistance, detect_candlestick_patterns,
    detect_candlestick_statuses, latest_trading_day, move_strength, percentage_move,
    rvol_state)
from scanner import WATCHLIST, scan_watchlist
from risk_engine import (
    SECTOR_ETFS, analyze_levels, analyze_market_context, analyze_trend, analyze_volume,
    build_conservative_decision, build_risk_alerts, classify_move_strength,
    historical_pattern_stats,
)
from trade_coach import (POSITION_TYPES, chart_snapshot_svg, coach_response,
                         profit_target_scenarios)

PREFERENCES_FILE = Path(__file__).with_name("dashboard_preferences.json")


def load_last_ticker() -> str:
    try:
        ticker = json.loads(PREFERENCES_FILE.read_text(encoding="utf-8")).get("last_ticker")
        return ticker if ticker in WATCHLIST else "NVDA"
    except (OSError, ValueError, TypeError):
        return "NVDA"


def save_last_ticker(ticker: str) -> None:
    if ticker not in WATCHLIST:
        return
    try:
        PREFERENCES_FILE.write_text(json.dumps({"last_ticker": ticker}, indent=2), encoding="utf-8")
    except OSError:
        pass

st.set_page_config(page_title="ApexFlow Trading Terminal", page_icon="▲", layout="wide", initial_sidebar_state="collapsed")


def running_on_streamlit_cloud() -> bool:
    """Keep the localhost chart bridge out of Streamlit's cloud process."""
    return (
        Path("/mount/src").exists()
        or os.environ.get("STREAMLIT_SHARING_MODE", "").lower() == "streamlit"
        or os.environ.get("STREAMLIT_RUNTIME_ENV", "").lower() == "cloud"
    )


if not running_on_streamlit_cloud():
    ensure_live_proxy()

st.markdown("""
<style>
:root{--bg:#070b11;--panel:#0d141e;--panel2:#111b28;--line:#243244;--text:#e8eef7;--muted:#8492a6;--cyan:#35c2ff;--green:#25d695;--red:#ff5c72;--amber:#ffb547}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,sans-serif}
[data-stale="true"]{opacity:1!important;filter:none!important;transition:none!important}
[data-testid="stAppViewContainer"],[data-testid="stVerticalBlock"],[data-testid="stPlotlyChart"]{transition:none!important;animation:none!important}
[data-testid="stHeader"],footer,#MainMenu{display:none}.block-container{max-width:100%;padding:.4rem .75rem .25rem}[data-testid="stVerticalBlock"]{gap:.36rem}[data-testid="column"]{padding:0 .1rem}
h1,h2,h3,p{margin:0}.terminal-head{height:35px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);margin-bottom:2px}.brand{font-size:19px;font-weight:850;letter-spacing:.4px}.brand b{color:var(--cyan)}.status{font-size:11px;color:var(--muted)}.live{color:var(--green);font-weight:800}.terminal-head .live{font-size:0}.terminal-head .live:after{content:"YAHOO FEED";font-size:11px}
div[data-baseweb="select"]>div{background:var(--panel)!important;border-color:var(--line)}.stSelectbox>div>div{min-height:36px}.stSelectbox label,.stSlider label{font-size:10px!important;color:var(--muted)!important}.stButton button{height:36px;border:1px solid var(--cyan);background:#0a2231;color:var(--cyan);font-weight:800}
.metric-grid{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:6px;padding-bottom:11px;box-sizing:content-box}.metric{height:73px;min-height:73px;box-sizing:border-box;background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:8px;padding:9px 10px;position:relative;overflow:hidden}.metric:before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--accent,var(--cyan))}.metric-label{font-size:10px;line-height:12px;letter-spacing:.65px;text-transform:uppercase;color:var(--muted);font-weight:750;white-space:nowrap}.metric-value{font-size:20px;line-height:27px;font-weight:850;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.metric-sub{font-size:9px;line-height:11px;color:var(--muted)}.up{color:var(--green)}.down{color:var(--red)}.neutral{color:var(--amber)}
.decision-card{display:grid;grid-template-columns:1.05fr .8fr 2.15fr 1.15fr 1.1fr;gap:0;background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden;margin:1px 0 6px}.decision-item{min-width:0;padding:7px 10px;border-right:1px solid var(--line)}.decision-item:last-child{border-right:0}.decision-label{font-size:8px;font-weight:850;letter-spacing:.7px;color:var(--muted);text-transform:uppercase}.decision-value{font-size:11px;line-height:1.25;font-weight:750;white-space:normal}.decision-action{font-size:14px;line-height:1.15;font-weight:900}.decision-call{color:var(--green)}.decision-put{color:var(--red)}.decision-wait{color:var(--amber)}.decision-warning{grid-column:1/-1;padding:5px 10px;border-top:1px solid var(--line);font-size:9.5px;font-weight:750;background:#fff5df;color:#8c5d00}.conflict-warning{grid-column:1/-1;padding:5px 10px;border-top:1px solid #f0bcc4;font-size:9.5px;font-weight:800;background:#fff0f2;color:#b3263b}
.context-strip{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin:-1px 0 5px}.context-item{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 7px;font-size:9px;min-width:0}.context-item b{display:block;font-size:9.5px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.context-item span{color:var(--muted)}
.best-setup{display:grid;grid-template-columns:1.35fr repeat(5,1fr);gap:0;background:var(--panel);border:1px solid var(--line);border-radius:7px;margin:-1px 0 5px;overflow:hidden}.best-cell{padding:4px 8px;border-right:1px solid var(--line);min-width:0}.best-cell:last-child{border-right:0}.best-cell span{display:block;color:var(--muted);font-size:7.5px;font-weight:800;text-transform:uppercase;letter-spacing:.5px}.best-cell b{display:block;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.best-title b{color:var(--cyan);letter-spacing:.5px}
.panel{height:100%;box-sizing:border-box;background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}.panel-title{height:30px;box-sizing:border-box;padding:7px 10px;border-bottom:1px solid var(--line);font-size:10px;font-weight:850;letter-spacing:.8px;color:#b8c4d4;text-transform:uppercase;display:flex;justify-content:space-between}.panel-body{padding:6px 8px}.scanner-wrap,.chart-wrap,.pattern-wrap{height:407px}
.scanner-table,.pattern-table,.movement-table,.alert-table{width:100%;border-collapse:collapse;table-layout:fixed}th{font-size:8px;color:#738197;text-align:left;padding:4px 3px;border-bottom:1px solid var(--line);text-transform:uppercase}td{font-size:10.5px;padding:5.5px 3px;border-bottom:1px solid #172231;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.ticker{font-weight:850;color:var(--cyan)}.score{font-weight:850}.scanner-table a{display:block;color:inherit;text-decoration:none;margin:-5.5px -3px;padding:5.5px 3px}.scanner-table tr{cursor:pointer;transition:background .12s ease}.scanner-table tbody tr:hover{background:rgba(53,194,255,.11)}.scanner-wrap .panel-body{height:calc(100% - 30px);overflow-y:auto}.scanner-reasons td{padding:2px 4px 5px!important;font-size:7.5px!important;line-height:1.25;color:#596b82;white-space:normal!important;background:#f5f8fc}.scanner-reasons b{color:#31435a}.pattern-table td{padding:3.2px 3px;font-size:9.5px}.pattern-table .hit{background:#12231f}.pattern-icon{font-size:13px;text-align:center}.bias-bullish{color:var(--green)}.bias-bearish{color:var(--red)}.bias-neutral,.bias-directional{color:var(--amber)}
.scanner-table tbody tr.selected{background:rgba(53,194,255,.16);box-shadow:inset 2px 0 var(--cyan)}
.pattern-wrap{position:relative;z-index:50;overflow:hidden!important}.pattern-wrap .panel-body{height:calc(100% - 30px);box-sizing:border-box;overflow-y:auto!important;overflow-x:hidden!important;scrollbar-width:thin}.pattern-table{overflow:visible!important}.pattern-name-cell{position:relative;overflow:visible!important;cursor:help}.pattern-table tr.confirmed{background:#e3f1ff!important;box-shadow:inset 3px 0 #248cff}.pattern-table tr.confirmed td{border-bottom-color:#b9d8f7}.status-confirmed{color:#0877d1;font-weight:900}.status-watch{color:#b87500;font-weight:800}.status-forming{color:#9a6500;font-weight:800}.status-complete{color:#16865f;font-weight:800}.pattern-time{color:#607086;font-variant-numeric:tabular-nums}.pattern-tip{display:none;position:fixed;z-index:1000;right:calc(28% + 8px);top:150px;width:282px;box-sizing:border-box;padding:10px;background:#fff;border:1px solid #bfcbd9;border-radius:8px;box-shadow:0 8px 24px rgba(19,32,51,.22);white-space:normal;color:#26364b;text-align:left}.pattern-table tr:hover .pattern-tip{display:block}.pattern-tip.tip-lift{top:150px;bottom:auto}.tip-head{display:flex;align-items:center;justify-content:space-between;font-size:11px;margin-bottom:3px}.tip-head span{font-size:8px;font-weight:900;letter-spacing:.5px;padding:2px 5px;border-radius:3px}.snapshot-label{font-size:7px;font-weight:900;letter-spacing:.7px;color:#708096;margin-top:2px}.bull{color:#087d57;background:#ddf6ec}.bear{color:#c72f45;background:#ffe5e9}.neutral-tip{color:#9a6500;background:#fff1cf}.formation-svg{display:block;width:100%;height:92px;background:#f8fafc;border:1px solid #e1e7ef;border-radius:5px;margin:4px 0}.tip-meaning{font-size:10px;line-height:1.35;margin:6px 1px}.tip-confirm{font-size:9px;line-height:1.35;color:#5f6f84;background:#f1f5f9;border-radius:4px;padding:6px}.tip-confirm b{color:#34465e}
.bottom-panel{height:239px}.movement-table td{padding:2px 4px;font-size:10px;line-height:1.05}.movement-table td:last-child{text-align:right;font-weight:800}.alert-table td{padding:3.7px;font-size:9.5px}.pri-high{color:var(--red)}.pri-med{color:var(--amber)}.pri-low{color:var(--muted)}
.reason-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px}.reason{height:26px;box-sizing:border-box;padding:5px 7px;background:#111c29;border:1px solid #1d2b3d;border-radius:5px;font-size:9.5px;display:flex;justify-content:space-between}.points{font-weight:900}.option-outlook{margin-top:5px;min-height:34px;box-sizing:border-box;padding:5px 8px;border:1px solid var(--line);border-radius:6px;display:flex;align-items:center;gap:10px}.option-outlook b{font-size:11px;white-space:nowrap}.option-outlook span{font-size:9px;line-height:1.25;color:#53647a}.option-call{border-color:var(--green);background:#eaf8f3}.option-call b{color:#087d57}.option-put{border-color:var(--red);background:#fff0f2}.option-put b{color:#c72f45}.option-wait{border-color:var(--amber);background:#fff8e8}.option-wait b{color:#9a6500}.score-total{margin-top:5px;padding:5px 8px;border:1px solid var(--cyan);border-radius:6px;display:flex;align-items:center;justify-content:space-between}.score-total strong{font-size:19px;color:var(--cyan)}.micro{font-size:8.5px;color:var(--muted)}.foot{font-size:8.5px;color:#657286;text-align:right}.mobile-section{display:none}.mobile-section summary{cursor:pointer;list-style:none;padding:11px 12px;font-size:11px;font-weight:850;letter-spacing:.65px;text-transform:uppercase;color:#384861}.mobile-section summary:after{content:"+";float:right;color:#248cff;font-size:16px}.mobile-section[open] summary:after{content:"âˆ’"}.mobile-section-body{padding:0 7px 8px}.desktop-only{display:block}
[data-testid="stPlotlyChart"]{height:407px!important;box-sizing:border-box;background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}.js-plotly-plot,.plot-container,.svg-container{height:405px!important}@media(max-width:1400px){.metric-value{font-size:16px}.metric-label{font-size:8px}}
@media(max-width:768px){
html,body,[data-testid="stAppViewContainer"]{overflow-x:hidden!important}.block-container{padding:.35rem .45rem 1rem}.terminal-head{height:auto;min-height:38px;gap:8px}.brand{font-size:17px}.brand span{display:none}.status{font-size:9px;text-align:right;white-space:nowrap}
[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:.45rem!important}[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important;min-width:0!important}
.st-key-terminal_controls [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{flex:1 1 calc(50% - .25rem)!important;width:calc(50% - .25rem)!important}.st-key-terminal_controls [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child,.st-key-terminal_controls [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:nth-child(4){flex-basis:100%!important;width:100%!important}
.stButton button,.stSelectbox>div>div{min-height:42px}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;padding:4px 0 9px}.metric-grid .metric:last-child{grid-column:1/-1}.metric{height:82px;min-height:82px;padding:10px}.metric-label{font-size:9px}.metric-value{font-size:19px;line-height:28px}.metric-sub{font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.decision-card{grid-template-columns:repeat(2,minmax(0,1fr))}.decision-item{border-bottom:1px solid var(--line)}.decision-item:nth-child(3){grid-column:1/-1}.decision-warning,.conflict-warning{grid-column:1/-1}.best-setup{grid-template-columns:1fr 1fr}.context-strip{grid-template-columns:1fr 1fr}.desktop-only{display:none!important}.mobile-section{display:block;background:var(--panel);border:1px solid var(--line);border-radius:9px;margin:5px 0;overflow:visible}.mobile-section .panel{border:0;box-shadow:none}.mobile-section .panel-title{display:none}.pattern-wrap,.bottom-panel{display:none!important}
.scanner-wrap,.pattern-wrap{height:auto;min-height:0}.panel-title{height:34px;padding:9px 8px;font-size:10px}.panel-body{padding:6px}.scanner-table td{font-size:11px;padding:7px 3px}.pattern-table td{font-size:9px;padding:5px 2px}.pattern-table th{font-size:7px;padding:4px 2px}.pattern-tip,.pattern-tip.tip-lift{position:fixed;left:8px;right:8px;top:12vh;bottom:auto;width:auto;max-height:76vh;overflow-y:auto;z-index:9999}
.bottom-panel{height:auto;min-height:220px}.movement-table td,.alert-table td{font-size:10px;padding:6px 4px}.reason-grid{grid-template-columns:1fr}.reason{height:auto;min-height:30px;font-size:10px;align-items:center}.option-outlook{align-items:flex-start;flex-direction:column;gap:3px}.option-outlook b{font-size:12px}.option-outlook span{font-size:10px}.score-total{padding:8px}.foot{text-align:left;font-size:9px;padding:5px 1px}
[data-testid="stDialog"] [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{flex-basis:100%!important;width:100%!important}.target-grid{grid-template-columns:repeat(3,minmax(0,1fr))!important}.target-card{padding:6px 3px!important}.target-card strong{font-size:14px!important}
}
@media(max-width:390px){.metric{padding:9px 8px}.metric-value{font-size:17px}.scanner-table td{font-size:10px}.pattern-table td{font-size:8px}.status{font-size:8px}.st-key-terminal_controls [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{flex-basis:100%!important;width:100%!important}}
</style>""", unsafe_allow_html=True)


def tone(value):
    return "up" if value > 0 else "down" if value < 0 else "neutral"


def metric(label, value, sub="", color="neutral"):
    accent = {"up":"var(--green)","down":"var(--red)","neutral":"var(--amber)","cyan":"var(--cyan)"}[color]
    return f'<div class="metric" style="--accent:{accent}"><div class="metric-label">{escape(label)}</div><div class="metric-value {color}">{escape(str(value))}</div><div class="metric-sub">{escape(sub)}</div></div>'


def simple_trade_instruction(action: str) -> str:
    return {
        "HIGH EDGE CALLS": "CALLS — READY",
        "MODERATE EDGE CALLS": "CALLS — READY",
        "HIGH EDGE PUTS": "PUTS — READY",
        "MODERATE EDGE PUTS": "PUTS — READY",
        "LOW EDGE": "WAIT — NOT READY",
        "NO EDGE": "WAIT — NO SETUP",
        "NO TRADE": "DO NOT TRADE",
    }.get(action, action)


@st.dialog("AI trade coach", width="large", icon=":material/smart_toy:")
def show_trade_coach(ticker: str, interval: str) -> None:
    history = get_analysis_data(ticker, interval=interval)
    if history.empty:
        st.error(f"No chart data is available for {ticker}.")
        return
    data = latest_trading_day(calculate_indicators(history))
    if data.empty:
        st.error(f"No current-session chart is available for {ticker}.")
        return

    latest = float(data["Close"].iloc[-1])
    volume = analyze_volume(data); trend = analyze_trend(data)
    levels = analyze_levels(data, interval, volume)
    support, resistance = levels["support"], levels["resistance"]
    def coach_benchmark(symbol):
        try:
            result=get_analysis_data(symbol,interval=interval)
            return latest_trading_day(calculate_indicators(result)) if not result.empty else data.iloc[0:0]
        except Exception:
            return data.iloc[0:0]
    spy=data if ticker=="SPY" else coach_benchmark("SPY")
    qqq=data if ticker=="QQQ" else coach_benchmark("QQQ")
    sector_symbol=SECTOR_ETFS.get(ticker,"QQQ")
    sector=data if ticker==sector_symbol else qqq if sector_symbol=="QQQ" else coach_benchmark(sector_symbol)
    market=analyze_market_context(data,spy,qqq,sector,sector_symbol)
    quality=assess_data_quality(data,interval)
    risk_settings={key:st.session_state.get(key,value) for key,value in RISK_DEFAULTS.items()}
    decision=build_conservative_decision(ticker,data,interval,market,trend,volume,levels,quality,risk_settings=risk_settings)
    expected=decision["expected"]
    analysis={"option_bias": decision["signal"],
              "option_reason": f"{decision['key_reason']}. Main risk: {decision['main_risk']}",
              "pattern": decision.get("pattern",{}).get("name","No clear pattern"),
              "signal": "CALL WATCH" if decision["direction"]=="CALL" else "PUT WATCH" if decision["direction"]=="PUT" else "WAIT",
              "probability": decision["confidence"], "action": decision["action"],
              "expected": expected, "decision": decision}
    movement={"Expected continuation": expected["continuation"], "Expected pullback": expected["pullback"],
              "Expected continuation move": expected["continuation_move"],
              "Expected move remaining": expected["remaining_move"]}
    last = data.iloc[-1]
    rvol = volume["rvol"]

    st.markdown("""
    <style>
    .coach-summary{padding:7px 9px;border:1px solid #ccd5e1;border-radius:7px;background:#f5f8fc;font-size:11px;color:#26364b;margin-bottom:7px}.coach-summary b{color:#0877d1}
    .target-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:5px}.target-card{padding:7px;border:1px solid #d7dee8;border-radius:6px;background:#f7f9fc;text-align:center}.target-card.best{border-color:#25a97a;background:#eaf8f3}.target-card span{display:block;font-size:9px;color:#607086}.target-card strong{font-size:15px;color:#132033}.coach-note{font-size:9px;line-height:1.3;color:#607086;margin-top:5px}
    </style>
    """, unsafe_allow_html=True)
    would_take="YES" if decision["execution_action"] in {"CALL","PUT"} else "NO"
    coach_instruction=simple_trade_instruction(decision["action"])
    st.markdown(
        f'<div class="coach-summary"><b>Take this trade? {would_take}. · {escape(coach_instruction)}</b><br>'
        f'Stock {decision["stock_score"]}/100 · Trade {decision["trade_score"]}/100 · '
        f'Readiness {decision["trade_readiness"]}% ({escape(decision["readiness_label"])})<br>'
        f'{escape(analysis["option_reason"])}</div>',
        unsafe_allow_html=True,
    )
    reason_lines="".join(f"<li>{escape(label)}</li>" for label,_ in sorted((item for item in decision["factor_breakdown"] if item[1]<0),key=lambda item:item[1])[:6])
    if not reason_lines: reason_lines="<li>No material negative factor detected.</li>"
    coach_trade=(f'<b>{escape(decision["signal"])}</b> · Entry: {escape(decision["entry_condition"])} · '
                 f'Stop: {escape(decision["invalidation"])} · T1 {decision["targets"]["1.5R"]:.2f} · '
                 f'T2 {decision["targets"]["2.5R"]:.2f} · T3 {decision["targets"]["4R"]:.2f} · '
                 f'Max account risk: ${decision["risk"]["max_risk_dollars"]:,.0f}')
    st.markdown(f'<div class="coach-summary"><b>Professional plan</b><ol style="margin:4px 0 5px 18px">{reason_lines}</ol>'
                f'<div>{coach_trade}</div><div class="coach-note">Expected continuation {expected["continuation_move"]:+.2f}% · '
                f'pullback {expected["pullback_move"]:+.2f}% · reversal {expected["reversal_move"]:+.2f}% · '
                f'remaining {expected["remaining_move"]:+.2f}% over roughly {escape(expected["estimated_time"])}. '
                f'{escape(decision["options"]["status"])}</div></div>',unsafe_allow_html=True)

    chart_col, position_col = st.columns([1.3, 1], gap="medium")
    with chart_col:
        st.markdown(chart_snapshot_svg(data, ticker, support, resistance, analysis), unsafe_allow_html=True)
        st.caption(f"{ticker} · {interval.upper()} · ${latest:.2f} · RVOL-TOD {rvol:.2f}× · {analysis['pattern']}")

    positions = st.session_state.setdefault("coach_positions", {})
    position = positions.get(ticker)
    with position_col:
        st.markdown("**Recently purchased position**")
        default_type = position.get("position_type", "Long shares") if position else "Long shares"
        default_entry = float(position.get("entry_price", 0.0)) if position else 0.0
        with st.form(f"coach_position_form_{ticker}", border=False):
            position_type = st.selectbox("Position type", POSITION_TYPES, index=POSITION_TYPES.index(default_type))
            entry_price = st.number_input("Price paid", min_value=0.0, value=default_entry, step=0.01, format="%.2f")
            saved = st.form_submit_button("Save purchase", width="stretch")
        if saved and entry_price > 0:
            position = {"position_type": position_type, "entry_price": float(entry_price)}
            positions[ticker] = position
        if position and float(position.get("entry_price", 0)) > 0:
            scenarios = profit_target_scenarios(float(position["entry_price"]), position["position_type"], analysis, movement, rvol)
            cards = "".join(
                f'<div class="target-card {"best" if percent==scenarios["plausible"] else ""}"><span>{percent}% PROFIT</span><strong>${price:.2f}</strong></div>'
                for percent, price in scenarios["targets"].items()
            )
            st.markdown(f'<div class="target-grid">{cards}</div><div class="coach-note"><b>Highlighted scenario:</b> {escape(scenarios["reason"])}<br>{escape(scenarios["note"])}</div>', unsafe_allow_html=True)
        else:
            st.caption("Enter the price paid to calculate 10%, 15%, and 20% sell-price scenarios.")

    messages_by_ticker = st.session_state.setdefault("coach_messages", {})
    messages = messages_by_ticker.setdefault(ticker, [{
        "role": "assistant",
        "content": f"Ask me about {ticker}'s chart, calls or puts, RVOL, risk levels, or a profit target.",
    }])
    message_area = st.container(height=205, border=True)
    with message_area:
        for message in messages[-12:]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    prompt = st.chat_input(
        f"Ask about {ticker}…",
        key=f"coach_chat_{ticker}",
        max_chars=500,
        submit_mode="disable",
    )
    if prompt:
        response = coach_response(prompt, ticker, latest, analysis, movement, support, resistance, rvol, position)
        messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ])
        with message_area:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                st.write(response)
    st.caption("Educational decision support only. Yahoo data may be delayed. Options are complex and a purchased option can lose its entire premium.")


st.session_state.setdefault("chart_focus", None)
st.session_state.setdefault("coach_positions", {})
st.session_state.setdefault("coach_messages", {})
st.session_state.setdefault("ticker_select", load_last_ticker())
st.session_state.setdefault("interval_select", "5m")
requested_symbol = st.query_params.get("symbol")
requested_interval = st.query_params.get("interval")
requested_focus = st.query_params.get("focus")
if requested_symbol in WATCHLIST:
    st.session_state["ticker_select"] = requested_symbol
    st.session_state["chart_focus"] = None
    save_last_ticker(requested_symbol)
if requested_interval in {"1m", "3m", "5m", "15m", "30m"}:
    st.session_state["interval_select"] = requested_interval
if requested_focus:
    try: st.session_state["chart_focus"] = int(requested_focus)
    except ValueError: pass
if requested_symbol or requested_interval or requested_focus:
    st.query_params.clear()

def clear_chart_focus():
    st.session_state["chart_focus"] = None

def ticker_changed():
    clear_chart_focus()
    save_last_ticker(st.session_state.get("ticker_select", "NVDA"))

RISK_DEFAULTS = {
    "account_size": 25000.0, "max_risk_pct": 0.5, "daily_loss_limit_pct": 1.5,
    "daily_profit_lock_pct": 2.0, "max_trades": 3, "stop_after_losses": 2,
    "daily_pnl": 0.0, "trades_today": 0, "consecutive_losses": 0,
    "manual_support": 0.0, "manual_resistance": 0.0,
}
for risk_key, risk_value in RISK_DEFAULTS.items():
    st.session_state.setdefault(risk_key, risk_value)

header_interval = st.session_state.get("interval_select", "5m").upper()
st.markdown(f'<div class="terminal-head"><div class="brand"><b>▲ APEX</b>FLOW <span style="color:#607086;font-size:11px">AI TRADING TERMINAL</span></div><div class="status"><span class="live">● LIVE</span> &nbsp; {header_interval} DATA &nbsp; | &nbsp; {datetime.now():%I:%M:%S %p ET}</div></div>', unsafe_allow_html=True)
with st.container(key="terminal_controls"):
    c1,c2,c3,c4,c5,c6,c7=st.columns([1.0,.55,1.18,1.55,.62,.52,.62],gap="small")
    with c1: ticker=st.selectbox("SYMBOL",WATCHLIST,key="ticker_select",on_change=ticker_changed,label_visibility="collapsed")
    with c2: interval=st.selectbox("INTERVAL",["1m","3m","5m","15m","30m"],key="interval_select",on_change=clear_chart_focus,label_visibility="collapsed")
    with c3: reference_choice=st.selectbox("REFERENCE",["Session open","Previous candle close","Recent swing high","VWAP"],label_visibility="collapsed")
    with c4: custom_threshold=st.slider("MOVE ALERT THRESHOLD (%)",.10,3.00,.50,.05,format="Alert ≥ %.2f%%",help="Creates an alert when the absolute price move reaches this percentage from the selected reference.",label_visibility="collapsed")
    with c5: coach_open=st.button("AI COACH",icon=":material/smart_toy:",width="stretch")
    with c6:
        with st.popover("RISK", width="stretch"):
            st.caption("Capital protection settings")
            st.number_input("Account size ($)", min_value=100.0, step=500.0, key="account_size")
            st.number_input("Maximum risk per trade (%)", min_value=.1, max_value=5.0, step=.1, key="max_risk_pct")
            st.number_input("Daily loss limit (%)", min_value=.1, max_value=10.0, step=.1, key="daily_loss_limit_pct")
            st.number_input("Daily profit lock (%)", min_value=.1, max_value=20.0, step=.1, key="daily_profit_lock_pct")
            st.number_input("Maximum trades per day", min_value=1, max_value=20, step=1, key="max_trades")
            st.number_input("Stop after consecutive losses", min_value=1, max_value=10, step=1, key="stop_after_losses")
            st.number_input("Today's P/L ($)", step=25.0, key="daily_pnl")
            st.number_input("Trades today", min_value=0, max_value=100, step=1, key="trades_today")
            st.number_input("Consecutive losses", min_value=0, max_value=20, step=1, key="consecutive_losses")
            st.number_input("Manual support (0 = auto)", min_value=0.0, step=.01, key="manual_support")
            st.number_input("Manual resistance (0 = auto)", min_value=0.0, step=.01, key="manual_resistance")
    with c7:
        if st.button("↻ REFRESH",width="stretch"): st.rerun()
if coach_open: show_trade_coach(ticker,interval)
st.markdown("""
<style>
:root{--bg:#edf1f6;--panel:#ffffff;--panel2:#f7f9fc;--line:#ccd5e1;--text:#132033;--muted:#607086}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important}
[data-testid="stPlotlyChart"]{background:var(--panel)!important}
.reason{background:#f2f5f9;border-color:#dbe2eb}.metric{background:linear-gradient(145deg,var(--panel2),var(--panel))}
.panel{box-shadow:0 1px 4px rgba(27,39,55,.08)}.panel-title{color:#384861!important}
.metric-label,th{color:#526176!important}.metric-sub,.micro,.foot,.status{color:#617086!important}
td{border-bottom-color:#e3e8ef}.pattern-table .hit{background:#e8f7f1}.scanner-table tbody tr:hover{background:#e5f5fc}
.pattern-jump{color:inherit;text-decoration:none;font-weight:750}.pattern-jump:hover{color:#0877d1;text-decoration:underline}.jump-mark{color:#248cff;font-size:9px;margin-left:3px}
</style>
""", unsafe_allow_html=True)

@st.fragment(run_every="15s")
def render_live_dashboard(ticker, interval, reference_choice, custom_threshold, chart_focus):
    history=get_analysis_data(ticker, interval=interval)
    if history.empty:
        st.error(f"No intraday data returned for {ticker}."); return
    data=latest_trading_day(calculate_indicators(history))
    if data.empty:
        st.error(f"No current-session data returned for {ticker}."); return
    last=data.iloc[-1]; latest=float(last["Close"])
    data_quality=assess_data_quality(data,interval)
    volume=analyze_volume(data); trend=analyze_trend(data)
    manual_support=st.session_state.get("manual_support") or None
    manual_resistance=st.session_state.get("manual_resistance") or None
    levels=analyze_levels(data,interval,volume,manual_support,manual_resistance)
    support,resistance=levels["support"],levels["resistance"]
    refs={"Session open":float(data["Open"].iloc[0]),"Previous candle close":float(data["Close"].iloc[-2]) if len(data)>1 else latest,"Recent swing high":float(data["High"].tail(20).max()),"VWAP":float(last["VWAP"])}
    move_pct=percentage_move(latest,refs[reference_choice]); pullback_pct=percentage_move(latest,float(data["High"].tail(20).max())); session_move=percentage_move(latest,float(data["Open"].iloc[0]))
    def benchmark(symbol):
        try:
            result=get_analysis_data(symbol,interval=interval)
            return latest_trading_day(calculate_indicators(result)) if not result.empty else data.iloc[0:0]
        except Exception:
            return data.iloc[0:0]
    spy=data if ticker=="SPY" else benchmark("SPY")
    qqq=data if ticker=="QQQ" else benchmark("QQQ")
    sector_symbol=SECTOR_ETFS.get(ticker,"QQQ")
    sector=data if ticker==sector_symbol else qqq if sector_symbol=="QQQ" else benchmark(sector_symbol)
    market=analyze_market_context(data,spy,qqq,sector,sector_symbol)
    risk_settings={key:st.session_state.get(key,value) for key,value in RISK_DEFAULTS.items()}
    decision=build_conservative_decision(ticker,data,interval,market,trend,volume,levels,data_quality,risk_settings=risk_settings)
    patterns=detect_candlestick_statuses(data)
    alerts=build_risk_alerts(ticker,interval,data,decision,market,volume,levels,data_quality,custom_threshold)
    expected=decision["expected"]
    prior_close=float(data["Close"].iloc[-2]) if len(data)>1 else latest
    movement={
        "Move from session open": percentage_move(latest,float(data["Open"].iloc[0])),
        "Move from previous candle close": percentage_move(latest,prior_close),
        "Move from recent swing high": percentage_move(latest,float(data["High"].tail(20).max())),
        "Move from recent swing low": percentage_move(latest,float(data["Low"].tail(20).min())),
        "Pullback from 20-bar high": percentage_move(latest,float(data["High"].tail(20).max())),
        "Recovery from 20-bar low": percentage_move(latest,float(data["Low"].tail(20).min())),
        "Distance from EMA9": percentage_move(latest,float(last["EMA9"])),
        "Distance from EMA20": percentage_move(latest,float(last["EMA20"])),
        "Distance from VWAP": percentage_move(latest,float(last["VWAP"])),
        "Distance from resistance": percentage_move(latest,resistance),
        "Distance from support": percentage_move(latest,support),
        "Distance from premarket high": percentage_move(latest,levels["premarket_high"]) if pd.notna(levels.get("premarket_high")) else 0.0,
        "Distance from premarket low": percentage_move(latest,levels["premarket_low"]) if pd.notna(levels.get("premarket_low")) else 0.0,
        "Expected continuation move": expected["continuation_move"],
        "Expected pullback move": expected["pullback_move"],
        "Expected reversal move": expected["reversal_move"],
        "Expected move remaining": expected["remaining_move"],
    }
    rvol=float(last["RVOL"]) if pd.notna(last["RVOL"]) else 0; strength_pct=min(100,int(abs(session_move)/2*100))
    rvol_samples=int(last.get("RVOL_BASELINE_SESSIONS",0)); rvol_method=str(last.get("RVOL_METHOD","20-bar fallback"))
    if rvol_method=="time-adjusted": rvol_sub=f"{rvol_state(rvol)} · {rvol_samples} prior"
    elif rvol_method=="last liquid candle": rvol_sub=f"{rvol_state(rvol)} · last liquid candle"
    else: rvol_sub="20-bar fallback"
    move_class=classify_move_strength(ticker,data,volume)
    direction=market["status"].upper()
    action_class="decision-call" if "CALL" in decision["action"] else "decision-put" if "PUT" in decision["action"] else "decision-wait"
    display_action=simple_trade_instruction(decision["action"])
    warning_html=f'<div class="decision-warning">âš  {escape(data_quality["warning"])}</div>' if data_quality.get("warning") else ''
    conflict_html=''.join(f'<div class="conflict-warning">SIGNAL CONFLICT - {escape(message)}</div>' for message in decision.get("conflicts",[]))
    if data_quality.get("warning"):
        warning_html=f'<div class="decision-warning">DATA WARNING - {escape(data_quality["warning"])}</div>'
    scanner_df=scan_watchlist()
    st.markdown(
        f'<div class="decision-card"><div class="decision-item"><div class="decision-label">Trade now?</div><div class="decision-action {action_class}">{escape(display_action)}</div></div>'
        f'<div class="decision-item"><div class="decision-label">Trade readiness</div><div class="decision-value">{decision["trade_readiness"]}% · {decision["readiness_label"]}<br><span class="micro">Stock {decision["stock_score"]} · Trade {decision["trade_score"]}</span></div></div>'
        f'<div class="decision-item"><div class="decision-label">Entry condition</div><div class="decision-value">{escape(decision["entry_condition"])}</div></div>'
        f'<div class="decision-item"><div class="decision-label">Invalidation level</div><div class="decision-value">{escape(decision["invalidation"])}</div></div>'
        f'<div class="decision-item"><div class="decision-label">First target</div><div class="decision-value">{escape(decision["first_target"])}<br><span class="micro">Risk: {escape(decision["main_risk"])}</span></div></div>'
        f'{warning_html}{conflict_html}</div>', unsafe_allow_html=True)
    eligible_setups=scanner_df[(scanner_df["Trade Score"]>=75) & (scanner_df["Trade Readiness"]>=75)] if not scanner_df.empty else scanner_df
    if not eligible_setups.empty:
        best=eligible_setups.iloc[0]
        best_html=(f'<div class="best-setup"><div class="best-cell best-title"><span>Today\'s best setup</span><b>{escape(str(best["Ticker"]))} · {escape(str(best["Signal"]))}</b></div>'
                   f'<div class="best-cell"><span>Stock / trade score</span><b>{int(best["Stock Score"])}/{int(best["Trade Score"])} · {escape(str(best["Trade Quality"]))}</b></div>'
                   f'<div class="best-cell"><span>Trade readiness</span><b>{int(best["Trade Readiness"])}%</b></div>'
                   f'<div class="best-cell"><span>Expected continuation</span><b>{float(best["Expected Continuation"]):+.2f}%</b></div>'
                   f'<div class="best-cell"><span>Trend / momentum</span><b>{int(best["Trend Strength"])} / {int(best["Momentum"])}</b></div>'
                   f'<div class="best-cell"><span>Risk reward</span><b>{float(best["Risk Reward"]):.1f}:1 · chain required</b></div></div>')
    else:
        best_html='<div class="best-setup"><div class="best-cell best-title"><span>Today\'s best setup</span><b>NO READY HIGH-QUALITY SETUP</b></div><div class="best-cell" style="grid-column:span 5"><span>Action</span><b>Protect capital · require trade score and readiness of at least 75</b></div></div>'
    st.markdown(best_html,unsafe_allow_html=True)
    pattern_name=decision.get("pattern",{}).get("name","No clear pattern"); pattern_bias=decision.get("pattern",{}).get("bias","Neutral")
    if decision.get("pattern",{}).get("status")=="FORMING": pattern_name=f"Forming {pattern_name}"
    tiles=[metric("Current Price",f"${latest:,.2f}",ticker,"cyan"),metric("Move %",f"{move_pct:+.2f}%",f"{move_class['label']} · {move_class['score']}%",tone(move_pct)),metric("Pullback %",f"{pullback_pct:+.2f}%","20-bar high",tone(pullback_pct)),metric("Stock Score",f"{decision['stock_score']}/100","Is the stock good?","up" if decision["stock_score"]>=75 else "down" if decision["stock_score"]<50 else "neutral"),metric("Trade Score",f"{decision['trade_score']}/100","Is the setup good?","up" if decision["trade_score"]>=75 else "down" if decision["trade_score"]<50 else "neutral"),metric("Readiness",f"{decision['trade_readiness']}%",decision["readiness_label"],"up" if decision["trade_readiness"]>=75 else "down" if decision["trade_readiness"]<50 else "neutral"),metric("RVOL-TOD",f"{rvol:.2f}×" if rvol else "N/A",f"{volume['quality']} · {decision['components']['Volume Intelligence']}/100","up" if 2<=rvol<=4 else "down" if rvol<.7 else "neutral"),metric("Market",direction,f"Alignment {decision['components']['Market Alignment']}/100","up" if direction=="BULLISH" else "down" if direction=="BEARISH" else "neutral"),metric("Setup Quality",decision["classification"],f"Grade {decision['grade']} · RR {expected['risk_reward']:.1f}:1","cyan"),metric("Candlestick",pattern_name,pattern_bias,"up" if pattern_bias=="Bullish" else "down" if pattern_bias=="Bearish" else "neutral")]
    st.markdown('<div class="metric-grid">'+''.join(tiles)+'</div>',unsafe_allow_html=True)
    lock_text="TRADING LOCKED" if decision["risk"]["locked"] else f"Max risk ${decision['risk']['max_risk_dollars']:,.0f}"
    market_detail=market["summary"]
    market_status=lambda value: {"Bullish":"BULL", "Bearish":"BEAR", "Weak / neutral":"NEUTRAL"}.get(value,value.upper())
    market_compact=(
        f"SPY {market_status(market['spy_status'])} · QQQ {market_status(market['qqq_status'])} · "
        f"{market['sector_symbol']} {market_status(market['sector_status'])} · {market['leadership']}"
    )
    st.markdown(
        '<div class="context-strip">'
        f'<div class="context-item" title="{escape(market_detail)}"><span>Market alignment · {market["alignment_score"]}/100</span><b>{escape(market_compact)}</b></div>'
        f'<div class="context-item"><span>Trend / momentum</span><b>{escape(trend["label"])} {decision["components"]["Trend Strength"]} · MOM {decision["components"]["Momentum"]}</b></div>'
        f'<div class="context-item"><span>Volume intelligence</span><b>{escape(volume["quality"])} · {decision["components"]["Volume Intelligence"]}/100</b></div>'
        f'<div class="context-item" title="Continuation {expected["continuation"]}% · Pullback {expected["pullback"]}% · Reversal {expected["reversal"]}%"><span>Expected move (heuristic)</span><b>Remain {expected["remaining_move"]:+.2f}% · {escape(expected["estimated_time"])}</b></div>'
        f'<div class="context-item"><span>Risk controls</span><b>{escape(lock_text)} · RR {expected["risk_reward"]:.1f}:1 · {escape(decision["classification"])}</b></div></div>',
        unsafe_allow_html=True)

    left,center,right=st.columns([1.18,2.55,1.47],gap="small")
    with left:
        rows=""
        for _,r in scanner_df.iterrows():
            mv=r.get("Move %"); mv_class=tone(mv) if pd.notna(mv) else "neutral"
            price=f'${r["Price"]:.2f}' if pd.notna(r["Price"]) else '—'; move=f'{mv:+.2f}%' if pd.notna(mv) else '—'
            symbol=escape(str(r["Ticker"])); href=f'?symbol={symbol}'; signal=escape(simple_trade_instruction(str(r["Signal"]).replace(" WATCH",""))); selected="selected" if symbol==ticker else ""
            reason_values=r.get("Top Reasons",[]); reason_values=reason_values if isinstance(reason_values,list) else []
            reason_text=" · ".join(escape(str(reason)) for reason in reason_values[:3]) or "No qualifying reasons"
            details=escape(f"Pattern: {r.get('Pattern','-')} | RS vs SPY: {r.get('Relative Strength','-')}% | Stock score: {r.get('Stock Score',0)} | Trade score: {r.get('Trade Score',0)} | Readiness: {r.get('Trade Readiness',0)}% | Trend: {r.get('Trend Strength',0)} | Momentum: {r.get('Momentum',0)} | Volume: {r.get('Volume Score',0)} | RR: {r.get('Risk Reward',0)}:1 | Reasons: {reason_text}")
            rvol_text=f'{r["RVOL"]:.1f}x' if pd.notna(r.get("RVOL")) else '—'; quality=escape(str(r.get("Trade Quality","NO TRADE")))
            quality_short="NT" if quality=="NO TRADE" else quality
            score_pair=f'{int(r.get("Stock Score",0))}·{int(r.get("Trade Score",0))}'
            readiness=f'{int(r.get("Trade Readiness",0))}%'
            rows+=f'<tr class="{selected}" title="{details}"><td class="ticker"><a href="{href}" target="_self">{symbol}</a></td><td><a href="{href}" target="_self">{price}</a></td><td class="{mv_class}"><a href="{href}" target="_self">{move}</a></td><td><a href="{href}" target="_self">{rvol_text}</a></td><td class="score"><a href="{href}" target="_self">{score_pair}</a></td><td class="score"><a href="{href}" target="_self">{readiness}</a></td><td><a href="{href}" target="_self">{signal}</a></td></tr>'
            if "CALL" in signal or "PUT" in signal:
                rows+=f'<tr class="scanner-reasons {selected}"><td colspan="7"><b>TOP 3:</b> {reason_text}</td></tr>'
        st.markdown(f'<div class="panel scanner-wrap"><div class="panel-title"><span>Live Scanner</span><span>{len(scanner_df)} symbols</span></div><div class="panel-body"><table class="scanner-table"><thead><tr><th>Symbol</th><th>Price</th><th>Move</th><th>RVOL</th><th>S·T</th><th>Ready</th><th>Signal</th></tr></thead><tbody>{rows}</tbody></table></div></div>',unsafe_allow_html=True)
    with center:
        render_live_chart(data, ticker, support, resistance, interval=interval, focus_time=chart_focus)
    with right:
        rows=""
        for pattern_index,p in enumerate(patterns):
            cls="confirmed" if p.get("confirmed") and p.get("status")=="CONFIRMED" else "hit" if p["detected"] else ""; bias_cls="bias-"+p["bias"].lower(); conf=f'{p["confidence"]}%' if p["detected"] else '—'
            event_time=p.get("confirmed_at") or p.get("detected_at"); time_text=event_time.strftime("%H:%M") if event_time is not None else "—"; status=p.get("status","—"); status_cls="status-"+status.lower() if status!="—" else ""
            pattern_time=p.get("detected_at"); confirmation_time=p.get("confirmed_at")
            actual_snapshot=market_snapshot_svg(data,p["name"],pattern_time,confirmation_time) if p["detected"] else None
            snapshot_label=f'ACTUAL {interval.upper()} CANDLES · PATTERN {pattern_time.strftime("%H:%M")}' if pattern_time is not None else "FORMATION GUIDE"
            if confirmation_time is not None and p.get("status")=="CONFIRMED": snapshot_label+=f' · CONFIRMED {confirmation_time.strftime("%H:%M")}'
            context_note=p.get("context_note","")
            intelligence=(f"Preceding trend: {trend['label']} ({trend['strength']}/100); "
                          f"location: {decision['location']['primary']}; volume: {volume['quality']}; "
                          f"action: {decision['action']}")
            context_note=f"{context_note}; {intelligence}" if context_note else intelligence
            if p["detected"]:
                stats=historical_pattern_stats(history,p["name"],interval)
                history_note=(f"{stats['label']}: n={stats['sample_size']}" if not stats["sufficient"] else
                              f"Observed n={stats['sample_size']}; continuation {stats['continuation_rate']:.1f}%; median {stats['median_move']:+.2f}%")
                context_note=f"{context_note}; {history_note}" if context_note else history_note
            hover_guide=guide_html(p["name"],p["bias"],p["confirmation"],lift=pattern_index>=8,actual_svg=actual_snapshot,snapshot_label=snapshot_label,context_note=context_note)
            if event_time is not None:
                focus_seconds=to_chart_timestamp(event_time,{"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800}[interval])
                pattern_label=f'<a class="pattern-jump" href="?symbol={ticker}&amp;interval={interval}&amp;focus={focus_seconds}" target="_self">{escape(p["name"])}<span class="jump-mark">↗</span></a>'
            else: pattern_label=escape(p["name"])
            rows+=f'<tr class="{cls}"><td class="pattern-icon {bias_cls}">{p["icon"]}</td><td class="pattern-name-cell">{pattern_label}{hover_guide}</td><td class="{bias_cls}">{escape(p["bias"][:4].upper())}</td><td>{conf}</td><td class="pattern-time">{time_text}</td><td class="{status_cls}">{status}</td></tr>'
        st.markdown(f'<div class="panel pattern-wrap"><div class="panel-title"><span>Candlestick Detection</span><span class="live">● ACTIVE</span></div><div class="panel-body"><table class="pattern-table"><thead><tr><th style="width:5%"></th><th style="width:39%">Pattern</th><th>Bias</th><th>Conf</th><th>Time</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></div></div>',unsafe_allow_html=True)

    pattern_rows=rows
    st.markdown(f'<details class="mobile-section"><summary>Candlestick Detection</summary><div class="mobile-section-body"><table class="pattern-table"><thead><tr><th style="width:5%"></th><th style="width:39%">Pattern</th><th>Bias</th><th>Conf</th><th>Time</th><th>Status</th></tr></thead><tbody>{pattern_rows}</tbody></table></div></details>',unsafe_allow_html=True)

    b1,b2,b3=st.columns([1.25,1.55,2],gap="small")
    with b1:
        rows=""
        for label,value in movement.items():
            color=tone(value); display=f"{value:+.2f}%"; rows+=f'<tr><td>{escape(label)}</td><td class="{color}">{display}</td></tr>'
        movement_rows=rows
        st.markdown(f'<div class="panel bottom-panel"><div class="panel-title"><span>Price Movement Analysis</span><span>Session</span></div><div class="panel-body"><table class="movement-table"><tbody>{rows}</tbody></table></div></div>',unsafe_allow_html=True)
    with b2:
        rows=""
        for a in alerts[:7]: rows+=f'<tr><td class="pri-{a["priority"].lower()}"><b>{a["priority"]}</b></td><td title="{escape(a["action_required"])}">{escape(a["reason"])}</td><td>{a["confidence"]}%</td><td>{a["continuation_probability"]}%</td><td>{a["pullback_risk"]}%</td></tr>'
        alert_rows=rows
        st.markdown(f'<div class="panel bottom-panel"><div class="panel-title"><span>AI Alerts</span><span>{len(alerts)} Triggered</span></div><div class="panel-body"><table class="alert-table"><thead><tr><th>Priority</th><th>Alert</th><th>Conf</th><th>Cont</th><th>Pull</th></tr></thead><tbody>{rows}</tbody></table></div></div>',unsafe_allow_html=True)
    with b3:
        reasons=''.join(f'<div class="reason"><span>{escape(label)}</span><span class="points {"up" if points>0 else "down"}">{points:+d}</span></div>' for label,points in decision["breakdown"][:10])
        option_class="option-call" if decision["direction"]=="CALL" else "option-put" if decision["direction"]=="PUT" else "option-wait"
        option_title=f"LOOK FOR {decision['direction']}S" if decision["execution_action"] in {"CALL","PUT"} else simple_trade_instruction(decision["action"])
        option_reason=f"Main risk: {decision['main_risk']}. Required entry: {decision['entry_condition']} Options: {decision['options']['status']}."
        option_outlook=f'<div class="option-outlook {option_class}"><b>{escape(option_title)}</b><span>{escape(option_reason)}</span></div>'
        explainable_body=f'<div class="reason-grid">{reasons}</div>{option_outlook}<div class="score-total"><span><b>TRADE SCORE · {escape(decision["classification"])} · GRADE {escape(decision["grade"])}</b><div class="micro">Seven setup-specific components · Stock {decision["stock_score"]}/100 · Readiness {decision["trade_readiness"]}% · heuristic, not win probability</div></span><strong>{decision["trade_score"]}/100</strong></div>'
        st.markdown(f'<div class="panel bottom-panel"><div class="panel-title"><span>Explainable AI · Why This Score?</span><span>Auditable</span></div><div class="panel-body">{explainable_body}</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="foot">Updated {datetime.now():%I:%M:%S %p} · Yahoo Finance data may be delayed · Decision support only, not trade instructions</div>',unsafe_allow_html=True)


    st.markdown(f'<details class="mobile-section"><summary>Price Movement Analysis</summary><div class="mobile-section-body"><table class="movement-table"><tbody>{movement_rows}</tbody></table></div></details>',unsafe_allow_html=True)
    st.markdown(f'<details class="mobile-section"><summary>AI Alerts Â· {len(alerts)} Triggered</summary><div class="mobile-section-body"><table class="alert-table"><thead><tr><th>Priority</th><th>Alert</th><th>Conf</th><th>Cont</th><th>Pull</th></tr></thead><tbody>{alert_rows}</tbody></table></div></details>',unsafe_allow_html=True)
    st.markdown(f'<details class="mobile-section"><summary>Explainable AI Â· Why This Score?</summary><div class="mobile-section-body">{explainable_body}</div></details>',unsafe_allow_html=True)


render_live_dashboard(ticker, interval, reference_choice, custom_threshold, st.session_state.get("chart_focus"))
