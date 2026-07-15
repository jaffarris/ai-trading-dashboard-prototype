"""Persistent TradingView Lightweight Charts component for Streamlit."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data_provider import to_chart_timestamp

CHART_HTML = """
<div class="live-chart-shell">
  <div class="chart-head">
    <strong class="chart-symbol"></strong>
    <div class="chart-tools">
      <button class="indicator-chip active" data-indicator="ema9" title="EMA 9: fast exponential average showing short-term momentum"><i style="background:#35c2ff"></i>EMA 9</button>
      <button class="indicator-chip active" data-indicator="ema20" title="EMA 20: slower exponential average showing the intraday trend"><i style="background:#ffb547"></i>EMA 20</button>
      <button class="indicator-chip active" data-indicator="vwap" title="VWAP: volume-weighted average price used as an intraday fair-value benchmark"><i style="background:#a778ff"></i>VWAP</button>
      <button class="indicator-chip active" data-indicator="volume" title="Volume: shares traded during each candle"><i style="background:#7189a6"></i>VOL</button>
      <span class="level-key" title="Green S is support; red R is resistance"><i class="support-dot"></i>S / <i class="resistance-dot"></i>R</span>
      <button class="auto-fit active" title="Automatically fit the price scale to the candles currently visible">AUTO FIT</button>
      <button class="reset-chart" title="Restore the full session and turn all indicators on">↺ RESET VIEW</button>
      <span class="chart-status">● YAHOO LIVE</span>
    </div>
  </div>
  <div class="chart-host"></div>
  <div class="chart-error"></div>
</div>
"""

CHART_CSS = """
.live-chart-shell{height:407px;position:relative;box-sizing:border-box;border:1px solid #ccd5e1;border-radius:9px;background:#fff;overflow:hidden;font-family:Inter,Segoe UI,sans-serif}
.chart-head{height:29px;box-sizing:border-box;padding:5px 8px;color:#243247;font-size:10px;letter-spacing:.3px;display:flex;align-items:center;justify-content:space-between;position:absolute;z-index:4;left:0;right:0;top:0;background:linear-gradient(#fff 72%,rgba(255,255,255,.2));pointer-events:none}.chart-tools{display:flex;align-items:center;gap:4px;pointer-events:auto}.indicator-chip,.auto-fit,.reset-chart{height:20px;box-sizing:border-box;border:1px solid #ccd5e1;border-radius:4px;background:#fff;color:#53647a;font:700 8px Inter,Segoe UI,sans-serif;padding:2px 5px;cursor:pointer}.indicator-chip{opacity:.45}.indicator-chip.active{opacity:1;background:#f5f8fb}.indicator-chip i,.level-key i{display:inline-block;width:7px;height:2px;margin-right:3px;vertical-align:middle}.indicator-chip:hover,.auto-fit:hover,.reset-chart:hover{border-color:#35aee8;background:#eaf7fd}.auto-fit{color:#607086}.auto-fit.active{color:#0877b5;border-color:#8dcceb;background:#eaf7fd}.reset-chart{color:#0877b5}.level-key{font-size:8px;font-weight:800;color:#607086;padding:0 2px}.support-dot{background:#25d695}.resistance-dot{background:#ff5c72}.chart-status{color:#138b62;font-size:8px;font-weight:800;margin-left:2px}.chart-host{position:absolute;inset:0}.chart-error{display:none;position:absolute;left:10px;bottom:7px;color:#d93f56;font-size:9px;background:rgba(255,255,255,.9);padding:3px 6px;border-radius:4px}
.session-layer{position:absolute;inset:0;z-index:2;pointer-events:none;overflow:hidden}.session-band{position:absolute;top:29px;bottom:23px;border-left:1px solid;border-right:1px solid;box-sizing:border-box}.session-band.premarket{background:rgba(245,158,11,.11);border-color:rgba(194,113,0,.25)}.session-band.afterhours{background:rgba(37,99,235,.10);border-color:rgba(37,99,235,.22)}.session-tag{position:absolute;top:3px;left:4px;font-size:7px;font-weight:900;letter-spacing:.6px}.premarket .session-tag{color:#9a5a00}.afterhours .session-tag{color:#2456b8}.chart-error{z-index:5}
@media(max-width:600px){.chart-head{height:48px;display:block;padding:4px 5px}.chart-symbol{display:block;height:16px;font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.chart-tools{gap:2px;width:100%;justify-content:space-between}.indicator-chip,.auto-fit,.reset-chart{height:22px;font-size:7px;padding:2px 3px}.chart-status,.level-key{display:none}.session-band{top:48px}.chart-error{left:5px;right:5px;font-size:8px}}
"""

CHART_JS = """
export default async function(component) {
  const { data, parentElement } = component;
  const host = parentElement.querySelector('.chart-host');
  const symbolLabel = parentElement.querySelector('.chart-symbol');
  const errorLabel = parentElement.querySelector('.chart-error');
  if (!parentElement.__apexChart) symbolLabel.textContent = `${data.ticker} · ${data.interval.toUpperCase()} · EXTENDED HOURS`;

  symbolLabel.textContent = `${data.ticker} - ${data.interval.toUpperCase()} - ${data.session_label}`;
  let state = parentElement.__apexChart;
  if (!state) {
    const localBridge = ['localhost','127.0.0.1','::1'].includes(window.location.hostname);
    parentElement.querySelector('.chart-status').textContent = localBridge ? '● YAHOO LIVE' : '● CLOUD 15S';
    parentElement.querySelector('.chart-status').textContent = 'YAHOO FEED';
    const LC = await import('https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.3/+esm');
    const chart = LC.createChart(host, {
      width: host.clientWidth, height: 405,
      layout: { background: { color: '#ffffff' }, textColor: '#526176', fontFamily: 'Inter, Segoe UI, sans-serif', fontSize: 10 },
      grid: { vertLines: { color: '#edf1f6' }, horzLines: { color: '#dfe5ed' } },
      crosshair: { mode: LC.CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#ccd5e1', scaleMargins: { top: .10, bottom: .22 } },
      timeScale: { borderColor: '#ccd5e1', timeVisible: true, secondsVisible: false, rightOffset: 3 },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });
    const candles = chart.addCandlestickSeries({
      upColor:'#25d695',downColor:'#ff5c72',borderVisible:false,wickUpColor:'#25d695',wickDownColor:'#ff5c72',
      autoscaleInfoProvider: (baseImplementation) => {
        if (!state?.autoFit || !state.currentCandles?.length) return baseImplementation();
        const visible = state.chart.timeScale().getVisibleLogicalRange();
        const last = state.currentCandles.length - 1;
        const firstIndex = visible ? Math.max(0, Math.floor(visible.from)) : 0;
        const lastIndex = visible ? Math.min(last, Math.ceil(visible.to)) : last;
        if (lastIndex < firstIndex) return baseImplementation();
        const shown = state.currentCandles.slice(firstIndex, lastIndex + 1);
        const minValue = Math.min(...shown.map(item => Number(item.low)).filter(Number.isFinite));
        const maxValue = Math.max(...shown.map(item => Number(item.high)).filter(Number.isFinite));
        if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue === maxValue) return baseImplementation();
        return { priceRange: { minValue, maxValue } };
      }
    });
    const ema9 = chart.addLineSeries({ color:'#35c2ff',lineWidth:2,priceLineVisible:false,lastValueVisible:false });
    const ema20 = chart.addLineSeries({ color:'#ffb547',lineWidth:2,priceLineVisible:false,lastValueVisible:false });
    const vwap = chart.addLineSeries({ color:'#a778ff',lineWidth:2,priceLineVisible:false,lastValueVisible:false });
    const volume = chart.addHistogramSeries({ priceFormat:{type:'volume'},priceScaleId:'',lastValueVisible:false,priceLineVisible:false });
    chart.priceScale('').applyOptions({ scaleMargins:{top:.72,bottom:0} });
    const resistanceLine = candles.createPriceLine({price:data.resistance,color:'#ff5c72',lineWidth:1,lineStyle:LC.LineStyle.Dotted,axisLabelVisible:true,title:'R'});
    const supportLine = candles.createPriceLine({price:data.support,color:'#25d695',lineWidth:1,lineStyle:LC.LineStyle.Dotted,axisLabelVisible:true,title:'S'});
    const sessionLayer = document.createElement('div'); sessionLayer.className = 'session-layer'; host.appendChild(sessionLayer);
    state = { chart,candles,ema9,ema20,vwap,volume,resistanceLine,supportLine,sessionLayer,sessionRanges:[],currentCandles:[],ticker:null,interval:null,dashboardInterval:null,lastTime:null,lastFocus:null,timer:null,localBridge,autoFit:true,autoFitFrame:null };
    parentElement.__apexChart = state;
    const drawSessionBands = () => {
      state.sessionLayer.innerHTML = '';
      for (const range of state.sessionRanges) {
        const left = state.chart.timeScale().timeToCoordinate(range.start);
        const right = state.chart.timeScale().timeToCoordinate(range.end);
        if (left == null || right == null || right < 0 || left > host.clientWidth) continue;
        const visibleLeft = Math.max(0,left), visibleRight = Math.min(host.clientWidth,right);
        const band = document.createElement('div'); band.className = `session-band ${range.kind}`;
        band.style.left = `${visibleLeft}px`; band.style.width = `${Math.max(1,visibleRight-visibleLeft)}px`;
        band.innerHTML = `<span class="session-tag">${range.kind === 'premarket' ? 'PREMARKET' : 'AFTER HOURS'}</span>`;
        state.sessionLayer.appendChild(band);
      }
    };
    state.drawSessionBands = drawSessionBands;
    state.applyDataset = (dataset, fitView=true) => {
      state.interval = dataset.interval;
      state.currentCandles = dataset.candles;
      state.candles.setData(dataset.candles); state.ema9.setData(dataset.ema9); state.ema20.setData(dataset.ema20);
      state.vwap.setData(dataset.vwap); state.volume.setData(dataset.volume);
      state.resistanceLine.applyOptions({price:dataset.resistance}); state.supportLine.applyOptions({price:dataset.support});
      state.sessionRanges = dataset.sessions || []; state.lastTime = Number(dataset.candles.at(-1)?.time);
      symbolLabel.textContent = `${state.ticker} · ${state.interval.toUpperCase()} · EXTENDED HOURS`;
      symbolLabel.textContent = `${state.ticker} - ${state.interval.toUpperCase()} - ${dataset.session_label}`;
      if (fitView) state.chart.timeScale().fitContent();
      requestAnimationFrame(state.drawSessionBands);
    };
    state.forceAutoFit = () => {
      state.autoFit = true;
      parentElement.querySelector('.auto-fit').classList.add('active');
      if (state.autoFitFrame) cancelAnimationFrame(state.autoFitFrame);
      state.autoFitFrame = requestAnimationFrame(() => {
        state.chart.priceScale('right').applyOptions({autoScale:true});
        state.candles.applyOptions({visible:true});
        state.autoFitFrame = null;
      });
    };
    chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
      if (state.autoFit) state.forceAutoFit();
      requestAnimationFrame(drawSessionBands);
    });
    new ResizeObserver(() => { chart.applyOptions({width:host.clientWidth}); requestAnimationFrame(drawSessionBands); }).observe(host);
    const seriesMap = {ema9,ema20,vwap,volume};
    parentElement.querySelectorAll('.indicator-chip').forEach((chip) => {
      chip.onclick = () => {
        chip.classList.toggle('active');
        seriesMap[chip.dataset.indicator].applyOptions({visible:chip.classList.contains('active')});
      };
    });
    parentElement.querySelector('.reset-chart').onclick = () => {
      chart.timeScale().fitContent();
      state.forceAutoFit();
      parentElement.querySelectorAll('.indicator-chip').forEach((chip) => {
        chip.classList.add('active'); seriesMap[chip.dataset.indicator].applyOptions({visible:true});
      });
    };
    parentElement.querySelector('.auto-fit').onclick = () => {
      state.forceAutoFit();
    };
  }

  if (state.ticker !== data.ticker || state.dashboardInterval !== data.interval) {
    state.ticker = data.ticker;
    state.dashboardInterval = data.interval;
    state.applyDataset(data,true);
  } else if (state.interval === data.interval) {
    const visibleRange = state.chart.timeScale().getVisibleRange();
    state.applyDataset(data,false);
    if (visibleRange) state.chart.timeScale().setVisibleRange(visibleRange);
    if (state.autoFit) state.forceAutoFit();
  }

  const focusKey = data.focus_time ? `${data.ticker}-${data.interval}-${data.focus_time}` : null;
  if (focusKey && state.lastFocus !== focusKey) {
    if (state.interval !== data.interval) state.applyDataset(data,false);
    const context = data.interval_seconds * 7;
    state.chart.timeScale().setVisibleRange({from:data.focus_time-context,to:data.focus_time+context});
    state.lastFocus = focusKey;
  }

  if (state.localBridge && !state.timer) {
    const poll = async () => {
      if (!parentElement.isConnected) { clearInterval(state.timer); return; }
      try {
        const response = await fetch(`http://localhost:8502/latest?ticker=${encodeURIComponent(state.ticker)}&interval=${encodeURIComponent(state.interval)}`, {cache:'no-store'});
        if (!response.ok) throw new Error(`Feed ${response.status}`);
        const point = await response.json();
        const incomingTime = Number(point.candle.time);
        if (Number.isFinite(state.lastTime) && incomingTime < state.lastTime) {
          errorLabel.style.display='none'; return;
        }
        state.candles.update(point.candle);
        if (point.ema9?.value != null && Number.isFinite(Number(point.ema9.value))) state.ema9.update(point.ema9);
        if (point.ema20?.value != null && Number.isFinite(Number(point.ema20.value))) state.ema20.update(point.ema20);
        if (point.vwap?.value != null && Number.isFinite(Number(point.vwap.value))) state.vwap.update(point.vwap);
        if (point.volume?.value != null && Number.isFinite(Number(point.volume.value))) state.volume.update(point.volume);
        state.resistanceLine.applyOptions({price:point.resistance}); state.supportLine.applyOptions({price:point.support});
        state.lastTime = incomingTime; errorLabel.style.display='none';
      } catch (error) {
        errorLabel.textContent = `Live update delayed: ${error.message}`; errorLabel.style.display='block';
      }
    };
    state.timer = setInterval(poll, 5000);
  }
}
"""

_live_chart = st.components.v2.component("apex_live_chart_v2", html=CHART_HTML, css=CHART_CSS, js=CHART_JS)


def _series(data: pd.DataFrame, column: str, interval_seconds: int) -> list[dict]:
    points = {}
    for index, value in data[column].dropna().items():
        timestamp = to_chart_timestamp(index, interval_seconds)
        points[timestamp] = {"time": timestamp, "value": float(value)}
    return list(points.values())


def render_live_chart(data: pd.DataFrame, ticker: str, support: float, resistance: float,
                      interval: str = "5m", focus_time: int | None = None) -> None:
    interval_seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800}[interval]
    volume_cap = max(1.0, float(data["Volume"].quantile(.95))) if not data.empty else 1.0
    candles, volumes = {}, {}
    for index, row in data.iterrows():
        timestamp = to_chart_timestamp(index, interval_seconds)
        candles[timestamp] = {"time": timestamp, "open": float(row["Open"]), "high": float(row["High"]),
                              "low": float(row["Low"]), "close": float(row["Close"])}
        volumes[timestamp] = {"time": timestamp, "value": min(float(row["Volume"]), volume_cap),
                              "actual": float(row["Volume"]),
                              "color": "rgba(37,214,149,.42)" if row["Close"] >= row["Open"] else "rgba(255,92,114,.42)"}
    sessions = []
    if not data.empty:
        local_index = data.index
        for session_date in sorted(set(local_index.date)):
            day_rows = data[local_index.date == session_date]
            premarket = day_rows[(day_rows.index.hour >= 4) & ((day_rows.index.hour < 9) | ((day_rows.index.hour == 9) & (day_rows.index.minute < 30)))]
            afterhours = day_rows[(day_rows.index.hour >= 16) & (day_rows.index.hour < 20)]
            for kind, frame in (("premarket", premarket), ("afterhours", afterhours)):
                if len(frame) >= 2:
                    sessions.append({"kind": kind,
                        "start": to_chart_timestamp(frame.index[0], interval_seconds),
                        "end": to_chart_timestamp(frame.index[-1], interval_seconds)})
    latest_minute = data.index[-1].hour * 60 + data.index[-1].minute if not data.empty else 0
    session_label = "EXTENDED HOURS" if latest_minute < 570 or latest_minute >= 960 else "REGULAR HOURS"
    _live_chart(key="persistent-price-chart-v2", height=407, data={"ticker": ticker, "candles": list(candles.values()),
        "interval": interval, "interval_seconds": interval_seconds, "focus_time": focus_time,
        "session_label": session_label,
        "ema9": _series(data, "EMA9", interval_seconds), "ema20": _series(data, "EMA20", interval_seconds),
        "vwap": _series(data, "VWAP", interval_seconds), "volume": list(volumes.values()),
        "support": support, "resistance": resistance, "sessions": sessions})
