#!/usr/bin/env python3
"""
OKX Market Scanner - public market data only, no API key.
Does not trade. Creates CSV/JSON market report.
"""

from __future__ import annotations
import csv, json, math, os, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

BASE_URL = "https://www.okx.com"
SYMBOLS = [
    "BTC-USDT", "SOL-USDT", "ETH-USDT", "LINK-USDT", "AVAX-USDT", "NEAR-USDT",
    "TIA-USDT", "INJ-USDT", "SUI-USDT", "DOGE-USDT", "XRP-USDT", "RENDER-USDT",
]
AMOUNT_USDT = float(os.environ.get("AMOUNT_USDT", "900"))
OUT_CSV = Path("okx_market_report.csv")
OUT_JSON = Path("okx_market_report.json")

@dataclass
class Candle:
    ts: int
    o: float
    h: float
    l: float
    c: float
    vol: float

def get(path: str, params: dict[str, Any]) -> Any:
    r = requests.get(f"{BASE_URL}{path}", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "0":
        raise RuntimeError(f"OKX error {data}")
    return data["data"]

def candles(inst: str, bar: str, limit: int = 100) -> list[Candle]:
    rows = get("/api/v5/market/candles", {"instId": inst, "bar": bar, "limit": str(limit)})
    out: list[Candle] = []
    for row in reversed(rows):
        out.append(Candle(int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])))
    return out

def ticker(inst: str) -> dict[str, Any]:
    return get("/api/v5/market/ticker", {"instId": inst})[0]

def orderbook(inst: str) -> dict[str, Any]:
    return get("/api/v5/market/books", {"instId": inst, "sz": "20"})[0]

def ema(values: list[float], period: int) -> float:
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def atr(cs: list[Candle], period: int = 14) -> float:
    trs, prev = [], cs[0].c
    for c in cs[1:]:
        trs.append(max(c.h - c.l, abs(c.h - prev), abs(c.l - prev)))
        prev = c.c
    return sum(trs[-period:]) / min(period, len(trs)) if trs else 0.0

def trend(last: float, ema20: float, label: str) -> str:
    return "bullish" if last >= ema20 else f"below_{label}_ema20"

def r(x: float, ref: float) -> float:
    if not math.isfinite(x):
        return x
    if ref >= 1000: return round(x, 0)
    if ref >= 100: return round(x, 2)
    if ref >= 1: return round(x, 3)
    if ref >= 0.1: return round(x, 4)
    return round(x, 5)

def fibs(low: float, high: float) -> dict[str, float]:
    rng = high - low
    if rng <= 0:
        return {}
    return {
        "0.236": low + 0.236*rng,
        "0.382": low + 0.382*rng,
        "0.500": low + 0.500*rng,
        "0.618": low + 0.618*rng,
        "0.786": low + 0.786*rng,
        "1.000": high,
    }

def analyze(inst: str) -> dict[str, Any]:
    t = ticker(inst)
    last = float(t["last"])
    high24 = float(t["high24h"])
    low24 = float(t["low24h"])
    vol_base = float(t.get("vol24h", 0) or 0)
    vol_quote = float(t.get("volCcy24h", 0) or 0)

    c1d, c4h, c15 = candles(inst, "1D", 80), candles(inst, "4H", 100), candles(inst, "15m", 80)
    close1d, close4h, close15 = [c.c for c in c1d], [c.c for c in c4h], [c.c for c in c15]

    e1d5, e1d10, e1d20 = ema(close1d, 5), ema(close1d, 10), ema(close1d, 20)
    e4h5, e4h10, e4h20 = ema(close4h, 5), ema(close4h, 10), ema(close4h, 20)
    e15_20 = ema(close15, 20)

    s1h, s1l = max(c.h for c in c1d[-30:]), min(c.l for c in c1d[-30:])
    s4h, s4l = max(c.h for c in c4h[-36:]), min(c.l for c in c4h[-36:])
    f = fibs(s4l, s4h)
    a4h = atr(c4h)

    ob = orderbook(inst)
    bids = [(float(p), float(sz)) for p, sz, *_ in ob.get("bids", [])[:10]]
    asks = [(float(p), float(sz)) for p, sz, *_ in ob.get("asks", [])[:10]]
    best_bid = bids[0][0] if bids else last
    best_ask = asks[0][0] if asks else last
    spread = best_ask - best_bid
    bid_wall = max(bids, key=lambda x: x[1])[0] if bids else last
    ask_wall = max(asks, key=lambda x: x[1])[0] if asks else last
    bid_total, ask_total = sum(sz for _, sz in bids), sum(sz for _, sz in asks)
    buy_ratio = bid_total / (bid_total + ask_total) if (bid_total + ask_total) else 0

    raw_above = [high24, s4h, f.get("0.618", math.nan), f.get("0.786", math.nan), e4h20, e1d20]
    raw_below = [low24, s4l, f.get("0.382", math.nan), f.get("0.236", math.nan), e4h20, e1d10]
    above = sorted({r(x, last) for x in raw_above if math.isfinite(x) and x > last})[:3]
    below = sorted({r(x, last) for x in raw_below if math.isfinite(x) and x < last}, reverse=True)[:3]

    trigger = above[0] if above else r(last * 1.005, last)
    limit = r(trigger + max(spread, last*0.001), last)
    sl = r(min(below[0] if below else last - 1.2*a4h, last - 1.1*a4h), last)
    tp1 = above[1] if len(above) > 1 else r(trigger + a4h, last)
    tp2 = above[2] if len(above) > 2 else r(trigger + 2*a4h, last)

    qty = AMOUNT_USDT / trigger if trigger else 0
    risk = max(0, (trigger - sl) * qty)
    rew1 = max(0, (tp1 - trigger) * qty)
    rew2 = max(0, (tp2 - trigger) * qty)
    rr1 = rew1 / risk if risk else 0
    rr2 = rew2 / risk if risk else 0

    status = "NO"
    if trend(last, e4h20, "4h") == "bullish" and trend(last, e15_20, "15m") == "bullish" and rr2 >= 1.7 and 15 <= risk <= 45 and rew2 >= 45 and buy_ratio >= 0.50:
        status = "WATCH_TRIGGER"

    return {
        "symbol": inst,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last": r(last, last),
        "high24h": r(high24, last),
        "low24h": r(low24, last),
        "vol24h_base": vol_base,
        "vol24h_quote": vol_quote,
        "trend_1d": trend(last, e1d20, "1d"),
        "trend_4h": trend(last, e4h20, "4h"),
        "trend_15m": trend(last, e15_20, "15m"),
        "ema_1d_5": r(e1d5, last), "ema_1d_10": r(e1d10, last), "ema_1d_20": r(e1d20, last),
        "ema_4h_5": r(e4h5, last), "ema_4h_10": r(e4h10, last), "ema_4h_20": r(e4h20, last),
        "swing_1d_high": r(s1h, last), "swing_1d_low": r(s1l, last),
        "swing_4h_high": r(s4h, last), "swing_4h_low": r(s4l, last),
        "above_alerts": "; ".join(map(str, above)),
        "below_alerts": "; ".join(map(str, below)),
        "best_ask": r(best_ask, last), "best_bid": r(best_bid, last), "spread": spread,
        "ask_wall_top10": r(ask_wall, last), "bid_wall_top10": r(bid_wall, last),
        "buy_ratio_top10": round(buy_ratio, 3),
        "status": status,
        "trigger_buy": trigger, "limit": limit, "sl": sl, "tp1": tp1, "tp2": tp2,
        "risk_usdt": round(risk, 2), "reward_tp1_usdt": round(rew1, 2), "reward_tp2_usdt": round(rew2, 2),
        "rr_tp1": round(rr1, 2), "rr_tp2": round(rr2, 2),
        "fb_4h_0.236": r(f.get("0.236", math.nan), last),
        "fb_4h_0.382": r(f.get("0.382", math.nan), last),
        "fb_4h_0.500": r(f.get("0.500", math.nan), last),
        "fb_4h_0.618": r(f.get("0.618", math.nan), last),
        "fb_4h_0.786": r(f.get("0.786", math.nan), last),
        "fb_4h_1.000": r(f.get("1.000", math.nan), last),
    }

def main() -> None:
    rows = []
    for inst in SYMBOLS:
        try:
            row = analyze(inst)
            rows.append(row)
            print(f"{row['symbol']} | price {row['last']} | status {row['status']} | trigger {row['trigger_buy']} | SL {row['sl']} | TP2 {row['tp2']} | risk ${row['risk_usdt']} | reward2 ${row['reward_tp2_usdt']}")
            time.sleep(0.25)
        except Exception as e:
            print(f"ERROR {inst}: {e}")

    if not rows:
        raise SystemExit("No rows generated")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")

if __name__ == "__main__":
    main()
