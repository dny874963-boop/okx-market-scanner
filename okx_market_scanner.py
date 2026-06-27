#!/usr/bin/env python3
"""
OKX Market Scanner - Swing 50 USD filter

Public market-data scanner only.
Does not trade. Does not use API keys.

Main rule:
- WATCH_TRIGGER only if:
  1) TP2 reward >= $50
  2) risk is between $15 and $35
  3) reward/risk >= 1:1.8
  4) trigger is close enough to current price

Outputs:
- okx_market_report.csv
- okx_market_report.json
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://www.okx.com"

SYMBOLS = [
    "BTC-USDT", "SOL-USDT", "ETH-USDT", "LINK-USDT",
    "AVAX-USDT", "NEAR-USDT", "TIA-USDT", "INJ-USDT",
    "SUI-USDT", "DOGE-USDT", "XRP-USDT", "RENDER-USDT",
    "APT-USDT", "OP-USDT", "ARB-USDT", "SEI-USDT",
    "DOT-USDT", "ADA-USDT", "WLD-USDT", "FET-USDT",
]

ACCOUNT_USDT = 900.0

# Swing rules
MIN_TP2_REWARD_USD = 50.0
MIN_RISK_USD = 15.0
MAX_RISK_USD = 35.0
MIN_REWARD_RISK = 1.8

# Scanner settings
BAR = "4H"
CANDLE_LIMIT = 100
BREAKOUT_LOOKBACK = 20
SWING_LOW_LOOKBACK = 10
ATR_PERIOD = 14

# Trigger/target construction
TRIGGER_BUFFER_PCT = 0.001       # 0.10% above recent resistance
ATR_STOP_MULT = 1.15
MIN_TP2_R_MULT = 2.0
MAX_TP2_R_MULT = 4.0
MAX_TP2_PCT_FROM_TRIGGER = 18.0  # avoid unrealistic TP2

# Alert only when the trigger is close enough.
# Example: price 100, trigger 103.2 => 3.2% away => still watchable.
MAX_DISTANCE_TO_TRIGGER_PCT = 3.5


@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirmed: bool


@dataclass
class ScanRow:
    symbol: str
    price: float | None
    status: str
    trigger: float | None
    stop_loss: float | None
    tp1: float | None
    tp2: float | None
    position_usdt: float
    quantity: float | None
    risk_usd: float | None
    reward2_usd: float | None
    reward_risk: float | None
    distance_to_trigger_pct: float | None
    atr_4h: float | None
    filters_passed: bool
    fail_reasons: str
    note: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def http_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = BASE_URL + path
    response = requests.get(url, params=params or {}, timeout=20)
    response.raise_for_status()
    data = response.json()

    if data.get("code") != "0":
        raise RuntimeError(f"OKX API error on {path}: {data}")

    return data


def fetch_spot_tickers() -> dict[str, float]:
    data = http_get("/api/v5/market/tickers", {"instType": "SPOT"})
    tickers: dict[str, float] = {}

    for item in data.get("data", []):
        inst_id = item.get("instId")
        last = item.get("last")
        if inst_id and last not in (None, ""):
            try:
                tickers[inst_id] = float(last)
            except ValueError:
                continue

    return tickers


def fetch_candles(symbol: str) -> list[Candle]:
    data = http_get(
        "/api/v5/market/candles",
        {"instId": symbol, "bar": BAR, "limit": str(CANDLE_LIMIT)},
    )

    candles: list[Candle] = []
    for row in data.get("data", []):
        # OKX format:
        # [ts, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
        if len(row) < 6:
            continue

        try:
            candles.append(
                Candle(
                    ts=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    confirmed=(len(row) < 9 or str(row[8]) == "1"),
                )
            )
        except (TypeError, ValueError):
            continue

    # OKX normally returns newest first. Sort oldest -> newest.
    candles.sort(key=lambda x: x.ts)

    # Prefer closed candles only, but keep all if too few are confirmed.
    confirmed = [c for c in candles if c.confirmed]
    if len(confirmed) >= max(BREAKOUT_LOOKBACK + 2, ATR_PERIOD + 2):
        candles = confirmed

    return candles


def calc_atr(candles: list[Candle], period: int = ATR_PERIOD) -> float | None:
    if len(candles) < period + 1:
        return None

    true_ranges: list[float] = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    return sum(true_ranges[-period:]) / period


def decimals_for_price(price: float) -> int:
    if price >= 1000:
        return 1
    if price >= 100:
        return 2
    if price >= 10:
        return 3
    if price >= 1:
        return 4
    if price >= 0.1:
        return 5
    return 6


def r(price: float | None, ref: float | None = None) -> float | None:
    if price is None:
        return None
    base = ref if ref is not None else price
    return round(price, decimals_for_price(abs(base)))


def safe_ratio(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def scan_symbol(symbol: str, tickers: dict[str, float]) -> ScanRow:
    try:
        candles = fetch_candles(symbol)
        time.sleep(0.12)

        min_needed = max(BREAKOUT_LOOKBACK + 2, ATR_PERIOD + 2, SWING_LOW_LOOKBACK + 2)
        if len(candles) < min_needed:
            return ScanRow(
                symbol=symbol,
                price=None,
                status="NO",
                trigger=None,
                stop_loss=None,
                tp1=None,
                tp2=None,
                position_usdt=ACCOUNT_USDT,
                quantity=None,
                risk_usd=None,
                reward2_usd=None,
                reward_risk=None,
                distance_to_trigger_pct=None,
                atr_4h=None,
                filters_passed=False,
                fail_reasons="not_enough_candles",
                note=f"Need at least {min_needed} candles; got {len(candles)}",
            )

        price = tickers.get(symbol, candles[-1].close)
        atr = calc_atr(candles)

        if price <= 0 or atr is None or atr <= 0:
            return ScanRow(
                symbol=symbol,
                price=price,
                status="NO",
                trigger=None,
                stop_loss=None,
                tp1=None,
                tp2=None,
                position_usdt=ACCOUNT_USDT,
                quantity=None,
                risk_usd=None,
                reward2_usd=None,
                reward_risk=None,
                distance_to_trigger_pct=None,
                atr_4h=atr,
                filters_passed=False,
                fail_reasons="bad_price_or_atr",
                note="Invalid price/ATR",
            )

        # Exclude the last closed candle from resistance/support calculation
        # to reduce false breakout from the current candle.
        prior = candles[:-1]
        resistance_window = prior[-BREAKOUT_LOOKBACK:]
        swing_window = prior[-SWING_LOW_LOOKBACK:]

        resistance = max(c.high for c in resistance_window)
        swing_low = min(c.low for c in swing_window)

        trigger = resistance * (1.0 + TRIGGER_BUFFER_PCT)

        # Stop uses the tighter of ATR stop and recent swing-low stop,
        # but must stay under trigger.
        atr_stop = trigger - (atr * ATR_STOP_MULT)
        stop_loss = max(swing_low, atr_stop)

        if stop_loss >= trigger:
            stop_loss = trigger * 0.97

        risk_per_coin = trigger - stop_loss
        if risk_per_coin <= 0:
            return ScanRow(
                symbol=symbol,
                price=price,
                status="NO",
                trigger=r(trigger, price),
                stop_loss=r(stop_loss, price),
                tp1=None,
                tp2=None,
                position_usdt=ACCOUNT_USDT,
                quantity=None,
                risk_usd=None,
                reward2_usd=None,
                reward_risk=None,
                distance_to_trigger_pct=None,
                atr_4h=r(atr, price),
                filters_passed=False,
                fail_reasons="invalid_stop",
                note="Stop-loss is not below trigger",
            )

        qty = ACCOUNT_USDT / trigger
        risk_usd = risk_per_coin * qty

        # TP1 = 1R.
        tp1 = trigger + risk_per_coin

        # TP2 must satisfy:
        # - at least MIN_TP2_REWARD_USD
        # - at least MIN_REWARD_RISK
        # - at least MIN_TP2_R_MULT
        min_tp2_by_rr = risk_per_coin * max(MIN_REWARD_RISK, MIN_TP2_R_MULT)
        min_tp2_by_usd = MIN_TP2_REWARD_USD / qty
        tp2_distance = max(min_tp2_by_rr, min_tp2_by_usd)
        tp2_r_mult = tp2_distance / risk_per_coin
        tp2 = trigger + tp2_distance

        reward2_usd = tp2_distance * qty
        reward_risk = safe_ratio(reward2_usd, risk_usd)
        distance_to_trigger_pct = ((trigger - price) / price) * 100.0

        fail_reasons: list[str] = []

        if risk_usd < MIN_RISK_USD:
            fail_reasons.append("risk_lt_15")
        if risk_usd > MAX_RISK_USD:
            fail_reasons.append("risk_gt_35")
        if reward2_usd < MIN_TP2_REWARD_USD:
            fail_reasons.append("reward2_lt_50")
        if reward_risk is None or reward_risk < MIN_REWARD_RISK:
            fail_reasons.append("rr_lt_1_8")
        if distance_to_trigger_pct < 0:
            fail_reasons.append("price_above_trigger")
        if distance_to_trigger_pct > MAX_DISTANCE_TO_TRIGGER_PCT:
            fail_reasons.append("trigger_too_far")
        if tp2_r_mult > MAX_TP2_R_MULT:
            fail_reasons.append("tp2_requires_too_many_R")
        if ((tp2 - trigger) / trigger) * 100.0 > MAX_TP2_PCT_FROM_TRIGGER:
            fail_reasons.append("tp2_too_far_pct")

        filters_passed = not fail_reasons
        status = "WATCH_TRIGGER" if filters_passed else "NO"

        note = "OK for swing watch" if filters_passed else "Rejected by swing filters"

        return ScanRow(
            symbol=symbol,
            price=r(price),
            status=status,
            trigger=r(trigger, price),
            stop_loss=r(stop_loss, price),
            tp1=r(tp1, price),
            tp2=r(tp2, price),
            position_usdt=round(ACCOUNT_USDT, 2),
            quantity=round(qty, 8),
            risk_usd=round(risk_usd, 2),
            reward2_usd=round(reward2_usd, 2),
            reward_risk=round(reward_risk, 2) if reward_risk is not None else None,
            distance_to_trigger_pct=round(distance_to_trigger_pct, 2),
            atr_4h=r(atr, price),
            filters_passed=filters_passed,
            fail_reasons=";".join(fail_reasons),
            note=note,
        )

    except Exception as exc:
        return ScanRow(
            symbol=symbol,
            price=None,
            status="NO",
            trigger=None,
            stop_loss=None,
            tp1=None,
            tp2=None,
            position_usdt=ACCOUNT_USDT,
            quantity=None,
            risk_usd=None,
            reward2_usd=None,
            reward_risk=None,
            distance_to_trigger_pct=None,
            atr_4h=None,
            filters_passed=False,
            fail_reasons="exception",
            note=str(exc),
        )


def write_csv(rows: list[ScanRow], path: str = "okx_market_report.csv") -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ScanRow.__dataclass_fields__.keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(rows: list[ScanRow], path: str = "okx_market_report.json") -> None:
    watch = [asdict(rw) for rw in rows if rw.status == "WATCH_TRIGGER"]

    payload = {
        "generated_at_utc": now_utc_iso(),
        "scanner": "OKX Market Scanner - Swing 50 USD filter",
        "does_trade": False,
        "account_usdt": ACCOUNT_USDT,
        "rules": {
            "min_tp2_reward_usd": MIN_TP2_REWARD_USD,
            "risk_usd_range": [MIN_RISK_USD, MAX_RISK_USD],
            "min_reward_risk": MIN_REWARD_RISK,
            "max_distance_to_trigger_pct": MAX_DISTANCE_TO_TRIGGER_PCT,
            "max_tp2_r_mult": MAX_TP2_R_MULT,
            "max_tp2_pct_from_trigger": MAX_TP2_PCT_FROM_TRIGGER,
        },
        "summary": {
            "symbols_scanned": len(rows),
            "watch_trigger_count": len(watch),
            "has_watch_trigger": bool(watch),
            "watch_symbols": [x["symbol"] for x in watch],
        },
        "WATCH_TRIGGER": watch,
        "rows": [asdict(rw) for rw in rows],
    }

    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_summary(rows: list[ScanRow]) -> None:
    print(f"Generated at UTC: {now_utc_iso()}")
    print(
        f"Rules: TP2 reward >= ${MIN_TP2_REWARD_USD:.0f} | "
        f"risk ${MIN_RISK_USD:.0f}-${MAX_RISK_USD:.0f} | "
        f"R/R >= {MIN_REWARD_RISK}"
    )
    print("-" * 100)

    for row in rows:
        risk = "NA" if row.risk_usd is None else f"${row.risk_usd}"
        reward = "NA" if row.reward2_usd is None else f"${row.reward2_usd}"
        rr = "NA" if row.reward_risk is None else f"{row.reward_risk}"
        trigger = "NA" if row.trigger is None else row.trigger
        sl = "NA" if row.stop_loss is None else row.stop_loss
        tp2 = "NA" if row.tp2 is None else row.tp2
        price = "NA" if row.price is None else row.price

        print(
            f"{row.symbol} | price {price} | status {row.status} | "
            f"trigger {trigger} | SL {sl} | TP2 {tp2} | "
            f"risk {risk} | reward2 {reward} | RR {rr} | "
            f"fail {row.fail_reasons or '-'}"
        )

    print("-" * 100)
    print("Wrote okx_market_report.csv")
    print("Wrote okx_market_report.json")


def main() -> None:
    tickers = fetch_spot_tickers()

    rows: list[ScanRow] = []
    for symbol in SYMBOLS:
        rows.append(scan_symbol(symbol, tickers))

    # Put real triggers at the top of reports.
    rows.sort(key=lambda x: (x.status != "WATCH_TRIGGER", x.symbol))

    write_csv(rows)
    write_json(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
