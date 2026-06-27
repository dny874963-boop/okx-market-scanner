#!/usr/bin/env python3
"""
OKX Market Scanner - Dynamic Swing 50 USD filter

Public market-data scanner only.
Does not trade. Does not use API keys.

What changed:
- No fixed SYMBOLS list.
- Fetches all OKX SPOT tickers.
- Scans the top USDT pairs by 24h quote volume.
- Keeps only Swing candidates that fit the rules.

WATCH_TRIGGER only if:
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
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://www.okx.com"

ACCOUNT_USDT = 900.0

MIN_TP2_REWARD_USD = 50.0
MIN_RISK_USD = 15.0
MAX_RISK_USD = 35.0
MIN_REWARD_RISK = 1.8

QUOTE = "USDT"
TOP_N_SYMBOLS_BY_VOLUME = 180
MIN_24H_QUOTE_VOLUME_USDT = 250_000.0

EXCLUDED_BASES = {
    "USDT", "USDC", "USDG", "DAI", "TUSD", "FDUSD", "USDK", "EURT", "PYUSD",
    "USTC", "LUNC", "BETH", "STETH", "WBTC", "WETH",
}
EXCLUDED_BASE_SUFFIXES = (
    "UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S",
)

BAR = "4H"
CANDLE_LIMIT = 100
BREAKOUT_LOOKBACK = 20
SWING_LOW_LOOKBACK = 10
ATR_PERIOD = 14

TRIGGER_BUFFER_PCT = 0.001
ATR_STOP_MULT = 1.15
MIN_TP2_R_MULT = 2.0
MAX_TP2_R_MULT = 4.0
MAX_TP2_PCT_FROM_TRIGGER = 18.0
MAX_DISTANCE_TO_TRIGGER_PCT = 3.5

REQUEST_SLEEP_SECONDS = 0.13


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
class MarketTicker:
    symbol: str
    price: float
    quote_volume_24h: float
    base: str
    quote: str


@dataclass
class ScanRow:
    symbol: str
    universe_rank: int | None
    price: float | None
    quote_volume_24h: float | None
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
    response = requests.get(BASE_URL + path, params=params or {}, timeout=25)
    response.raise_for_status()
    data = response.json()
    if data.get("code") != "0":
        raise RuntimeError(f"OKX API error on {path}: {data}")
    return data


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def split_symbol(symbol: str) -> tuple[str, str] | None:
    parts = symbol.split("-")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def is_allowed_swing_symbol(symbol: str) -> bool:
    parts = split_symbol(symbol)
    if not parts:
        return False
    base, quote = parts
    if quote != QUOTE:
        return False
    if base in EXCLUDED_BASES:
        return False
    if any(base.endswith(suffix) for suffix in EXCLUDED_BASE_SUFFIXES):
        return False
    return True


def fetch_dynamic_universe() -> list[MarketTicker]:
    data = http_get("/api/v5/market/tickers", {"instType": "SPOT"})
    tickers: list[MarketTicker] = []

    for item in data.get("data", []):
        symbol = item.get("instId", "")
        if not is_allowed_swing_symbol(symbol):
            continue

        parts = split_symbol(symbol)
        if not parts:
            continue
        base, quote = parts

        price = parse_float(item.get("last"))
        if price <= 0:
            continue

        quote_volume = parse_float(item.get("volCcy24h"))
        if quote_volume <= 0:
            quote_volume = parse_float(item.get("vol24h")) * price

        if quote_volume < MIN_24H_QUOTE_VOLUME_USDT:
            continue

        tickers.append(MarketTicker(symbol, price, quote_volume, base, quote))

    tickers.sort(key=lambda x: x.quote_volume_24h, reverse=True)
    return tickers[:TOP_N_SYMBOLS_BY_VOLUME]


def fetch_candles(symbol: str) -> list[Candle]:
    data = http_get(
        "/api/v5/market/candles",
        {"instId": symbol, "bar": BAR, "limit": str(CANDLE_LIMIT)},
    )

    candles: list[Candle] = []
    for row in data.get("data", []):
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

    candles.sort(key=lambda x: x.ts)
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
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
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


def round_price(price: float | None, ref: float | None = None) -> float | None:
    if price is None:
        return None
    base = ref if ref is not None else price
    return round(price, decimals_for_price(abs(base)))


def safe_ratio(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def error_row(symbol: str, rank: int | None, price: float | None, volume: float | None, reason: str, note: str) -> ScanRow:
    return ScanRow(
        symbol=symbol,
        universe_rank=rank,
        price=round_price(price) if price else None,
        quote_volume_24h=round(volume, 2) if volume else None,
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
        fail_reasons=reason,
        note=note,
    )


def scan_symbol(ticker: MarketTicker, universe_rank: int) -> ScanRow:
    symbol = ticker.symbol
    try:
        candles = fetch_candles(symbol)
        time.sleep(REQUEST_SLEEP_SECONDS)

        min_needed = max(BREAKOUT_LOOKBACK + 2, ATR_PERIOD + 2, SWING_LOW_LOOKBACK + 2)
        if len(candles) < min_needed:
            return error_row(symbol, universe_rank, ticker.price, ticker.quote_volume_24h, "not_enough_candles", f"Need at least {min_needed} candles; got {len(candles)}")

        price = ticker.price if ticker.price > 0 else candles[-1].close
        atr = calc_atr(candles)
        if price <= 0 or atr is None or atr <= 0:
            return error_row(symbol, universe_rank, price, ticker.quote_volume_24h, "bad_price_or_atr", "Invalid price/ATR")

        prior = candles[:-1]
        resistance = max(c.high for c in prior[-BREAKOUT_LOOKBACK:])
        swing_low = min(c.low for c in prior[-SWING_LOW_LOOKBACK:])

        trigger = resistance * (1.0 + TRIGGER_BUFFER_PCT)
        atr_stop = trigger - (atr * ATR_STOP_MULT)
        stop_loss = max(swing_low, atr_stop)

        if stop_loss >= trigger:
            stop_loss = trigger * 0.97

        risk_per_coin = trigger - stop_loss
        if risk_per_coin <= 0:
            return error_row(symbol, universe_rank, price, ticker.quote_volume_24h, "invalid_stop", "Stop-loss is not below trigger")

        qty = ACCOUNT_USDT / trigger
        risk_usd = risk_per_coin * qty

        tp1 = trigger + risk_per_coin
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

        return ScanRow(
            symbol=symbol,
            universe_rank=universe_rank,
            price=round_price(price),
            quote_volume_24h=round(ticker.quote_volume_24h, 2),
            status="WATCH_TRIGGER" if filters_passed else "NO",
            trigger=round_price(trigger, price),
            stop_loss=round_price(stop_loss, price),
            tp1=round_price(tp1, price),
            tp2=round_price(tp2, price),
            position_usdt=round(ACCOUNT_USDT, 2),
            quantity=round(qty, 8),
            risk_usd=round(risk_usd, 2),
            reward2_usd=round(reward2_usd, 2),
            reward_risk=round(reward_risk, 2) if reward_risk is not None else None,
            distance_to_trigger_pct=round(distance_to_trigger_pct, 2),
            atr_4h=round_price(atr, price),
            filters_passed=filters_passed,
            fail_reasons=";".join(fail_reasons),
            note="OK for swing watch" if filters_passed else "Rejected by swing filters",
        )

    except Exception as exc:
        return error_row(symbol, universe_rank, ticker.price, ticker.quote_volume_24h, "exception", str(exc))


def write_csv(rows: list[ScanRow], path: str = "okx_market_report.csv") -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ScanRow.__dataclass_fields__.keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(rows: list[ScanRow], universe_size: int, path: str = "okx_market_report.json") -> None:
    watch = [asdict(rw) for rw in rows if rw.status == "WATCH_TRIGGER"]
    near_misses = [
        asdict(rw)
        for rw in rows
        if rw.status == "NO"
        and rw.risk_usd is not None
        and rw.reward2_usd is not None
        and rw.reward_risk is not None
        and MIN_RISK_USD <= rw.risk_usd <= MAX_RISK_USD
        and rw.reward2_usd >= MIN_TP2_REWARD_USD
        and rw.reward_risk >= MIN_REWARD_RISK
    ][:25]

    payload = {
        "generated_at_utc": now_utc_iso(),
        "scanner": "OKX Market Scanner - Dynamic Swing 50 USD filter",
        "does_trade": False,
        "account_usdt": ACCOUNT_USDT,
        "universe": {
            "mode": "dynamic_top_usdt_spot_by_24h_quote_volume",
            "quote": QUOTE,
            "top_n_symbols_by_volume": TOP_N_SYMBOLS_BY_VOLUME,
            "min_24h_quote_volume_usdt": MIN_24H_QUOTE_VOLUME_USDT,
            "symbols_scanned_after_filters": universe_size,
        },
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
            "near_miss_count": len(near_misses),
        },
        "WATCH_TRIGGER": watch,
        "NEAR_MISSES": near_misses,
        "rows": [asdict(rw) for rw in rows],
    }

    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_summary(rows: list[ScanRow], universe_size: int) -> None:
    print(f"Generated at UTC: {now_utc_iso()}")
    print(f"Universe: dynamic top {TOP_N_SYMBOLS_BY_VOLUME} OKX SPOT {QUOTE} pairs by 24h quote volume; min volume ${MIN_24H_QUOTE_VOLUME_USDT:,.0f}")
    print(f"Symbols scanned: {universe_size}")
    print(f"Rules: TP2 reward >= ${MIN_TP2_REWARD_USD:.0f} | risk ${MIN_RISK_USD:.0f}-${MAX_RISK_USD:.0f} | R/R >= {MIN_REWARD_RISK} | trigger distance <= {MAX_DISTANCE_TO_TRIGGER_PCT}%")
    print("-" * 120)

    for row in rows:
        risk = "NA" if row.risk_usd is None else f"${row.risk_usd}"
        reward = "NA" if row.reward2_usd is None else f"${row.reward2_usd}"
        rr = "NA" if row.reward_risk is None else f"{row.reward_risk}"
        trigger = "NA" if row.trigger is None else row.trigger
        sl = "NA" if row.stop_loss is None else row.stop_loss
        tp2 = "NA" if row.tp2 is None else row.tp2
        price = "NA" if row.price is None else row.price
        volume = "NA" if row.quote_volume_24h is None else f"${row.quote_volume_24h:,.0f}"

        print(f"{row.symbol} | rank {row.universe_rank} | vol {volume} | price {price} | status {row.status} | trigger {trigger} | SL {sl} | TP2 {tp2} | risk {risk} | reward2 {reward} | RR {rr} | fail {row.fail_reasons or '-'}")

    print("-" * 120)
    print("Wrote okx_market_report.csv")
    print("Wrote okx_market_report.json")


def main() -> None:
    universe = fetch_dynamic_universe()
    rows: list[ScanRow] = []

    for rank, ticker in enumerate(universe, start=1):
        rows.append(scan_symbol(ticker, rank))

    rows.sort(
        key=lambda x: (
            x.status != "WATCH_TRIGGER",
            abs(x.distance_to_trigger_pct) if x.distance_to_trigger_pct is not None else 9999,
            x.universe_rank if x.universe_rank is not None else 9999,
        )
    )

    write_csv(rows)
    write_json(rows, universe_size=len(universe))
    print_summary(rows, universe_size=len(universe))


if __name__ == "__main__":
    main()
