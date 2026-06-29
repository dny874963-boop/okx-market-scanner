#!/usr/bin/env python3
"""
OKX Market Scanner - SOL 94 realistic trigger model

Public market-data scanner only.
No API keys. No trading actions.

Purpose:
- Find realistic spot swing triggers.
- Prioritize SOL-style setups: ~900 USDT position, realistic TP2 reward around 85-100 USDT.
- Use support/resistance, ATR, EMA, RSI and Fibonacci pullback/extension levels.
- Output short trigger_report.txt for easy copy/paste into ChatGPT.

Outputs:
- okx_market_report.csv
- okx_market_report.json
- trigger_report.txt
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://www.okx.com"

# Account model based on the profitable SOL trade:
# ~824 USDT cost, ~91 USDT profit, ~11% move.
ACCOUNT_USDT = 900.0

QUOTE = "USDT"
BAR = "4H"
CANDLE_LIMIT = 150
DAILY_BAR = "1D"
DAILY_LIMIT = 120

TOP_N_SYMBOLS_BY_VOLUME = 180
MIN_24H_QUOTE_VOLUME_USDT = 250_000.0

# Target/risk model
GENERAL_MIN_REWARD_USD = 55.0
SOL_MIN_REWARD_USD = 75.0
SOL_IDEAL_REWARD_USD = 90.0
SOL_MAX_REASONABLE_REWARD_USD = 115.0

MIN_RISK_USD = 10.0
MAX_RISK_USD = 38.0
SOL_MAX_RISK_USD = 42.0

MIN_REWARD_RISK = 2.20
SOL_MIN_REWARD_RISK = 2.60

MAX_DISTANCE_TO_TRIGGER_PCT = 4.0
SOL_MAX_DISTANCE_TO_TRIGGER_PCT = 5.0

MIN_TARGET_PCT = 4.5
MAX_TARGET_PCT = 15.0
SOL_IDEAL_TARGET_PCT_LOW = 8.0
SOL_IDEAL_TARGET_PCT_HIGH = 12.5

ATR_PERIOD = 14
RSI_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 50
BREAKOUT_LOOKBACK = 20
RECENT_SUPPORT_LOOKBACK = 12
FIB_LOOKBACK = 72
DAILY_RESISTANCE_LOOKBACK = 45

TRIGGER_BUFFER_PCT = 0.0015
RETEST_TRIGGER_BUFFER_PCT = 0.0030
ATR_STOP_MULT = 1.20
REQUEST_SLEEP_SECONDS = 0.13

PRIORITY_SYMBOLS = [
    "SOL-USDT",
    "LINK-USDT",
    "RENDER-USDT",
    "TON-USDT",
    "AVAX-USDT",
    "XRP-USDT",
    "SUI-USDT",
    "ETH-USDT",
]

EXCLUDED_BASES = {
    "USDT", "USDC", "USDG", "DAI", "TUSD", "FDUSD", "USDK", "EURT", "PYUSD",
    "USTC", "LUNC", "BETH", "STETH", "WBTC", "WETH",
}

EXCLUDED_BASE_SUFFIXES = (
    "UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S",
)


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
    universe_rank: int | None = None


@dataclass
class ScanRow:
    symbol: str
    universe_rank: int | None
    priority_rank: int | None
    setup_type: str
    score: float
    price: float | None
    quote_volume_24h: float | None
    status: str
    filters_passed: bool
    trigger: float | None
    stop_loss: float | None
    tp1: float | None
    tp2: float | None
    position_usdt: float
    quantity: float | None
    risk_usd: float | None
    target_reward_usd: float | None
    reward_risk: float | None
    distance_to_trigger_pct: float | None
    target_pct: float | None
    stop_pct: float | None
    atr_4h: float | None
    rsi_4h: float | None
    ema20_4h: float | None
    ema50_4h: float | None
    fib_50: float | None
    fib_618: float | None
    fib_786: float | None
    recent_support: float | None
    recent_resistance: float | None
    daily_resistance: float | None
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


def fetch_all_tickers() -> list[MarketTicker]:
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

        tickers.append(MarketTicker(symbol, price, quote_volume, base, quote))

    tickers.sort(key=lambda x: x.quote_volume_24h, reverse=True)

    for idx, ticker in enumerate(tickers, start=1):
        ticker.universe_rank = idx

    return tickers


def fetch_dynamic_universe() -> list[MarketTicker]:
    all_tickers = fetch_all_tickers()

    selected: dict[str, MarketTicker] = {}

    for ticker in all_tickers:
        if ticker.quote_volume_24h >= MIN_24H_QUOTE_VOLUME_USDT:
            selected[ticker.symbol] = ticker
        if len(selected) >= TOP_N_SYMBOLS_BY_VOLUME:
            break

    # Force scan priority symbols, especially SOL.
    by_symbol = {ticker.symbol: ticker for ticker in all_tickers}
    for symbol in PRIORITY_SYMBOLS:
        if symbol in by_symbol:
            selected[symbol] = by_symbol[symbol]

    universe = list(selected.values())
    universe.sort(
        key=lambda x: (
            0 if x.symbol == "SOL-USDT" else 1,
            PRIORITY_SYMBOLS.index(x.symbol) if x.symbol in PRIORITY_SYMBOLS else 999,
            x.universe_rank or 999999,
        )
    )
    return universe


def fetch_candles(symbol: str, bar: str = BAR, limit: int = CANDLE_LIMIT) -> list[Candle]:
    data = http_get(
        "/api/v5/market/candles",
        {"instId": symbol, "bar": bar, "limit": str(limit)},
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
    min_needed = max(FIB_LOOKBACK + 2, EMA_SLOW + 2, ATR_PERIOD + 2)
    if len(confirmed) >= min_needed:
        return confirmed

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


def calc_ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None

    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for value in values[period:]:
        ema = value * k + ema * (1 - k)
    return ema


def calc_rsi(values: list[float], period: int = RSI_PERIOD) -> float | None:
    if len(values) < period + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))

    if len(gains) < period:
        return None

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


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
    if price is None or not math.isfinite(price):
        return None
    base = abs(ref if ref is not None else price)
    return round(price, decimals_for_price(base))


def round_num(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def pct(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return (a / b - 1.0) * 100.0


def safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def fibonacci_levels(candles: list[Candle]) -> dict[str, float] | None:
    if len(candles) < FIB_LOOKBACK:
        return None

    window = candles[-FIB_LOOKBACK:]
    swing_low = min(c.low for c in window)
    swing_high = max(c.high for c in window)

    if swing_high <= swing_low:
        return None

    rng = swing_high - swing_low

    return {
        "swing_low": swing_low,
        "swing_high": swing_high,
        "fib_50": swing_high - 0.500 * rng,
        "fib_618": swing_high - 0.618 * rng,
        "fib_786": swing_high - 0.786 * rng,
        "ext_1272": swing_high + 0.272 * rng,
        "ext_1618": swing_high + 0.618 * rng,
    }


def choose_tp_levels(
    trigger: float,
    fib: dict[str, float] | None,
    daily_resistance: float | None,
    setup_type: str,
    symbol: str,
) -> tuple[float | None, float | None]:
    candidates: list[float] = []

    if fib:
        candidates.extend([
            fib["swing_high"],
            fib["ext_1272"],
            fib["ext_1618"],
        ])

    if daily_resistance:
        candidates.append(daily_resistance)

    # Only targets above trigger.
    candidates = sorted({x for x in candidates if x > trigger * 1.01})

    if not candidates:
        return None, None

    # TP1 = nearest reasonable resistance.
    tp1 = candidates[0]

    # TP2 = realistic target, not fantasy target.
    upper_pct = SOL_IDEAL_TARGET_PCT_HIGH if symbol == "SOL-USDT" else MAX_TARGET_PCT
    lower_pct = SOL_IDEAL_TARGET_PCT_LOW if symbol == "SOL-USDT" else MIN_TARGET_PCT

    preferred: list[float] = []
    backup: list[float] = []

    for level in candidates:
        target_pct = pct(level, trigger)
        if target_pct is None:
            continue

        if lower_pct <= target_pct <= upper_pct:
            preferred.append(level)
        elif MIN_TARGET_PCT <= target_pct <= MAX_TARGET_PCT:
            backup.append(level)

    if preferred:
        # Choose the nearest target inside the realistic zone.
        tp2 = preferred[0]
    elif backup:
        tp2 = backup[0]
    else:
        # No clean resistance target inside the range.
        return tp1, None

    if tp2 < tp1:
        tp1 = tp2

    return tp1, tp2


def build_candidate(
    ticker: MarketTicker,
    candles: list[Candle],
    daily_candles: list[Candle],
    setup_type: str,
) -> ScanRow:
    symbol = ticker.symbol
    price = ticker.price
    closes = [c.close for c in candles]

    atr = calc_atr(candles)
    ema20 = calc_ema(closes, EMA_FAST)
    ema50 = calc_ema(closes, EMA_SLOW)
    rsi = calc_rsi(closes)
    fib = fibonacci_levels(candles)

    recent_resistance = max(c.high for c in candles[-BREAKOUT_LOOKBACK - 1:-1])
    recent_support = min(c.low for c in candles[-RECENT_SUPPORT_LOOKBACK:])
    daily_resistance = None
    if len(daily_candles) >= DAILY_RESISTANCE_LOOKBACK + 1:
        daily_resistance = max(c.high for c in daily_candles[-DAILY_RESISTANCE_LOOKBACK - 1:-1])

    fail: list[str] = []

    if atr is None or atr <= 0:
        fail.append("אין ATR מספיק")
    if ema20 is None or ema50 is None:
        fail.append("אין EMA מספיק")
    if rsi is None:
        fail.append("אין RSI מספיק")
    if fib is None:
        fail.append("אין פיבונאצ'י מספיק")

    if fail:
        return error_row(ticker, setup_type, "; ".join(fail))

    assert atr is not None
    assert ema20 is not None
    assert ema50 is not None
    assert rsi is not None
    assert fib is not None

    close = closes[-1]

    # Setup logic
    if setup_type == "FIB_RECOVERY":
        fib_zone_low = min(fib["fib_786"], fib["fib_618"])
        fib_zone_high = max(fib["fib_50"], fib["fib_618"])
        in_fib_zone = fib_zone_low <= close <= fib_zone_high * 1.025

        trigger = max(
            close * (1.0 + RETEST_TRIGGER_BUFFER_PCT),
            max(c.high for c in candles[-4:]) * (1.0 + TRIGGER_BUFFER_PCT),
        )

        support_stop = min(recent_support, fib["fib_786"])
        atr_stop = trigger - ATR_STOP_MULT * atr
        stop_loss = min(support_stop * 0.997, atr_stop)

        if not in_fib_zone:
            fail.append("לא באזור פיבונאצ'י 0.618-0.786/התאוששות")
        if close < ema20 * 0.985 and close < ema50 * 0.985:
            fail.append("המחיר עדיין חלש מול EMA20/EMA50")

    elif setup_type == "BREAKOUT":
        trigger = recent_resistance * (1.0 + TRIGGER_BUFFER_PCT)

        support_stop = max(recent_support * 0.997, trigger - ATR_STOP_MULT * atr)
        stop_loss = support_stop

        if close < ema20 or ema20 < ema50 * 0.98:
            fail.append("אין מגמת 4H מספיק חזקה")
        if price > trigger:
            fail.append("כבר מעל הטריגר — לא לרדוף")
    else:
        return error_row(ticker, setup_type, "setup_type לא מוכר")

    tp1, tp2 = choose_tp_levels(trigger, fib, daily_resistance, setup_type, symbol)
    if tp2 is None:
        fail.append("אין יעד TP2 ריאלי מעל הטריגר")

    quantity = ACCOUNT_USDT / trigger if trigger > 0 else None
    risk_usd = None
    target_reward_usd = None
    reward_risk = None
    distance_to_trigger_pct = None
    target_pct_value = None
    stop_pct_value = None

    if quantity and stop_loss and tp2:
        risk_per_coin = max(trigger - stop_loss, 0.0)
        reward_per_coin = max(tp2 - trigger, 0.0)

        risk_usd = quantity * risk_per_coin
        target_reward_usd = quantity * reward_per_coin
        reward_risk = safe_div(target_reward_usd, risk_usd)
        distance_to_trigger_pct = ((trigger / price) - 1.0) * 100.0 if price > 0 else None
        target_pct_value = pct(tp2, trigger)
        stop_pct_value = pct(stop_loss, trigger)

    # Filters
    min_reward = SOL_MIN_REWARD_USD if symbol == "SOL-USDT" else GENERAL_MIN_REWARD_USD
    min_rr = SOL_MIN_REWARD_RISK if symbol == "SOL-USDT" else MIN_REWARD_RISK
    max_risk = SOL_MAX_RISK_USD if symbol == "SOL-USDT" else MAX_RISK_USD
    max_distance = SOL_MAX_DISTANCE_TO_TRIGGER_PCT if symbol == "SOL-USDT" else MAX_DISTANCE_TO_TRIGGER_PCT

    if ticker.quote_volume_24h < MIN_24H_QUOTE_VOLUME_USDT and symbol not in PRIORITY_SYMBOLS:
        fail.append("נזילות 24h נמוכה")

    if distance_to_trigger_pct is None:
        fail.append("אין מרחק לטריגר")
    elif distance_to_trigger_pct < -0.20:
        fail.append("המחיר כבר עבר את הטריגר")
    elif distance_to_trigger_pct > max_distance:
        fail.append("הטריגר רחוק מדי")

    if risk_usd is None:
        fail.append("אין חישוב סיכון")
    elif risk_usd < MIN_RISK_USD:
        fail.append("סטופ צפוף מדי/סיכון נמוך מדי")
    elif risk_usd > max_risk:
        fail.append("סיכון גבוה מדי")

    if target_reward_usd is None:
        fail.append("אין חישוב יעד")
    elif target_reward_usd < min_reward:
        fail.append("יעד דולר לא מספיק")

    if reward_risk is None:
        fail.append("אין יחס סיכוי-סיכון")
    elif reward_risk < min_rr:
        fail.append("יחס סיכוי-סיכון נמוך")

    if target_pct_value is None:
        fail.append("אין אחוז יעד")
    elif target_pct_value < MIN_TARGET_PCT:
        fail.append("יעד קרוב מדי")
    elif target_pct_value > MAX_TARGET_PCT:
        fail.append("יעד רחוק מדי/לא ריאלי")

    # RSI guardrail: avoid very overheated entries.
    if rsi > 76:
        fail.append("RSI גבוה מדי — לא לרדוף")

    score = score_candidate(
        symbol=symbol,
        setup_type=setup_type,
        reward_usd=target_reward_usd,
        reward_risk=reward_risk,
        distance_to_trigger_pct=distance_to_trigger_pct,
        target_pct_value=target_pct_value,
        rsi=rsi,
        close=close,
        ema20=ema20,
        ema50=ema50,
        fail_count=len(fail),
    )

    filters_passed = len(fail) == 0

    if filters_passed:
        status = "WATCH_TRIGGER"
    else:
        status = "NO"

    note = make_note(symbol, setup_type, target_reward_usd, reward_risk, target_pct_value, rsi, close, ema20, ema50)

    return ScanRow(
        symbol=symbol,
        universe_rank=ticker.universe_rank,
        priority_rank=PRIORITY_SYMBOLS.index(symbol) + 1 if symbol in PRIORITY_SYMBOLS else None,
        setup_type=setup_type,
        score=round_num(score, 2) or 0.0,
        price=round_price(price),
        quote_volume_24h=round_num(ticker.quote_volume_24h, 2),
        status=status,
        filters_passed=filters_passed,
        trigger=round_price(trigger, price),
        stop_loss=round_price(stop_loss, price),
        tp1=round_price(tp1, price),
        tp2=round_price(tp2, price),
        position_usdt=ACCOUNT_USDT,
        quantity=round_num(quantity, 8),
        risk_usd=round_num(risk_usd, 2),
        target_reward_usd=round_num(target_reward_usd, 2),
        reward_risk=round_num(reward_risk, 2),
        distance_to_trigger_pct=round_num(distance_to_trigger_pct, 2),
        target_pct=round_num(target_pct_value, 2),
        stop_pct=round_num(stop_pct_value, 2),
        atr_4h=round_price(atr, price),
        rsi_4h=round_num(rsi, 2),
        ema20_4h=round_price(ema20, price),
        ema50_4h=round_price(ema50, price),
        fib_50=round_price(fib["fib_50"], price),
        fib_618=round_price(fib["fib_618"], price),
        fib_786=round_price(fib["fib_786"], price),
        recent_support=round_price(recent_support, price),
        recent_resistance=round_price(recent_resistance, price),
        daily_resistance=round_price(daily_resistance, price) if daily_resistance else None,
        fail_reasons="; ".join(fail),
        note=note,
    )


def score_candidate(
    symbol: str,
    setup_type: str,
    reward_usd: float | None,
    reward_risk: float | None,
    distance_to_trigger_pct: float | None,
    target_pct_value: float | None,
    rsi: float,
    close: float,
    ema20: float,
    ema50: float,
    fail_count: int,
) -> float:
    score = 0.0

    if symbol == "SOL-USDT":
        score += 35.0
    elif symbol in PRIORITY_SYMBOLS:
        score += 8.0

    if setup_type == "FIB_RECOVERY":
        score += 12.0
    elif setup_type == "BREAKOUT":
        score += 8.0

    if reward_usd is not None:
        if symbol == "SOL-USDT":
            # Best score near the historical SOL model: 85-100 USDT target.
            score += max(0.0, 25.0 - abs(reward_usd - SOL_IDEAL_REWARD_USD) * 0.45)
            if SOL_IDEAL_REWARD_USD <= reward_usd <= SOL_MAX_REASONABLE_REWARD_USD:
                score += 6.0
        else:
            score += min(20.0, reward_usd / 5.0)

    if reward_risk is not None:
        score += min(22.0, reward_risk * 5.0)

    if distance_to_trigger_pct is not None:
        if 0.0 <= distance_to_trigger_pct <= 1.5:
            score += 14.0
        elif distance_to_trigger_pct <= 3.5:
            score += 8.0
        elif distance_to_trigger_pct <= 5.0:
            score += 4.0

    if target_pct_value is not None:
        if symbol == "SOL-USDT" and SOL_IDEAL_TARGET_PCT_LOW <= target_pct_value <= SOL_IDEAL_TARGET_PCT_HIGH:
            score += 12.0
        elif MIN_TARGET_PCT <= target_pct_value <= MAX_TARGET_PCT:
            score += 6.0

    if close >= ema20 >= ema50:
        score += 8.0
    elif close >= ema20:
        score += 4.0

    if 42 <= rsi <= 68:
        score += 6.0
    elif 35 <= rsi <= 75:
        score += 3.0

    score -= fail_count * 7.0
    return max(0.0, score)


def make_note(
    symbol: str,
    setup_type: str,
    reward_usd: float | None,
    reward_risk: float | None,
    target_pct_value: float | None,
    rsi: float,
    close: float,
    ema20: float,
    ema50: float,
) -> str:
    parts: list[str] = []

    if symbol == "SOL-USDT":
        parts.append("SOL עדיפות: מודל יעד דומה לעסקה הרווחית ~90 USDT.")

    if setup_type == "FIB_RECOVERY":
        parts.append("כניסה רק באישור התאוששות מאזור פיבונאצ'י, לא קנייה עיוורת.")
    elif setup_type == "BREAKOUT":
        parts.append("כניסה רק בפריצה מעל התנגדות 4H, לא לרדוף אחרי מחיר שכבר ברח.")

    if reward_usd is not None:
        parts.append(f"יעד מחושב: {reward_usd:.2f} USDT.")
    if reward_risk is not None:
        parts.append(f"יחס: 1:{reward_risk:.2f}.")
    if target_pct_value is not None:
        parts.append(f"מהלך יעד: {target_pct_value:.2f}%.")

    trend = "מעל EMA20/EMA50" if close >= ema20 >= ema50 else "לא מגמה נקייה"
    parts.append(f"RSI {rsi:.1f}; {trend}.")

    return " ".join(parts)


def error_row(ticker: MarketTicker, setup_type: str, reason: str) -> ScanRow:
    return ScanRow(
        symbol=ticker.symbol,
        universe_rank=ticker.universe_rank,
        priority_rank=PRIORITY_SYMBOLS.index(ticker.symbol) + 1 if ticker.symbol in PRIORITY_SYMBOLS else None,
        setup_type=setup_type,
        score=0.0,
        price=round_price(ticker.price),
        quote_volume_24h=round_num(ticker.quote_volume_24h, 2),
        status="NO",
        filters_passed=False,
        trigger=None,
        stop_loss=None,
        tp1=None,
        tp2=None,
        position_usdt=ACCOUNT_USDT,
        quantity=None,
        risk_usd=None,
        target_reward_usd=None,
        reward_risk=None,
        distance_to_trigger_pct=None,
        target_pct=None,
        stop_pct=None,
        atr_4h=None,
        rsi_4h=None,
        ema20_4h=None,
        ema50_4h=None,
        fib_50=None,
        fib_618=None,
        fib_786=None,
        recent_support=None,
        recent_resistance=None,
        daily_resistance=None,
        fail_reasons=reason,
        note="אין מספיק נתונים טכניים נקיים לסריקה.",
    )


def scan_symbol(ticker: MarketTicker) -> list[ScanRow]:
    try:
        candles_4h = fetch_candles(ticker.symbol, BAR, CANDLE_LIMIT)
        time.sleep(REQUEST_SLEEP_SECONDS)

        daily_candles = fetch_candles(ticker.symbol, DAILY_BAR, DAILY_LIMIT)
        time.sleep(REQUEST_SLEEP_SECONDS)

        min_needed = max(FIB_LOOKBACK + 2, EMA_SLOW + 2, ATR_PERIOD + 2)
        if len(candles_4h) < min_needed:
            return [error_row(ticker, "ALL", "אין מספיק נרות 4H")]

        rows = [
            build_candidate(ticker, candles_4h, daily_candles, "FIB_RECOVERY"),
            build_candidate(ticker, candles_4h, daily_candles, "BREAKOUT"),
        ]

        # Keep the best setup per symbol.
        rows.sort(key=lambda r: (r.filters_passed, r.score), reverse=True)
        return [rows[0]]

    except Exception as exc:
        return [error_row(ticker, "ERROR", f"שגיאה בסריקה: {type(exc).__name__}: {exc}")]


def row_to_dict(row: ScanRow) -> dict[str, Any]:
    return asdict(row)


def write_csv(rows: list[ScanRow], path: Path) -> None:
    data = [row_to_dict(r) for r in rows]
    if not data:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def write_json(rows: list[ScanRow], path: Path) -> None:
    passed = [r for r in rows if r.filters_passed]
    rejected = [r for r in rows if not r.filters_passed]

    passed.sort(key=lambda r: (r.symbol != "SOL-USDT", -r.score))
    rejected.sort(key=lambda r: (r.symbol != "SOL-USDT", -r.score))

    report = {
        "generated_at_utc": now_utc_iso(),
        "model": "SOL_94_REALISTIC_V1",
        "disclaimer": "בדיקה ידנית בלבד. לא פקודת קנייה ולא ייעוץ פיננסי.",
        "account_usdt": ACCOUNT_USDT,
        "model_basis": {
            "known_from_uploaded_history": {
                "best_trade_symbol": "SOL-USDT",
                "approx_entry": 66.5365,
                "approx_exit": 73.9160,
                "approx_return_pct": 11.09,
                "approx_profit_usdt": 91.37,
                "note": "מבוסס על היסטוריית עסקאות שהועלתה. אין בקובץ נרות מחיר, לכן פיבונאצ'י היסטורי לעסקה לא אומת מתוך הקובץ.",
            },
            "scanner_target": "לתעדף טריגרים עם יעד ריאלי, בעיקר SOL סביב 85-100 USDT רווח פוטנציאלי על 900 USDT.",
        },
        "WATCH_TRIGGER": [row_to_dict(r) for r in passed[:20]],
        "SOL_PRIORITY": [row_to_dict(r) for r in rows if r.symbol == "SOL-USDT"],
        "REJECTED_SAMPLE": [row_to_dict(r) for r in rejected[:30]],
        "ALL_ROWS": [row_to_dict(r) for r in rows],
    }

    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_trigger_txt(rows: list[ScanRow], path: Path) -> None:
    passed = [r for r in rows if r.filters_passed]
    passed.sort(key=lambda r: (r.symbol != "SOL-USDT", -r.score))

    lines: list[str] = [
        "OKX WATCH TRIGGERS",
        "בדיקה ידנית בלבד. לא פקודת קנייה ולא ייעוץ פיננסי.",
        f"מודל: SOL_94_REALISTIC_V1 | זמן UTC: {now_utc_iso()}",
        "",
    ]

    if not passed:
        lines.append("אין טריגרים נקיים כרגע.")
        sol_rows = [r for r in rows if r.symbol == "SOL-USDT"]
        if sol_rows:
            sol = sorted(sol_rows, key=lambda r: r.score, reverse=True)[0]
            lines.extend([
                "",
                "SOL נבדק אבל לא עבר:",
                f"מחיר: {sol.price}",
                f"טריגר: {sol.trigger}",
                f"סטופ: {sol.stop_loss}",
                f"TP2: {sol.tp2}",
                f"סיכון: {sol.risk_usd}",
                f"יעד: {sol.target_reward_usd}",
                f"יחס: {sol.reward_risk}",
                f"סיבות פסילה: {sol.fail_reasons}",
            ])
    else:
        for item in passed[:12]:
            lines.extend([
                f"--- {item.symbol} | {item.setup_type} | score {item.score} ---",
                f"מחיר עכשיו: {item.price}",
                f"כניסה רק אם נוגע/פורץ טריגר: {item.trigger}",
                f"סטופ: {item.stop_loss}",
                f"TP1: {item.tp1}",
                f"TP2: {item.tp2}",
                f"סיכון משוער: {item.risk_usd} USDT",
                f"יעד משוער: {item.target_reward_usd} USDT",
                f"יחס סיכוי/סיכון: 1:{item.reward_risk}",
                f"מרחק לטריגר: {item.distance_to_trigger_pct}%",
                f"פיבו: 0.50={item.fib_50}, 0.618={item.fib_618}, 0.786={item.fib_786}",
                f"הערה: {item.note}",
                "",
            ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Fetching OKX universe...")
    universe = fetch_dynamic_universe()
    print(f"Scanning {len(universe)} symbols...")

    rows: list[ScanRow] = []

    for idx, ticker in enumerate(universe, start=1):
        print(f"[{idx}/{len(universe)}] {ticker.symbol}")
        rows.extend(scan_symbol(ticker))

    rows.sort(key=lambda r: (r.symbol != "SOL-USDT", not r.filters_passed, -r.score))

    write_csv(rows, Path("okx_market_report.csv"))
    write_json(rows, Path("okx_market_report.json"))
    write_trigger_txt(rows, Path("trigger_report.txt"))

    passed_count = sum(1 for r in rows if r.filters_passed)
    print(f"Done. WATCH_TRIGGER count: {passed_count}")
    print("Wrote okx_market_report.csv, okx_market_report.json, trigger_report.txt")


if __name__ == "__main__":
    main()
