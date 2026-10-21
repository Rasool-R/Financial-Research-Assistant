from datetime import datetime, timezone
from typing import Any

import pandas as pd

from financial_assistant.tools.market_data import (
    INTERVAL_DELTAS,
    get_market_data,
)
from financial_assistant.tools.technical_indicators import (
    add_technical_indicators,
)

# sma_50 is the longest indicator window used by the summary.
MIN_CANDLES = 50

INDICATOR_COLUMNS = [
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
]


def build_data_quality(
    latest_candle_start,
    interval: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Describe how fresh the latest closed candle is."""

    now = now or datetime.now(timezone.utc)
    delta = INTERVAL_DELTAS[interval]
    candle_end = latest_candle_start + delta
    age = now - candle_end
    age_hours = round(age.total_seconds() / 3600, 2)
    is_stale = age > 2 * delta

    note = None
    if is_stale:
        note = (
            f"The latest closed candle ended {age_hours} hours ago. "
            "The market may be closed (e.g. weekend or holiday) or the "
            "data may be delayed; do not treat these prices as live."
        )

    return {
        "as_of": now.isoformat(),
        "latest_candle_closed": True,
        "latest_candle_end": candle_end.isoformat(),
        "data_age_hours": age_hours,
        "is_stale": is_stale,
        "note": note,
    }


def get_market_summary(
    symbol: str,
    interval: str = "1h",
    outputsize: int = 100,
) -> dict[str, Any]:
    """Return a structured summary of recent market conditions."""

    df = get_market_data(
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
    )

    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"Not enough closed candles for {symbol} ({interval}): "
            f"got {len(df)}, need at least {MIN_CANDLES}. "
            "Increase outputsize."
        )

    df = add_technical_indicators(df)

    latest = df.iloc[-1]

    missing_indicators = [
        column for column in INDICATOR_COLUMNS if pd.isna(latest[column])
    ]

    if missing_indicators:
        raise ValueError(
            f"Indicators could not be computed: {missing_indicators}"
        )

    data_quality = build_data_quality(latest["datetime"], interval)
    previous = df.iloc[-2] if len(df) >= 2 else None

    if latest["sma_20"] > latest["sma_50"]:
        trend = "bullish"
    elif latest["sma_20"] < latest["sma_50"]:
        trend = "bearish"
    else:
        trend = "neutral"

    rsi = latest["rsi_14"]

    if rsi >= 70:
        rsi_condition = "overbought"
    elif rsi <= 30:
        rsi_condition = "oversold"
    else:
        rsi_condition = "neutral"

    if latest["macd"] > latest["macd_signal"]:
        macd_momentum = "bullish"
    elif latest["macd"] < latest["macd_signal"]:
        macd_momentum = "bearish"
    else:
        macd_momentum = "neutral"

    latest_candle = {
        "datetime": latest["datetime"].isoformat(),
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "close": float(latest["close"]),
    }

    indicators = {
        "sma_20": float(latest["sma_20"]),
        "sma_50": float(latest["sma_50"]),
        "ema_12": float(latest["ema_12"]),
        "ema_26": float(latest["ema_26"]),
        "rsi_14": float(rsi),
        "macd": float(latest["macd"]),
        "macd_signal": float(latest["macd_signal"]),
        "macd_histogram": float(latest["macd_histogram"]),
    }

    price_change = None

    if previous is not None and previous["close"] != 0:
        price_change = (
            (latest["close"] - previous["close"])
            / previous["close"]
        ) * 100

    return {
        "symbol": symbol.strip().upper(),
        "interval": interval,
        "latest_candle": latest_candle,
        "indicators": indicators,
        "market_conditions": {
            "trend": trend,
            "rsi_condition": rsi_condition,
            "macd_momentum": macd_momentum,
            "last_candle_change_percent": (
                float(price_change)
                if price_change is not None
                else None
            ),
        },
        "data_points": len(df),
        "data_source": "Twelve Data",
        "data_quality": data_quality,
    }