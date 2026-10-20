from typing import Any

from financial_assistant.tools.market_data import get_market_data
from financial_assistant.tools.technical_indicators import (
    add_technical_indicators,
)


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

    df = add_technical_indicators(df)

    latest = df.iloc[-1]
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
    }