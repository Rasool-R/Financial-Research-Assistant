from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from financial_assistant.tools.market_data import (
    INTERVAL_DELTAS,
    get_market_data,
)
from financial_assistant.tools.market_structure import (
    CANDLES_PER_24H,
    describe_bollinger,
    describe_volatility,
    key_levels,
    last_sma_cross,
    percent_change,
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
    "atr_14",
    "bb_upper",
    "bb_middle",
    "bb_lower",
]

# Timeframes used to confirm (or contradict) the requested one.
HIGHER_TIMEFRAMES = {
    "1h": ["4h", "1day"],
    "4h": ["1day"],
    "1day": [],
}


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
    """Fetch candles and return a structured summary of market conditions."""

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

    return build_market_summary(add_technical_indicators(df), symbol, interval)


def build_market_summary(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Summarize candles that already carry indicator columns.

    Performs no I/O and looks only at the rows it is given, so a backtest can
    pass a slice that ends in the past without any look-ahead.
    """

    latest = df.iloc[-1]

    missing_indicators = [
        column for column in INDICATOR_COLUMNS if pd.isna(latest[column])
    ]

    if missing_indicators:
        raise ValueError(
            f"Indicators could not be computed: {missing_indicators}"
        )

    data_quality = build_data_quality(latest["datetime"], interval, now)
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


    close = float(latest["close"])
    atr = float(latest["atr_14"])

    price_action = {
        "change_24h_percent": percent_change(df, CANDLES_PER_24H[interval]),
        "last_sma_cross": last_sma_cross(df),
        "close_vs_sma_20_atr": (
            (close - float(latest["sma_20"])) / atr if atr > 0 else None
        ),
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
        "volatility": describe_volatility(df, atr, close),
        "bollinger": describe_bollinger(latest),
        "key_levels": key_levels(df, atr),
        "price_action": price_action,
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


def summarize_timeframe(summary: dict[str, Any]) -> dict[str, Any]:
    """Compact view of another timeframe's summary."""

    conditions = summary["market_conditions"]

    return {
        "interval": summary["interval"],
        "trend": conditions["trend"],
        "rsi_condition": conditions["rsi_condition"],
        "macd_momentum": conditions["macd_momentum"],
        "rsi_14": summary["indicators"]["rsi_14"],
        "close": summary["latest_candle"]["close"],
        "sma_20": summary["indicators"]["sma_20"],
        "sma_50": summary["indicators"]["sma_50"],
        "is_stale": summary["data_quality"]["is_stale"],
    }


def timeframe_alignment(main_trend: str, higher: dict[str, dict]) -> str:
    """aligned_bullish / aligned_bearish / mixed / single_timeframe."""

    if not higher:
        return "single_timeframe"

    trends = {main_trend, *(item["trend"] for item in higher.values())}

    if trends == {"bullish"}:
        return "aligned_bullish"

    if trends == {"bearish"}:
        return "aligned_bearish"

    return "mixed"


def get_multi_timeframe_summary(
    symbol: str,
    interval: str = "1h",
    outputsize: int = 100,
) -> dict[str, Any]:
    """Market summary for ``interval`` plus higher-timeframe confirmation.

    The requested timeframe must succeed. A higher timeframe that cannot be
    loaded is listed under ``multi_timeframe.unavailable`` instead of
    failing the whole summary.
    """

    summary = get_market_summary(symbol, interval, outputsize)

    timeframes = HIGHER_TIMEFRAMES.get(interval, [])
    higher: dict[str, dict] = {}
    unavailable: list[dict[str, str]] = []

    def load(timeframe: str):
        try:
            return timeframe, get_market_summary(
                symbol, timeframe, outputsize
            ), None
        except (RuntimeError, ValueError) as exc:
            return timeframe, None, str(exc)

    if timeframes:
        with ThreadPoolExecutor(max_workers=len(timeframes)) as pool:
            outcomes = list(pool.map(load, timeframes))

        for timeframe, result, error in outcomes:
            if error is not None:
                unavailable.append({"interval": timeframe, "error": error})
            else:
                higher[timeframe] = summarize_timeframe(result)

    trend = summary["market_conditions"]["trend"]

    summary["multi_timeframe"] = {
        "higher_timeframes": higher,
        "alignment": timeframe_alignment(trend, higher),
        "conflicting_timeframes": [
            name for name, item in higher.items() if item["trend"] != trend
        ],
        "unavailable": unavailable,
    }

    return summary