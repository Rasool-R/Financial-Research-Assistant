"""Price-structure features computed from candles that already carry
indicators: swing levels, SMA crosses, recent change and volatility context.

All functions are pure and return plain Python values, so they can be
unit-tested without any API access.
"""

import numpy as np
import pandas as pd

SWING_WINDOW = 3  # candles on each side that must be lower/higher
LEVEL_LOOKBACK = 60  # candles searched for support and resistance
ATR_MEDIAN_WINDOW = 50

ELEVATED_VOLATILITY_RATIO = 1.25
SUBDUED_VOLATILITY_RATIO = 0.75

CANDLES_PER_24H = {"1h": 24, "4h": 6, "1day": 1}


def find_swing_points(
    df: pd.DataFrame,
    window: int = SWING_WINDOW,
) -> tuple[list[dict], list[dict]]:
    """Return confirmed swing highs and swing lows.

    A swing high is a candle whose high is the highest within ``window``
    candles on both sides. The last ``window`` candles cannot be confirmed
    yet. For flat tops/bottoms only the first candle of the plateau counts.
    """

    df = df.reset_index(drop=True)
    size = 2 * window + 1

    highs = df["high"]
    lows = df["low"]

    is_high = (highs.rolling(size, center=True).max() == highs) & (
        highs.shift(1) != highs
    )
    is_low = (lows.rolling(size, center=True).min() == lows) & (
        lows.shift(1) != lows
    )

    def points(mask: pd.Series, prices: pd.Series) -> list[dict]:
        return [
            {
                "index": int(i),
                "datetime": df["datetime"].iloc[i].isoformat(),
                "price": float(prices.iloc[i]),
            }
            for i in np.flatnonzero(mask.to_numpy())
        ]

    return points(is_high, highs), points(is_low, lows)


def _level(point: dict, close: float, atr: float) -> dict:
    distance = abs(point["price"] - close)

    return {
        "price": point["price"],
        "datetime": point["datetime"],
        "distance": distance,
        "distance_atr": distance / atr if atr > 0 else None,
    }


def key_levels(
    df: pd.DataFrame,
    atr: float,
    lookback: int = LEVEL_LOOKBACK,
    window: int = SWING_WINDOW,
) -> dict:
    """Nearest swing support/resistance around the latest close.

    ``support`` / ``resistance`` are None when no such swing level exists in
    the lookback window (for example when price is at a new high).
    """

    recent = df.iloc[-lookback:].reset_index(drop=True)
    close = float(recent["close"].iloc[-1])

    swing_highs, swing_lows = find_swing_points(recent, window)

    above = [p for p in swing_highs if p["price"] > close]
    below = [p for p in swing_lows if p["price"] < close]

    resistance = min(above, key=lambda p: p["price"]) if above else None
    support = max(below, key=lambda p: p["price"]) if below else None

    return {
        "support": _level(support, close, atr) if support else None,
        "resistance": _level(resistance, close, atr) if resistance else None,
        "lookback_high": float(recent["high"].max()),
        "lookback_low": float(recent["low"].min()),
        "lookback_candles": len(recent),
    }


def last_sma_cross(df: pd.DataFrame) -> dict | None:
    """Most recent SMA 20 / SMA 50 crossover, or None if there was none."""

    diff = (df["sma_20"] - df["sma_50"]).reset_index(drop=True).dropna()

    # an exact tie keeps the previous direction instead of counting as a cross
    side = np.sign(diff).replace(0, np.nan).ffill().dropna()

    if len(side) < 2:
        return None

    flips = side.ne(side.shift()).to_numpy(copy=True)
    flips[0] = False
    positions = side.index[flips]

    if len(positions) == 0:
        return None

    position = positions[-1]

    return {
        "direction": "bullish" if side.loc[position] > 0 else "bearish",
        "candles_ago": int(len(df) - 1 - position),
    }


def percent_change(df: pd.DataFrame, candles: int) -> float | None:
    """Close-to-close change over the last ``candles`` candles, in percent."""

    if candles < 1 or len(df) <= candles:
        return None

    past = float(df["close"].iloc[-1 - candles])

    if past == 0:
        return None

    return (float(df["close"].iloc[-1]) - past) / past * 100


def describe_volatility(df: pd.DataFrame, atr: float, close: float) -> dict:
    """ATR in price units, in percent of price, and versus its recent median."""

    history = df["atr_14"].dropna().iloc[-ATR_MEDIAN_WINDOW:]
    median = float(history.median()) if len(history) else 0.0
    ratio = atr / median if median > 0 else None

    if ratio is None:
        condition = "unknown"
    elif ratio >= ELEVATED_VOLATILITY_RATIO:
        condition = "elevated"
    elif ratio <= SUBDUED_VOLATILITY_RATIO:
        condition = "subdued"
    else:
        condition = "normal"

    return {
        "atr_14": atr,
        "atr_percent": atr / close * 100 if close else None,
        "atr_vs_median": ratio,
        "condition": condition,
    }


def describe_bollinger(latest: pd.Series) -> dict:
    """Band values, %B, bandwidth and where the close sits inside the bands."""

    upper = float(latest["bb_upper"])
    middle = float(latest["bb_middle"])
    lower = float(latest["bb_lower"])
    close = float(latest["close"])
    width = upper - lower

    if width > 0:
        percent_b = (close - lower) / width

        if percent_b >= 1:
            position = "above_upper_band"
        elif percent_b <= 0:
            position = "below_lower_band"
        elif percent_b >= 0.8:
            position = "near_upper_band"
        elif percent_b <= 0.2:
            position = "near_lower_band"
        else:
            position = "inside_bands"
    else:
        percent_b = None
        position = "undefined"

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "percent_b": percent_b,
        "bandwidth_percent": width / middle * 100 if middle else None,
        "position": position,
    }
