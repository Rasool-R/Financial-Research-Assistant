import pandas as pd

REQUIRED_COLUMNS = {"datetime", "open", "high", "low", "close"}


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLC market data."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Market data cannot be empty.")

    result = df.copy()
    close = result["close"]

    # Simple and Exponential Moving Averages
    result["sma_20"] = close.rolling(window=20).mean()
    result["sma_50"] = close.rolling(window=50).mean()

    result["ema_12"] = close.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    result["ema_26"] = close.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    # Relative Strength Index (RSI)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    relative_strength = avg_gain / avg_loss

    result["rsi_14"] = 100 - (
        100 / (1 + relative_strength)
    )

    # Handle edge cases for RSI
    result.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "rsi_14",
    ] = 100

    result.loc[
        (avg_gain == 0) & (avg_loss > 0),
        "rsi_14",
    ] = 0

    result.loc[
        (avg_gain == 0) & (avg_loss == 0),
        "rsi_14",
    ] = 50

    # MACD
    result["macd"] = result["ema_12"] - result["ema_26"]

    result["macd_signal"] = result["macd"].ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    result["macd_histogram"] = (
        result["macd"] - result["macd_signal"]
    )

    # Average True Range (ATR) with Wilder smoothing
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["atr_14"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    # Bollinger Bands (20 periods, 2 population standard deviations)
    rolling_std = close.rolling(window=20).std(ddof=0)

    result["bb_middle"] = result["sma_20"]
    result["bb_upper"] = result["sma_20"] + 2 * rolling_std
    result["bb_lower"] = result["sma_20"] - 2 * rolling_std

    return result