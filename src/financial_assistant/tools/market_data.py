import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.twelvedata.com/time_series"

SUPPORTED_SYMBOLS = {"XAU/USD", "EUR/USD"}

SUPPORTED_INTERVALS = {
    "1h",
    "4h",
    "1day",
}

INTERVAL_DELTAS = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1day": timedelta(days=1),
}

OHLC_COLUMNS = ["datetime", "open", "high", "low", "close"]


def drop_incomplete_candle(
    df: pd.DataFrame,
    interval: str,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Remove the last candle if it has not closed yet.

    Twelve Data also returns the candle that is still forming. Indicators
    computed on it would change with every tick, so it is excluded.
    """

    if df.empty:
        return df

    now = now or datetime.now(timezone.utc)
    last_candle_end = df["datetime"].iloc[-1] + INTERVAL_DELTAS[interval]

    if last_candle_end > now:
        return df.iloc[:-1].reset_index(drop=True)

    return df


def get_market_data(
    symbol: str,
    interval: str = "1h",
    outputsize: int = 100,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Fetch historical OHLC candles for a supported forex or metal symbol.

    By default the candle that is still forming is dropped, so the result
    contains at most ``outputsize`` closed candles.
    """

    symbol = symbol.strip().upper()

    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(
            f"Unsupported symbol: {symbol}. "
            f"Supported symbols: {sorted(SUPPORTED_SYMBOLS)}"
        )

    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Unsupported interval: {interval}. "
            f"Supported intervals: {sorted(SUPPORTED_INTERVALS)}"
        )

    if not 1 <= outputsize <= 5000:
        raise ValueError("outputsize must be between 1 and 5000.")

    api_key = os.getenv("TWELVE_DATA_API_KEY")

    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY is not configured.")

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "timezone": "UTC",
        "order": "asc",
        "apikey": api_key,
    }

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch market data: {exc}"
        ) from exc

    if payload.get("status") == "error":
        raise RuntimeError(
            f"Twelve Data API error: {payload.get('message', payload)}"
        )

    values = payload.get("values")

    if not values:
        raise RuntimeError(
            f"No market data returned for {symbol}."
        )

    df = pd.DataFrame(values)

    missing_columns = set(OHLC_COLUMNS) - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            f"API response is missing columns: {missing_columns}"
        )

    df = df[OHLC_COLUMNS].copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(subset=OHLC_COLUMNS)
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )

    if drop_incomplete:
        df = drop_incomplete_candle(df, interval)

    if df.empty:
        raise RuntimeError("No valid OHLC candles remain after cleaning.")

    return df
