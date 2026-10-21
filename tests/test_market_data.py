from datetime import datetime, timezone

import pandas as pd
import pytest

from financial_assistant.tools import market_summary as ms
from financial_assistant.tools.market_data import drop_incomplete_candle


def make_candles(n: int, start: str = "2026-01-01", freq: str = "h"):
    close = pd.Series([100.0 + i for i in range(n)])

    return pd.DataFrame(
        {
            "datetime": pd.date_range(start, periods=n, freq=freq, tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    )


def test_drop_incomplete_candle_removes_forming_candle():
    df = make_candles(5)  # last candle starts 04:00 and ends 05:00
    now = datetime(2026, 1, 1, 4, 30, tzinfo=timezone.utc)

    result = drop_incomplete_candle(df, "1h", now=now)

    assert len(result) == 4


def test_drop_incomplete_candle_keeps_closed_candle():
    df = make_candles(5)
    now = datetime(2026, 1, 1, 5, 0, tzinfo=timezone.utc)

    result = drop_incomplete_candle(df, "1h", now=now)

    assert len(result) == 5


def test_drop_incomplete_candle_respects_interval():
    df = make_candles(3, freq="4h")  # last candle: 08:00-12:00
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    assert len(drop_incomplete_candle(df, "4h", now=now)) == 2


def test_build_data_quality_fresh_data_is_not_stale():
    latest_start = pd.Timestamp("2026-01-01 10:00", tz="UTC")
    now = datetime(2026, 1, 1, 11, 20, tzinfo=timezone.utc)

    quality = ms.build_data_quality(latest_start, "1h", now=now)

    assert quality["is_stale"] is False
    assert quality["note"] is None
    assert quality["latest_candle_closed"] is True


def test_build_data_quality_flags_weekend_gap_as_stale():
    latest_start = pd.Timestamp("2026-01-02 21:00", tz="UTC")  # Friday
    now = datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc)  # Sunday

    quality = ms.build_data_quality(latest_start, "1h", now=now)

    assert quality["is_stale"] is True
    assert "market may be closed" in quality["note"]


def test_market_summary_rejects_too_few_candles(monkeypatch):
    monkeypatch.setattr(
        ms, "get_market_data", lambda **kwargs: make_candles(30)
    )

    with pytest.raises(ValueError, match="Not enough closed candles"):
        ms.get_market_summary("XAU/USD", "1h", outputsize=100)


def test_market_summary_includes_data_quality(monkeypatch):
    monkeypatch.setattr(
        ms, "get_market_data", lambda **kwargs: make_candles(100)
    )

    summary = ms.get_market_summary("XAU/USD", "1h", outputsize=100)

    assert summary["market_conditions"]["trend"] == "bullish"
    assert "data_quality" in summary
    assert summary["data_quality"]["latest_candle_closed"] is True
