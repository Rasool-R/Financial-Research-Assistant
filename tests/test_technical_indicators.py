import numpy as np
import pandas as pd
import pytest

from financial_assistant.tools.technical_indicators import (
    add_technical_indicators,
)


def frame(close, high=None, low=None):
    close = pd.Series(close, dtype=float)

    return pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2026-01-01", periods=len(close), freq="h", tz="UTC"
            ),
            "open": close,
            "high": close + 1 if high is None else pd.Series(high, dtype=float),
            "low": close - 1 if low is None else pd.Series(low, dtype=float),
            "close": close,
        }
    )


def last(df, column):
    return add_technical_indicators(df).iloc[-1][column]


def test_missing_columns_and_empty_input_are_rejected():
    with pytest.raises(ValueError, match="Missing required columns"):
        add_technical_indicators(pd.DataFrame({"close": [1.0]}))

    with pytest.raises(ValueError, match="cannot be empty"):
        add_technical_indicators(frame([]))


def test_input_frame_is_not_modified():
    df = frame(range(100, 160))

    add_technical_indicators(df)

    assert "rsi_14" not in df.columns


def test_moving_averages_on_a_linear_series():
    df = frame(range(1, 101))  # close = 1..100

    assert last(df, "sma_20") == pytest.approx(np.mean(range(81, 101)))
    assert last(df, "sma_50") == pytest.approx(np.mean(range(51, 101)))


def test_rsi_extremes():
    assert last(frame(range(100, 200)), "rsi_14") == 100
    assert last(frame(range(200, 100, -1)), "rsi_14") == 0
    assert last(frame([100] * 100), "rsi_14") == 50


def test_rsi_matches_the_textbook_wilder_calculation():
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 1, 300))

    delta = np.diff(prices)
    gains = np.clip(delta, 0, None)
    losses = -np.clip(delta, None, 0)
    avg_gain, avg_loss = gains[:14].mean(), losses[:14].mean()
    for k in range(14, len(delta)):
        avg_gain = (avg_gain * 13 + gains[k]) / 14
        avg_loss = (avg_loss * 13 + losses[k]) / 14
    expected = 100 - 100 / (1 + avg_gain / avg_loss)

    assert last(frame(prices), "rsi_14") == pytest.approx(expected, abs=0.01)


def test_atr_equals_the_candle_range_when_there_are_no_gaps():
    # close is flat, every candle spans 99..101
    assert last(frame([100] * 60), "atr_14") == pytest.approx(2.0)


def test_atr_reacts_to_a_gap():
    close = [100] * 30 + [110]
    high = [101] * 30 + [110]
    low = [99] * 30 + [110]

    # the gap candle has no range itself, but its true range is 10
    expected = 13 / 14 * 2.0 + 1 / 14 * 10.0

    assert last(frame(close, high, low), "atr_14") == pytest.approx(expected)


def test_atr_needs_a_full_window():
    atr = add_technical_indicators(frame([100] * 30))["atr_14"]

    assert atr.iloc[:13].isna().all()
    assert not np.isnan(atr.iloc[13])


def test_bollinger_bands_collapse_on_a_flat_series():
    result = add_technical_indicators(frame([100] * 40)).iloc[-1]

    assert result["bb_upper"] == result["bb_middle"] == result["bb_lower"] == 100


def test_bollinger_bands_use_two_population_standard_deviations():
    close = [99, 101] * 20  # 20-candle window: 10 x 99 and 10 x 101

    result = add_technical_indicators(frame(close)).iloc[-1]

    assert result["bb_middle"] == pytest.approx(100)
    assert result["bb_upper"] == pytest.approx(102)  # std is exactly 1
    assert result["bb_lower"] == pytest.approx(98)


def test_bollinger_needs_a_full_window():
    bands = add_technical_indicators(frame([100] * 25))["bb_upper"]

    assert bands.iloc[:19].isna().all()
    assert not np.isnan(bands.iloc[19])
