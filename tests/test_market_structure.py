import numpy as np
import pandas as pd
import pytest

from financial_assistant.tools import market_structure as ms

# close zigzag with two swing highs (10 at i=3, 12 at i=10) and one swing
# low (5 at i=6); high = close + 0.5, low = close - 0.5
ZIGZAG = [5, 6, 7, 10, 7, 6, 5, 6, 7, 8, 12, 8, 7, 6, 5]


def frame(close, spread=0.5):
    close = pd.Series(close, dtype=float)

    return pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2026-01-01", periods=len(close), freq="h", tz="UTC"
            ),
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
        }
    )


def test_swing_points_are_found_and_unconfirmed_edges_ignored():
    highs, lows = ms.find_swing_points(frame(ZIGZAG), window=3)

    assert [(p["index"], p["price"]) for p in highs] == [(3, 10.5), (10, 12.5)]
    assert [(p["index"], p["price"]) for p in lows] == [(6, 4.5)]


def test_a_flat_top_counts_once():
    df = frame([1, 3, 3, 1], spread=0)

    highs, _ = ms.find_swing_points(df, window=1)

    assert [p["index"] for p in highs] == [1]


def test_a_straight_trend_has_no_swing_points():
    highs, lows = ms.find_swing_points(frame(range(1, 40)))

    assert highs == [] and lows == []


def test_key_levels_pick_the_nearest_support_and_resistance():
    levels = ms.key_levels(frame(ZIGZAG), atr=0.5)

    assert levels["resistance"]["price"] == 10.5  # nearest swing high above 5
    assert levels["resistance"]["distance"] == pytest.approx(5.5)
    assert levels["resistance"]["distance_atr"] == pytest.approx(11.0)
    assert levels["support"]["price"] == 4.5
    assert levels["support"]["distance_atr"] == pytest.approx(1.0)
    assert levels["lookback_high"] == 12.5
    assert levels["lookback_low"] == 4.5
    assert levels["lookback_candles"] == len(ZIGZAG)


def test_no_resistance_when_price_is_above_every_swing_high():
    levels = ms.key_levels(frame([*ZIGZAG, 20]), atr=0.5)

    assert levels["resistance"] is None
    assert levels["support"]["price"] == 4.5


def test_no_support_when_price_is_below_every_swing_low():
    levels = ms.key_levels(frame([*ZIGZAG, 1]), atr=0.5)

    assert levels["support"] is None
    assert levels["resistance"]["price"] == 10.5


def test_zero_atr_gives_no_distance_in_atr_units():
    levels = ms.key_levels(frame(ZIGZAG), atr=0.0)

    assert levels["support"]["distance_atr"] is None


def test_key_levels_only_look_at_the_lookback_window():
    # the last 4 candles are 8, 7, 6, 5: the 12 at index 10 is outside
    levels = ms.key_levels(frame(ZIGZAG), atr=0.5, lookback=4)

    assert levels["lookback_candles"] == 4
    assert levels["lookback_high"] == 8.5


def cross_frame(diff):
    return pd.DataFrame(
        {"sma_20": diff, "sma_50": [0.0] * len(diff)}, dtype=float
    )


def test_last_sma_cross_bullish():
    result = ms.last_sma_cross(cross_frame([np.nan, np.nan, -1, -1, 1, 1, 1]))

    assert result == {"direction": "bullish", "candles_ago": 2}


def test_last_sma_cross_reports_the_most_recent_one():
    result = ms.last_sma_cross(cross_frame([-1, 1, -1]))

    assert result == {"direction": "bearish", "candles_ago": 0}


@pytest.mark.parametrize(
    "diff",
    [[1, 1, 1, 1], [-1, -1], [np.nan, np.nan], [1.0], [-1, 0, -1]],
)
def test_no_sma_cross(diff):
    assert ms.last_sma_cross(cross_frame(diff)) is None


def test_an_exact_tie_keeps_the_previous_direction():
    result = ms.last_sma_cross(cross_frame([1, 0, -1]))

    assert result == {"direction": "bearish", "candles_ago": 0}


def test_percent_change():
    df = frame([100, 105, 110])

    assert ms.percent_change(df, 1) == pytest.approx(4.7619, abs=1e-3)
    assert ms.percent_change(df, 2) == pytest.approx(10.0)


def test_percent_change_is_none_without_enough_history_or_a_base():
    assert ms.percent_change(frame([100, 110]), 2) is None
    assert ms.percent_change(frame([100, 110]), 0) is None
    assert ms.percent_change(frame([0, 110]), 1) is None


def test_24h_candle_counts_cover_every_interval():
    assert ms.CANDLES_PER_24H == {"1h": 24, "4h": 6, "1day": 1}


def atr_frame(values):
    return pd.DataFrame({"atr_14": values})


@pytest.mark.parametrize(
    "atr, condition",
    [(4.0, "elevated"), (2.0, "normal"), (1.0, "subdued"), (2.4, "normal")],
)
def test_volatility_condition(atr, condition):
    df = atr_frame([2.0] * 59 + [atr])

    result = ms.describe_volatility(df, atr=atr, close=100.0)

    assert result["condition"] == condition
    assert result["atr_percent"] == pytest.approx(atr)


def test_volatility_is_unknown_without_atr_history():
    result = ms.describe_volatility(atr_frame([np.nan] * 5), 1.0, 100.0)

    assert result["condition"] == "unknown"
    assert result["atr_vs_median"] is None


def bands(close, lower=90.0, middle=100.0, upper=110.0):
    return pd.Series(
        {
            "close": close,
            "bb_lower": lower,
            "bb_middle": middle,
            "bb_upper": upper,
        }
    )


@pytest.mark.parametrize(
    "close, position, percent_b",
    [
        (112, "above_upper_band", 1.1),
        (110, "above_upper_band", 1.0),
        (107, "near_upper_band", 0.85),
        (100, "inside_bands", 0.5),
        (93, "near_lower_band", 0.15),
        (90, "below_lower_band", 0.0),
        (85, "below_lower_band", -0.25),
    ],
)
def test_bollinger_position(close, position, percent_b):
    result = ms.describe_bollinger(bands(close))

    assert result["position"] == position
    assert result["percent_b"] == pytest.approx(percent_b)
    assert result["bandwidth_percent"] == pytest.approx(20.0)


def test_flat_bollinger_bands_are_undefined_not_a_division_error():
    result = ms.describe_bollinger(bands(100, 100, 100, 100))

    assert result["position"] == "undefined"
    assert result["percent_b"] is None
