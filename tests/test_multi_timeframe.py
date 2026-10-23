import pandas as pd
import pytest

from financial_assistant.tools import market_summary as ms
from tests.test_market_data import make_candles


def down_candles(n=100):
    df = make_candles(n)
    close = pd.Series([300.0 - i for i in range(n)])
    df["open"], df["high"], df["low"], df["close"] = (
        close,
        close + 1,
        close - 1,
        close,
    )
    return df


def install(monkeypatch, by_interval):
    """Patch candle loading; a value may be a frame or an exception."""

    calls = []

    def loader(**kwargs):
        calls.append(kwargs["interval"])
        result = by_interval[kwargs["interval"]]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(ms, "get_market_data", loader)
    return calls


def test_higher_timeframes_by_interval():
    assert ms.HIGHER_TIMEFRAMES == {
        "1h": ["4h", "1day"],
        "4h": ["1day"],
        "1day": [],
    }


@pytest.mark.parametrize(
    "main, higher, expected",
    [
        ("bullish", {"4h": "bullish", "1day": "bullish"}, "aligned_bullish"),
        ("bearish", {"4h": "bearish"}, "aligned_bearish"),
        ("bullish", {"4h": "bullish", "1day": "bearish"}, "mixed"),
        ("bullish", {"4h": "neutral"}, "mixed"),
        ("neutral", {"4h": "neutral"}, "mixed"),
        ("bullish", {}, "single_timeframe"),
    ],
)
def test_timeframe_alignment(main, higher, expected):
    items = {name: {"trend": trend} for name, trend in higher.items()}

    assert ms.timeframe_alignment(main, items) == expected


def test_all_timeframes_agree(monkeypatch):
    up = make_candles(100)
    calls = install(monkeypatch, {"1h": up, "4h": up, "1day": up})

    summary = ms.get_multi_timeframe_summary("XAU/USD", "1h")

    multi = summary["multi_timeframe"]
    assert multi["alignment"] == "aligned_bullish"
    assert multi["conflicting_timeframes"] == []
    assert multi["unavailable"] == []
    assert set(multi["higher_timeframes"]) == {"4h", "1day"}
    assert sorted(calls) == ["1day", "1h", "4h"]


def test_conflicting_timeframe_is_named(monkeypatch):
    install(
        monkeypatch,
        {"1h": make_candles(100), "4h": make_candles(100), "1day": down_candles()},
    )

    multi = ms.get_multi_timeframe_summary("XAU/USD", "1h")["multi_timeframe"]

    assert multi["alignment"] == "mixed"
    assert multi["conflicting_timeframes"] == ["1day"]
    assert multi["higher_timeframes"]["1day"]["trend"] == "bearish"


def test_daily_interval_is_a_single_timeframe(monkeypatch):
    calls = install(monkeypatch, {"1day": make_candles(100)})

    multi = ms.get_multi_timeframe_summary("XAU/USD", "1day")["multi_timeframe"]

    assert multi["alignment"] == "single_timeframe"
    assert multi["higher_timeframes"] == {}
    assert calls == ["1day"]


def test_failing_higher_timeframe_is_recorded_not_raised(monkeypatch):
    up = make_candles(100)
    install(
        monkeypatch,
        {"1h": up, "4h": RuntimeError("rate limited"), "1day": up},
    )

    multi = ms.get_multi_timeframe_summary("XAU/USD", "1h")["multi_timeframe"]

    assert multi["unavailable"] == [
        {"interval": "4h", "error": "rate limited"}
    ]
    assert set(multi["higher_timeframes"]) == {"1day"}
    assert multi["alignment"] == "aligned_bullish"  # judged on what exists


def test_higher_timeframe_with_too_little_history_is_unavailable(monkeypatch):
    up = make_candles(100)
    install(monkeypatch, {"1h": up, "4h": up, "1day": make_candles(30)})

    multi = ms.get_multi_timeframe_summary("XAU/USD", "1h")["multi_timeframe"]

    assert [u["interval"] for u in multi["unavailable"]] == ["1day"]
    assert "Not enough closed candles" in multi["unavailable"][0]["error"]


def test_all_higher_timeframes_failing_still_gives_a_summary(monkeypatch):
    install(
        monkeypatch,
        {
            "1h": make_candles(100),
            "4h": RuntimeError("down"),
            "1day": RuntimeError("down"),
        },
    )

    multi = ms.get_multi_timeframe_summary("XAU/USD", "1h")["multi_timeframe"]

    assert multi["alignment"] == "single_timeframe"
    assert len(multi["unavailable"]) == 2


def test_the_requested_timeframe_must_succeed(monkeypatch):
    up = make_candles(100)
    install(monkeypatch, {"1h": RuntimeError("no data"), "4h": up, "1day": up})

    with pytest.raises(RuntimeError, match="no data"):
        ms.get_multi_timeframe_summary("XAU/USD", "1h")


def test_summary_contains_the_new_sections(monkeypatch):
    install(monkeypatch, {"1h": make_candles(100)})

    summary = ms.get_market_summary("XAU/USD", "1h")

    assert set(summary["volatility"]) == {
        "atr_14", "atr_percent", "atr_vs_median", "condition"
    }
    assert set(summary["bollinger"]) == {
        "upper", "middle", "lower", "percent_b", "bandwidth_percent", "position"
    }
    assert {"support", "resistance", "lookback_high", "lookback_low"} <= set(
        summary["key_levels"]
    )
    assert set(summary["price_action"]) == {
        "change_24h_percent", "last_sma_cross", "close_vs_sma_20_atr"
    }
    # 24 one-hour candles back on a +1 per candle series: 176 -> 199
    assert summary["price_action"]["change_24h_percent"] == pytest.approx(
        (199 - 175) / 175 * 100
    )


def test_summarize_timeframe_is_compact():
    summary = {
        "interval": "4h",
        "market_conditions": {
            "trend": "bullish", "rsi_condition": "neutral",
            "macd_momentum": "bearish",
        },
        "indicators": {"rsi_14": 55.0, "sma_20": 10.0, "sma_50": 9.0},
        "latest_candle": {"close": 10.5},
        "data_quality": {"is_stale": False},
    }

    assert ms.summarize_timeframe(summary) == {
        "interval": "4h", "trend": "bullish", "rsi_condition": "neutral",
        "macd_momentum": "bearish", "rsi_14": 55.0, "close": 10.5,
        "sma_20": 10.0, "sma_50": 9.0, "is_stale": False,
    }
