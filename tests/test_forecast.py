import json
from datetime import datetime, timezone

import pytest

from financial_assistant.forecast import (
    DEFAULT_HORIZON_CANDLES,
    append_record,
    build_forecast_record,
    horizon_for,
    load_records,
    parse_forecast,
    rule_based_bias,
    unknown_forecast,
)

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


def test_parse_a_valid_line_and_remove_it_from_the_text():
    text = "Trend is up.\nMomentum agrees.\nFORECAST: bias=bullish; confidence=medium"

    forecast, cleaned = parse_forecast(text)

    assert forecast == {"bias": "bullish", "confidence": "medium"}
    assert cleaned == "Trend is up.\nMomentum agrees."
    assert "FORECAST" not in cleaned


@pytest.mark.parametrize(
    "line",
    [
        "forecast: bias=Bearish; confidence=HIGH",
        "FORECAST:bias=bearish;confidence=high",
        "FORECAST : bias = bearish , confidence = high",
        "**FORECAST: bias=bearish; confidence=high**",
        "> FORECAST: bias=bearish; confidence=high.",
        "`FORECAST: bias=bearish; confidence=high`",
    ],
)
def test_parse_tolerates_case_spacing_and_markdown(line):
    forecast, _ = parse_forecast(f"text\n{line}")

    assert forecast == {"bias": "bearish", "confidence": "high"}


@pytest.mark.parametrize(
    "text",
    [
        "No forecast here.",
        "",
        "FORECAST: bias=sideways; confidence=high",
        "FORECAST: bias=bullish; confidence=certain",
        "FORECAST: bias=bullish",
        "My FORECAST: bias=bullish; confidence=high is inline, not a line",
    ],
)
def test_parse_rejects_missing_or_invalid_lines(text):
    forecast, cleaned = parse_forecast(text)

    assert forecast is None
    assert cleaned == text


def test_the_last_forecast_line_wins():
    text = (
        "FORECAST: bias=bullish; confidence=low\n"
        "Second thoughts.\n"
        "FORECAST: bias=bearish; confidence=high"
    )

    forecast, cleaned = parse_forecast(text)

    assert forecast == {"bias": "bearish", "confidence": "high"}
    assert "FORECAST" not in cleaned


def test_horizons():
    assert DEFAULT_HORIZON_CANDLES == {"1h": 24, "4h": 6, "1day": 5}
    assert horizon_for("4h") == 6
    assert unknown_forecast("1day") == {
        "bias": "unknown",
        "confidence": "unknown",
        "horizon_candles": 5,
    }


def summary(trend="bullish", momentum="bullish", alignment=None, stale=False):
    data = {
        "symbol": "XAU/USD",
        "interval": "1h",
        "latest_candle": {"datetime": "2026-10-23T10:00:00+00:00", "close": 4400.5},
        "market_conditions": {"trend": trend, "macd_momentum": momentum},
        "volatility": {"atr_percent": 0.2},
        "data_quality": {"is_stale": stale},
    }
    if alignment:
        data["multi_timeframe"] = {"alignment": alignment}
    return data


@pytest.mark.parametrize(
    "trend, momentum, alignment, expected",
    [
        ("bullish", "bullish", "aligned_bullish", ("bullish", "high")),
        ("bullish", "bullish", "mixed", ("bullish", "medium")),
        ("bullish", "bullish", "aligned_bearish", ("bullish", "medium")),
        ("bullish", "bullish", None, ("bullish", "medium")),
        ("bearish", "bearish", "aligned_bearish", ("bearish", "high")),
        ("bearish", "bearish", "single_timeframe", ("bearish", "medium")),
        ("bullish", "bearish", "aligned_bullish", ("neutral", "low")),
        ("bearish", "bullish", None, ("neutral", "low")),
        ("neutral", "neutral", None, ("neutral", "low")),
        ("neutral", "bullish", None, ("neutral", "low")),
    ],
)
def test_rule_based_bias(trend, momentum, alignment, expected):
    result = rule_based_bias(summary(trend, momentum, alignment))

    assert (result["bias"], result["confidence"]) == expected


def result(bias="bullish", **summary_kwargs):
    return {
        "forecast": {"bias": bias, "confidence": "medium", "horizon_candles": 24},
        "market_summary": summary(**summary_kwargs),
        "review": {"verdict": "PASS"},
        "revision_count": 1,
    }


def test_build_forecast_record():
    record = build_forecast_record(
        result(alignment="aligned_bullish"), now=NOW
    )

    assert record == {
        "created_at": "2026-10-23T12:00:00+00:00",
        "symbol": "XAU/USD",
        "interval": "1h",
        "bias": "bullish",
        "confidence": "medium",
        "horizon_candles": 24,
        "rule_bias": "bullish",
        "rule_confidence": "high",
        "trend": "bullish",
        "alignment": "aligned_bullish",
        "reference_candle_start": "2026-10-23T10:00:00+00:00",
        "reference_price": 4400.5,
        "atr_percent": 0.2,
        "is_stale": False,
        "review_verdict": "PASS",
        "revisions": 1,
    }


def test_neutral_forecasts_are_recorded_too():
    assert build_forecast_record(result("neutral"), now=NOW)["bias"] == "neutral"


def test_stale_flag_is_recorded():
    assert build_forecast_record(result(stale=True), now=NOW)["is_stale"] is True


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"forecast": {"bias": "unknown"}, "market_summary": summary()},
        {"forecast": {"bias": "bullish", "confidence": "low"}},
        {"market_summary": summary()},
    ],
)
def test_nothing_to_record_returns_none(bad):
    assert build_forecast_record(bad) is None


def test_log_round_trip_appends_and_creates_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "forecasts.jsonl"
    first = build_forecast_record(result("bullish"), now=NOW)
    second = build_forecast_record(result("bearish"), now=NOW)

    append_record(path, first)
    append_record(path, second)

    records, skipped = load_records(path)
    assert records == [first, second]
    assert skipped == 0
    assert path.read_text().count("\n") == 2  # one JSON object per line


def test_unreadable_lines_are_skipped_and_counted(tmp_path):
    path = tmp_path / "log.jsonl"
    good = build_forecast_record(result(), now=NOW)
    path.write_text(
        json.dumps(good) + "\n"
        "{not json\n"
        "\n"
        '{"symbol": "XAU/USD"}\n'  # valid JSON, missing fields
        "[1, 2]\n"
        + json.dumps(good) + "\n",
        encoding="utf-8",
    )

    records, skipped = load_records(path)

    assert len(records) == 2
    assert skipped == 3
