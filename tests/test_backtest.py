from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from financial_assistant import backtest as bt

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hits, n, expected",
    [
        (8, 10, 0.109375),  # 2 * (1 + 10 + 45) / 1024
        (5, 10, 1.0),
        (10, 10, 2 / 1024),
        (0, 10, 2 / 1024),
        (15, 20, 0.04138946533203125),
    ],
)
def test_exact_binomial_p_value(hits, n, expected):
    assert bt.binomial_two_sided_p(hits, n) == pytest.approx(expected)


def test_binomial_p_value_handles_large_n_without_underflow():
    p_value = bt.binomial_two_sided_p(2600, 5000)

    assert 0 < p_value < 0.05
    assert bt.binomial_two_sided_p(0, 0) is None


def test_wilson_interval_matches_the_textbook_value():
    low, high = bt.wilson_interval(8, 10)

    assert (round(low, 3), round(high, 3)) == (0.49, 0.943)
    assert bt.wilson_interval(0, 0) == (None, None)
    assert bt.wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_forward_return_percent():
    closes = np.array([100.0, 110.0, 121.0])

    assert bt.forward_return_percent(closes, 0, 1) == pytest.approx(10.0)
    assert bt.forward_return_percent(closes, 0, 2) == pytest.approx(21.0)
    assert bt.forward_return_percent(closes, 1, 2) is None  # not closed yet
    assert bt.forward_return_percent(closes, -1, 1) is None
    assert bt.forward_return_percent(np.array([0.0, 1.0]), 0, 1) is None


# --------------------------------------------------------------------------
# Scoring samples
# --------------------------------------------------------------------------


def sample(bias, r, index=None, **extra):
    data = {"bias": bias, "forward_return_percent": r, "symbol": "X", "interval": "1h"}
    if index is not None:
        data["reference_index"] = index
    data.update(extra)
    return data


def spaced(calls, horizon=24):
    """Samples whose horizons never overlap."""

    return [
        sample(bias, r, index=i * horizon, **extra)
        for i, (bias, r, extra) in enumerate(calls)
    ]


def test_hit_rate_counts_only_directional_calls():
    samples = [
        sample("bullish", 1.0),
        sample("bullish", -1.0),
        sample("bearish", -2.0),
        sample("neutral", 5.0),
    ]

    report = bt.evaluate_samples(samples, horizon=24)
    stats = report["methods"]["system"]

    assert (stats["n"], stats["hits"]) == (3, 2)
    assert stats["hit_rate"] == pytest.approx(2 / 3)
    assert report["samples"] == 4
    assert report["neutral"]["calls"] == 1


def test_a_flat_outcome_is_not_a_hit():
    report = bt.evaluate_samples(
        [sample("bullish", 0.0), sample("bearish", 0.0)], horizon=24
    )

    assert report["methods"]["system"]["hits"] == 0


def test_mean_follow_return_flips_the_sign_for_bearish_calls():
    samples = [sample("bullish", 2.0), sample("bearish", -4.0)]

    stats = bt.evaluate_samples(samples, horizon=24)["methods"]["system"]

    assert stats["mean_follow_return_percent"] == pytest.approx(3.0)  # (2 + 4) / 2


def test_samples_without_an_outcome_are_ignored():
    report = bt.evaluate_samples(
        [sample("bullish", None), sample("bullish", 1.0)], horizon=24
    )

    assert report["samples"] == 1


def test_always_bullish_baseline_uses_the_same_dates_as_the_calls():
    samples = [
        sample("bearish", -1.0),  # market fell
        sample("bearish", 2.0),   # market rose
        sample("bullish", 1.0),
        sample("neutral", 9.0),   # not a directional call: not in the baseline
    ]

    methods = bt.evaluate_samples(samples, horizon=24)["methods"]

    assert methods["always_bullish"]["n"] == 3
    assert methods["always_bullish"]["hits"] == 2  # r > 0 on two of the three


def test_rule_based_and_trend_baselines_are_scored_when_present():
    samples = [
        sample("bullish", 1.0, rule_bias="bullish", trend="bullish"),
        sample("bullish", -1.0, rule_bias="bearish", trend="neutral"),
        sample("bearish", -1.0, rule_bias="neutral", trend="bearish"),
    ]

    methods = bt.evaluate_samples(samples, horizon=24)["methods"]

    assert (methods["rule_based"]["n"], methods["rule_based"]["hits"]) == (2, 2)
    assert (methods["trend_only"]["n"], methods["trend_only"]["hits"]) == (2, 2)


def test_baselines_are_omitted_when_the_data_is_missing():
    methods = bt.evaluate_samples([sample("bullish", 1.0)], horizon=24)["methods"]

    assert set(methods) == {"system", "always_bullish"}


def test_neutral_calls_are_judged_against_the_expected_move():
    # expected move = 1.0% ATR * sqrt(16) = 4%, band = half of it = 2%
    samples = [
        sample("neutral", 1.5, atr_percent=1.0),
        sample("neutral", -1.9, atr_percent=1.0),
        sample("neutral", 2.5, atr_percent=1.0),
        sample("neutral", 0.1),  # no ATR: cannot be scored
    ]

    neutral = bt.evaluate_samples(samples, horizon=16)["neutral"]

    assert neutral == {
        "calls": 4,
        "scored": 3,
        "correct": 2,
        "accuracy": pytest.approx(2 / 3),
    }


def test_hit_rate_by_confidence():
    samples = [
        sample("bullish", 1.0, confidence="high"),
        sample("bullish", 1.0, confidence="high"),
        sample("bullish", -1.0, confidence="low"),
    ]

    by_confidence = bt.evaluate_samples(samples, horizon=24)["by_confidence"]

    assert by_confidence["high"] == {"n": 2, "hits": 2, "hit_rate": 1.0}
    assert by_confidence["low"]["hit_rate"] == 0.0
    assert by_confidence["medium"]["n"] == 0


def test_empty_input_is_handled():
    report = bt.evaluate_samples([], horizon=24)

    assert report["methods"]["system"]["hit_rate"] is None
    assert report["conclusion"] == "insufficient_data"
    assert "Only 0 directional forecasts" in report["warnings"][0]


# --------------------------------------------------------------------------
# Conclusions, overlap and warnings
# --------------------------------------------------------------------------


def calls(hits, total, bias="bullish"):
    """``total`` independent bullish calls of which ``hits`` were right."""

    return spaced(
        [(bias, 1.0 if i < hits else -1.0, {}) for i in range(total)]
    )


@pytest.mark.parametrize(
    "hits, total, conclusion",
    [
        (45, 60, "better_than_chance"),
        (30, 60, "not_distinguishable_from_chance"),
        (12, 60, "worse_than_chance"),
        (9, 10, "insufficient_data"),  # looks great but n is far too small
    ],
)
def test_conclusion(hits, total, conclusion):
    report = bt.evaluate_samples(calls(hits, total), horizon=24)

    assert report["conclusion"] == conclusion


def test_a_lucky_small_sample_is_never_called_better_than_chance():
    report = bt.evaluate_samples(calls(10, 10), horizon=24)

    assert report["methods"]["system"]["hit_rate"] == 1.0
    assert report["conclusion"] == "insufficient_data"
    assert any("at least 30" in w for w in report["warnings"])


def test_overlapping_forecasts_count_once():
    # 60 hourly forecasts on one series, each judged over 24 candles
    samples = [sample("bullish", 1.0, index=i) for i in range(60)]

    assert bt.count_non_overlapping(samples, horizon=24) == 3  # 0, 24, 48


def test_overlap_is_counted_per_series():
    samples = [
        sample("bullish", 1.0, index=0, symbol="A"),
        sample("bullish", 1.0, index=1, symbol="B"),
        sample("bullish", 1.0, index=2, symbol="A"),
    ]

    assert bt.count_non_overlapping(samples, horizon=24) == 2


def test_overlapping_samples_block_a_conclusion_and_warn():
    samples = [sample("bullish", 1.0, index=i) for i in range(60)]

    report = bt.evaluate_samples(samples, horizon=24)

    assert report["directional_calls"] == 60
    assert report["independent_directional_calls"] == 3
    assert report["conclusion"] == "insufficient_data"
    assert any("overlap" in w for w in report["warnings"])
    assert any("too optimistic" in w for w in report["warnings"])


def test_conclusion_uses_only_independent_forecasts():
    """40 blocks of 24 overlapping forecasts; every block shares one outcome.

    Counting all 960 forecasts, 22 winning blocks look significant (55%
    over 960 samples), but there are really only 40 independent outcomes and
    22 of 40 is indistinguishable from a coin flip.
    """

    samples = []
    for block in range(40):
        outcome = 1.0 if block < 22 else -1.0
        for step in range(24):
            samples.append(sample("bullish", outcome, index=block * 24 + step))

    report = bt.evaluate_samples(samples, horizon=24)

    table = report["methods"]["system"]
    assert table["n"] == 960
    assert table["p_value"] < 0.05  # the optimistic number in the table
    assert report["independent"]["n"] == 40
    assert report["independent"]["hits"] == 22
    assert report["independent"]["p_value"] > 0.05
    assert report["conclusion"] == "not_distinguishable_from_chance"
    assert any("uses only the independent" in w for w in report["warnings"])


def test_text_report_shows_the_independent_numbers_when_samples_overlap():
    samples = [sample("bullish", 1.0, index=i) for i in range(240)]

    text = bt.format_report(bt.evaluate_samples(samples, horizon=24), "t")

    assert "Independent forecasts only (n=10)" in text


def test_independent_stats_equal_the_table_when_nothing_overlaps():
    report = bt.evaluate_samples(calls(45, 60), horizon=24)

    assert report["independent"]["n"] == report["methods"]["system"]["n"]
    assert report["independent"]["p_value"] == pytest.approx(
        report["methods"]["system"]["p_value"]
    )
    assert "Independent forecasts only" not in bt.format_report(report, "t")


def test_warns_when_the_calls_do_not_beat_always_up():
    # 40 bullish calls, 30 right; the market rose in 30 of those 40 as well
    report = bt.evaluate_samples(calls(30, 40), horizon=24)

    assert any("did not beat simply predicting 'up'" in w for w in report["warnings"])


def test_no_beat_warning_when_the_calls_clearly_beat_the_drift():
    samples = spaced(
        [("bearish", -1.0, {}) for _ in range(35)]
        + [("bullish", 1.0, {}) for _ in range(5)]
    )  # 100% hits; always-up would be right only 5/40 times

    report = bt.evaluate_samples(samples, horizon=24)

    assert not any("did not beat" in w for w in report["warnings"])


def test_format_report_contains_the_key_facts():
    report = bt.evaluate_samples(calls(45, 60), horizon=24)

    text = bt.format_report(report, "Live forecasts, 1h")

    assert "== Live forecasts, 1h ==" in text
    assert "next 24 candles" in text
    assert "75.0%" in text
    assert "always_bullish" in text
    assert "significantly above 50%" in text


def test_format_report_for_insufficient_data():
    text = bt.format_report(bt.evaluate_samples([], horizon=24), "t")

    assert "INSUFFICIENT DATA" in text
    assert "!" in text  # the warning is printed


# --------------------------------------------------------------------------
# Resolving logged forecasts
# --------------------------------------------------------------------------


def candles(n=200, start="2026-10-01 00:00", interval="h"):
    """close = 100, 101, 102, ..."""

    close = pd.Series([100.0 + i for i in range(n)])

    return pd.DataFrame(
        {
            "datetime": pd.date_range(start, periods=n, freq=interval, tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    )


def record(index, bias="bullish", interval="1h", **extra):
    start = candles().loc[index, "datetime"].isoformat()
    data = {
        "symbol": "XAU/USD",
        "interval": interval,
        "bias": bias,
        "confidence": "medium",
        "rule_bias": "bullish",
        "trend": "bullish",
        "atr_percent": 0.5,
        "reference_candle_start": start,
        "is_stale": False,
    }
    data.update(extra)
    return data


class Fetch:
    def __init__(self, frame=None):
        self.frame = candles() if frame is None else frame
        self.requests = []

    def __call__(self, symbol, interval, outputsize):
        self.requests.append((symbol, interval, outputsize))
        return self.frame


def test_a_forecast_is_scored_against_the_candle_horizon_candles_later():
    fetch = Fetch()

    result = bt.resolve_forecasts(
        [record(10)], fetch, horizon=5, now=NOW
    )

    report = result["intervals"]["1h"]
    assert report["samples"] == 1
    assert report["horizon_candles"] == 5
    stats = report["methods"]["system"]
    assert (stats["n"], stats["hits"]) == (1, 1)
    # close[10] = 110, close[15] = 115
    assert stats["mean_follow_return_percent"] == pytest.approx(5 / 110 * 100)


def test_a_wrong_call_is_a_miss():
    result = bt.resolve_forecasts(
        [record(10, bias="bearish")], Fetch(), horizon=5, now=NOW
    )

    assert result["intervals"]["1h"]["methods"]["system"]["hits"] == 0


def test_forecasts_whose_horizon_has_not_passed_are_pending():
    result = bt.resolve_forecasts(
        [record(10), record(198)], Fetch(), horizon=5, now=NOW
    )

    report = result["intervals"]["1h"]
    assert report["samples"] == 1
    assert report["pending"] == 1


def test_forecasts_missing_from_the_downloaded_candles_are_unmatched():
    old = record(10, reference_candle_start="2020-01-01T00:00:00+00:00")

    report = bt.resolve_forecasts([old], Fetch(), horizon=5, now=NOW)[
        "intervals"
    ]["1h"]

    assert report["unmatched"] == 1
    assert report["samples"] == 0


def test_duplicate_forecasts_for_the_same_candle_count_once():
    result = bt.resolve_forecasts(
        [record(10), record(10), record(10, bias="bearish")],
        Fetch(),
        horizon=5,
        now=NOW,
    )

    assert result["meta"]["duplicates_ignored"] == 2
    assert result["intervals"]["1h"]["samples"] == 1


def test_stale_forecasts_are_excluded_unless_requested():
    stale = record(10, is_stale=True)

    excluded = bt.resolve_forecasts([stale], Fetch(), horizon=5, now=NOW)
    included = bt.resolve_forecasts(
        [stale], Fetch(), horizon=5, include_stale=True, now=NOW
    )

    assert excluded["meta"]["stale_excluded"] == 1
    assert excluded["intervals"] == {}
    assert included["intervals"]["1h"]["samples"] == 1


def test_runs_without_a_direction_are_not_scored():
    result = bt.resolve_forecasts(
        [record(10, bias="unknown")], Fetch(), horizon=5, now=NOW
    )

    assert result["meta"]["without_directional_forecast"] == 1
    assert result["intervals"] == {}


def test_each_interval_uses_its_own_default_horizon():
    frames = {"1h": candles(), "4h": candles(interval="4h")}

    def fetch(symbol, interval, outputsize):
        return frames[interval]

    four_hour = record(10, interval="4h")
    four_hour["reference_candle_start"] = (
        frames["4h"].loc[10, "datetime"].isoformat()
    )

    result = bt.resolve_forecasts([record(10), four_hour], fetch, now=NOW)

    assert result["intervals"]["1h"]["horizon_candles"] == 24
    assert result["intervals"]["4h"]["horizon_candles"] == 6


def test_enough_history_is_requested_to_reach_the_oldest_forecast():
    fetch = Fetch()
    oldest = record(0)  # 2026-10-01, about 22 days before NOW

    bt.resolve_forecasts([oldest, record(50)], fetch, horizon=24, now=NOW)

    (_, _, outputsize), = fetch.requests
    assert outputsize >= 22 * 24 + 24  # age in hourly candles + horizon
    assert outputsize <= bt.MAX_FETCH_CANDLES


def test_history_request_is_clamped_to_the_api_limit():
    fetch = Fetch()
    ancient = record(0, reference_candle_start="2019-01-01T00:00:00+00:00")

    bt.resolve_forecasts([ancient], fetch, horizon=24, now=NOW)

    assert fetch.requests[0][2] == bt.MAX_FETCH_CANDLES


def test_naive_timestamps_in_the_log_are_treated_as_utc():
    naive = record(10, reference_candle_start="2026-10-01T10:00:00")

    report = bt.resolve_forecasts([naive], Fetch(), horizon=5, now=NOW)[
        "intervals"
    ]["1h"]

    assert report["samples"] == 1


# --------------------------------------------------------------------------
# Replay of the rule-based bias
# --------------------------------------------------------------------------


def walk(n, seed, drift=0.0, noise=4.0):
    rng = np.random.default_rng(seed)
    close = 4000 + np.cumsum(rng.normal(drift, noise, n))

    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close,
        }
    )


def trending(n, seed, run=300, drift=2.0, noise=4.0):
    """Long, persistent up and down trends: a trend-follower has a real edge."""

    rng = np.random.default_rng(seed)
    signs = np.repeat(np.where(np.arange(n // run + 1) % 2 == 0, 1, -1), run)[:n]
    close = 4000 + np.cumsum(signs * drift + rng.normal(0, noise, n))

    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close,
        }
    )


def test_replay_finds_no_edge_in_a_random_walk():
    report = bt.replay(walk(4000, seed=1), "XAU/USD", "1h")

    assert report["directional_calls"] >= bt.MIN_SAMPLE
    assert report["conclusion"] == "not_distinguishable_from_chance"


def test_replay_finds_a_real_edge_when_trends_persist():
    report = bt.replay(trending(5000, seed=1), "XAU/USD", "1h")

    stats = report["methods"]["rule_based"]
    assert report["conclusion"] == "better_than_chance"
    assert stats["hit_rate"] > 0.8
    # the market itself is flat on average, so "always up" is a coin flip
    assert report["methods"]["always_bullish"]["hit_rate"] < 0.6


def test_replay_decisions_never_depend_on_the_future():
    frame = walk(2600, seed=7)

    short = bt.replay_samples(frame.iloc[:1500], "XAU/USD", "1h", 24, 24)
    long = bt.replay_samples(frame, "XAU/USD", "1h", 24, 24)

    assert len(short) > 40
    for early, later in zip(short, long, strict=False):
        assert early["reference_index"] == later["reference_index"]
        for key in ("bias", "confidence", "trend", "atr_percent"):
            assert early[key] == later[key], key


def test_replay_sample_count_and_spacing():
    frame = walk(500, seed=3)

    samples = bt.replay_samples(frame, "XAU/USD", "1h", horizon=24, step=24)

    expected = list(range(bt.MIN_CANDLES - 1, 500 - 24, 24))
    assert [s["reference_index"] for s in samples] == expected
    assert all(s["forward_return_percent"] is not None for s in samples)


def test_replay_outcome_is_read_horizon_candles_later():
    frame = candles(300)  # close = 100 + i

    sample = bt.replay_samples(frame, "XAU/USD", "1h", horizon=10, step=50)[0]

    i = sample["reference_index"]
    assert sample["forward_return_percent"] == pytest.approx(
        ((100 + i + 10) - (100 + i)) / (100 + i) * 100
    )


def test_replay_defaults_step_to_the_horizon_so_samples_do_not_overlap():
    report = bt.replay(walk(1200, seed=2), "XAU/USD", "1h")

    assert report["step_candles"] == report["horizon_candles"] == 24
    assert (
        report["independent_directional_calls"] == report["directional_calls"]
    )


def test_replay_with_a_small_step_reports_that_samples_overlap():
    report = bt.replay(walk(1500, seed=2), "XAU/USD", "1h", step=1)

    assert (
        report["independent_directional_calls"] < report["directional_calls"]
    )
    assert any("too optimistic" in w for w in report["warnings"])


def test_replay_needs_enough_candles():
    report = bt.replay(walk(60, seed=1), "XAU/USD", "1h")

    assert report["samples"] == 0
    assert report["conclusion"] == "insufficient_data"
    assert report["first_reference"] is None


def test_replay_is_labelled_as_rule_based_only():
    report = bt.replay(walk(300, seed=1), "XAU/USD", "4h")

    assert report["mode"] == "replay"
    assert report["primary"] == "rule_based"
    assert report["horizon_candles"] == 6
    assert "No LLM" in report["scope_note"]
    assert "rule_based" in report["methods"]
