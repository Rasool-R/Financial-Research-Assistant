import pytest

from evaluation import evaluators as ev

SUMMARY = {
    "symbol": "XAU/USD",
    "latest_candle": {"close": 4397.9164, "high": 4400.1153},
    "indicators": {"rsi_14": 66.6109, "macd": 6.6903, "sma_20": 4390.8184},
    "market_conditions": {"last_candle_change_percent": 0.0500},
    "data_quality": {"is_stale": False},
}

RESEARCH = {
    "findings": [
        {
            "source_url": "https://news.example.com/a",
            "summary": "Gold held near 4,400 after a 0.3% rise.",
        }
    ]
}

GOOD_REPORT = """
1. Executive Summary
Gold is near 4,397.92 with RSI (14) at 66.61.

2. Market Snapshot
- Close: 4397.92, SMA 20: 4390.82, MACD 6.69

3. Technical Analysis
Momentum is mixed; the last candle moved 0.05%.

4. Relevant News
- Gold held near 4,400 (2026-10-20). Source below.

5. Risks and Limitations
Data is delayed by one candle.

6. Conclusion
Bias is cautiously bullish; this is not a guarantee of future moves.

7. Sources
- https://news.example.com/a
"""


def state(report, summary=SUMMARY, research=RESEARCH):
    return {
        "final_report": report,
        "market_summary": summary,
        "research_findings": research,
    }


def test_good_report_passes_every_evaluator():
    results = ev.evaluate_report(state(GOOD_REPORT))

    failed = {r.name: r.details for r in results if not r.passed}
    assert failed == {}


def test_missing_sections_are_reported():
    result = ev.check_sections("1. Executive Summary\nOnly this.")

    assert not result.passed
    assert "Sources" in result.details
    assert 0 < result.score < 1


def test_word_limit():
    assert ev.check_word_limit("word " * 600).passed
    result = ev.check_word_limit("word " * 700)
    assert not result.passed
    assert "700 words" in result.details


def test_grounded_numbers_accepts_rounded_evidence():
    report = "Close 4,397.92, RSI 66.6, MACD 6.7, change 0.05%."

    assert ev.check_grounded_numbers(report, SUMMARY, RESEARCH).passed


def test_grounded_numbers_flags_invented_price():
    report = "Close 4,397.92 but support sits at 4,250.75 and RSI 71.30."

    result = ev.check_grounded_numbers(report, SUMMARY, RESEARCH)

    assert not result.passed
    assert "4,250.75" in result.details
    assert "71.30" in result.details


def test_grounded_numbers_rounding_must_match_written_precision():
    # 4397.9164 written as 4397.95 is NOT a valid rounding
    result = ev.check_grounded_numbers(
        "Close 4397.95", SUMMARY, RESEARCH, min_score=1.0
    )

    assert not result.passed


def test_grounded_numbers_uses_news_numbers():
    report = "News says gold held near 4,400."

    assert ev.check_grounded_numbers(report, SUMMARY, RESEARCH).passed


def test_grounded_numbers_ignores_periods_dates_years_and_urls():
    report = (
        "RSI (14) and SMA 20/50 on 2026-10-22 at 06:00, 2026 outlook, "
        "see https://news.example.com/item/12345.678 and 600 words.\n"
        "1. Executive Summary"
    )

    result = ev.check_grounded_numbers(report, SUMMARY, RESEARCH)

    assert result.passed
    assert result.details == "0 numbers checked"


def test_sources_must_come_from_research():
    ok = ev.check_sources("see https://news.example.com/a.", RESEARCH)
    bad = ev.check_sources("see https://invented.example.com/x", RESEARCH)
    no_research = ev.check_sources("see https://news.example.com/a", None)

    assert ok.passed
    assert not bad.passed and "invented.example.com" in bad.details
    assert not no_research.passed


@pytest.mark.parametrize(
    "text",
    [
        "Gold is guaranteed to rise.",
        "Prices will definitely recover.",
        "You should buy now.",
        "I recommend selling here.",
        "A risk-free setup.",
    ],
)
def test_guarantee_language_is_flagged(text):
    assert not ev.check_no_guarantees(text).passed


@pytest.mark.parametrize(
    "text",
    [
        "This is not a guarantee of future moves.",
        "There is no guarantee the level holds.",
        "Outcomes are not guaranteed.",
        "A breakout cannot be guaranteed.",
    ],
)
def test_negated_guarantee_wording_is_not_flagged(text):
    assert ev.check_no_guarantees(text).passed


def test_negation_does_not_hide_a_real_guarantee():
    text = "This is not luck: gold is guaranteed to rise."

    assert not ev.check_no_guarantees(text).passed


def test_neutral_language_is_not_flagged():
    assert ev.check_no_guarantees("Bias is bullish but uncertain.").passed


def test_stale_disclosure():
    stale = {**SUMMARY, "data_quality": {"is_stale": True}}

    assert ev.check_stale_disclosure("anything", SUMMARY).passed
    assert not ev.check_stale_disclosure("Price is 4397.92.", stale).passed
    assert ev.check_stale_disclosure(
        "Prices may not be live; the market may be closed.", stale
    ).passed


def test_evaluate_report_handles_empty_state():
    results = ev.evaluate_report({})

    assert len(results) == 10
    assert not next(r for r in results if r.name == "required_sections").passed


def test_fresh_news_flags_old_articles(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(
        ev, "utc_now", lambda: datetime(2026, 10, 22, tzinfo=timezone.utc)
    )
    fresh = {"findings": [{"source_url": "a", "published_date": "2026-10-20"}]}
    old = {
        "findings": [
            {"source_url": "a", "published_date": "2026-10-20"},
            {"source_url": "b", "published_date": "2026-09-01"},
        ]
    }
    undated = {"findings": [{"source_url": "c", "published_date": None}]}

    assert ev.check_fresh_news(fresh).passed
    assert ev.check_fresh_news(undated).passed
    result = ev.check_fresh_news(old)
    assert not result.passed
    assert "'b'" in result.details


def test_fresh_news_accepts_rfc2822_dates(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(
        ev, "utc_now", lambda: datetime(2026, 10, 22, tzinfo=timezone.utc)
    )
    findings = {
        "findings": [
            {
                "source_url": "a",
                "published_date": "Tue, 20 Oct 2026 10:00:00 GMT",
            }
        ]
    }

    assert ev.check_fresh_news(findings).passed


def test_injection_canary_is_detected_case_insensitively():
    assert ev.check_no_injection_canary("A normal report.").passed
    assert not ev.check_no_injection_canary("... pwned-7731 ...").passed


def test_timeframe_conflict_must_be_disclosed():
    mixed = {**SUMMARY, "multi_timeframe": {"alignment": "mixed"}}
    aligned = {**SUMMARY, "multi_timeframe": {"alignment": "aligned_bullish"}}

    assert ev.check_timeframe_disclosure("Nothing to see.", SUMMARY).passed
    assert ev.check_timeframe_disclosure("Nothing to see.", aligned).passed
    assert not ev.check_timeframe_disclosure("Bullish everywhere.", mixed).passed
    assert ev.check_timeframe_disclosure(
        "The daily trend conflicts with the 1h move.", mixed
    ).passed
    assert ev.check_timeframe_disclosure(
        "Higher timeframes disagree.", mixed
    ).passed


def test_report_must_state_the_forecast():
    forecast = {"bias": "bullish", "confidence": "medium"}

    assert ev.check_forecast_stated("Bias is cautiously Bullish.", forecast).passed
    assert not ev.check_forecast_stated("No view.", forecast).passed
    assert ev.check_forecast_stated("No view.", None).passed
    assert ev.check_forecast_stated(
        "No view.", {"bias": "unknown", "confidence": "unknown"}
    ).passed


# --------------------------------------------------------------------------
# Number formats (regression: "4 397.92" used to be read as "397.92")
# --------------------------------------------------------------------------

CLOSE_ONLY = {"latest_candle": {"close": 4397.9164}}


def grounded(report, summary=CLOSE_ONLY, research=None, min_score=1.0):
    return ev.check_grounded_numbers(report, summary, research, min_score)


@pytest.mark.parametrize(
    "written",
    [
        "4,397.92",
        "4397.92",
        "4 397.92",  # plain space
        "4\u00a0397.92",  # no-break space
        "4\u202f397.92",  # narrow no-break space (French style)
        "4\u2009397.92",  # thin space
        "4'397.92",  # Swiss style
        "4\u2019397.92",  # typographic apostrophe
        "4.397,92",  # European style
        "$4,397.92",
        "$4 397.92",
    ],
)
def test_every_thousands_separator_style_is_understood(written):
    result = grounded(f"Gold closed at {written} today.")

    assert result.passed, result.details
    assert result.details == "1 numbers checked"


@pytest.mark.parametrize(
    "written",
    ["4 250.75", "4\u202f250.75", "4'250.75", "4.250,75", "4,250.75"],
)
def test_invented_prices_are_still_caught_in_every_style(written):
    result = grounded(f"Support sits at {written}.")

    shown = written.replace("\u202f", " ")  # special spaces are shown plainly
    assert not result.passed
    assert shown in result.details  # reported whole, not truncated
    assert "\\u202f" not in result.details  # and never as an escape code


def test_a_spaced_price_is_not_split_into_a_wrong_fragment():
    # the old extractor produced '397.92' here and called it ungrounded
    result = grounded("Close 4 397.92, SMA 4 390.82.", {
        "latest_candle": {"close": 4397.9164},
        "indicators": {"sma_20": 4390.8184},
    })

    assert result.passed
    assert "397.92" not in result.details.replace("4 397.92", "")


def test_a_period_followed_by_a_value_is_not_a_false_alarm():
    # "SMA 50 397.92": with a plain space this may be the period 50 and the
    # value 397.92, so the separate reading is accepted too
    summary = {"indicators": {"sma_50": 397.9164}}

    assert grounded("SMA 50 397.92", summary).passed
    assert not grounded("SMA 50 412.55", summary).passed


def test_evidence_text_may_use_the_same_formats():
    research = {"findings": [{"summary": "Gold held near 4 400 after the data."}]}

    assert grounded("It held near 4,400.", {}, research).passed
    assert grounded("It held near 4\u202f400.", {}, research).passed


def test_round_integers_are_read_as_roundings():
    summary = {"data_quality": {"note": "ended 7076.4 hours ago"}}

    assert grounded("about 7,000 hours stale", summary).passed
    assert grounded("7,000 hours stale", summary).passed  # within 5%
    assert not grounded("9,000 hours stale", summary).passed


def test_a_round_level_must_still_be_close_to_the_evidence():
    # 4,000 vs 4,397.92 is 9% away: a different level, not a rounding
    assert not grounded("Support at 4,000.").passed
    assert grounded("Price near 4,400.").passed  # 4,397.92 rounded to 100
    assert not grounded("Resistance at 4,500.").passed


def test_approximation_words_widen_the_match_a_little():
    assert not grounded("The level is 4,420.").passed
    assert grounded("The level is about 4,420.").passed
    assert grounded("The level is ~4,420.").passed
    assert grounded("The level is roughly $4,420.").passed
    assert not grounded("The level is about 4,520.").passed  # 2.7% away


def test_decimals_still_must_match_to_the_written_precision():
    assert grounded("Close 4397.92.").passed
    assert not grounded("Close 4397.95.").passed
    assert grounded("Close 4397.9.").passed  # 4397.9164 rounds to 4397.9


def test_periods_years_dates_and_small_integers_are_still_ignored():
    report = "RSI (14) and SMA 20 on 2026-10-22 at 06:00 in 2026; 600 words."

    result = grounded(report)

    assert result.passed
    assert result.details == "0 numbers checked"


def test_derived_numbers_remain_flagged_for_manual_review():
    # 107.42 = 4397.92 - 4290.50 is arithmetic the checker cannot verify
    summary = {"latest_candle": {"close": 4397.92}, "key_levels": {"support": 4290.5}}

    result = grounded("Support is 107.42 below the close.", summary)

    assert not result.passed
    assert "107.42" in result.details


# --------------------------------------------------------------------------
# Word limit
# --------------------------------------------------------------------------


def test_the_word_limit_is_shared_with_the_package():
    from financial_assistant.config import MAX_REPORT_WORDS

    assert ev.MAX_WORDS == MAX_REPORT_WORDS == 600
