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

    assert len(results) == 6
    assert not next(r for r in results if r.name == "required_sections").passed
