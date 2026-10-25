import json

from financial_assistant.nodes import reviewer_agent as module
from tests.fakes import FakeModel

MARKET_SUMMARY = {"symbol": "XAU/USD", "interval": "1h"}


def review_json(verdict, critical=(), other=(), recommendations=()):
    return json.dumps(
        {
            "verdict": verdict,
            "critical_issues": list(critical),
            "other_issues": list(other),
            "recommendations": list(recommendations),
        }
    )


def run_reviewer(monkeypatch, replies):
    model = FakeModel(replies)
    monkeypatch.setattr(module, "get_model", lambda: model)

    result = module.reviewer_agent(
        {"final_report": "report text", "market_summary": MARKET_SUMMARY}
    )

    return result, model


def test_reviewer_returns_structured_pass(monkeypatch):
    result, _ = run_reviewer(monkeypatch, [review_json("PASS")])

    assert result["review"]["verdict"] == "PASS"
    assert result["review_feedback"].startswith("VERDICT: PASS")
    assert "errors" not in result


def test_reviewer_forces_revision_when_critical_issues_exist(monkeypatch):
    result, _ = run_reviewer(
        monkeypatch,
        [review_json("PASS", critical=["RSI is wrong"])],
    )

    assert result["review"]["verdict"] == "NEEDS_REVISION"
    assert "- RSI is wrong" in result["review_feedback"]


def test_reviewer_retries_unparseable_output(monkeypatch):
    result, model = run_reviewer(
        monkeypatch, ["not json", review_json("PASS")]
    )

    assert len(model.calls) == 2
    assert result["review"]["verdict"] == "PASS"


def test_reviewer_returns_unreviewed_when_output_stays_invalid(monkeypatch):
    result, _ = run_reviewer(monkeypatch, ["bad", "still bad"])

    assert result["review"]["verdict"] == "UNREVIEWED"
    assert "could not be parsed" in result["review"]["other_issues"][0]
    assert result["errors"][0].startswith("reviewer_agent")


def test_format_review_lists_empty_sections_as_none():
    text = module.format_review(
        {
            "verdict": "PASS",
            "critical_issues": [],
            "other_issues": [],
            "recommendations": [],
        }
    )

    assert text.count("- None") == 3


def test_reviewer_is_told_todays_date_and_that_web_text_is_untrusted(
    monkeypatch,
):
    from datetime import datetime, timezone

    monkeypatch.setattr(
        module,
        "utc_now",
        lambda: datetime(2026, 10, 22, tzinfo=timezone.utc),
    )

    _, model = run_reviewer(monkeypatch, [review_json("PASS")])

    system, human = model.calls[0]
    assert "Today's date (UTC): 2026-10-22" in human.content
    assert "untrusted" in system.content


def test_reviewer_sees_levels_volatility_and_timeframes(monkeypatch):
    summary = {
        **MARKET_SUMMARY,
        "key_levels": {"support": {"price": 4290.5}},
        "volatility": {"atr_14": 12.3},
        "multi_timeframe": {"alignment": "mixed"},
    }
    model = FakeModel([review_json("PASS")])
    monkeypatch.setattr(module, "get_model", lambda: model)

    module.reviewer_agent({"final_report": "report", "market_summary": summary})

    human = model.calls[0][1].content
    assert "4290.5" in human
    assert "atr_14" in human
    assert '"alignment": "mixed"' in human
    assert "multi-timeframe" in model.calls[0][0].content


def test_reviewer_sees_the_forecast(monkeypatch):
    model = FakeModel([review_json("PASS")])
    monkeypatch.setattr(module, "get_model", lambda: model)

    module.reviewer_agent(
        {
            "final_report": "report",
            "market_summary": MARKET_SUMMARY,
            "forecast": {"bias": "bearish", "confidence": "low"},
        }
    )

    system, human = model.calls[0]
    assert '"bias": "bearish"' in human.content
    assert "contradicts the" in system.content


# --------------------------------------------------------------------------
# Word limit enforced by counting
# --------------------------------------------------------------------------


def run_with_report(monkeypatch, report, replies):
    model = FakeModel(replies)
    monkeypatch.setattr(module, "get_model", lambda: model)

    return module.reviewer_agent(
        {"final_report": report, "market_summary": MARKET_SUMMARY}
    )


def test_an_over_long_report_must_be_revised_even_if_the_llm_says_pass(
    monkeypatch,
):
    result = run_with_report(
        monkeypatch, "word " * 650, [review_json("PASS")]
    )

    assert result["review"]["verdict"] == "NEEDS_REVISION"
    issue = result["review"]["critical_issues"][0]
    assert "650 words" in issue
    assert "600-word limit" in issue
    assert "at least 50 words" in issue
    assert "NEEDS_REVISION" in result["review_feedback"]


def test_a_report_at_the_limit_is_accepted(monkeypatch):
    result = run_with_report(monkeypatch, "word " * 600, [review_json("PASS")])

    assert result["review"]["verdict"] == "PASS"
    assert result["review"]["critical_issues"] == []


def test_the_length_issue_comes_first_and_keeps_the_llms_issues(monkeypatch):
    result = run_with_report(
        monkeypatch,
        "word " * 601,
        [review_json("NEEDS_REVISION", critical=["RSI value is wrong"])],
    )

    issues = result["review"]["critical_issues"]
    assert len(issues) == 2
    assert "601 words" in issues[0]
    assert issues[1] == "RSI value is wrong"


def test_length_is_still_enforced_when_the_llm_review_fails(monkeypatch):
    result = run_with_report(monkeypatch, "word " * 700, ["bad", "still bad"])

    review = result["review"]
    assert review["verdict"] == "NEEDS_REVISION"  # not UNREVIEWED
    assert "700 words" in review["critical_issues"][0]
    assert "could not be parsed" in review["other_issues"][0]
    assert result["errors"][0].startswith("reviewer_agent")


def test_a_short_report_with_a_failed_review_is_still_unreviewed(monkeypatch):
    result = run_with_report(monkeypatch, "short report", ["bad", "still bad"])

    assert result["review"]["verdict"] == "UNREVIEWED"
    assert result["review"]["critical_issues"] == []


def test_the_report_agent_is_told_the_same_limit():
    from financial_assistant.nodes import report_agent

    assert "under 600 words" in report_agent.SYSTEM_PROMPT
    assert "aim for about 500" in report_agent.SYSTEM_PROMPT


def test_reviewer_and_evaluator_count_words_the_same_way():
    from evaluation.evaluators import check_word_limit

    for words in (599, 600, 601):
        text = ("word\n" * words).strip()
        assert bool(module.report_length_issues(text)) is (
            not check_word_limit(text).passed
        )
