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
