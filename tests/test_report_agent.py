from financial_assistant.nodes.report_agent import (
    compact_research,
    limit_text,
)


def test_limit_text_preserves_short_text():
    assert limit_text("Short report", 20) == "Short report"


def test_limit_text_truncates_long_text():
    result = limit_text("x" * 100, 20)

    assert result.startswith("x" * 20)
    assert result.endswith("[Truncated for brevity.]")
    assert len(result) > 20


def test_compact_research_limits_findings():
    research = {
        "symbol": "XAU/USD",
        "summary": "Gold market news summary",
        "findings": [
            {
                "title": f"News {i}",
                "published_date": "2026-10-20",
                "source_url": f"https://example.com/{i}",
                "summary": "Market update",
            }
            for i in range(5)
        ],
        "limitations": [],
    }

    result = compact_research(research)

    assert result["symbol"] == "XAU/USD"
    assert len(result["findings"]) == 3
    assert result["findings"][0]["source_url"] == (
        "https://example.com/0"
    )


def test_compact_research_handles_missing_data():
    result = compact_research(None)

    assert result["findings"] == []
    assert result["limitations"] == []


def run_report(monkeypatch, state):
    import json

    from financial_assistant.nodes import report_agent as module
    from tests.fakes import FakeModel

    model = FakeModel(["report body"])
    monkeypatch.setattr(module, "get_model", lambda: model)

    base_state = {
        "market_summary": {"symbol": "XAU/USD", "interval": "1h"},
        "market_analysis": "analysis one",
        "analysis_findings": "analysis two",
        "research_findings": None,
    }
    base_state.update(state)

    result = module.report_agent(base_state)
    evidence = json.loads(model.calls[0][1].content)

    return result, evidence


def test_first_report_has_no_revision_context(monkeypatch):
    result, evidence = run_report(monkeypatch, {})

    assert result["revision_count"] == 0
    assert "previous_report" not in evidence
    assert "reviewer_feedback" not in evidence


def test_revision_receives_previous_report_and_feedback(monkeypatch):
    state = {
        "final_report": "old draft",
        "revision_count": 0,
        "review": {
            "verdict": "NEEDS_REVISION",
            "critical_issues": ["wrong RSI"],
            "other_issues": [],
            "recommendations": ["fix RSI"],
        },
    }

    result, evidence = run_report(monkeypatch, state)

    assert result["revision_count"] == 1
    assert evidence["previous_report"] == "old draft"
    assert evidence["reviewer_feedback"]["critical_issues"] == ["wrong RSI"]


def test_pipeline_warnings_are_passed_to_the_report(monkeypatch):
    _, evidence = run_report(
        monkeypatch, {"errors": ["research_agent: Web search failed"]}
    )

    assert evidence["pipeline_warnings"] == [
        "research_agent: Web search failed"
    ]


def test_report_gets_todays_date_and_news_sentiment(monkeypatch):
    from datetime import datetime, timezone

    from financial_assistant.nodes import report_agent as module

    monkeypatch.setattr(
        module,
        "utc_now",
        lambda: datetime(2026, 10, 22, 23, 59, tzinfo=timezone.utc),
    )
    research = {
        "symbol": "XAU/USD",
        "summary": "s",
        "findings": [
            {
                "title": "T",
                "published_date": "2026-10-20",
                "source_url": "https://example.com/a",
                "relevance": "high",
                "sentiment": "bearish",
                "summary": "Gold fell.",
            }
        ],
        "limitations": [],
    }

    _, evidence = run_report(monkeypatch, {"research_findings": research})

    assert evidence["as_of"] == "2026-10-22"
    finding = evidence["web_research"]["findings"][0]
    assert finding["relevance"] == "high"
    assert finding["sentiment"] == "bearish"
