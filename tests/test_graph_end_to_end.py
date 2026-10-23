"""Run the real graph and real nodes; only network access is faked."""

import json
from datetime import datetime, timezone

from financial_assistant import graph as graph_module
from financial_assistant.nodes import (
    analysis_agent,
    market_agent,
    report_agent,
    research_agent,
    reviewer_agent,
)
from financial_assistant.tools import market_summary
from tests.fakes import FakeResponse
from tests.test_market_data import make_candles

# The fixture news is dated 2026-10-20; freeze "now" so the recency filter
# keeps behaving the same no matter when the tests run.
FROZEN_NOW = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)

SEARCH_RESULTS = [
    {
        "title": "Gold climbs",
        "url": "https://example.com/gold",
        "published_date": "2026-10-20",
        "content": "Gold rose on Tuesday.",
    }
]

RESEARCH_JSON = json.dumps(
    {
        "symbol": "XAU/USD",
        "summary": "Gold is higher.",
        "findings": [
            {
                "title": "Gold climbs",
                "published_date": "2026-10-20",
                "source_url": "https://example.com/gold",
                "relevance": "high",
                "sentiment": "bullish",
                "summary": "Gold rose.",
            }
        ],
        "limitations": [],
    }
)


def review(verdict, critical=()):
    return json.dumps(
        {
            "verdict": verdict,
            "critical_issues": list(critical),
            "other_issues": [],
            "recommendations": [],
        }
    )


class RoutingModel:
    """Answers according to which agent's system prompt it receives."""

    def __init__(self, review_replies):
        self.review_replies = list(review_replies)
        self.report_inputs = []

    def invoke(self, messages):
        system = messages[0].content

        if "trend and momentum analyst" in system:
            return FakeResponse("market analysis")
        if "risk and levels analyst" in system:
            return FakeResponse("independent analysis")
        if "financial news research agent" in system:
            return FakeResponse(RESEARCH_JSON)
        if "final reporting agent" in system:
            evidence = json.loads(messages[1].content)
            self.report_inputs.append(evidence)
            return FakeResponse(f"report v{len(self.report_inputs)}")
        if "quality reviewer" in system:
            return FakeResponse(self.review_replies.pop(0))

        raise AssertionError(f"unexpected agent: {system[:60]}")


def run_real_graph(monkeypatch, model, search=None, market_data=None):
    for module in (
        market_agent,
        analysis_agent,
        research_agent,
        report_agent,
        reviewer_agent,
    ):
        monkeypatch.setattr(module, "get_model", lambda: model)

    monkeypatch.setattr(research_agent, "utc_now", lambda: FROZEN_NOW)
    monkeypatch.setattr(
        market_summary,
        "get_market_data",
        market_data or (lambda **kw: make_candles(100)),
    )
    monkeypatch.setattr(
        research_agent,
        "search_web",
        search or (lambda **kw: SEARCH_RESULTS),
    )

    return graph_module.build_graph().invoke(
        {
            "user_query": "Analyze gold",
            "symbol": "XAU/USD",
            "interval": "1h",
        }
    )


def test_full_workflow_with_one_revision(monkeypatch):
    model = RoutingModel(
        [
            review("NEEDS_REVISION", critical=["RSI value is wrong"]),
            review("PASS"),
        ]
    )

    result = run_real_graph(monkeypatch, model)

    assert result["final_report"] == "report v2"
    assert result["revision_count"] == 1
    assert result["review"]["verdict"] == "PASS"
    assert "errors" not in result or result["errors"] == []

    first, second = model.report_inputs
    assert "previous_report" not in first
    assert second["previous_report"] == "report v1"
    assert second["reviewer_feedback"]["critical_issues"] == [
        "RSI value is wrong"
    ]
    assert first["market_data"]["multi_timeframe"]["alignment"] == (
        "aligned_bullish"
    )
    assert first["market_data"]["key_levels"]["lookback_candles"] == 60
    assert "trend_and_momentum_analysis" in first
    assert "risk_and_levels_analysis" in first
    assert first["web_research"]["findings"][0]["source_url"] == (
        "https://example.com/gold"
    )


def test_full_workflow_survives_search_outage(monkeypatch):
    def search_down(**kwargs):
        raise RuntimeError("Tavily is down")

    model = RoutingModel([review("PASS")])

    result = run_real_graph(monkeypatch, model, search=search_down)

    assert result["final_report"] == "report v1"
    assert any("Web search failed" in error for error in result["errors"])
    # the report writer was told about the problem
    assert model.report_inputs[0]["pipeline_warnings"] == result["errors"]


def test_full_workflow_survives_a_failing_higher_timeframe(monkeypatch):
    def market_data(**kwargs):
        if kwargs["interval"] == "4h":
            raise RuntimeError("Twelve Data rate limit")
        return make_candles(100)

    model = RoutingModel([review("PASS")])

    result = run_real_graph(monkeypatch, model, market_data=market_data)

    assert result["final_report"] == "report v1"
    unavailable = result["market_summary"]["multi_timeframe"]["unavailable"]
    assert [item["interval"] for item in unavailable] == ["4h"]
    assert any("4h data unavailable" in e for e in result["errors"])
    # the report writer was told that confirmation is incomplete
    assert model.report_inputs[0]["pipeline_warnings"] == result["errors"]
