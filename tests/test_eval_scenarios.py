import json
from unittest.mock import patch

import pytest

from evaluation.run_eval import run_scenario
from evaluation.scenarios import SCENARIOS, SCENARIOS_BY_NAME
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


def summary_for(scenario):
    with patch.object(
        market_summary,
        "get_market_data",
        lambda **kw: scenario.make_candles(kw["interval"]),
    ):
        return market_summary.get_multi_timeframe_summary(
            scenario.symbol, scenario.interval, outputsize=100
        )


def test_scenario_names_are_unique():
    assert len(SCENARIOS_BY_NAME) == len(SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_fixtures_behave_as_labelled(scenario):
    summary = summary_for(scenario)

    if scenario.expected_trend:
        assert summary["market_conditions"]["trend"] == scenario.expected_trend

    if scenario.expected_alignment:
        multi = summary["multi_timeframe"]
        assert multi["alignment"] == scenario.expected_alignment
        assert multi["unavailable"] == []

    assert summary["data_quality"]["is_stale"] is scenario.expect_stale


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_fixtures_are_deterministic(scenario):
    first = scenario.make_candles(scenario.interval)["close"].tolist()
    second = scenario.make_candles(scenario.interval)["close"].tolist()

    assert first == second


class FixtureModel:
    """Fake LLM that cites the first news item of the running scenario."""

    def __init__(self, news):
        self.news = news
        self.research_prompts = []

    def invoke(self, messages):
        system = messages[0].content

        if "financial news research agent" in system:
            self.research_prompts.append(messages[1].content)
            item = self.news[0]
            return FakeResponse(
                json.dumps(
                    {
                        "symbol": "X",
                        "summary": "News summary.",
                        "findings": [
                            {
                                "title": item["title"],
                                "published_date": None,
                                "source_url": item["url"],
                                "relevance": "high",
                                "sentiment": "neutral",
                                "summary": "Summary.",
                            }
                        ],
                        "limitations": [],
                    }
                )
            )
        if "final reporting agent" in system:
            return FakeResponse("1. Executive Summary\nTiny report.")
        if "quality reviewer" in system:
            return FakeResponse(
                json.dumps(
                    {
                        "verdict": "PASS",
                        "critical_issues": [],
                        "other_issues": [],
                        "recommendations": [],
                    }
                )
            )
        return FakeResponse("analysis")


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_run_scenario_uses_frozen_data_with_the_real_graph(
    monkeypatch, scenario
):
    model = FixtureModel(scenario.news or [{"title": "-", "url": "-"}])

    for module in (
        market_agent,
        analysis_agent,
        research_agent,
        report_agent,
        reviewer_agent,
    ):
        monkeypatch.setattr(module, "get_model", lambda: model)

    record = run_scenario(scenario, graph=graph_module.build_graph())

    assert record["trend_matches_fixture"] is True
    assert record["reviewer_verdict"] == "PASS"
    assert len(record["evaluators"]) == 9
    assert record["errors"] == []
    # the tiny fake report is missing most sections, so the run must fail
    assert record["passed"] is False
    sections = next(
        e for e in record["evaluators"] if e["name"] == "required_sections"
    )
    assert sections["passed"] is False


def install_model(monkeypatch, model):
    for module in (
        market_agent,
        analysis_agent,
        research_agent,
        report_agent,
        reviewer_agent,
    ):
        monkeypatch.setattr(module, "get_model", lambda: model)


def test_old_article_is_removed_before_the_llm_sees_it(monkeypatch):
    scenario = SCENARIOS_BY_NAME["gold_uptrend_with_news"]
    model = FixtureModel(scenario.news)
    install_model(monkeypatch, model)

    record = run_scenario(scenario, graph=graph_module.build_graph())

    prompt = model.research_prompts[0]
    assert "gold-edges-higher" in prompt
    assert "old-gold-selloff" not in prompt
    fresh = next(
        e for e in record["evaluators"] if e["name"] == "fresh_news"
    )
    assert fresh["passed"] is True


def test_injected_news_is_sent_as_data_inside_the_delimiters(monkeypatch):
    scenario = SCENARIOS_BY_NAME["gold_prompt_injection_in_news"]
    model = FixtureModel(scenario.news)
    install_model(monkeypatch, model)

    run_scenario(scenario, graph=graph_module.build_graph())

    prompt = model.research_prompts[0]
    start = prompt.index("<search_results>")
    end = prompt.index("</search_results>")
    assert "PWNED-7731" in prompt[start:end]
    assert "PWNED-7731" not in prompt[:start]
    assert "PWNED-7731" not in prompt[end:]
