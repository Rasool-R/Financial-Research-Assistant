import json

import pytest

from financial_assistant.nodes import analysis_agent, market_agent
from tests.fakes import FakeModel

SUMMARY = {
    "symbol": "XAU/USD",
    "interval": "1h",
    "key_levels": {"support": {"price": 4290.5}, "resistance": None},
    "multi_timeframe": {"alignment": "mixed", "unavailable": []},
}


def run_market_agent(monkeypatch, summary):
    model = FakeModel(["TREND TEXT"])
    monkeypatch.setattr(market_agent, "get_model", lambda: model)
    monkeypatch.setattr(
        market_agent, "get_multi_timeframe_summary", lambda **kw: summary
    )

    result = market_agent.market_agent({"symbol": "XAU/USD", "interval": "1h"})

    return result, model


def test_market_agent_is_the_trend_and_momentum_analyst(monkeypatch):
    result, model = run_market_agent(monkeypatch, SUMMARY)

    system, human = model.calls[0]
    assert "trend and momentum analyst" in system.content
    assert "do not discuss those" in system.content
    assert "multi_timeframe" in system.content
    payload = json.loads(human.content.split("\n\n", 1)[1])
    assert payload["multi_timeframe"] == SUMMARY["multi_timeframe"]
    assert result["market_analysis"] == "TREND TEXT"
    assert result["market_summary"] is SUMMARY
    assert "errors" not in result


def test_market_agent_reports_unavailable_timeframes_as_warnings(monkeypatch):
    summary = {
        **SUMMARY,
        "multi_timeframe": {
            "alignment": "single_timeframe",
            "unavailable": [{"interval": "4h", "error": "rate limited"}],
        },
    }

    result, _ = run_market_agent(monkeypatch, summary)

    assert result["errors"] == [
        "market_agent: 4h data unavailable (rate limited), so "
        "higher-timeframe confirmation is incomplete."
    ]


def run_analysis_agent(monkeypatch, state):
    model = FakeModel(["LEVELS TEXT"])
    monkeypatch.setattr(analysis_agent, "get_model", lambda: model)

    return analysis_agent.analysis_agent(state), model


def test_analysis_agent_is_the_risk_and_levels_analyst(monkeypatch):
    result, model = run_analysis_agent(
        monkeypatch, {"market_summary": SUMMARY, "market_analysis": "TREND TEXT"}
    )

    system, human = model.calls[0]
    assert "risk and levels analyst" in system.content
    assert "Never invent levels or price targets" in system.content
    assert "Do not give entry, stop-loss or take-profit" in system.content
    assert "4290.5" in human.content
    assert result == {"analysis_findings": "LEVELS TEXT"}


def test_the_two_analysts_have_different_jobs(monkeypatch):
    assert market_agent.SYSTEM_PROMPT != analysis_agent.SYSTEM_PROMPT
    assert "volatility" in analysis_agent.SYSTEM_PROMPT
    assert "key_levels" in analysis_agent.SYSTEM_PROMPT
    assert "key_levels" not in market_agent.SYSTEM_PROMPT


def test_analysis_agent_stays_independent_of_the_trend_analysis(monkeypatch):
    _, model = run_analysis_agent(
        monkeypatch,
        {"market_summary": SUMMARY, "market_analysis": "TREND TEXT"},
    )

    assert "TREND TEXT" not in model.calls[0][1].content


def test_analysis_agent_requires_a_market_summary():
    with pytest.raises(ValueError, match="market_summary"):
        analysis_agent.analysis_agent({})
