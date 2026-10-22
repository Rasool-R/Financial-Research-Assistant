import threading

import pytest
import requests
from langgraph.types import RetryPolicy

from financial_assistant import graph as graph_module
from financial_assistant.retry import is_transient_error


class Recorder:
    """Records which fake nodes ran and with which state."""

    def __init__(self):
        self.calls = {}
        self.report_states = []

    def count(self, name):
        return self.calls.get(name, 0)


def install_fake_nodes(monkeypatch, recorder, verdicts, research_errors=None):
    """Replace every node with a fake and return a freshly built graph."""

    verdict_iter = iter(verdicts)

    def tick(name):
        recorder.calls[name] = recorder.count(name) + 1

    def market_agent(state):
        tick("market_agent")
        return {"market_summary": {"symbol": "XAU/USD"}, "market_analysis": "m"}

    def analysis_agent(state):
        tick("analysis_agent")
        assert state["market_summary"]
        return {"analysis_findings": "a"}

    def research_agent(state):
        tick("research_agent")
        update = {"research_findings": {"findings": []}}
        if research_errors:
            update["errors"] = research_errors
        return update

    def report_agent(state):
        tick("report_agent")
        recorder.report_states.append(dict(state))
        # the join must wait for both branches
        assert state["analysis_findings"] and state["research_findings"]
        is_revision = state.get("review", {}).get("verdict") == "NEEDS_REVISION"
        return {
            "final_report": f"draft {recorder.count('report_agent')}",
            "revision_count": state.get("revision_count", 0) + 1
            if is_revision
            else 0,
        }

    def reviewer_agent(state):
        tick("reviewer_agent")
        verdict = next(verdict_iter)
        return {
            "review": {"verdict": verdict, "critical_issues": ["x"]},
            "review_feedback": verdict,
        }

    for name, fn in {
        "market_agent": market_agent,
        "analysis_agent": analysis_agent,
        "research_agent": research_agent,
        "report_agent": report_agent,
        "reviewer_agent": reviewer_agent,
    }.items():
        monkeypatch.setattr(graph_module, name, fn)

    return graph_module.build_graph()


def test_graph_runs_once_when_review_passes(monkeypatch):
    recorder = Recorder()
    graph = install_fake_nodes(monkeypatch, recorder, ["PASS"])

    result = graph.invoke({"symbol": "XAU/USD"})

    assert recorder.count("report_agent") == 1
    assert recorder.count("reviewer_agent") == 1
    assert result["final_report"] == "draft 1"
    assert result["review"]["verdict"] == "PASS"


def test_graph_revises_until_review_passes(monkeypatch):
    recorder = Recorder()
    graph = install_fake_nodes(
        monkeypatch, recorder, ["NEEDS_REVISION", "PASS"]
    )

    result = graph.invoke({"symbol": "XAU/USD"})

    assert recorder.count("report_agent") == 2
    assert recorder.count("reviewer_agent") == 2
    assert result["final_report"] == "draft 2"
    assert result["revision_count"] == 1
    # the writer saw the reviewer's verdict on the second pass
    assert recorder.report_states[1]["review"]["verdict"] == "NEEDS_REVISION"
    # market/research are not re-run during a revision
    assert recorder.count("market_agent") == 1
    assert recorder.count("research_agent") == 1


def test_graph_stops_after_max_revisions(monkeypatch):
    monkeypatch.setattr(graph_module, "MAX_REVISIONS", 2)
    recorder = Recorder()
    graph = install_fake_nodes(monkeypatch, recorder, ["NEEDS_REVISION"] * 10)

    result = graph.invoke({"symbol": "XAU/USD"})

    assert recorder.count("report_agent") == 3  # first draft + 2 revisions
    assert recorder.count("reviewer_agent") == 3
    assert result["revision_count"] == 2
    assert result["review"]["verdict"] == "NEEDS_REVISION"


def test_graph_does_not_loop_when_reviewer_could_not_review(monkeypatch):
    recorder = Recorder()
    graph = install_fake_nodes(monkeypatch, recorder, ["UNREVIEWED"])

    graph.invoke({"symbol": "XAU/USD"})

    assert recorder.count("report_agent") == 1


def test_errors_from_nodes_are_accumulated(monkeypatch):
    recorder = Recorder()
    graph = install_fake_nodes(
        monkeypatch, recorder, ["PASS"], research_errors=["research down"]
    )

    result = graph.invoke({"symbol": "XAU/USD"})

    assert result["errors"] == ["research down"]


def test_market_and_research_run_in_parallel(monkeypatch):
    barrier = threading.Barrier(2, timeout=5)
    seen = []

    def market_agent(state):
        barrier.wait()  # only passes if research_agent runs at the same time
        seen.append("market")
        return {"market_summary": {"symbol": "XAU/USD"}, "market_analysis": "m"}

    def research_agent(state):
        barrier.wait()
        seen.append("research")
        return {"research_findings": {"findings": []}}

    monkeypatch.setattr(graph_module, "market_agent", market_agent)
    monkeypatch.setattr(graph_module, "research_agent", research_agent)
    monkeypatch.setattr(
        graph_module, "analysis_agent", lambda s: {"analysis_findings": "a"}
    )
    monkeypatch.setattr(
        graph_module, "report_agent", lambda s: {"final_report": "r"}
    )
    monkeypatch.setattr(
        graph_module,
        "reviewer_agent",
        lambda s: {"review": {"verdict": "PASS"}},
    )

    result = graph_module.build_graph().invoke({"symbol": "XAU/USD"})

    assert sorted(seen) == ["market", "research"]
    assert result["final_report"] == "r"


def test_fatal_nodes_retry_transient_errors(monkeypatch):
    fast_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.01,
        jitter=False,
        retry_on=is_transient_error,
    )
    monkeypatch.setattr(graph_module, "RETRY_POLICY", fast_policy)

    attempts = []

    def flaky_market_agent(state):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("fetch failed") from requests.ConnectionError()
        return {"market_summary": {"symbol": "XAU/USD"}, "market_analysis": "m"}

    monkeypatch.setattr(graph_module, "market_agent", flaky_market_agent)
    monkeypatch.setattr(
        graph_module, "analysis_agent", lambda s: {"analysis_findings": "a"}
    )
    monkeypatch.setattr(
        graph_module,
        "research_agent",
        lambda s: {"research_findings": {"findings": []}},
    )
    monkeypatch.setattr(
        graph_module, "report_agent", lambda s: {"final_report": "r"}
    )
    monkeypatch.setattr(
        graph_module,
        "reviewer_agent",
        lambda s: {"review": {"verdict": "PASS"}},
    )

    result = graph_module.build_graph().invoke({"symbol": "XAU/USD"})

    assert len(attempts) == 3
    assert result["final_report"] == "r"


def test_fatal_nodes_do_not_retry_permanent_errors(monkeypatch):
    fast_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.01,
        jitter=False,
        retry_on=is_transient_error,
    )
    monkeypatch.setattr(graph_module, "RETRY_POLICY", fast_policy)

    attempts = []

    def broken_market_agent(state):
        attempts.append(1)
        raise ValueError("Unsupported symbol")

    monkeypatch.setattr(graph_module, "market_agent", broken_market_agent)
    monkeypatch.setattr(
        graph_module,
        "research_agent",
        lambda s: {"research_findings": {"findings": []}},
    )

    with pytest.raises(ValueError, match="Unsupported symbol"):
        graph_module.build_graph().invoke({"symbol": "BAD"})

    assert len(attempts) == 1
