from financial_assistant.nodes.research_agent import (
    prepare_search_results,
    sanitize_research_output,
)
from financial_assistant.schemas import ResearchFinding, ResearchOutput
from tests.fakes import FakeModel


def test_prepare_search_results_removes_duplicate_urls():
    results = [
        {"title": "Gold news", "url": "https://example.com/1", "content": "News"},
        {"title": "Duplicate", "url": "https://example.com/1", "content": "Duplicate"},
        {"title": "Currency news", "url": "https://example.com/2", "content": "News"},
    ]

    prepared = prepare_search_results(results)

    assert len(prepared) == 2
    assert prepared[0]["title"] == "Gold news"


def test_prepare_search_results_limits_content_length():
    results = [
        {
            "title": "Long article",
            "url": "https://example.com/long",
            "content": "x" * 2000,
        }
    ]

    prepared = prepare_search_results(results)

    assert len(prepared[0]["content"]) == 900


def test_research_output_schema():
    output = ResearchOutput(
        symbol="XAU/USD",
        summary="News sentiment is mixed.",
        findings=[
            ResearchFinding(
                title="Gold market update",
                published_date="2026-10-20",
                source_url="https://example.com/gold",
                relevance="high",
                summary="The article discusses recent gold market conditions.",
            )
        ],
        limitations=[],
    )

    assert output.symbol == "XAU/USD"
    assert len(output.findings) == 1
    assert output.model_dump()["findings"][0]["source_url"] == (
        "https://example.com/gold"
    )


SEARCH_RESULTS = [
    {
        "title": "Real headline",
        "url": "https://example.com/real",
        "published_date": "2026-10-20",
        "content": "Gold rose.",
    }
]


def make_finding(url, **overrides):
    data = {
        "title": "LLM rewritten title",
        "published_date": "1999-01-01",
        "source_url": url,
        "relevance": "high",
        "summary": "x" * 700,
    }
    data.update(overrides)
    return ResearchFinding(**data)


def test_sanitize_drops_findings_with_unknown_urls():
    output = ResearchOutput(
        symbol="WRONG",
        summary="s" * 1000,
        findings=[
            make_finding("https://example.com/real"),
            make_finding("https://invented.example.com/fake"),
        ],
    )

    result = sanitize_research_output(output, SEARCH_RESULTS, "XAU/USD")

    assert result.symbol == "XAU/USD"
    assert len(result.findings) == 1
    assert result.findings[0].source_url == "https://example.com/real"
    assert any("discarded" in item for item in result.limitations)


def test_sanitize_copies_title_and_date_from_source():
    output = ResearchOutput(
        symbol="XAU/USD",
        summary="ok",
        findings=[make_finding("https://example.com/real")],
    )

    finding = sanitize_research_output(
        output, SEARCH_RESULTS, "XAU/USD"
    ).findings[0]

    assert finding.title == "Real headline"
    assert finding.published_date == "2026-10-20"


def test_sanitize_enforces_length_limits():
    output = ResearchOutput(
        symbol="XAU/USD",
        summary="s" * 1000,
        findings=[make_finding("https://example.com/real")],
    )

    result = sanitize_research_output(output, SEARCH_RESULTS, "XAU/USD")

    assert len(result.summary) == 800
    assert len(result.findings[0].summary) == 500


def test_sanitize_removes_duplicate_urls():
    output = ResearchOutput(
        symbol="XAU/USD",
        summary="ok",
        findings=[
            make_finding("https://example.com/real"),
            make_finding("https://example.com/real"),
        ],
    )

    result = sanitize_research_output(output, SEARCH_RESULTS, "XAU/USD")

    assert len(result.findings) == 1


VALID_JSON = (
    '{"symbol": "XAU/USD", "summary": "Gold is higher.", "findings": ['
    '{"title": "t", "published_date": null, '
    '"source_url": "https://example.com/real", "relevance": "high", '
    '"summary": "Gold rose."}], "limitations": []}'
)


def patch_research(monkeypatch, model):
    from financial_assistant.nodes import research_agent as module

    monkeypatch.setattr(module, "search_web", lambda **kwargs: SEARCH_RESULTS)
    monkeypatch.setattr(module, "get_model", lambda: model)
    return module


def test_research_agent_retries_after_unparseable_answer(monkeypatch):
    model = FakeModel(["not json at all", VALID_JSON])
    module = patch_research(monkeypatch, model)

    result = module.research_agent({"symbol": "XAU/USD", "user_query": "gold"})

    assert len(model.calls) == 2
    assert len(result["research_findings"]["findings"]) == 1


def test_research_agent_falls_back_when_parsing_keeps_failing(monkeypatch):
    model = FakeModel(["bad", "still bad"])
    module = patch_research(monkeypatch, model)

    result = module.research_agent({"symbol": "XAU/USD", "user_query": "gold"})

    findings = result["research_findings"]
    assert findings["findings"] == []
    assert "could not be parsed" in findings["limitations"][0]


def test_research_agent_degrades_when_web_search_fails(monkeypatch):
    from financial_assistant.nodes import research_agent as module

    def broken_search(**kwargs):
        raise RuntimeError("Tavily is down")

    monkeypatch.setattr(module, "search_web", broken_search)

    result = module.research_agent({"symbol": "XAU/USD", "user_query": "gold"})

    assert result["research_findings"]["findings"] == []
    assert "Web search failed" in result["research_findings"]["limitations"][0]
    assert result["errors"] and "research_agent" in result["errors"][0]


def test_research_agent_retries_transient_search_errors(monkeypatch):
    import requests

    from financial_assistant import retry
    from financial_assistant.nodes import research_agent as module

    attempts = []

    def flaky_search(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("network") from requests.ConnectionError()
        return []

    monkeypatch.setattr(module, "search_web", flaky_search)
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)

    result = module.research_agent({"symbol": "XAU/USD", "user_query": "gold"})

    assert len(attempts) == 3
    assert result["research_findings"]["summary"] == (
        "No relevant news results were found."
    )
