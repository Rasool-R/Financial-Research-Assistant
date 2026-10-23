import pytest

from financial_assistant.tools import web_search as module


class FakeTavily:
    last_kwargs = None
    response = {"results": []}

    def __init__(self, api_key):
        self.api_key = api_key

    def search(self, **kwargs):
        FakeTavily.last_kwargs = kwargs
        return FakeTavily.response


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    FakeTavily.last_kwargs = None
    FakeTavily.response = {"results": []}
    monkeypatch.setenv("TAVILY_API_KEY", "key")
    monkeypatch.setattr(module, "TavilyClient", FakeTavily)


def test_days_is_sent_to_tavily_with_the_news_topic():
    module.search_web("gold price", max_results=3, days=7)

    assert FakeTavily.last_kwargs == {
        "query": "gold price",
        "max_results": 3,
        "topic": "news",
        "search_depth": "basic",
        "days": 7,
    }


def test_days_is_omitted_when_not_requested():
    module.search_web("gold price")

    assert "days" not in FakeTavily.last_kwargs


def test_results_keep_the_search_score():
    FakeTavily.response = {
        "results": [
            {
                "title": "T",
                "url": "https://example.com/a",
                "content": "C",
                "published_date": "2026-10-20",
                "score": 0.87,
            }
        ]
    }

    results = module.search_web("gold")

    assert results == [
        {
            "title": "T",
            "url": "https://example.com/a",
            "content": "C",
            "published_date": "2026-10-20",
            "score": 0.87,
        }
    ]


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="days"):
        module.search_web("gold", days=0)
    with pytest.raises(ValueError, match="max_results"):
        module.search_web("gold", max_results=11)
    with pytest.raises(ValueError, match="empty"):
        module.search_web("   ")


def test_missing_api_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        module.search_web("gold")


def test_provider_errors_become_runtime_errors(monkeypatch):
    class Broken(FakeTavily):
        def search(self, **kwargs):
            raise ConnectionError("boom")

    monkeypatch.setattr(module, "TavilyClient", Broken)

    with pytest.raises(RuntimeError, match="Web search failed") as exc:
        module.search_web("gold")

    assert isinstance(exc.value.__cause__, ConnectionError)
