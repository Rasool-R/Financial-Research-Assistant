import pytest

from financial_assistant import config


@pytest.fixture(autouse=True)
def base_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_URL_BASE", raising=False)


def test_get_model_reads_openai_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    model = config.get_model()

    assert model.openai_api_base == "https://example.com/v1"


def test_get_model_falls_back_to_legacy_url_base(monkeypatch):
    monkeypatch.setenv("OPENAI_URL_BASE", "https://legacy.example.com/v1")

    model = config.get_model()

    assert model.openai_api_base == "https://legacy.example.com/v1"


def test_get_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        config.get_model()
