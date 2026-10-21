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
