import json

from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model


SYSTEM_PROMPT = """
You are the final reporting agent in a financial research assistant.

Write a concise, evidence-based report in English.

Rules:
- Use only the supplied evidence.
- Never invent prices, indicators, news, dates, or sources.
- Preserve source URLs exactly.
- Separate current market data from historical news.
- Distinguish observations from interpretations.
- Explain conflicting signals without treating them as errors.
- Do not present historical prices as current prices.
- Do not make guaranteed predictions or personalized financial recommendations.
- Identify important data limitations.
- If data_quality.is_stale is true, state clearly that the prices may not
  be live and mention data_quality.note in Risks and Limitations.
- Avoid repeating information across sections.
- Keep the report under 600 words.

Use these sections:
1. Executive Summary
2. Market Snapshot
3. Technical Analysis
4. Relevant News
5. Risks and Limitations
6. Conclusion
7. Sources

Use compact bullets instead of large tables.
Include only the most relevant news.
"""


def limit_text(value: str, max_chars: int) -> str:
    """Limit text size before sending it to the model."""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n[Truncated for brevity.]"


def compact_research(findings) -> dict:
    """Keep only the most useful research fields."""
    if not isinstance(findings, dict):
        return {
            "summary": "Research findings are unavailable.",
            "findings": [],
            "limitations": [],
        }

    items = []

    for item in findings.get("findings", [])[:3]:
        if not isinstance(item, dict):
            continue

        items.append({
            "title": item.get("title"),
            "published_date": item.get("published_date"),
            "source_url": item.get("source_url"),
            "summary": limit_text(
                str(item.get("summary") or ""),
                350,
            ),
        })

    return {
        "symbol": findings.get("symbol"),
        "summary": limit_text(
            str(findings.get("summary") or ""),
            500,
        ),
        "findings": items,
        "limitations": findings.get("limitations", [])[:3],
    }


def report_agent(state: dict) -> dict:
    """Generate a concise report from compact evidence."""

    market_summary = state.get("market_summary")
    market_analysis = state.get("market_analysis")
    analysis_findings = state.get("analysis_findings")
    research_findings = state.get("research_findings")

    if not market_summary:
        raise ValueError("market_summary is missing from the graph state.")

    if not market_analysis or not analysis_findings:
        raise ValueError("Technical analysis results are missing.")

    compact_market = {
        "symbol": market_summary.get("symbol"),
        "interval": market_summary.get("interval"),
        "latest_candle": market_summary.get("latest_candle"),
        "indicators": market_summary.get("indicators"),
        "market_conditions": market_summary.get("market_conditions"),
        "data_points": market_summary.get("data_points"),
        "data_source": market_summary.get("data_source"),
        "data_quality": market_summary.get("data_quality"),
    }

    compact_news = compact_research(research_findings)

    evidence = {
        "user_query": state.get("user_query", ""),
        "market_data": compact_market,
        "market_agent_analysis": limit_text(
            str(market_analysis),
            1800,
        ),
        "independent_technical_analysis": limit_text(
            str(analysis_findings),
            1800,
        ),
        "web_research": compact_news,
    }

    model = get_model()

    response = model.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(
            evidence,
            ensure_ascii=False,
            separators=(",", ":"),
        )),
    ])

    return {"final_report": response.content}
