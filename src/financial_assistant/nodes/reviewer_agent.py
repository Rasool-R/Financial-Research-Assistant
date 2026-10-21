import json

from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model


SYSTEM_PROMPT = """
You are an independent quality reviewer for a financial research report.

Review the report using the supplied market data and research evidence.

Check for:
- Numerical inconsistencies with the market data.
- Unsupported factual claims.
- Outdated or irrelevant news.
- Incorrect technical interpretations.
- Missing report sections.
- Guaranteed predictions or personalized financial advice.
- Stale market data (data_quality.is_stale) presented as live prices.

Rules:
- Do not invent facts.
- Distinguish critical errors from minor concerns.
- Do not treat historical news as current market data.
- Preserve uncertainty when evidence is inconclusive.
- Write in English and be concise.

Return exactly this structure:

VERDICT: PASS or NEEDS_REVISION

CRITICAL ISSUES:
- List material factual errors, or None.

OTHER ISSUES:
- List minor problems, or None.

RECOMMENDATIONS:
- List specific improvements, or None.

Keep the entire review under 250 words.
"""


def limit_text(value: str, max_chars: int) -> str:
    """Limit text size while preserving a truncation notice."""
    if len(value) <= max_chars:
        return value

    return value[:max_chars] + "\n[Content truncated for review.]"


def reviewer_agent(state: dict) -> dict:
    """Review the final report using compact supporting evidence."""

    final_report = state.get("final_report")
    market_summary = state.get("market_summary")
    research_findings = state.get("research_findings", "")

    if not final_report:
        raise ValueError("final_report is missing from the graph state.")

    if not market_summary:
        raise ValueError("market_summary is missing from the graph state.")

    # Send only the market fields needed for factual verification.
    compact_market_data = {
        "symbol": market_summary.get("symbol"),
        "interval": market_summary.get("interval"),
        "latest_candle": market_summary.get("latest_candle"),
        "indicators": market_summary.get("indicators"),
        "market_conditions": market_summary.get("market_conditions"),
        "data_points": market_summary.get("data_points"),
        "data_source": market_summary.get("data_source"),
        "data_quality": market_summary.get("data_quality"),
    }

    market_evidence = json.dumps(
        compact_market_data,
        ensure_ascii=False,
        indent=2,
    )

    research_evidence = limit_text(
        str(research_findings),
        max_chars=3500,
    )

    report_to_review = limit_text(
        str(final_report),
        max_chars=6000,
    )

    model = get_model()

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Market data:\n{market_evidence}\n\n"
                    f"Research evidence:\n{research_evidence}\n\n"
                    f"Final report to review:\n{report_to_review}"
                )
            ),
        ]
    )

    return {
        "review_feedback": response.content,
    }
