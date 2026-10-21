from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model


SYSTEM_PROMPT = """
You are the final reporting agent in a multi-agent financial
research assistant.

Your task is to combine technical analysis and web research
into one clear, evidence-based report.

Requirements:
- Write the entire report in English.
- Use only the information provided in the input.
- Do not invent prices, indicators, news, dates, or sources.
- Preserve the distinction between technical observations
  and interpretations.
- Separate current market data from historical news.
- Explicitly flag news that may be outdated or inconsistent
  with the current market data.
- Do not treat historical support or resistance levels as
  current levels unless the supplied evidence confirms this.
- Preserve source URLs exactly as provided.
- If evidence is insufficient, clearly state the limitation.
- Highlight agreements and conflicts between technical
  indicators and reported news.
- Do not make guaranteed predictions or personalized
  financial recommendations.

Use this structure:

1. Executive Summary
2. Current Market Snapshot
3. Technical Analysis
4. Relevant News and Fundamental Context
5. Technical and News Alignment
6. Risks and Data Limitations
7. Sources

Keep the report concise, professional, and easy to read.
"""


def report_agent(state: dict) -> dict:
    """Generate the final financial research report."""

    market_summary = state.get("market_summary")
    market_analysis = state.get("market_analysis")
    analysis_findings = state.get("analysis_findings")
    research_findings = state.get("research_findings")

    if not market_summary:
        raise ValueError("market_summary is missing from the graph state.")

    if not market_analysis or not analysis_findings:
        raise ValueError("Technical analysis results are missing.")

    if not research_findings:
        research_findings = (
            "No research findings were provided. "
            "The report must disclose this limitation."
        )

    model = get_model()

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User query: {state.get('user_query', '')}\n"
                    f"Instrument: {state.get('symbol', 'XAU/USD')}\n"
                    f"Interval: {state.get('interval', '1h')}\n\n"
                    f"Market summary:\n{market_summary}\n\n"
                    f"Market agent report:\n{market_analysis}\n\n"
                    f"Independent technical analysis:\n"
                    f"{analysis_findings}\n\n"
                    f"Web research findings:\n{research_findings}\n\n"
                    "Generate the final report using the required "
                    "structure. Do not introduce unsupported facts."
                )
            ),
        ]
    )

    return {
        "final_report": response.content,
    }
