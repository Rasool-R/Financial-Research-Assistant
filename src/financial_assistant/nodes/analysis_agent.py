import json

from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model

SYSTEM_PROMPT = """
You are an independent technical analysis agent.

Analyze the supplied market data independently.

Requirements:
- Use only the supplied numerical data.
- Evaluate moving averages, RSI, and MACD separately.
- Identify bullish and bearish signals.
- Explain conflicts between indicators.
- Do not invent values or external events.
- Do not make guaranteed predictions.
- Do not provide personalized financial advice.
- Write the analysis in English.
- Keep the output concise and structured.
"""


def analysis_agent(state: dict) -> dict:
    """Independently analyze market indicators."""

    market_summary = state.get("market_summary")

    if not market_summary:
        raise ValueError(
            "market_summary is missing from the graph state."
        )

    model = get_model()

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Analyze the following market data:\n\n"
                    f"{json.dumps(market_summary, indent=2)}"
                )
            ),
        ]
    )

    return {
        "analysis_findings": response.content,
    }
