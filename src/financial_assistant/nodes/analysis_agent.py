import json

from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model

SYSTEM_PROMPT = """
You are the risk and levels analyst in a financial research assistant.

Analyze the supplied market summary. A separate analyst covers trend and
momentum, so focus on volatility, price levels and scenarios.

Requirements:
- Use only the supplied numbers. Every price level you mention must come
  from the summary: key_levels, bollinger, the moving averages in
  indicators, or the latest candle. Never invent levels or price targets.
- Volatility: describe the ATR (atr_percent and atr_vs_median) and the
  Bollinger Bands (position and bandwidth_percent).
- Key levels: give the nearest support and resistance with their distance
  in ATR units, and the lookback high and low. If support or resistance is
  null, say that no such swing level exists in the lookback window.
- Bull case and bear case: describe what would strengthen each view, using
  conditional wording such as "if price closes above ...".
- Invalidation: name the levels at which the current picture would be wrong.
- Do not give entry, stop-loss or take-profit instructions.
- Do not make guaranteed predictions.
- Do not provide personalized financial advice.
- Write the analysis in English.
- Keep the output concise and structured.
"""


def analysis_agent(state: dict) -> dict:
    """Analyze volatility, key levels and scenarios from the market summary."""

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
