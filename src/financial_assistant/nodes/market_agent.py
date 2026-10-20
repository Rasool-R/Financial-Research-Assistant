import json

from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model
from financial_assistant.tools.market_summary import get_market_summary


SYSTEM_PROMPT = """
You are a financial market research analyst.

Analyze the provided market data carefully.

Rules:
- Use only the data provided in the user message.
- Never invent prices, indicators, events, or external facts.
- Explain trend, RSI, and MACD separately.
- Identify agreements or conflicts between indicators.
- Distinguish observations from interpretations.
- Do not present the analysis as a guaranteed prediction.
- Do not give personalized financial advice.
- Keep the report concise and structured.
"""


def market_agent(state: dict) -> dict:
    """Analyze a supported financial instrument using market indicators."""

    symbol = state.get("symbol", "XAU/USD")
    interval = state.get("interval", "1h")

    market_summary = get_market_summary(
        symbol=symbol,
        interval=interval,
        outputsize=100,
    )

    model = get_model()

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Analyze the following market summary:\n\n"
                    f"{json.dumps(market_summary, indent=2)}"
                )
            ),
        ]
    )

    return {
        "market_summary": market_summary,
        "market_analysis": response.content,
    }
