import json

from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model
from financial_assistant.tools.market_summary import (
    get_multi_timeframe_summary,
)

SYSTEM_PROMPT = """
You are the trend and momentum analyst in a financial research assistant.

Analyze the provided market summary. A separate analyst covers volatility,
price levels and scenarios, so do not discuss those.

Rules:
- Use only the data provided in the user message.
- Never invent prices, indicators, events, or external facts.
- Explain the trend (SMA 20 versus SMA 50 and price_action.last_sma_cross),
  momentum (RSI and MACD) and recent price action
  (price_action.change_24h_percent) separately.
- Use multi_timeframe to say whether the higher timeframes confirm or
  conflict with the main timeframe. If multi_timeframe.unavailable is not
  empty, say that higher-timeframe confirmation is incomplete.
- Identify agreements or conflicts between indicators.
- End with a directional bias (bullish, bearish or neutral) and a
  confidence level (low, medium or high). Use high confidence only when
  trend, momentum and the higher timeframes agree.
- Distinguish observations from interpretations.
- If data_quality.is_stale is true, say that the prices may not be live.
- Do not present the analysis as a guaranteed prediction.
- Do not give personalized financial advice.
- Keep the analysis concise and structured.
"""


def market_agent(state: dict) -> dict:
    """Analyze a supported financial instrument using market indicators."""

    symbol = state.get("symbol", "XAU/USD")
    interval = state.get("interval", "1h")

    market_summary = get_multi_timeframe_summary(
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

    update = {
        "market_summary": market_summary,
        "market_analysis": response.content,
    }

    unavailable = market_summary["multi_timeframe"]["unavailable"]

    if unavailable:
        update["errors"] = [
            f"market_agent: {item['interval']} data unavailable "
            f"({item['error']}), so higher-timeframe confirmation is "
            "incomplete."
            for item in unavailable
        ]

    return update
