import json

from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model
from financial_assistant.forecast import (
    horizon_for,
    parse_forecast,
    unknown_forecast,
)
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
- Give a directional bias (bullish, bearish or neutral) and a confidence
  level (low, medium or high) for the forecast horizon stated in the
  request. Use high confidence only when trend, momentum and the higher
  timeframes agree.
- The very last line of your answer must be exactly:
  FORECAST: bias=<bullish|bearish|neutral>; confidence=<low|medium|high>
  Write nothing after it. It is read by software.
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

    horizon = horizon_for(interval)

    model = get_model()

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Forecast horizon: the next {horizon} candles of the "
                    f"{interval} timeframe.\n\n"
                    "Analyze the following market summary:\n\n"
                    f"{json.dumps(market_summary, indent=2)}"
                )
            ),
        ]
    )

    parsed, analysis = parse_forecast(response.content)

    errors = []

    if parsed is None:
        forecast = unknown_forecast(interval)
        errors.append(
            "market_agent: the forecast line was missing or invalid, so "
            "this run has no directional forecast and cannot be backtested."
        )
    else:
        forecast = {**parsed, "horizon_candles": horizon}

    update = {
        "market_summary": market_summary,
        "market_analysis": analysis,
        "forecast": forecast,
    }

    unavailable = market_summary["multi_timeframe"]["unavailable"]

    errors.extend(
        f"market_agent: {item['interval']} data unavailable "
        f"({item['error']}), so higher-timeframe confirmation is "
        "incomplete."
        for item in unavailable
    )

    if errors:
        update["errors"] = errors

    return update
