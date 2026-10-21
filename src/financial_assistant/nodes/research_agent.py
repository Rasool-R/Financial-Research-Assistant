import json

from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model
from financial_assistant.tools.web_search import search_web


SYSTEM_PROMPT = """
You are a financial news research agent.

Your task is to research recent news relevant to the requested
financial instrument.

Rules:
- Use only the search results provided to you.
- Never invent news, dates, URLs, or source names.
- Distinguish reported facts from opinions and speculation.
- Prefer recent and directly relevant sources.
- Mention publication dates when available.
- Cite sources using their exact URLs.
- If sources disagree, explain the disagreement.
- Do not treat news headlines as verified facts.
- Do not make guaranteed market predictions.
- Do not provide personalized financial advice.
- Write the report in English.
- Keep the report concise and structured.
"""


def research_agent(state: dict) -> dict:
    """Research recent news related to the requested instrument."""

    symbol = state.get("symbol", "XAU/USD")
    user_query = state.get("user_query", "")

    query = (
        f"Recent financial news about {symbol}. "
        f"Research question: {user_query}"
    )

    search_results = search_web(
        query=query,
        max_results=5,
    )

    if not search_results:
        return {
            "research_findings": (
                "No relevant web search results were found."
            )
        }

    model = get_model()

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Instrument: {symbol}\n"
                    f"Research question: {user_query}\n\n"
                    "Search results:\n"
                    f"{json.dumps(search_results, indent=2)}"
                )
            ),
        ]
    )

    return {
        "research_findings": response.content,
    }
