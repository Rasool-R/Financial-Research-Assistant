import os
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def search_web(
    query: str,
    max_results: int = 5,
    days: int | None = None,
) -> list[dict[str, Any]]:
    """Search recent news and return concise source information.

    ``days`` limits results to articles published in the last N days.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY is not configured.")

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10.")

    if days is not None and days < 1:
        raise ValueError("days must be at least 1.")

    client = TavilyClient(api_key=api_key)

    search_kwargs: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "topic": "news",
        "search_depth": "basic",
    }

    if days is not None:
        search_kwargs["days"] = days

    try:
        response = client.search(**search_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Web search failed: {exc}"
        ) from exc

    results = []

    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "published_date": item.get("published_date"),
                "score": item.get("score"),
            }
        )

    return results
