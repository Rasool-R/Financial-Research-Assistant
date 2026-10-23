"""Short keyword queries for news search.

Search engines work better with a few focused keyword queries than with one
long natural-language sentence, and different queries surface different
drivers (policy, currency, yields, ...).
"""

MAX_USER_QUERY_CHARS = 150

QUERIES_BY_SYMBOL = {
    "XAU/USD": [
        "gold price",
        "gold Federal Reserve rate outlook",
        "gold US dollar real yields",
    ],
    "EUR/USD": [
        "EUR/USD euro dollar",
        "ECB interest rate euro outlook",
        "US dollar Federal Reserve outlook",
    ],
}


def build_queries(symbol: str, user_query: str = "") -> list[str]:
    """Queries to run for an instrument, plus the user's own question."""

    queries = list(
        QUERIES_BY_SYMBOL.get(symbol) or [f"{symbol} news", f"{symbol} outlook"]
    )

    extra = " ".join((user_query or "").split())[:MAX_USER_QUERY_CHARS].strip()

    if extra and extra.lower() not in {query.lower() for query in queries}:
        queries.append(extra)

    return queries
