from financial_assistant.tools.market_data import SUPPORTED_SYMBOLS
from financial_assistant.tools.news_queries import (
    MAX_USER_QUERY_CHARS,
    QUERIES_BY_SYMBOL,
    build_queries,
)


def test_every_supported_symbol_has_keyword_queries():
    assert set(QUERIES_BY_SYMBOL) == SUPPORTED_SYMBOLS


def test_base_queries_are_short_keywords():
    for queries in QUERIES_BY_SYMBOL.values():
        assert all(len(q.split()) <= 6 for q in queries)


def test_user_question_is_appended_once():
    queries = build_queries("XAU/USD", "  Is gold   overbought? ")

    assert queries[-1] == "Is gold overbought?"
    assert len(queries) == len(QUERIES_BY_SYMBOL["XAU/USD"]) + 1


def test_empty_or_duplicate_user_question_adds_nothing():
    base = QUERIES_BY_SYMBOL["EUR/USD"]

    assert build_queries("EUR/USD", "") == base
    assert build_queries("EUR/USD", "   ") == base
    assert build_queries("EUR/USD", base[0].upper()) == base


def test_long_user_question_is_truncated():
    queries = build_queries("XAU/USD", "x" * 1000)

    assert len(queries[-1]) == MAX_USER_QUERY_CHARS


def test_unknown_symbol_gets_generic_queries():
    assert build_queries("BTC/USD") == ["BTC/USD news", "BTC/USD outlook"]


def test_returned_list_is_a_copy():
    build_queries("XAU/USD").append("mutated")

    assert "mutated" not in QUERIES_BY_SYMBOL["XAU/USD"]
