from typing import Any, TypedDict


class FinancialAssistantState(TypedDict, total=False):
    """Shared state for the financial research workflow."""

    # User request
    user_query: str

    # Market configuration
    symbol: str
    interval: str

    # Market agent outputs
    market_summary: dict[str, Any]
    market_analysis: str

    # Future agent outputs
    research_findings: str
    final_report: str
    review_feedback: str

    # Workflow metadata
    errors: list[str]