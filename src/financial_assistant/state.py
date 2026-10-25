import operator
from typing import Annotated, Any, TypedDict


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
    forecast: dict[str, Any]

    analysis_findings: str

    # Research agent output
    research_findings: dict[str, Any]

    # Report and review
    final_report: str
    review: dict[str, Any]
    review_feedback: str
    revision_count: int

    # Non-fatal problems collected from parallel/degrading nodes
    errors: Annotated[list[str], operator.add]
