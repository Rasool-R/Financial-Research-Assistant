from langgraph.graph import END, START, StateGraph

from financial_assistant.config import MAX_REVISIONS
from financial_assistant.nodes.market_agent import market_agent
from financial_assistant.nodes.analysis_agent import analysis_agent
from financial_assistant.nodes.research_agent import research_agent
from financial_assistant.nodes.report_agent import report_agent
from financial_assistant.nodes.reviewer_agent import reviewer_agent
from financial_assistant.retry import RETRY_POLICY
from financial_assistant.state import FinancialAssistantState


def route_after_review(state: FinancialAssistantState) -> str:
    """Send the report back for revision, or finish the workflow."""

    verdict = (state.get("review") or {}).get("verdict")

    if (
        verdict == "NEEDS_REVISION"
        and state.get("revision_count", 0) < MAX_REVISIONS
    ):
        return "report_agent"

    return END


def build_graph():
    """Build the workflow.

    market_agent -> analysis_agent  --+
                                      +--> report_agent -> reviewer_agent
    research_agent -------------------+          ^               |
                                                 +--NEEDS_REVISION

    Both branches start from START and run in parallel.

    - market/analysis/report failures stop the workflow after transient
      errors have been retried (there is no report without them);
    - research and reviewer degrade gracefully: they retry internally and,
      if they still fail, record the problem in ``errors`` and continue.
    """

    graph_builder = StateGraph(FinancialAssistantState)

    graph_builder.add_node(
        "market_agent", market_agent, retry_policy=RETRY_POLICY
    )
    graph_builder.add_node(
        "analysis_agent", analysis_agent, retry_policy=RETRY_POLICY
    )
    graph_builder.add_node("research_agent", research_agent)
    graph_builder.add_node(
        "report_agent", report_agent, retry_policy=RETRY_POLICY
    )
    graph_builder.add_node("reviewer_agent", reviewer_agent)

    graph_builder.add_edge(START, "market_agent")
    graph_builder.add_edge(START, "research_agent")
    graph_builder.add_edge("market_agent", "analysis_agent")
    graph_builder.add_edge(
        ["analysis_agent", "research_agent"], "report_agent"
    )
    graph_builder.add_edge("report_agent", "reviewer_agent")
    graph_builder.add_conditional_edges(
        "reviewer_agent",
        route_after_review,
        ["report_agent", END],
    )

    return graph_builder.compile()


financial_graph = build_graph()
