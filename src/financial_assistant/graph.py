from langgraph.graph import END, START, StateGraph

from financial_assistant.nodes.market_agent import market_agent
from financial_assistant.state import FinancialAssistantState


def build_graph():
    """Build and compile the financial research graph."""

    graph_builder = StateGraph(FinancialAssistantState)

    graph_builder.add_node("market_agent", market_agent)

    graph_builder.add_edge(START, "market_agent")
    graph_builder.add_edge("market_agent", END)

    return graph_builder.compile()


financial_graph = build_graph()
