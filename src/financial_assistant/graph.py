from langgraph.graph import END, START, StateGraph

from financial_assistant.nodes.market_agent import market_agent
from financial_assistant.nodes.analysis_agent import analysis_agent
from financial_assistant.nodes.research_agent import research_agent
from financial_assistant.nodes.report_agent import report_agent
from financial_assistant.state import FinancialAssistantState


def build_graph():
    graph_builder = StateGraph(FinancialAssistantState)

    graph_builder.add_node("market_agent", market_agent)
    graph_builder.add_node("analysis_agent", analysis_agent)
    graph_builder.add_node("research_agent", research_agent)
    graph_builder.add_node("report_agent", report_agent)

    graph_builder.add_edge(START, "market_agent")
    graph_builder.add_edge("market_agent", "analysis_agent")
    graph_builder.add_edge("analysis_agent", "research_agent")
    graph_builder.add_edge("research_agent", "report_agent")
    graph_builder.add_edge("report_agent", END)

    return graph_builder.compile()


financial_graph = build_graph()