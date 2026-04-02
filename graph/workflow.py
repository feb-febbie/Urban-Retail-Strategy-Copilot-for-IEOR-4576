"""
LangGraph workflow definition for the Urban Retail Strategy Copilot.

Architecture: Orchestrator-Worker pattern
  - lead_strategist  (orchestrator) ─┐
  - data_engineer    (worker)        │  cyclic until phase = "done"
  - market_researcher (worker)       │
                                     └─→ END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.data_engineer import data_engineer_node
from agents.lead_strategist import lead_strategist_node
from agents.market_researcher import market_researcher_node
from graph.state import AgentState


def _router(state: AgentState) -> str:
    """
    Conditional edge function: reads state['next'] to determine routing.
    Returns the name of the next node (or END sentinel).
    """
    destination = state.get("next", "done")
    if destination in ("done", "end", ""):
        return END
    return destination


def build_graph():
    """
    Compile and return the LangGraph StateGraph.

    Graph edges:
        START → lead_strategist
        lead_strategist → data_engineer       (phase=init)
        lead_strategist → market_researcher   (after EDA)
        lead_strategist → END                 (phase=done)
        data_engineer   → lead_strategist
        market_researcher → lead_strategist
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("lead_strategist", lead_strategist_node)
    graph.add_node("data_engineer", data_engineer_node)
    graph.add_node("market_researcher", market_researcher_node)

    # Entry point
    graph.add_edge(START, "lead_strategist")

    # Lead Strategist is the orchestrator: uses conditional routing
    graph.add_conditional_edges(
        "lead_strategist",
        _router,
        {
            "data_engineer": "data_engineer",
            "market_researcher": "market_researcher",
            "lead_strategist": "lead_strategist",
            END: END,
        },
    )

    # Workers always return to the Lead Strategist
    graph.add_edge("data_engineer", "lead_strategist")
    graph.add_edge("market_researcher", "lead_strategist")

    return graph.compile()


# Module-level compiled graph (created once, reused per Streamlit session)
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
