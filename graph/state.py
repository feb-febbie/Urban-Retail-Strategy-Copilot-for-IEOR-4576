"""
LangGraph shared state definition for the Urban Retail Strategy Copilot.

All fields are JSON-serializable. DataFrames are stored as JSON strings.
Matplotlib figures are stored as base64-encoded PNG strings.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── User input ──────────────────────────────────────────────────────────
    user_query: str           # e.g. "quiet reading café in Manhattan"
    business_type: str        # "quiet_cafe" | "bar" | "retail"

    # ── Graph control ───────────────────────────────────────────────────────
    next: str                 # routing key: "data_engineer" | "market_researcher" | "done"
    phase: str                # "init" | "analyzing" | "validating" | "done"
    iteration: int            # how many times lead_strategist has been called
    max_iterations: int

    # ── Data layer (populated by Data Engineer) ─────────────────────────────
    mta_records: int          # number of MTA station records fetched
    complaints_records: int   # number of 311 records fetched
    panel_df_json: str        # JSON-serialized panel DataFrame (ZIP-level)
    data_summary: str         # human-readable summary of collected data

    # ── Analysis layer (populated by Lead Strategist EDA) ──────────────────
    analysis_summary: str     # text summary of EDA findings
    candidates: list[dict]    # ranked list of ZIP candidate dicts
    rejected_zips: list[str]  # ZIPs rejected by zoning constraints
    visualization_b64: str    # base64-encoded scatter plot PNG

    # ── Zoning layer (populated by Market Researcher) ───────────────────────
    zoning_results: dict[str, str]  # zip_code → zoning context text
    zoning_verdicts: dict[str, str] # zip_code → "pass" | "fail" | "caution"

    # ── Output ──────────────────────────────────────────────────────────────
    final_hypothesis: str     # the strategic memo / recommendation
    status_logs: list[str]    # running log shown in the UI
    error: str                # error message if something went wrong
    poi_warning: str          # non-fatal POI data quality warning for the user


def initial_state(user_query: str, business_type: str = "quiet_cafe") -> AgentState:
    """Create the initial state for a new analysis run."""
    return AgentState(
        user_query=user_query,
        business_type=business_type,
        next="lead_strategist",
        phase="init",
        iteration=0,
        max_iterations=4,
        rejected_zips=[],
        status_logs=[f"🚀 Starting analysis for: '{user_query}'"],
        candidates=[],
        zoning_results={},
        zoning_verdicts={},
    )
