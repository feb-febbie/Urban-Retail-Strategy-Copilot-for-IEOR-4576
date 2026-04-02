"""
Lead Strategist Agent Node – the orchestrator.

Responsibilities:
  1. (Phase: init)      → Route to Data Engineer to collect NYC data.
  2. (Phase: analyzing) → Write + execute Python EDA code via execute_python tool.
                          Uses Claude tool-use loop to iteratively refine analysis.
  3. (Phase: validating)→ Evaluate zoning verdicts from Market Researcher.
                          Accept best candidate or loop back for more zoning checks.
  4. (Phase: done)      → Format and output the final strategic hypothesis memo.

The node uses Claude claude-sonnet-4-6 with explicit tool use. Claude can call
`execute_python` multiple times to iteratively analyse the data before routing.
"""
from __future__ import annotations

import base64
import io
import json
import numpy as np
import pandas as pd

import config
from graph.state import AgentState
from tools.python_executor import execute_python, format_tool_result
from tools.llm_client import get_llm_client

# ── Tool schemas for Claude ────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "execute_python",
        "description": (
            "Execute Python code for data analysis. "
            "The following variables are pre-loaded:\n"
            "  df   – pandas DataFrame with columns: zip_code, neighborhood, "
            "total_exits, station_count, total_complaints, noise_complaints, "
            "construction_complaints, community_district, low_confidence\n"
            "  pd, np, plt, gpd, stats – standard libraries\n"
            "Assign to 'result' (dict), 'candidates' (list of dicts), "
            "'fig' (matplotlib Figure) to return outputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python code to execute for analysis.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "route_to_market_researcher",
        "description": (
            "Send the top candidate ZIP codes to the Market Researcher agent "
            "for zoning law validation via the RAG knowledge base."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_zip_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of top ZIP codes to validate (most promising first).",
                },
                "analysis_summary": {
                    "type": "string",
                    "description": "Brief summary of EDA findings to record in state.",
                },
            },
            "required": ["candidate_zip_codes", "analysis_summary"],
        },
    },
    {
        "name": "finalize_hypothesis",
        "description": "Output the final strategic memo as the analysis result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "description": (
                        "The full strategic recommendation memo, formatted in markdown. "
                        "Must include: selected ZIP, rejected hypothesis, quantitative evidence, "
                        "zoning validation, confidence level, and limitations."
                    ),
                },
            },
            "required": ["hypothesis"],
        },
    },
]

# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_ANALYZING_BY_TYPE = {
    "quiet_cafe": """You are the Lead Strategist for an Urban Retail Analytics team.
Your task: perform rigorous EDA on Manhattan ZIP-code-level data to identify the optimal
location for a quiet café or reading space.

TARGET PROFILE: High foot traffic (customers passing by) + Low noise complaints (peaceful atmosphere).
A quiet café cannot operate well next to loud bars or construction. Low complaints_per_1k is a strong positive.

PRE-LOADED VARIABLES (do NOT import these — they are already in scope):
  df, pd, np, plt, gpd, stats, norm
  Any `import` statement for these will cause a ModuleNotFoundError and fail the analysis.

CRITICAL CODE FORMATTING RULES:
- Write code with NO leading spaces or indentation on the first line.
- Every line must be syntactically valid Python with consistent 4-space indentation inside blocks.
- Do not wrap code in a function — write it as a flat script.
- Do NOT add any import statements. All libraries are pre-loaded.

You must write and execute Python code to:
1. Drop ZIPs where total_exits < 5000 to prevent ratio blow-up: df = df[df['total_exits'] >= 5000].copy()
2. Winsorize using np.clip() ONLY — do NOT use stats.mstats.winsorize() (returns a masked array incompatible with pandas):
   w_exits = df['total_exits'].clip(upper=df['total_exits'].quantile(0.95))
   w_complaints = df['total_complaints'].clip(upper=df['total_complaints'].quantile(0.95))
3. Compute complaints_per_1k = (winsorized_complaints / winsorized_exits) * 1000
4. Normalize all metrics using the pre-loaded `norm(series)` helper — always returns a Series, safe for zero-variance columns: traffic_norm=norm(log_exits), noise_norm=norm(complaints_per_1k), synergy_norm=norm(df['synergy_score']), sat_norm=norm(df['competitor_density'])
5. score = 0.35*traffic_norm - 0.30*noise_norm + 0.25*synergy_norm - 0.10*sat_norm
6. Target quadrant: HIGH traffic (≥ Q3 log_exits) × LOW noise (≤ Q1 complaints_per_1k).
7. Flag low_confidence ZIPs; deprioritize but do not remove them.
8. Generate a scatter plot (Traffic vs Normalized Noise) with quadrant lines. Save as `fig`.
9. Set `candidates` (top 5 dicts): {zip_code, neighborhood, total_exits, complaints_per_1k, score, station_count, low_confidence}
10. Set `result` dict with thresholds and totals.

Use print() to explain each step.
IMPORTANT: You MUST call the execute_python tool with the code — do NOT write code in a text message.
After execute_python succeeds, call route_to_market_researcher.

ANALYTICAL NOTE: ZIP 10036 (Times Square) has the highest exits but also very high noise —
show explicitly why it falls in the HIGH/HIGH quadrant and is deprioritised for a quiet café.
""",

    "bar": """You are the Lead Strategist for an Urban Retail Analytics team.
Your task: perform rigorous EDA on Manhattan ZIP-code-level data to identify the optimal
location for a bar or nightlife venue.

TARGET PROFILE: High evening foot traffic + Commercially tolerant zoning + Cultural synergy.

The panel DataFrame `df` has these columns:
  total_exits, noise_complaints, total_complaints, station_count, low_confidence,
  synergy_score (POI density of Korean restaurants/karaoke/cultural anchors),
  competitor_density (existing bars/nightclubs per km²),
  lq_competitor (Location Quotient: >1.2 = established hub, <0.8 = untapped market)

KEY INSIGHTS FOR NIGHTLIFE:
- noise_per_1k = residential noise complaints / foot traffic. LOW = commercially tolerant area.
- synergy_score = density of synergy anchors (Korean BBQ, karaoke). HIGH = built-in audience.
- lq_competitor: a hub (>1.2) has existing audience but high competition; untapped (<0.8) has
  lower competition but requires more marketing. Both are valid strategies, name the tradeoff.

PRE-LOADED VARIABLES (do NOT import these — they are already in scope):
  df, pd, np, plt, gpd, stats, norm
  Any `import` statement for these will cause a ModuleNotFoundError and fail the analysis.

CRITICAL CODE FORMATTING RULES:
- Write code with NO leading spaces or indentation on the first line.
- Every line must be syntactically valid Python with consistent 4-space indentation inside blocks.
- Do not wrap code in a function — write it as a flat script.
- Do NOT add any import statements. All libraries are pre-loaded.

You must write and execute Python code to:
1. Drop ZIPs where total_exits < 5000: df = df[df['total_exits'] >= 5000].copy()
2. Winsorize using np.clip() ONLY — do NOT use stats.mstats.winsorize() (returns a masked array incompatible with pandas):
   w_exits = df['total_exits'].clip(upper=df['total_exits'].quantile(0.95))
   w_noise = df['noise_complaints'].clip(upper=df['noise_complaints'].quantile(0.95))
3. Compute noise_per_1k = (w_noise / w_exits) * 1000
4. Apply log1p to exits: log_exits = np.log1p(w_exits)
5. Normalize using the pre-loaded `norm(series)` helper — always returns a Series, safe for zero-variance columns: traffic_norm=norm(log_exits), noise_norm=norm(noise_per_1k), synergy_norm=norm(df['synergy_score']), sat_norm=norm(df['competitor_density'])
6. Compute: score = 0.35*traffic_norm - 0.30*noise_norm + 0.25*synergy_norm - 0.10*sat_norm
7. Flag hubs (lq_competitor > 1.2) and untapped ZIPs (lq_competitor < 0.8).
8. Generate a scatter plot (Traffic vs Residential Noise Sensitivity, sized by synergy_score).
   Annotate top synergy ZIPs. Label quadrants. Save as `fig`.
9. Set `candidates` (top 5 dicts): {zip_code, neighborhood, total_exits, complaints_per_1k,
   score, station_count, low_confidence, synergy_score, lq_competitor}
10. Set `result` dict: include the top synergy ZIP and whether it is a hub or untapped.

STRICT OUTPUT RULE: Your FIRST action must be a call to execute_python. Do NOT write any
explanatory text, greeting, or plan before the tool call. Output ONLY the tool call.
After execute_python succeeds, call route_to_market_researcher.
""",

    "retail": """You are the Lead Strategist for an Urban Retail Analytics team.
Your task: perform rigorous EDA on Manhattan ZIP-code-level data to identify the optimal
location for a retail store.

TARGET PROFILE: High foot traffic (shoppers) + Moderate noise environment (commercial street, not chaotic).
Retail benefits from busy streets but not from chaotic, high-complaint zones that deter shoppers.

PRE-LOADED VARIABLES (do NOT import these — they are already in scope):
  df, pd, np, plt, gpd, stats, norm
  Any `import` statement for these will cause a ModuleNotFoundError and fail the analysis.

CRITICAL CODE FORMATTING RULES:
- Write code with NO leading spaces or indentation on the first line.
- Every line must be syntactically valid Python with consistent 4-space indentation inside blocks.
- Do not wrap code in a function — write it as a flat script.
- Do NOT add any import statements. All libraries are pre-loaded.

You must write and execute Python code to:
1. Drop ZIPs where total_exits < 5000: df = df[df['total_exits'] >= 5000].copy()
2. Winsorize using np.clip() ONLY — do NOT use stats.mstats.winsorize() (returns a masked array incompatible with pandas):
   w_exits = df['total_exits'].clip(upper=df['total_exits'].quantile(0.95))
   w_complaints = df['total_complaints'].clip(upper=df['total_complaints'].quantile(0.95))
3. Compute complaints_per_1k = (winsorized_complaints / winsorized_exits) * 1000
4. Normalize all metrics using the pre-loaded `norm(series)` helper — always returns a Series, safe for zero-variance columns: traffic_norm=norm(log_exits), noise_norm=norm(complaints_per_1k), synergy_norm=norm(df['synergy_score']), sat_norm=norm(df['competitor_density'])
5. score = 0.35*traffic_norm - 0.30*noise_norm + 0.25*synergy_norm - 0.10*sat_norm
6. Target quadrant: HIGH traffic (≥ Q3 exits) × MODERATE noise (≤ median complaints_per_1k).
7. Flag low_confidence ZIPs; deprioritize but do not remove them.
8. Generate a scatter plot (Traffic vs Normalized Noise) with quadrant lines. Save as `fig`.
9. Set `candidates` (top 5 dicts): {zip_code, neighborhood, total_exits, complaints_per_1k, score, station_count, low_confidence}
10. Set `result` dict with thresholds and totals.

Use print() to explain each step.
IMPORTANT: You MUST call the execute_python tool with the code — do NOT write code in a text message.
After execute_python succeeds, call route_to_market_researcher.
""",
}

# Default fallback for unknown business types
_SYSTEM_ANALYZING_BY_TYPE["default"] = _SYSTEM_ANALYZING_BY_TYPE["retail"]


_SYSTEM_VALIDATING_BY_TYPE = {
    "quiet_cafe": """You are the Lead Strategist for an Urban Retail Analytics team.
You have completed the quantitative EDA and received zoning verdicts from the Market Researcher.

Your task: evaluate the combined evidence and finalize a recommendation for a quiet café or bookstore.
Call finalize_hypothesis with a strategic memo in markdown that follows this EXACT structure:

**Opening (consulting-style, decisive):**
Lead with: "Recommendation: Launch in ZIP XXXXX (Neighborhood). It offers the best balance of
high daytime foot traffic, low environmental noise, and clear zoning feasibility for a quiet,
experience-driven retail concept."

**Key Insight section (required):**
Include a "## Key Insight" section with this framing:
"The optimal location for a quiet café is not the highest-traffic area, but the location that
balances strong pedestrian flow with minimal environmental noise. While [highest-traffic ZIP]
has the most exits, its higher noise levels reduce suitability — making [recommended ZIP] the
best overall trade-off between demand and experience quality."

**Times Square rejection (required if 10036 appears in candidates):**
Explicitly note: "High-traffic areas such as Times Square (10036) fall into the high traffic /
high noise quadrant and are unsuitable for a quiet, experience-driven concept."

**Scoring formula (required, explain it):**
State: score = 0.35·traffic − 0.30·noise + 0.25·synergy − 0.10·saturation
Define each term. Define synergy as "density of libraries, bookshops, and cultural anchors per km²."
Add: "All features are min-max normalized before scoring, ensuring comparability across scales."

**Trade-off statement (required):**
Include: "This recommendation reflects a deliberate trade-off: the highest-traffic ZIP is bypassed
in favour of a quieter location with better experience-environment fit for the concept."

**Candidate table:**
For each candidate: ZIP, neighborhood, total exits, complaints per 1k exits, composite score,
zoning verdict. Describe noise as "Noise Environment" (Low/Moderate/High) — not as a percentile.

**Zoning notes:**
If a ZIP received "caution" due to manufacturing overlay or mixed-use zoning, describe it as:
"Mixed-use zoning with manufacturing legacy — may impose unexpected restrictions on low-impact
retail." Do NOT reproduce internal tag names like 'quiet_residential_cafe' verbatim.

**Conclusion:**
Name which ZIPs were eliminated by noise profile vs. which by zoning. Separate these two reasons.

**Limitations:**
7-day data window, no rent data, not legal advice, on-site zoning verification recommended.

CRITICAL SCORING RULE: The recommended ZIP must have the highest composite score among
candidates with a non-"fail" zoning verdict. A zoning "pass" does NOT override the
objective function. If the highest-scoring non-fail candidate has score < 0.05, state the
EDA was inconclusive rather than fabricating a recommendation.
""",

    "bar": """You are the Lead Strategist for an Urban Retail Analytics team.
You have completed the quantitative EDA and received zoning verdicts from the Market Researcher.

Your task: evaluate the combined evidence and finalize a recommendation for a bar/nightlife venue.
Call finalize_hypothesis with a strategic memo in markdown that follows this EXACT structure:

**Opening (consulting-style, decisive):**
Lead with: "Recommendation: Launch in ZIP XXXXX (Neighborhood). It is the only top-tier candidate
that combines strong demand signals with clear regulatory feasibility for late-night nightlife."

**Key Insight section (required):**
Include a "## Key Insight" section with this framing:
"The optimal location is not the highest-demand area, but the highest-demand area that remains
feasible under regulatory constraints. While several ZIP codes show strong traffic signals,
zoning restrictions eliminate them as viable nightlife locations, leaving [ZIP] as the best
risk-adjusted choice."

**Scoring formula (required, explain it):**
State: score = 0.35·traffic − 0.30·noise + 0.25·synergy − 0.10·saturation
Define each term. Define synergy as "density of Korean restaurants, karaoke venues, and
culturally aligned nightlife anchors per km²."
Add: "All features are min-max normalized before scoring, ensuring comparability across scales."

**Commercial Tolerance (required framing):**
Describe noise profiles as "Commercial Tolerance" (high = low complaints per 1k exits = good for
nightlife). Add: "Commercial tolerance is derived from normalized noise complaints per 1,000
exits. Moderate levels indicate active nightlife ecosystems; extreme levels signal regulatory risk."
Do NOT use "residential sensitivity percentile" — it reads backward to an executive audience.

**Trade-off statement (required):**
Include: "This recommendation reflects a deliberate trade-off: slightly lower peak traffic is
accepted in exchange for significantly lower regulatory risk and higher execution certainty."

**Candidate table:**
For each candidate: ZIP, neighborhood, traffic rank, commercial tolerance (High/Medium/Low),
synergy score, saturation (LQ), composite score, zoning verdict.

**Conclusion:**
State explicitly which high-scoring ZIPs were eliminated by regulatory risk (not lack of demand).
If Times Square (10036) has PASS zoning, acknowledge it as a valid alternative with its trade-offs.

**Limitations:**
7-day data window, no rent data, not legal advice, SLA license required separately.

CRITICAL SCORING RULE: The recommended ZIP must have the highest composite score among
candidates with a non-"fail" zoning verdict. A zoning "pass" does NOT override the
objective function. If the highest-scoring non-fail candidate has score < 0.05, state the
EDA was inconclusive rather than fabricating a recommendation.
""",

    "retail": """You are the Lead Strategist for an Urban Retail Analytics team.
You have completed the quantitative EDA and received zoning verdicts from the Market Researcher.

Your task: evaluate the combined evidence and finalize a recommendation for a retail store.
Call finalize_hypothesis with a strategic memo in markdown that:
  - Opens with a decisive consulting-style recommendation: "Recommendation: Launch in ZIP XXXXX
    (Neighborhood). It offers the best combination of foot traffic, noise environment, and
    zoning feasibility for a retail concept."
  - Includes a "## Key Insight" section explaining why the top-ranked ZIP beats higher-traffic
    alternatives (trade-off between volume and noise/zoning feasibility).
  - Explicitly states the scoring formula:
    score = 0.35·traffic − 0.30·noise + 0.25·synergy − 0.10·saturation
    Define each term. Define synergy as "density of restaurants, cafés, and tourist attractions
    per km²." Add: "All features are min-max normalized before scoring."
  - Includes a candidate table: ZIP, neighborhood, total exits, complaints per 1k, score, zoning.
  - Notes which ZIPs were eliminated by noise vs. which by zoning (separate these reasons).
  - Notes confidence level and key limitations (7-day window, no rent data, not legal advice).

CRITICAL SCORING RULE: The recommended ZIP must have the highest composite score among
candidates with a non-"fail" zoning verdict. A zoning "pass" does NOT override the
objective function. If the highest-scoring non-fail candidate has score < 0.05, state the
EDA was inconclusive rather than fabricating a recommendation.
""",
}
_SYSTEM_VALIDATING_BY_TYPE["default"] = _SYSTEM_VALIDATING_BY_TYPE["retail"]


# ── Main node function ─────────────────────────────────────────────────────────

def lead_strategist_node(state: AgentState) -> dict:
    """LangGraph node: orchestrates the full analysis lifecycle."""
    try:
        phase = state.get("phase", "init")
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", config.MAX_ITERATIONS)
        logs = list(state.get("status_logs", []))

        # ── Guard: max iterations ──────────────────────────────────────────
        if iteration >= max_iter and phase != "done":
            logs.append(f"🧭 [Lead Strategist] Max iterations ({max_iter}) reached. Forcing finalisation.")
            return _force_finalize(state, logs)

        # ── Phase dispatch ─────────────────────────────────────────────────
        if phase == "init":
            return _handle_init(state, logs)
        elif phase == "analyzing":
            return _handle_analyzing(state, logs)
        elif phase == "validating":
            return _handle_validating(state, logs)
        else:
            return {"status_logs": logs, "next": "done"}

    except Exception as exc:
        logs = list(state.get("status_logs", []))
        logs.append(f"❌ [Lead Strategist] Fatal error: {exc}")
        return {
            "status_logs": logs,
            "next": "done",
            "phase": "done",
            "error": str(exc),
            "final_hypothesis": f"## Analysis Failed\n\n**Error:** {exc}\n\nCheck that an LLM provider is configured (Gemini ADC, or an API key in Cloud Run secrets).",
        }


# ── Phase handlers ─────────────────────────────────────────────────────────────

def _fallback_candidates(panel_json: str, business_type: str) -> list[dict]:
    """
    Deterministic fallback: rank ZIP codes from the panel without LLM.
    Used when the LLM fails to produce candidates via execute_python.
    """
    try:
        df = pd.read_json(io.StringIO(panel_json))
        if df.empty:
            return []

        # Winsorize exits and complaints at 95th pct
        # Drop ZIPs with near-zero traffic — they blow up the noise ratio
        # (e.g. 135 complaints / 1 exit * 1000 = 135,000 which destroys normalization)
        MIN_EXITS = 5_000
        df = df[df["total_exits"] >= MIN_EXITS].copy()
        if df.empty:
            return []

        exits_cap = df["total_exits"].quantile(0.95)
        df["w_exits"] = df["total_exits"].clip(upper=exits_cap)
        comp_col = "noise_complaints" if business_type == "bar" else "total_complaints"
        if comp_col not in df.columns:
            comp_col = "total_complaints"
        comp_cap = df[comp_col].quantile(0.95)
        df["w_comp"] = df[comp_col].clip(upper=comp_cap)

        # Log-transform traffic to model diminishing returns
        # (doubling exits from 50k→100k matters less than 5k→10k)
        df["log_exits"] = np.log1p(df["w_exits"])

        # Normalized noise (safe — exits guaranteed >= 5000 by floor above)
        df["complaints_per_1k"] = (df["w_comp"] / df["w_exits"]) * 1000

        # Quadrant thresholds (on log scale for consistency)
        traffic_q3 = df["log_exits"].quantile(0.75)
        noise_q1 = df["complaints_per_1k"].quantile(0.25)

        target = df[(df["log_exits"] >= traffic_q3) & (df["complaints_per_1k"] <= noise_q1)].copy()
        if target.empty:
            target = df.nlargest(5, "log_exits")

        # Min-max normalize all features to [0, 1] so weights are scale-independent
        def _norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-9)

        target["traffic_norm"] = _norm(target["log_exits"])
        target["noise_norm"] = _norm(target["complaints_per_1k"])
        target["synergy_norm"] = _norm(target["synergy_score"]) if "synergy_score" in target.columns else 0.0
        target["sat_norm"] = _norm(target["competitor_density"]) if "competitor_density" in target.columns else 0.0

        # Weighted scoring:
        # Traffic (0.35): primary revenue driver (log-scaled for diminishing returns)
        # Noise (−0.30): critical constraint — quiet environment / regulatory risk
        # Synergy (0.25): complementary business spillover from nearby POIs
        # Saturation (−0.10): penalized but less critical due to differentiation potential
        target["score"] = (
            0.35 * target["traffic_norm"]
            - 0.30 * target["noise_norm"]
            + 0.25 * target["synergy_norm"]
            - 0.10 * target["sat_norm"]
        )
        target = target.nlargest(5, "score")

        candidates = []
        for _, row in target.iterrows():
            zip_code = str(row.get("zip_code", ""))
            candidates.append({
                "zip_code": zip_code,
                "neighborhood": config.ZIP_NAMES.get(zip_code, zip_code),
                "total_exits": int(row.get("total_exits", 0)),
                "complaints_per_1k": round(float(row.get("complaints_per_1k", 0)), 2),
                "score": round(float(row.get("score", 0)), 4),
                "station_count": int(row.get("station_count", 1)),
                "low_confidence": bool(row.get("low_confidence", False)),
                "synergy_score": round(float(row.get("synergy_score", 0)), 2),
                "lq_competitor": round(float(row.get("lq_competitor", 0)), 2),
            })
        return candidates
    except Exception:
        return []


def _compute_sensitivity(panel_json: str, candidates: list[dict], business_type: str) -> str:
    """
    Sensitivity analysis: perturb each weight by ±10% across 30 configurations and
    check whether the top-3 ranked ZIPs remain in the top-5.
    Returns a human-readable stability note.
    """
    try:
        if not candidates:
            return ""
        df = pd.read_json(io.StringIO(panel_json))
        if df.empty or len(df) < 4:
            return ""

        # Rebuild the same features used in _fallback_candidates (including MIN_EXITS floor)
        df = df[df["total_exits"] >= 5_000].copy()
        if df.empty:
            return ""
        exits_cap = df["total_exits"].quantile(0.95)
        df["w_exits"] = df["total_exits"].clip(upper=exits_cap)
        comp_col = "noise_complaints" if business_type == "bar" else "total_complaints"
        if comp_col not in df.columns:
            comp_col = "total_complaints"
        comp_cap = df[comp_col].quantile(0.95)
        df["w_comp"] = df[comp_col].clip(upper=comp_cap)
        df["log_exits"] = np.log1p(df["w_exits"])
        df["complaints_per_1k"] = (df["w_comp"] / df["w_exits"]) * 1000

        def _norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-9)

        df["traffic_norm"] = _norm(df["log_exits"])
        df["noise_norm"] = _norm(df["complaints_per_1k"])
        df["synergy_norm"] = _norm(df["synergy_score"]) if "synergy_score" in df.columns else 0.0
        df["sat_norm"] = _norm(df["competitor_density"]) if "competitor_density" in df.columns else 0.0
        df["zip_code"] = df["zip_code"].astype(str)

        base_weights = [0.35, -0.30, 0.25, -0.10]
        top3_zips = {str(c["zip_code"]) for c in candidates[:3]}

        # Generate 30 weight configs by perturbing each weight ±10%
        stable_count = 0
        total = 0
        for i in range(4):
            for delta in (-0.10, -0.05, 0.05, 0.10):
                perturbed = base_weights.copy()
                perturbed[i] = round(perturbed[i] * (1 + delta), 4)
                df["_score"] = (
                    perturbed[0] * df["traffic_norm"]
                    + perturbed[1] * df["noise_norm"]
                    + perturbed[2] * df["synergy_norm"]
                    + perturbed[3] * df["sat_norm"]
                )
                top5 = set(df.nlargest(5, "_score")["zip_code"].astype(str))
                if top3_zips.issubset(top5):
                    stable_count += 1
                total += 1

        pct = int(100 * stable_count / total)
        return (
            f"Ranking stable in **{stable_count}/{total}** weight perturbation tests "
            f"({pct}% — ±10% on each weight). "
            + ("Top candidates are robust." if pct >= 70 else "Some sensitivity to weight choice — review manually.")
        )
    except Exception:
        return ""


def _generate_eda_chart(panel_json: str, business_type: str, candidates: list[dict]) -> str:
    """
    Generate a clean quadrant scatter chart from panel data.
    Returns base64-encoded PNG, or empty string on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.read_json(io.StringIO(panel_json))
        if df.empty:
            return ""

        comp_col = "noise_complaints" if business_type == "bar" else "total_complaints"
        if comp_col not in df.columns:
            comp_col = "total_complaints"

        exits_cap = df["total_exits"].quantile(0.95)
        df["w_exits"] = df["total_exits"].clip(upper=exits_cap)
        comp_cap = df[comp_col].quantile(0.95)
        df["w_comp"] = df[comp_col].clip(upper=comp_cap)
        df["complaints_per_1k"] = (df["w_comp"] / (df["w_exits"] + 1)) * 1000

        traffic_q3 = df["w_exits"].quantile(0.75)
        noise_q1 = df["complaints_per_1k"].quantile(0.25)

        candidate_zips = {str(c["zip_code"]) for c in candidates}
        colors = ["#2563eb" if str(row["zip_code"]) in candidate_zips else "#94a3b8"
                  for _, row in df.iterrows()]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df["w_exits"], df["complaints_per_1k"], c=colors, s=80, alpha=0.85, zorder=3)

        # Quadrant lines
        ax.axvline(traffic_q3, color="#ef4444", linestyle="--", linewidth=1.2, label=f"Traffic Q3 ({int(traffic_q3):,})")
        ax.axhline(noise_q1, color="#22c55e", linestyle="--", linewidth=1.2, label=f"Noise Q1 ({noise_q1:.1f})")

        # Cap Y-axis at a readable percentile (exclude extreme low-traffic outliers)
        y_cap = df["complaints_per_1k"].quantile(0.90) * 1.3
        ax.set_ylim(-y_cap * 0.05, y_cap)
        ax.set_xlim(-max(df["w_exits"]) * 0.05, max(df["w_exits"]) * 1.08)

        # ZIP labels
        for _, row in df.iterrows():
            if row["complaints_per_1k"] <= y_cap:
                ax.annotate(
                    str(row["zip_code"]),
                    (row["w_exits"], row["complaints_per_1k"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points",
                )

        # Target quadrant shading
        ax.axvspan(traffic_q3, ax.get_xlim()[1], ymin=0,
                   ymax=(noise_q1 - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                   alpha=0.07, color="#2563eb")
        ax.text(traffic_q3 + (ax.get_xlim()[1] - traffic_q3) * 0.05, noise_q1 * 0.5,
                "✓ Target\nHigh traffic\nLow noise", fontsize=8, color="#1d4ed8",
                va="center")

        ax.set_xlabel("Weekly Subway Exits (winsorized)", fontsize=11)
        ax.set_ylabel("Complaints per 1k Exits (winsorized)", fontsize=11)
        ax.set_title(f"Manhattan ZIP Quadrant Analysis — Traffic vs Normalized Noise", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def _handle_init(state: AgentState, logs: list) -> dict:
    logs.append(
        f"🧭 [Lead Strategist] Initiating analysis for: '{state.get('user_query', '')}'\n"
        f"   Business type: {state.get('business_type', 'quiet_cafe')}\n"
        f"   → Delegating to Data Engineer to collect NYC Open Data."
    )
    return {
        "status_logs": logs,
        "next": "data_engineer",
        "phase": "init",
        "iteration": state.get("iteration", 0) + 1,
    }


_CODE_EXTRACTION_PROMPT_BY_TYPE = {
    "quiet_cafe": """Write Python code in a single ```python block to analyse the panel DataFrame `df`.

Available columns: total_exits, total_complaints, synergy_score, competitor_density,
lq_competitor, station_count, low_confidence, zip_code, neighborhood.

The code MUST:
1. Drop ZIPs where total_exits < 5000 (near-zero traffic causes ratio blow-up: e.g. 135 complaints / 1 exit * 1000 = 135,000 which destroys normalization).
2. Winsorize total_exits and total_complaints at the 95th percentile.
2. Apply log1p to winsorized exits: log_exits = np.log1p(winsorized_exits)
   (models diminishing returns — doubling traffic from 50k→100k matters less than 5k→10k)
3. Compute complaints_per_1k = (winsorized_complaints / (winsorized_exits + 1)) * 1000
4. Target: HIGH traffic (≥ Q3 log_exits) × LOW noise (≤ Q1 complaints_per_1k).
5. Normalise each metric to [0,1] using min-max scaling so weights are scale-independent.
6. Compute weighted composite score:
   score = 0.35*traffic_norm - 0.30*noise_norm + 0.25*synergy_norm - 0.10*saturation_norm
   where:
     traffic_norm    = normalised log_exits
     noise_norm      = normalised complaints_per_1k
     synergy_norm    = normalised synergy_score (libraries, bookshops nearby)
     saturation_norm = normalised competitor_density
   IMPORTANT: use (x - x.min()) / (x.max() - x.min() + 1e-9) to avoid division by zero when all values are equal (e.g. all synergy_score = 0)
7. Flag ZIPs where lq_competitor > 1.5 as "saturated hub" and lq_competitor < 0.5 as "untapped".
8. Set `candidates` = top 5 dicts: zip_code (str), neighborhood, total_exits (int),
   complaints_per_1k (float), score (float), station_count (int), low_confidence (bool),
   synergy_score (float), lq_competitor (float)
9. Set `result` = dict with thresholds.
10. Print findings summary.

After the code block output ONLY this JSON:
{"zip_codes": ["10025", "10017", ...], "analysis_summary": "one sentence"}
""",
    "bar": """Write Python code in a single ```python block to analyse the panel DataFrame `df`.

Available columns: total_exits, noise_complaints, total_complaints, synergy_score,
competitor_density, lq_competitor, station_count, low_confidence, zip_code, neighborhood.

The code MUST:
1. Drop ZIPs where total_exits < 5000 (near-zero traffic causes ratio blow-up: 135 complaints / 1 exit * 1000 = 135,000 which destroys normalization).
2. Winsorize total_exits and noise_complaints at the 95th percentile.
2. Apply log1p to winsorized exits: log_exits = np.log1p(winsorized_exits)
   (models diminishing returns — a venue in a 200k-exit hub isn't twice as good as a 100k hub)
3. Compute noise_per_1k = (winsorized_noise_complaints / (winsorized_exits + 1)) * 1000
   LOW noise_per_1k in a HIGH-traffic zone = commercially tolerant area (good for a bar).
4. Normalise each metric to [0,1] using min-max scaling so weights are scale-independent.
5. Compute weighted composite score using this formula:
   score = 0.35*traffic_norm - 0.30*noise_norm + 0.25*synergy_norm - 0.10*saturation_norm
   where:
     traffic_norm    = normalised log_exits
     noise_norm      = normalised noise_per_1k (higher = more residential risk)
     synergy_norm    = normalised synergy_score (Korean restaurants, karaoke, etc.)
     saturation_norm = normalised competitor_density
   IMPORTANT: use (x - x.min()) / (x.max() - x.min() + 1e-9) to avoid division by zero when all values are equal (e.g. all synergy_score = 0)
5. Flag ZIPs where lq_competitor > 1.5 as "established hub" and lq_competitor < 0.5 as "untapped".
6. Set `candidates` = top 5 dicts: zip_code (str), neighborhood, total_exits (int),
   complaints_per_1k (float), score (float), station_count (int), low_confidence (bool),
   synergy_score (float), lq_competitor (float)
7. Set `result` = dict with thresholds and top synergy ZIPs.
8. Print: which ZIPs are hubs vs untapped, and the top synergy ZIP (likely Koreatown area).
   Do NOT pre-reject Times Square — evaluate it against synergy and noise scores objectively.

After the code block output ONLY this JSON:
{"zip_codes": ["10036", "10001", ...], "analysis_summary": "one sentence"}
""",
    "retail": """Write Python code in a single ```python block to analyse the panel DataFrame `df`.

Available columns: total_exits, total_complaints, synergy_score, competitor_density,
lq_competitor, station_count, low_confidence, zip_code, neighborhood.

The code MUST:
1. Drop ZIPs where total_exits < 5000 (near-zero traffic causes ratio blow-up: e.g. 135 complaints / 1 exit * 1000 = 135,000 which destroys normalization).
2. Winsorize total_exits and total_complaints at the 95th percentile.
2. Apply log1p to winsorized exits: log_exits = np.log1p(winsorized_exits)
   (models diminishing returns — a 150k-exit ZIP isn't proportionally better than an 80k-exit ZIP)
3. Compute complaints_per_1k = (winsorized_complaints / (winsorized_exits + 1)) * 1000
4. Target: HIGH traffic (≥ Q3 log_exits) × MODERATE noise (≤ median complaints_per_1k).
5. Normalise each metric to [0,1] using min-max scaling so weights are scale-independent.
6. Compute weighted composite score:
   score = 0.35*traffic_norm - 0.30*noise_norm + 0.25*synergy_norm - 0.10*saturation_norm
   where:
     traffic_norm    = normalised log_exits
     noise_norm      = normalised complaints_per_1k
     synergy_norm    = normalised synergy_score (restaurants, attractions nearby)
     saturation_norm = normalised competitor_density
   IMPORTANT: use (x - x.min()) / (x.max() - x.min() + 1e-9) to avoid division by zero when all values are equal (e.g. all synergy_score = 0)
6. Flag ZIPs where lq_competitor > 1.5 as "saturated hub" and lq_competitor < 0.5 as "untapped".
7. Set `candidates` = top 5 dicts: zip_code (str), neighborhood, total_exits (int),
   complaints_per_1k (float), score (float), station_count (int), low_confidence (bool),
   synergy_score (float), lq_competitor (float)
8. Set `result` = dict with thresholds.
9. Print findings summary.

After the code block output ONLY this JSON:
{"zip_codes": ["10025", "10017", ...], "analysis_summary": "one sentence"}
""",
}


def _code_extraction_eda(
    state: AgentState, df: "pd.DataFrame", logs: list
) -> tuple:
    """
    Fallback EDA: ask LLM to write code in a markdown block, extract and run it.
    Returns (candidates, visualization_b64, analysis_summary, routing_action).
    """
    import re

    business_type = state.get("business_type", "quiet_cafe")
    extraction_prompt = _CODE_EXTRACTION_PROMPT_BY_TYPE.get(
        business_type, _CODE_EXTRACTION_PROMPT_BY_TYPE["retail"]
    )
    user_msg = (
        f"Business type: {business_type}\n"
        f"DataFrame columns: {df.columns.tolist()}\n"
        f"Head:\n{df.head(5).to_string(index=False)}\n\n"
        + extraction_prompt
    )

    candidates: list[dict] = []
    visualization_b64 = ""
    analysis_summary = ""
    routing_action = None

    try:
        resp = get_llm_client().messages_create(
            system="You are a Python data analyst. Follow instructions exactly.",
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=3000,
        )
        raw = " ".join(
            b.text for b in resp.content if b.type == "text"
        ).strip()

        # Extract ```python block
        code_match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            logs.append("   🔧 execute_python called (code-extraction mode)")
            exec_result = execute_python(code, df)
            if exec_result.get("candidates"):
                candidates = exec_result["candidates"]
            if exec_result.get("visualization_b64"):
                visualization_b64 = exec_result["visualization_b64"]
            if exec_result.get("stdout"):
                logs.append(f"      stdout: {exec_result['stdout'].strip()[:400]}")

        # Extract routing JSON — validate zip_codes are real 5-digit strings
        json_match = re.search(r'\{"zip_codes".*?\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                raw_zips = data.get("zip_codes", [])
                # Filter: must be 5-digit numeric strings (not Python expressions)
                zip_codes = [str(z) for z in raw_zips if re.match(r"^\d{5}$", str(z))]
                analysis_summary = data.get("analysis_summary", "")
            except Exception:
                zip_codes = []

            # If LLM JSON had bad zips, derive from candidates list
            if not zip_codes and candidates:
                zip_codes = [str(c["zip_code"]) for c in candidates[:5]]

            if zip_codes:
                routing_action = "market_researcher"
                logs.append(f"   🗺️  Routing to Market Researcher for: {', '.join(zip_codes)}")
                # Reorder candidates to match zip_codes if possible
                if candidates and zip_codes:
                    zip_map = {str(c["zip_code"]): c for c in candidates}
                    ordered = [zip_map[z] for z in zip_codes if z in zip_map]
                    candidates = ordered or candidates

        # Ensure at least 3 candidates by relaxing quadrant filter if needed
        if len(candidates) < 3:
            fallback = _fallback_candidates(
                state.get("panel_df_json", ""), state.get("business_type", "quiet_cafe")
            )
            # Merge: keep existing, fill from fallback
            existing_zips = {str(c["zip_code"]) for c in candidates}
            for c in fallback:
                if str(c["zip_code"]) not in existing_zips:
                    candidates.append(c)
                    existing_zips.add(str(c["zip_code"]))
                if len(candidates) >= 5:
                    break
            if candidates and not routing_action:
                routing_action = "market_researcher"
                zip_codes = [str(c["zip_code"]) for c in candidates[:5]]
                logs.append(f"   🗺️  Routing to Market Researcher for: {', '.join(zip_codes)}")

    except Exception as exc:
        logs.append(f"   ❌ Code-extraction EDA failed: {exc}")

    return candidates, visualization_b64, analysis_summary, routing_action


def _handle_analyzing(state: AgentState, logs: list) -> dict:
    """
    Claude writes and executes EDA code via the execute_python tool.
    Multi-turn loop: Claude can call execute_python up to MAX_TOOL_ROUNDS times.
    """
    logs.append(
        f"🧭 [Lead Strategist] Starting EDA (iteration {state.get('iteration', 0)+1}). "
        f"Data summary: {state.get('data_summary', 'N/A')}"
    )

    # Load panel DataFrame from state
    panel_json = state.get("panel_df_json", "")
    if not panel_json:
        logs.append("   ❌ No panel data available. Cannot run EDA.")
        return {
            "status_logs": logs,
            "next": "done",
            "phase": "done",
            "error": "No panel data",
        }

    try:
        df = pd.read_json(io.StringIO(panel_json))
    except Exception as exc:
        logs.append(f"   ❌ Failed to load panel: {exc}")
        return {"status_logs": logs, "next": "done", "phase": "done", "error": str(exc)}

    # Build the initial message for Claude
    user_msg = (
        f"User query: '{state.get('user_query', '')}'\n"
        f"Business type: {state.get('business_type', 'quiet_cafe')}\n"
        f"Data summary: {state.get('data_summary', '')}\n\n"
        f"DataFrame shape: {df.shape}\n"
        f"Columns: {df.columns.tolist()}\n"
        f"Head:\n{df.head(5).to_string(index=False)}\n\n"
        f"Please perform the complete EDA now and generate candidates."
    )

    messages = [{"role": "user", "content": user_msg}]

    # State to accumulate across tool rounds
    candidates: list[dict] = []
    visualization_b64 = ""
    analysis_summary = ""
    routing_action = None
    error_count = 0

    for _round in range(config.MAX_TOOL_ROUNDS):
        try:
            business_type = state.get("business_type", "quiet_cafe")
            system_analyzing = _SYSTEM_ANALYZING_BY_TYPE.get(
                business_type, _SYSTEM_ANALYZING_BY_TYPE["default"]
            )
            # Force execute_python on round 0; after that let the model choose freely
            # (it may want to call route_to_market_researcher or finalize next).
            resp = get_llm_client().messages_create(
                system=system_analyzing,
                messages=messages,
                tools=_TOOLS,
                max_tokens=4096,
                force_tool="execute_python" if _round == 0 else None,
            )
        except Exception as exc:
            err_str = str(exc)
            # Groq sometimes rejects tool_use with 400 (legacy format) — retry as plain code
            if "tool_use_failed" in err_str or "400" in err_str:
                logs.append("   ⚠️  Tool-use rejected (model compat issue) — retrying as code extraction")
                candidates, visualization_b64, analysis_summary, routing_action = _code_extraction_eda(
                    state, df, logs
                )
            else:
                logs.append(f"   ❌ LLM API error: {exc}")
            break

        # Append assistant turn
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []

        def _harvest_exec(exec_result: dict, source: str) -> None:
            """Extract candidates and chart from an execute_python result into enclosing scope."""
            nonlocal candidates, visualization_b64
            if exec_result.get("candidates"):
                raw_cands = exec_result["candidates"]
                valid_cands = [
                    c for c in raw_cands
                    if c.get("score") is not None and str(c.get("score", "")) != "nan"
                ]
                all_zero = valid_cands and all(
                    abs(float(c.get("score", 0))) < 1e-9 for c in valid_cands
                )
                if not valid_cands or all_zero:
                    reason = "all-zero scores" if all_zero else "NaN scores"
                    logs.append(f"   ⚠️  {source}: {reason} — using deterministic fallback ranking")
                    candidates = _fallback_candidates(panel_json, state.get("business_type", "quiet_cafe"))
                else:
                    candidates = valid_cands
            if exec_result.get("visualization_b64"):
                visualization_b64 = exec_result["visualization_b64"]

        import re as _re

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                text = block.text.strip()
                logs.append(f"   💬 {text[:200]}")

                # LLM sometimes writes code in a markdown block instead of calling execute_python.
                # Detect this, extract the code, and run it so scoring doesn't silently skip.
                if not candidates:
                    code_match = _re.search(r"```python\s*(.*?)```", text, _re.DOTALL)
                    if code_match:
                        extracted = code_match.group(1).strip()
                        logs.append("   🔧 Code detected in text block — executing automatically")
                        exec_result = execute_python(extracted, df)
                        if exec_result.get("stdout"):
                            logs.append(f"      stdout: {exec_result['stdout'].strip()[:400]}")
                        if not exec_result.get("success"):
                            logs.append(f"   ⚠️  Text-block code error: {exec_result.get('error','')[:200]}")
                        _harvest_exec(exec_result, "text-block code")

            elif block.type == "tool_use":
                if block.name == "execute_python":
                    code = block.input.get("code", "")
                    logs.append(f"   🔧 execute_python called (round {_round + 1})")

                    exec_result = execute_python(code, df)
                    _harvest_exec(exec_result, f"execute_python round {_round + 1}")
                    if exec_result.get("stdout"):
                        logs.append(f"      stdout: {exec_result['stdout'].strip()[:400]}")

                    # If code errored, increment error count for retry budget
                    if not exec_result.get("success"):
                        error_count += 1
                        logs.append(
                            f"   ⚠️  Code error (attempt {error_count}): {exec_result.get('error','')[:200]}"
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": format_tool_result(exec_result),
                        }
                    )

                elif block.name == "route_to_market_researcher":
                    zip_codes = [str(z) for z in block.input.get("candidate_zip_codes", [])]
                    analysis_summary = block.input.get("analysis_summary", "")
                    routing_action = "market_researcher"
                    logs.append(
                        f"   🗺️  Routing to Market Researcher for: {', '.join(zip_codes)}"
                    )
                    # Authoritative source: rebuild candidates from panel for the LLM's chosen ZIPs.
                    # This fixes the state disconnect where execute_python computes one set of ZIPs
                    # but route_to_market_researcher specifies a different (correct) set.
                    if zip_codes and panel_json:
                        zip_set = {c["zip_code"]: c for c in candidates}
                        panel_df_lookup = pd.read_json(io.StringIO(panel_json))
                        panel_df_lookup["zip_code"] = panel_df_lookup["zip_code"].astype(str)
                        ordered = []
                        for z in zip_codes:
                            if z in zip_set:
                                ordered.append(zip_set[z])
                            else:
                                # Build a candidate dict from raw panel row (skip low-traffic ZIPs)
                                row = panel_df_lookup[
                                    (panel_df_lookup["zip_code"] == z) &
                                    (panel_df_lookup["total_exits"] >= 5000)
                                ]
                                if not row.empty:
                                    r = row.iloc[0]
                                    ordered.append({
                                        "zip_code": z,
                                        "neighborhood": config.ZIP_NAMES.get(z, z),
                                        "total_exits": int(r.get("total_exits", 0)),
                                        "complaints_per_1k": round(float(r.get("noise_complaints", r.get("total_complaints", 0))) / max(float(r.get("total_exits", 1)), 1) * 1000, 2),
                                        "score": 0.0,  # not yet scored; will be ranked by fallback
                                        "station_count": int(r.get("station_count", 1)),
                                        "low_confidence": bool(r.get("low_confidence", False)),
                                        "synergy_score": round(float(r.get("synergy_score", 0)), 2),
                                        "lq_competitor": round(float(r.get("lq_competitor", 0)), 2),
                                    })
                        if ordered:
                            candidates = ordered

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Routing acknowledged. Market Researcher will validate zoning.",
                        }
                    )

                elif block.name == "finalize_hypothesis":
                    routing_action = "finalize"
                    hypothesis = block.input.get("hypothesis", "")
                    logs.append("   📝 Lead Strategist finalizing hypothesis directly.")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Hypothesis recorded.",
                        }
                    )
                    return {
                        "status_logs": logs,
                        "candidates": candidates,
                        "visualization_b64": visualization_b64,
                        "analysis_summary": analysis_summary,
                        "final_hypothesis": hypothesis,
                        "next": "done",
                        "phase": "done",
                        "iteration": state.get("iteration", 0) + 1,
                    }

        # Append tool results for next round
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if resp.stop_reason == "end_turn" or routing_action:
            break

    # ── Always regenerate chart deterministically for clean visuals ───────────
    panel_json = state.get("panel_df_json", "")
    business_type = state.get("business_type", "quiet_cafe")
    clean_chart = _generate_eda_chart(panel_json, business_type, candidates)
    if clean_chart:
        visualization_b64 = clean_chart

    # ── Sensitivity analysis ───────────────────────────────────────────────
    ranking_stability = _compute_sensitivity(panel_json, candidates, business_type)
    if ranking_stability:
        logs.append(f"   📊 Sensitivity check: {ranking_stability}")

    # ── Return routing update ──────────────────────────────────────────────
    updates = {
        "status_logs": logs,
        "iteration": state.get("iteration", 0) + 1,
    }
    if candidates:
        updates["candidates"] = candidates
    if ranking_stability:
        updates["ranking_stability"] = ranking_stability
    if visualization_b64:
        updates["visualization_b64"] = visualization_b64
    if analysis_summary:
        updates["analysis_summary"] = analysis_summary

    if routing_action == "market_researcher" or candidates:
        # Route to Market Researcher whether the LLM called the tool explicitly
        # or wrote code in a text block (routing_action stays None in that path).
        updates["next"] = "market_researcher"
        updates["phase"] = "analyzing"
    else:
        # LLM didn't produce candidates — try deterministic fallback from panel data
        fallback = _fallback_candidates(
            state.get("panel_df_json", ""),
            state.get("business_type", "quiet_cafe"),
        )
        if fallback:
            updates["candidates"] = fallback
            updates["next"] = "market_researcher"
            updates["phase"] = "analyzing"
            logs.append(
                f"   ⚠️  LLM did not produce candidates; using deterministic fallback ranking. "
                f"Top ZIP: {fallback[0]['zip_code']}"
            )
        else:
            updates["next"] = "done"
            updates["phase"] = "done"
            updates["final_hypothesis"] = (
                "## Analysis Failed\n\nThe EDA could not produce candidate ZIP codes.\n\n"
                "The data collection succeeded but the analysis step failed to rank locations. "
                "Check the Agent Log tab for details."
            )

    return updates


def _handle_validating(state: AgentState, logs: list) -> dict:
    """
    Evaluate zoning verdicts and either finalize or loop for more candidates.
    """
    verdicts = state.get("zoning_verdicts", {})
    candidates = state.get("candidates", [])
    rejected = state.get("rejected_zips", [])
    iteration = state.get("iteration", 0)

    logs.append(
        f"🧭 [Lead Strategist] Evaluating zoning verdicts (iteration {iteration+1}): "
        f"{json.dumps(verdicts, indent=None)}"
    )

    # Guard: if Market Researcher returned nothing, route back to retry (max once)
    if not verdicts and iteration < 4:
        logs.append(
            "   ⚠️  Zoning verdicts are empty — Market Researcher likely failed. "
            "Routing back to retry zoning check."
        )
        return {
            "status_logs": logs,
            "next": "market_researcher",
            "phase": "validating",
            "iteration": iteration + 1,
        }

    # Guard: if every candidate has score ≤ 0.05 the EDA silently failed (chatty LLM, default=0).
    # Abort rather than recommend a location with fabricated evidence.
    top_score = max((float(c.get("score", 0)) for c in candidates), default=0.0)
    if candidates and top_score <= 0.05:
        logs.append(
            "   ❌ EDA integrity check failed: top candidate score is "
            f"{top_score:.3f} (≤ 0.05). The scoring step likely did not run. "
            "Aborting recommendation to prevent a zero-evidence output."
        )
        return {
            "status_logs": logs,
            "next": "done",
            "phase": "done",
            "final_hypothesis": (
                "## Analysis Aborted — EDA Did Not Produce Valid Scores\n\n"
                "The quantitative scoring step returned a maximum score of "
                f"`{top_score:.3f}` across all candidates, which indicates the "
                "Python EDA code did not execute correctly (the Lead Strategist "
                "may have described the analysis in text rather than running it).\n\n"
                "**Please re-run the analysis.** If the problem persists, check "
                "the Agent Log tab for the EDA output."
            ),
        }

    # Build context for Claude
    candidates_summary = json.dumps(candidates[:8], indent=2, default=str)
    verdicts_summary = json.dumps(verdicts, indent=2)
    zoning_ctx = "\n\n".join(
        f"=== ZIP {z} ===\n{v[:600]}"
        for z, v in state.get("zoning_results", {}).items()
    )

    user_msg = (
        f"User query: '{state.get('user_query', '')}'\n"
        f"Business type: {state.get('business_type', 'quiet_cafe')}\n\n"
        f"Ranked candidates from EDA:\n{candidates_summary}\n\n"
        f"Zoning verdicts: {verdicts_summary}\n"
        f"Rejected ZIPs (failed zoning): {rejected}\n\n"
        f"Retrieved zoning context:\n{zoning_ctx}\n\n"
        f"Analysis summary: {state.get('analysis_summary', '')}\n\n"
        "Now call finalize_hypothesis with the complete strategic memo."
    )

    messages = [{"role": "user", "content": user_msg}]
    hypothesis = ""

    for _round in range(3):
        try:
            business_type = state.get("business_type", "quiet_cafe")
            system_validating = _SYSTEM_VALIDATING_BY_TYPE.get(
                business_type, _SYSTEM_VALIDATING_BY_TYPE["default"]
            )
            resp = get_llm_client().messages_create(
                system=system_validating,
                messages=messages,
                tools=[t for t in _TOOLS if t["name"] == "finalize_hypothesis"],
                max_tokens=4096,
                force_tool="finalize_hypothesis" if _round == 0 else None,
            )
        except Exception as exc:
            err_str = str(exc)
            # Groq 400 tool_use_failed: the hypothesis is embedded in failed_generation
            if "tool_use_failed" in err_str or "failed_generation" in err_str:
                import re as _re
                # The error body contains: 'failed_generation': '<function=finalize_hypothesis>{...}'
                # Extract the JSON payload after the function tag
                fg_match = _re.search(r'<function=finalize_hypothesis>(\{.*\})', err_str, _re.DOTALL)
                if fg_match:
                    try:
                        payload = json.loads(fg_match.group(1))
                        hypothesis = payload.get("hypothesis", "")
                        logs.append("   📝 Hypothesis extracted from failed_generation (Groq compat).")
                    except json.JSONDecodeError:
                        logs.append("   ❌ Could not parse hypothesis JSON from failed_generation.")
                else:
                    logs.append(f"   ❌ LLM tool_use_failed and no hypothesis extractable: {err_str[:200]}")
            else:
                logs.append(f"   ❌ LLM API error: {exc}")
            break

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                hypothesis = block.text.strip()  # fallback if no tool call
                logs.append(f"   💬 {block.text.strip()[:200]}")

            elif block.type == "tool_use" and block.name == "finalize_hypothesis":
                hypothesis = block.input.get("hypothesis", "")
                logs.append("   📝 Final hypothesis generated.")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Hypothesis recorded successfully.",
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if hypothesis and "finalize_hypothesis" in str(resp.content):
            break
        if resp.stop_reason == "end_turn":
            break

    logs.append("🧭 [Lead Strategist] ✅ Analysis complete.")

    return {
        "status_logs": logs,
        "final_hypothesis": hypothesis,
        "next": "done",
        "phase": "done",
        "iteration": iteration + 1,
    }


def _force_finalize(state: AgentState, logs: list) -> dict:
    """
    Emergency fallback: generate a simple hypothesis from available state data.
    Used when max iterations are reached without a clean completion.
    """
    candidates = state.get("candidates", [])
    verdicts = state.get("zoning_verdicts", {})
    top = next(
        (c for c in candidates if verdicts.get(c["zip_code"]) != "fail"),
        candidates[0] if candidates else None,
    )

    if top:
        hypothesis = (
            f"## Strategic Recommendation (Auto-finalized)\n\n"
            f"**Primary Candidate:** ZIP {top['zip_code']} "
            f"({config.ZIP_NAMES.get(top['zip_code'], '')})\n\n"
            f"**Quantitative evidence:** Total exits {top.get('total_exits', 0):,.0f} | "
            f"Normalized noise {top.get('complaints_per_1k', 0):.2f}/1k exits | "
            f"Score {top.get('score', 0):.3f}\n\n"
            f"**Zoning verdict:** {verdicts.get(top['zip_code'], 'not evaluated')}\n\n"
            f"*Note: Auto-finalized due to iteration limit. Manual review recommended.*"
        )
    else:
        hypothesis = (
            "## No Feasible Candidate Found\n\n"
            "The constraint-satisfaction search exhausted all candidate ZIPs without "
            "finding a location that satisfies both quantitative targets (High Traffic / "
            "Low Normalized Noise) and zoning requirements. "
            "Consider relaxing the noise threshold or expanding the search area."
        )

    logs.append(f"🧭 [Lead Strategist] Force-finalized: {hypothesis[:100]}…")
    return {
        "status_logs": logs,
        "final_hypothesis": hypothesis,
        "next": "done",
        "phase": "done",
    }
