"""
Market Researcher Agent Node

Responsibilities:
  1. Receive a ranked list of candidate ZIP codes from the Lead Strategist.
  2. Query the local RAG knowledge base (FAISS + zoning documents) for each ZIP.
  3. Use Claude to interpret the zoning context and issue a structured verdict.
  4. Return verdicts and zoning excerpts back to the Lead Strategist.
"""
from __future__ import annotations

import config
from graph.state import AgentState
from tools.rag_tools import get_rag
from tools.llm_client import get_llm_client

_SYSTEM_PROMPT_CAFE = """You are the Market Researcher for an Urban Retail Strategy team.
You have retrieved NYC zoning and community district information from a local knowledge base.

Your job: given a ZIP code and retrieved zoning context, produce a concise structured verdict.

Output a JSON object with exactly these fields:
{
  "verdict": "pass" | "caution" | "fail",
  "zoning_type": "e.g. C4-2 along Broadway",
  "cafe_permitted": true | false,
  "key_finding": "1-2 sentences with the most relevant zoning fact",
  "demographic_fit": "high" | "medium" | "low",
  "risk_flags": ["list of any zoning risks or complications"]
}

verdict rules:
  - "pass": Café/retail explicitly permitted, clean zoning, good demographics.
  - "caution": Permitted but with complications (special district, variance needed, etc.).
  - "fail": Residential-only zoning, or strict restrictions that prohibit café use.
"""

_SYSTEM_PROMPT_BAR = """You are the Market Researcher for an Urban Retail Strategy team.
You have retrieved NYC zoning and community district information from a local knowledge base.

Your job: given a ZIP code and retrieved zoning context, produce a zoning verdict for a
BAR or LATE-NIGHT NIGHTLIFE VENUE (K-pop club, bar with dancing, cabaret, venue open past midnight).

NYC nightlife licensing is adversarial. Assume zoning is HOSTILE to nightlife unless the
retrieved text EXPLICITLY proves otherwise.

STEP 1 — METADATA GATE (deterministic, do this FIRST):
The context begins with a [ZONING METADATA] block containing explicit_allow and explicit_deny arrays.
- If your business type (nightclub, late_night_bar, cabaret, bar_with_dancing) appears in
  explicit_deny → verdict must be "caution" or "fail". Do NOT override this with semantic text.
- If it appears only in explicit_allow → may qualify for "pass" if the semantic text confirms.
- If the metadata block is absent or both lists are empty → proceed to Step 2.

STEP 2 — SEMANTIC VERIFICATION:
Read the zoning text and apply these rules.

Output a JSON object with exactly these fields:
{
  "verdict": "pass" | "caution" | "fail",
  "zoning_type": "e.g. C6-4 commercial",
  "bar_permitted": true | false,
  "key_finding": "1-2 sentences citing the specific fact (metadata or text) that determines the verdict",
  "demographic_fit": "high" | "medium" | "low",
  "risk_flags": ["list of risks: CB opposition, SLA history, residential overlay, noise enforcement"]
}

STRICT verdict rules for nightlife:
  - "pass": explicit_allow contains bar/nightclub/cabaret AND zoning text confirms C6-2+ or C4
    with no residential overlay AND no documented CB opposition to late-night venues.
  - "caution": Commercial zoning permits bars in general BUT any of: high residential density,
    CB opposition history, C4-2 on a residential-adjacent block, scale restrictions, or the
    metadata explicit_deny list flags nightclub/cabaret even if allow list has bar.
  - "fail": explicit_deny contains nightclub/late_night_bar, OR residential-only zoning,
    OR no zoning text retrieved at all.

IMPORTANT: Upper West Side (10023, 10024, 10025) and Upper East Side (10021, 10028) are
"caution" at minimum for late-night nightlife — Community Board 7 and CB8 have a documented
history of blocking late-night SLA applications. A C4-2 zoning designation alone is not
sufficient for a "pass" for a nightclub or K-pop club.
"""

_SYSTEM_PROMPT_RETAIL = _SYSTEM_PROMPT_CAFE  # retail zoning rules mirror café

SYSTEM_PROMPT = _SYSTEM_PROMPT_CAFE  # default (overridden per-call below)


def market_researcher_node(state: AgentState) -> dict:
    """
    LangGraph node: checks zoning for each candidate ZIP via RAG + Claude.
    """
    logs = list(state.get("status_logs", []))
    candidates = state.get("candidates", [])
    business_type = state.get("business_type", "quiet_cafe")
    rejected_zips = list(state.get("rejected_zips", []))
    existing_verdicts = dict(state.get("zoning_verdicts", {}))
    existing_results = dict(state.get("zoning_results", {}))

    if not candidates:
        logs.append("📚 [Market Researcher] No candidates to check. Returning.")
        return {
            "status_logs": logs,
            "next": "lead_strategist",
            "phase": "validating",
        }

    rag = get_rag()

    # Check top candidates not yet evaluated (ensure zip_codes are strings)
    to_check = [
        str(c["zip_code"]) for c in candidates
        if str(c["zip_code"]) not in existing_verdicts
    ][:config.MAX_CANDIDATES_TO_CHECK]

    if not to_check:
        logs.append("📚 [Market Researcher] All candidates already evaluated.")
        return {
            "status_logs": logs,
            "next": "lead_strategist",
            "phase": "validating",
        }

    logs.append(
        f"📚 [Market Researcher] Querying RAG knowledge base for ZIPs: {', '.join(to_check)}"
    )

    new_verdicts = dict(existing_verdicts)
    new_results = dict(existing_results)

    for zip_code in [str(z) for z in to_check]:
        # ── Retrieve relevant zoning context ──────────────────────────────
        chunks = rag.retrieve_for_zip(zip_code, query=business_type, k=4)
        context = rag.format_context(chunks)
        new_results[zip_code] = context

        # ── Ask Claude to issue a verdict ──────────────────────────────────
        neighborhood = config.ZIP_NAMES.get(zip_code, zip_code)
        prompt = (
            f"ZIP Code: {zip_code} ({neighborhood})\n"
            f"Business concept: {business_type.replace('_', ' ')}\n\n"
            f"Retrieved zoning context:\n{context}\n\n"
            f"Issue your structured JSON verdict."
        )

        system_prompt = {
            "bar": _SYSTEM_PROMPT_BAR,
            "retail": _SYSTEM_PROMPT_RETAIL,
        }.get(business_type, _SYSTEM_PROMPT_CAFE)

        try:
            resp = get_llm_client().messages_create(
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            raw = resp.content[0].text.strip()

            # Extract JSON from response
            import json, re
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                verdict_data = json.loads(json_match.group())
                verdict = verdict_data.get("verdict", "caution")
                key_finding = verdict_data.get("key_finding", "")
                zoning_type = verdict_data.get("zoning_type", "")
                risk_flags = verdict_data.get("risk_flags", [])
                demo_fit = verdict_data.get("demographic_fit", "medium")
            else:
                # Fallback parse
                verdict = "caution"
                key_finding = raw[:300]
                zoning_type = "Unknown"
                risk_flags = []
                demo_fit = "medium"

        except Exception as exc:
            # Graceful fallback: use RAG content directly without LLM
            verdict = "caution"
            key_finding = f"RAG retrieved zoning context (LLM error: {exc}). Manual review recommended."
            zoning_type = "See context"
            risk_flags = []
            demo_fit = "medium"

        new_verdicts[zip_code] = verdict

        verdict_emoji = {"pass": "✅", "caution": "⚠️", "fail": "❌"}.get(verdict, "❓")
        log_line = (
            f"   {verdict_emoji} ZIP {zip_code} ({neighborhood}): {verdict.upper()} | "
            f"{zoning_type} | {key_finding[:120]}"
        )
        if risk_flags:
            log_line += f" | Risks: {', '.join(risk_flags[:2])}"
        logs.append(log_line)

        if verdict == "fail":
            rejected_zips.append(zip_code)
            logs.append(f"      → ZIP {zip_code} added to rejected list.")

    logs.append(
        f"📚 [Market Researcher] ✅ Zoning check complete: "
        f"{sum(1 for v in new_verdicts.values() if v == 'pass')} pass, "
        f"{sum(1 for v in new_verdicts.values() if v == 'caution')} caution, "
        f"{sum(1 for v in new_verdicts.values() if v == 'fail')} fail."
    )

    return {
        "status_logs": logs,
        "zoning_results": new_results,
        "zoning_verdicts": new_verdicts,
        "rejected_zips": rejected_zips,
        "next": "lead_strategist",
        "phase": "validating",
    }
