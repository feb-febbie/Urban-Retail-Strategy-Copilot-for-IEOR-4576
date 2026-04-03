"""
Urban Retail Strategy Copilot – Streamlit Frontend
"""
from __future__ import annotations

import base64
import os

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Retail Strategy Copilot",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 48px 40px 40px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(229,57,53,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.65);
    margin: 0 0 24px 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.85);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 8px;
    margin-bottom: 8px;
}

/* ── How it works cards ── */
.step-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 24px;
    height: 100%;
    position: relative;
    transition: box-shadow 0.2s;
}
.step-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    color: white;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 12px;
}
.step-title { font-size: 1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 8px; }
.step-body { font-size: 0.88rem; color: #5f6368; line-height: 1.6; }
.step-tech {
    margin-top: 12px;
    padding: 10px 12px;
    background: #f8f9fa;
    border-radius: 8px;
    font-size: 0.8rem;
    color: #3c4043;
    font-family: 'Courier New', monospace;
    border-left: 3px solid #0f3460;
}

/* ── Formula card ── */
.formula-card {
    background: linear-gradient(135deg, #f8f9ff 0%, #eef1ff 100%);
    border: 1px solid #c5cae9;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 24px 0;
}
.formula-title { font-size: 0.85rem; font-weight: 600; color: #3949ab; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
.formula-text { font-size: 1.15rem; font-weight: 600; color: #1a237e; font-family: 'Courier New', monospace; margin-bottom: 4px; }
.formula-note { font-size: 0.82rem; color: #5c6bc0; margin-top: 8px; }

/* ── Data source pills ── */
.source-pill {
    display: inline-block;
    background: #e8f0fe;
    color: #1967d2;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px;
}

/* ── Example query cards ── */
.query-card {
    background: #fff;
    border: 1px solid #e8eaed;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    cursor: pointer;
}
.query-concept { font-size: 0.9rem; font-weight: 600; color: #1a1a2e; }
.query-insight { font-size: 0.82rem; color: #5f6368; margin-top: 2px; }

/* ── Agent log ── */
.agent-log {
    background: transparent;
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 16px 18px;
    font-family: 'Courier New', monospace;
    font-size: 12.5px;
    line-height: 1.7;
    max-height: 580px;
    overflow-y: auto;
    color: inherit;
}

/* ── Run button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    padding: 14px;
    letter-spacing: 0.3px;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover { opacity: 0.88; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e8eaed;
    border-radius: 10px;
    padding: 16px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #fafafa; }
</style>
""", unsafe_allow_html=True)

# ── Load secrets / env vars ────────────────────────────────────────────────────
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "NYC_OPEN_DATA_APP_TOKEN" in st.secrets:
        os.environ["NYC_OPEN_DATA_APP_TOKEN"] = st.secrets["NYC_OPEN_DATA_APP_TOKEN"]
except Exception:
    pass

import config  # noqa: E402
from graph.state import initial_state  # noqa: E402
from graph.workflow import get_graph   # noqa: E402

DEFAULT_QUERY = "premium quiet reading café in Manhattan"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗽 Urban Retail Strategy Copilot")
    st.caption("Multi-agent spatial & regulatory analytics for Manhattan")
    st.divider()

    st.markdown("**⚙️ LLM Provider**")
    st.caption("App uses the first available provider:")

    api_key = st.text_input("Anthropic API Key (paid)", value=os.environ.get("ANTHROPIC_API_KEY", ""), type="password", help="console.anthropic.com")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    groq_key = st.text_input("Groq API Key (free)", value=os.environ.get("GROQ_API_KEY", ""), type="password", help="console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    google_key = st.text_input("Google Gemini API Key (free)", value=os.environ.get("GOOGLE_API_KEY", ""), type="password", help="aistudio.google.com")
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key

    st.caption("**No key?** Run `gcloud auth application-default login` to use Vertex AI free via ADC.")

    try:
        from tools.llm_client import get_llm_client, reset_llm_client
        if api_key or groq_key or google_key:
            reset_llm_client()
        client = get_llm_client()
        provider_emoji = {"anthropic": "🟣", "groq": "🟢", "gemini": "🔵", "gemini_native": "🔵", "ollama": "🟠"}
        st.success(f"{provider_emoji.get(client.provider, '⚪')} **{client.provider}** · `{client.model}`")
    except RuntimeError as e:
        st.error(str(e))

    st.divider()
    st.markdown("**📊 Analysis Parameters**")
    rolling_days = st.slider("Rolling window (days)", 7, 60, 30)
    config.ROLLING_DAYS = rolling_days

    st.divider()
    st.markdown("**🧠 Agent Architecture**")
    st.markdown("""
| Agent | Role |
|-------|------|
| 🧭 Lead Strategist | Orchestrator · EDA · Memo |
| 👷 Data Engineer | MTA · 311 · POI · Spatial |
| 📚 Market Researcher | Zoning RAG · SLA Risk |

| Type | Window | Synergy POIs |
|------|--------|-------------|
| ☕ Café | Daytime | Libraries, bookshops |
| 🍸 Bar | Thu–Sat night | Korean venues, karaoke |
| 🛍️ Retail | Full week | Restaurants, attractions |
""")

    st.divider()
    st.caption("Data: NYC Open Data · OSM Overpass · ZCTA Shapefiles · Zoning RAG")


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🗽 Urban Retail Strategy Copilot</div>
    <div class="hero-subtitle">
        Where should I open my store in Manhattan? — answered with real data, spatial analytics,
        and multi-agent AI reasoning. Not a guess.
    </div>
    <span class="hero-badge">🤖 LangGraph Multi-Agent</span>
    <span class="hero-badge">📡 Live NYC Open Data</span>
    <span class="hero-badge">🗺️ Geopandas Spatial Join</span>
    <span class="hero-badge">📚 FAISS Zoning RAG</span>
    <span class="hero-badge">⚖️ MCDA Scoring</span>
</div>
""", unsafe_allow_html=True)


# ── Session state — must initialize before any read ────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False
if "user_has_typed" not in st.session_state:
    st.session_state.user_has_typed = False

# ── LLM check ──────────────────────────────────────────────────────────────────
try:
    from tools.llm_client import get_llm_client
    get_llm_client()
except RuntimeError:
    st.warning(
        "**No LLM provider detected.** Options:\n"
        "- Add an API key in the sidebar: **Anthropic** (paid) · **Groq** (free) · **Google Gemini** (free)\n"
        "- **Vertex AI (no key):** run `gcloud auth application-default login` then restart\n"
        "- **Ollama** (local, no key): https://ollama.com → `ollama pull llama3.2`",
        icon="🔑",
    )
    st.stop()

# ── Query input ─────────────────────────────────────────────────────────────────
st.markdown("## 🚀 Run Your Analysis")
col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input(
        "Describe your business concept",
        placeholder="e.g., premium quiet reading café in Manhattan",
        value=DEFAULT_QUERY,
        help="Be specific — the system adapts data collection, time windows, and POI synergy to your concept.",
    )

with col2:
    business_type = st.selectbox(
        "Business type",
        options=["quiet_cafe", "bar", "retail"],
        format_func=lambda x: {
            "quiet_cafe": "☕ Quiet Café / Bookshop",
            "bar": "🍸 Bar / Nightlife",
            "retail": "🛍️ Premium Retail",
        }[x],
        help=(
            "Controls the full data strategy:\n"
            "• Café → Mon–Sat daytime MTA + library/bookshop POI synergy\n"
            "• Bar → Thu–Sat 8PM–4AM MTA + Korean/nightlife POI synergy\n"
            "• Retail → Full 7-day MTA + restaurant/attraction POI synergy"
        ),
    )

# Detect if user has changed the query from the default
if user_query != DEFAULT_QUERY:
    st.session_state.user_has_typed = True

run_button = st.button(
    "⚡ Run Full Analysis",
    type="primary",
    use_container_width=True,
    disabled=not user_query.strip(),
)

# Collapse How It Works the moment user clicks Run (before expander renders)
if run_button:
    st.session_state.user_has_typed = True

# ── Live journal placeholder — appears right under the run button ────────────
log_area = st.empty()

st.markdown("---")

# ── How it works — collapsed once user starts typing or after a run ───────────
_how_it_works_expanded = not (
    st.session_state.user_has_typed or st.session_state.result is not None
)
with st.expander("ℹ️ How It Works", expanded=_how_it_works_expanded):
    st.caption("Three specialized AI agents collaborate in a LangGraph orchestrator-worker pipeline to answer one precise question.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
<div class="step-card">
    <div class="step-number">1</div>
    <div class="step-title">📡 Collect — Live Data Ingestion</div>
    <div class="step-body">
        The <b>Data Engineer agent</b> fetches three real data sources and adapts the query
        window to your business type — daytime for cafés, Thu–Sat 8PM–4AM for bars.
        <br><br>
        • <b>MTA Subway ridership</b> — time-sliced exit counts per station (NYC Open Data SODA API)<br>
        • <b>311 Noise complaints</b> — residential + commercial complaint density per ZIP<br>
        • <b>OpenStreetMap POIs</b> — competitor density &amp; cultural synergy anchors via Overpass API<br>
        • <b>ZCTA shapefiles</b> — geopandas spatial join maps all data to ZIP polygons
    </div>
    <div class="step-tech">38 Manhattan ZIPs · ~600K exits · ~2,500 complaints · ~1,000 POIs</div>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="step-card">
    <div class="step-number">2</div>
    <div class="step-title">📊 Explore — Quantitative EDA</div>
    <div class="step-body">
        The <b>Lead Strategist agent</b> writes and executes Python code to score every ZIP
        using a spatial Multi-Criteria Decision Analysis (MCDA) model — standard in urban
        site selection research.
        <br><br>
        • 95th-percentile <b>winsorization</b> removes outlier distortion<br>
        • <b>log₁₊ transform</b> on exits — compresses the 5K→500K range so Grand Central
          doesn't dominate the objective function<br>
        • <b>Min-max normalization</b> makes all weights scale-independent<br>
        • <b>Sensitivity analysis</b> — top-3 ZIPs tested across 16 weight perturbations (±10%)
    </div>
    <div class="step-tech">score = 0.35·traffic − 0.30·noise + 0.25·synergy − 0.10·saturation</div>
</div>
""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div class="step-card">
    <div class="step-number">3</div>
    <div class="step-title">⚖️ Validate — Zoning & Regulatory Check</div>
    <div class="step-body">
        The <b>Market Researcher agent</b> validates every top candidate against NYC zoning law
        using a FAISS vector store of curated Community District profiles — no raw PDF parsing,
        no hallucination risk from OCR errors.
        <br><br>
        • <b>Metadata-gated RAG</b>: explicit allow/deny arrays checked <i>before</i> semantic text,
          preventing the model from inferring "café zoning = K-pop club zoning"<br>
        • Verdicts: ✅ Pass · ⚠️ Caution · ❌ Fail<br>
        • Lead Strategist writes the final memo: highest-scoring <i>feasible</i> ZIP wins
    </div>
    <div class="step-tech">FAISS · sentence-transformers · all-MiniLM-L6-v2 · 6 CDs indexed</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="formula-card">
    <div class="formula-title">🧮 Scoring Formula — Weighted Linear Combination (WLC / MCDA)</div>
    <div class="formula-text">score = 0.35 · traffic − 0.30 · noise + 0.25 · synergy − 0.10 · saturation</div>
    <div class="formula-note">
        All inputs log-transformed (traffic) and min-max normalized to [0,1] before scoring.
        Weights follow an Analytic Hierarchy Process (AHP) pairwise priority ordering.
        Traffic coefficient reflects year-one revenue dependence on foot traffic;
        noise penalty reflects Community Board friction risk for SLA licensing;
        synergy captures cultural co-location demand; saturation applies a light agglomeration discount.
    </div>
    <div style="margin-top:16px;padding-top:14px;border-top:1px solid #c5cae9;">
        <div class="formula-title" style="margin-bottom:8px;">⚖️ How the Score and Zoning Verdict Work Together</div>
        <div class="formula-note" style="font-size:0.88rem;line-height:1.7;color:#1a237e;">
            The formula and the zoning verdict serve <b>two separate roles</b> — and both are required to produce a recommendation.<br><br>
            <b>Step 1 — Score ranks every ZIP</b> using live data. A higher score means better foot traffic,
            quieter environment, and stronger cultural synergy for your concept. This is pure data math —
            it knows nothing about what the law allows.<br><br>
            <b>Step 2 — Zoning verdict gates the ranked list.</b> The Market Researcher checks each top
            candidate against NYC Community District zoning profiles via RAG. Each ZIP receives:
            ✅ <b>Pass</b> (permitted) · ⚠️ <b>Caution</b> (conditional) · ❌ <b>Fail</b> (prohibited or high regulatory risk).<br><br>
            <b>The winner = highest-scoring ZIP with a non-Fail verdict.</b> This is constrained optimization:
            maximize demand and experience quality, subject to regulatory feasibility.
            A ZIP can score #1 on data and still be rejected — Times Square (10036) is the canonical example:
            top foot traffic, but Fail for quiet retail due to C6-4 high-intensity zoning and extreme noise.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("**Example Queries**")
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        st.markdown("""
<div class="query-card">
    <div class="query-concept">☕ Premium quiet reading café</div>
    <div class="query-insight">→ UWS 10025 · daytime traffic · low noise · library synergy · C4-2 zoning PASS</div>
</div>
<div class="query-card">
    <div class="query-concept">🍸 K-pop club / nightlife bar</div>
    <div class="query-insight">→ Chelsea 10001 · Thu–Sat night traffic · Korean POI synergy · C6-4 zoning PASS</div>
</div>
""", unsafe_allow_html=True)
    with col_ex2:
        st.markdown("""
<div class="query-card">
    <div class="query-concept">📚 American comics bookstore</div>
    <div class="query-insight">→ UWS 10023 · daytime traffic · low noise · bookshop synergy · C4-2 PASS</div>
</div>
<div class="query-card">
    <div class="query-concept">🛍️ Luxury boutique retail</div>
    <div class="query-insight">→ UES 10021/10028 · 7-day traffic · low saturation · attraction synergy</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Run ─────────────────────────────────────────────────────────────────────────
if run_button:
    st.session_state.result = None
    st.session_state.running = True
    st.session_state.user_has_typed = True  # collapse How It Works during run

    graph = get_graph()
    init = initial_state(user_query=user_query.strip(), business_type=business_type)
    final_state = None
    all_logs: list[str] = []
    last_log_count = 0

    with st.spinner(""):  # keeps browser tab spinning; journal below shows live status
        try:
            for event in graph.stream(init, stream_mode="updates"):
                for node_name, node_output in event.items():
                    new_logs = node_output.get("status_logs", [])
                    if len(new_logs) > last_log_count:
                        all_logs = new_logs
                        last_log_count = len(new_logs)

                    log_area.markdown(
                        """
<div style="border:2px solid #0f3460;border-radius:12px;padding:18px 20px;margin:8px 0;">
  <div style="font-size:1rem;font-weight:700;color:#0f3460;margin-bottom:10px;letter-spacing:0.2px;">
    🔄 &nbsp;ANALYSIS IN PROGRESS — agents are working…
  </div>
  <div style="font-size:0.78rem;color:#5f6368;margin-bottom:12px;">
    This takes ~60 seconds. Data is being fetched live from MTA, 311, and OpenStreetMap.
  </div>
  <div class="agent-log">"""
                        + "<br>".join(all_logs[-18:])
                        + "</div></div>",
                        unsafe_allow_html=True,
                    )

                    if final_state is None:
                        final_state = dict(init)
                    final_state.update(node_output)

        except Exception as exc:
            st.error(f"❌ Analysis failed: {exc}")
            st.session_state.running = False
            st.stop()

    log_area.empty()
    st.session_state.result = final_state
    st.session_state.running = False
    st.rerun()


# ── Results ─────────────────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result

    st.success("✅ Analysis complete!")

    if result.get("poi_warning"):
        st.warning(result["poi_warning"], icon="⚠️")

    tab_hypothesis, tab_chart, tab_data, tab_log = st.tabs(
        ["📋 Strategic Memo", "📊 EDA Chart", "🗂️ Candidate Data", "📡 Agent Log"]
    )

    with tab_hypothesis:
        hypothesis = result.get("final_hypothesis", "")
        if hypothesis:
            st.markdown(hypothesis)
        else:
            st.warning("No hypothesis generated. Check the Agent Log tab.")

        verdicts = result.get("zoning_verdicts", {})
        if verdicts:
            st.divider()
            st.subheader("🏛️ Zoning Validation Results")
            verdict_cols = st.columns(min(len(verdicts), 5))
            verdict_emoji = {"pass": "✅", "caution": "⚠️", "fail": "❌"}
            for i, (zip_code, verdict) in enumerate(verdicts.items()):
                col = verdict_cols[i % len(verdict_cols)]
                col.metric(
                    label=f"ZIP {zip_code}",
                    value=f"{verdict_emoji.get(verdict, '❓')} {verdict.upper()}",
                    delta=config.ZIP_NAMES.get(zip_code, ""),
                    delta_color="off",
                )

    with tab_chart:
        viz_b64 = result.get("visualization_b64", "")
        if viz_b64:
            img_bytes = base64.b64decode(viz_b64)
            st.image(img_bytes, caption="Composite Score Quadrant Analysis (Traffic × Noise × Synergy × Saturation)", use_container_width=True)
            st.download_button(
                label="⬇️ Download Chart (PNG)",
                data=img_bytes,
                file_name="eda_quadrant_analysis.png",
                mime="image/png",
            )
        else:
            st.info("No chart generated. Check the Agent Log tab for EDA output details.")

    with tab_data:
        import pandas as pd

        candidates = result.get("candidates", [])
        if candidates:
            df_cand = pd.DataFrame(candidates)
            col_map = {
                "zip_code": "ZIP Code",
                "neighborhood": "Neighborhood",
                "total_exits": "Exits (time-sliced)",
                "complaints_per_1k": "Noise / 1k Exits",
                "score": "Composite Score",
                "station_count": "Subway Stations",
                "low_confidence": "Low Confidence",
                "synergy_score": "Synergy Score (POI/km²)",
                "lq_competitor": "Location Quotient",
                "competitor_density": "Competitor Density",
            }
            df_display = df_cand.rename(columns={k: v for k, v in col_map.items() if k in df_cand.columns})

            for col in ["Exits (time-sliced)"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{int(x):,}" if x is not None else "—")
            for col in ["Noise / 1k Exits", "Composite Score", "Synergy Score (POI/km²)", "Location Quotient", "Competitor Density"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{float(x):.3f}" if x is not None else "—")

            st.subheader("🏆 Ranked Candidates")
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No candidate ranking data available.")

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("MTA Station Records", f"{result.get('mta_records', 0):,}")
        col_b.metric("311 Complaint Records", f"{result.get('complaints_records', 0):,}")
        mta_window = {
            "quiet_cafe": f"Daytime ({config.ROLLING_DAYS}-day)",
            "bar": "Thu–Sat 8PM–4AM",
            "retail": f"{config.ROLLING_DAYS}-day full",
        }.get(business_type, f"{config.ROLLING_DAYS} days")
        col_c.metric("MTA Window", mta_window)

        if result.get("ranking_stability"):
            st.info(f"📊 **Ranking robustness:** {result['ranking_stability']}")

        if result.get("data_summary"):
            with st.expander("📋 Full data summary"):
                st.text(result["data_summary"])

    with tab_log:
        st.subheader("📡 Full Agent Execution Log")
        logs = result.get("status_logs", [])
        if logs:
            st.markdown(
                "<div class='agent-log'>"
                + "<br>".join(logs)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No logs available.")

        if result.get("error"):
            st.error(f"⚠️ Error recorded: {result['error']}")
