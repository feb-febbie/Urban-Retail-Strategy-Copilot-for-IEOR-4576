# 🗽 Urban Retail Strategy Copilot

### *"Where should I open my store?" — answered like a real data analyst, not a guess.*

> **Live demo:** [http://70.111.84.196:8080](http://70.111.84.196:8080)
>
> Open this link on any device (phone, tablet, laptop) to try the app live.

---

## 📖 What is this?

Imagine you want to open a **K-pop club in Manhattan**.

Not just anywhere — you need:
- High **late-night foot traffic** (Thursday–Saturday after 8PM)
- A neighborhood that **tolerates commercial noise** (won't get your liquor license blocked)
- **Korean cultural anchors nearby** (BBQ restaurants, karaoke bars) to reduce customer acquisition cost
- **C6 or C4 zoning** that actually permits bars and cabarets

Instead of guessing, this project builds an **AI analyst team** that reasons with real data:

> **Collect → Explore → Hypothesize** — the first three steps of a real data analyst workflow.

This is not a chatbot giving opinions. It is a **multi-agent system that earns its answer.**

---

## 🧠 How it works (plain English)

Three specialized AI agents collaborate in a **LangGraph orchestrator-worker pipeline**:

| Agent | Role | Analogy |
|-------|------|---------|
| 🧭 **Lead Strategist** | Orchestrator — directs the other agents, runs EDA, writes the final memo | Senior analyst |
| 👷 **Data Engineer** | Collects live MTA, 311, and OSM data; performs spatial joins | Data engineer |
| 📚 **Market Researcher** | Validates zoning law for each candidate ZIP via RAG | Regulatory counsel |

The agents loop until a defensible recommendation emerges or the iteration budget is exhausted.

---

## ✅ Step 1 — Collect (5 pts)

**Location:** `agents/data_engineer.py` · `tools/nyc_data_tools.py`

The Data Engineer agent fetches **four real data sources at runtime** — nothing is hardcoded.

### What gets collected

| Source | Method | What it provides |
|--------|--------|-----------------|
| **MTA Subway Turnstile** | NYC Open Data SODA API (`data.ny.gov`) | Exit counts per station per time window |
| **NYC 311 Complaints** | NYC Open Data SODA API (`data.cityofnewyork.us`) | Noise + construction complaint density per ZIP |
| **OpenStreetMap POIs** | Overpass API (live OSM query) | Competitor density + cultural synergy anchors per km² |
| **NYC ZCTA Shapefiles** | NYC Open Data GeoJSON endpoint | ZIP Code polygon boundaries for spatial joins |

**~38 Manhattan ZIPs · ~600K subway exits · ~2,500 complaints · ~1,000 POIs per run.**

### Dynamic behavior (not scripted)

The agent adapts its entire data strategy based on the business concept:

| Business Type | MTA Window | Complaint Types | Synergy POIs |
|--------------|-----------|----------------|-------------|
| ☕ Quiet Café / Bookshop | Mon–Sat daytime | Residential noise, construction | Libraries, bookshops |
| 🍸 Bar / K-pop Club | Thu–Sat 8PM–4AM | Commercial noise, street noise | Korean restaurants, karaoke |
| 🛍️ Premium Retail | Full 7-day | All noise types | Restaurants, tourist attractions |

**The same query "open a K-pop club" vs "open a quiet reading café" triggers completely different API calls, time filters, and POI tag sets.** This satisfies the dynamic collection requirement — the data retrieved is determined by the question, not by the code.

### Spatial join

After collection, `tools/nyc_data_tools.py → perform_spatial_join()` uses **geopandas** to spatially join subway stations and POI points into ZCTA ZIP polygons, producing a clean ZIP-level panel DataFrame ready for EDA.

### Second data retrieval method — RAG

The Market Researcher uses a **FAISS vector store** (`tools/rag_tools.py`) built from hand-curated NYC Community District profiles and zoning resolution excerpts (`data/zoning_knowledge.py`). This is a second, distinct retrieval method: semantic vector search over a local document corpus, separate from the API calls above.

---

## ✅ Step 2 — Explore & Analyze (5 pts)

**Location:** `agents/lead_strategist.py → _handle_analyzing()` · `tools/python_executor.py`

The Lead Strategist **writes Python code and executes it** against the live panel data via `tools/python_executor.py`. This is a sandboxed `exec()` environment with `pandas`, `numpy`, `matplotlib`, `geopandas`, and `scipy` pre-loaded.

### What the EDA actually computes

**1. Filter** — Drop ZIPs with fewer than 5,000 exits (prevents ratio blow-up from low-count ZIPs).

**2. Winsorize** — Cap exits and complaints at the 95th percentile using `Series.clip()`. Prevents construction-spike outliers from dominating the score.

**3. Log-transform exits** — `log1p(total_exits)`. Manhattan ZIPs span 5K → 500K exits (100× range). Without compression, Grand Central dominates every scoring model regardless of weights. `log1p` also models real economics: the difference between 5K and 50K exits matters more to a small business than the difference between 50K and 500K.

**4. Normalize** — Min-max normalize all features to [0, 1] so weights are truly scale-independent.

**5. Score** — Weighted Linear Combination (WLC) within a Multi-Criteria Decision Analysis (MCDA) framework:

```
score = 0.35 · traffic − 0.30 · noise + 0.25 · synergy − 0.10 · saturation
```

Weights follow an **Analytic Hierarchy Process (AHP)** pairwise priority ordering:
- `traffic +0.35` — foot traffic is the primary year-one revenue driver
- `noise −0.30` — high residential complaints → Community Board friction → SLA license risk
- `synergy +0.25` — cultural co-location reduces customer acquisition cost
- `saturation −0.10` — light penalty; agglomeration economics mean some clustering is positive

**6. Quadrant analysis** — Segments ZIPs into four quadrants (High/Low Traffic × High/Low Noise). The target quadrant for a café is High Traffic / Low Noise. For a bar, high commercial noise is acceptable; extreme residential noise is not.

**7. Location Quotient (LQ)** — `competitor_count_in_zip / (total_competitors / total_area)`. LQ > 1.2 = established hub; LQ < 0.8 = untapped market. Both are valid strategies with different trade-offs named in the memo.

**8. Sensitivity analysis** — Top-3 candidates tested across **16 weight perturbation scenarios (±10% on each weight)**. A 16/16 stable ranking means the recommendation is robust to reasonable disagreements about the weights. A 0/16 means two candidates are in genuine tension — the memo names the trade-off explicitly.

**9. Visualization** — Generates a matplotlib scatter plot (Traffic vs Noise, sized by synergy score, annotated with top ZIPs). Returned as base64 PNG, displayed in the Streamlit UI.

### EDA is dynamic

A "quiet café" query targets the High Traffic / Low Noise quadrant and penalizes noise heavily. A "K-pop club" query accepts moderate commercial noise and rewards Korean POI synergy. **Different questions produce different EDA code, different quadrant targets, and different visualizations.**

---

## ✅ Step 3 — Hypothesize (5 pts)

**Location:** `agents/lead_strategist.py → _handle_validating()` · `agents/market_researcher.py`

The system does not stop at a ranked list. It validates the top candidates against NYC zoning law and produces a structured strategic memo.

### Zoning validation via metadata-gated RAG

The Market Researcher (`agents/market_researcher.py`) retrieves zoning context for each candidate ZIP from the FAISS vector store. Critically, each document chunk carries **explicit allow/deny metadata arrays**:

```python
{
  "explicit_allow": ["bar", "nightclub", "cabaret", "tavern"],
  "explicit_deny":  ["low_impact_quiet_retail", "heavy_manufacturing"],
  ...
}
```

`tools/rag_tools.py → format_context()` prepends these arrays **before the semantic text** in every prompt. The Market Researcher is instructed to check `explicit_deny` first — if the business type appears there, the verdict is CAUTION or FAIL regardless of what the semantic text says. This eliminates the failure mode where the model reads "C4-2 commercial zoning permits retail" and infers it also permits late-night cabarets.

Verdicts: ✅ Pass · ⚠️ Caution · ❌ Fail

### The final recommendation

The Lead Strategist receives the ranked candidates + zoning verdicts and calls `finalize_hypothesis` with a structured memo that must include:
- A consulting-style opening naming the recommended ZIP
- The scoring formula with all terms defined
- A candidate comparison table
- A "Key Insight" explaining why the winner beats higher-traffic alternatives
- An explicit trade-off statement (what was sacrificed for regulatory safety)
- Quantitative evidence for every claim

**The recommendation is the highest-scoring ZIP with a non-Fail zoning verdict** — not just the highest traffic, not just the cleanest zoning. The system is doing constrained optimization: maximizing demand subject to regulatory feasibility.

### Iterative refinement loop

The pipeline loops:

```
Lead Strategist (init) → Data Engineer → Lead Strategist (EDA)
  → Market Researcher → Lead Strategist (validate) → [done]
```

If zoning verdicts come back empty (API failure), the system retries. If the EDA produces zero-score candidates (code execution failed silently), a score floor guard aborts rather than fabricating a recommendation. The loop has a maximum iteration budget with a forced-finalize fallback.

---

## 🗂️ Grading Checklist

### Core Requirements (10 pts)

| Requirement | Status | Location |
|-------------|--------|----------|
| **Frontend** | ✅ Streamlit — live UI with tabs, metrics, chart display | `app.py` |
| **Agent framework** | ✅ LangGraph `StateGraph` with conditional routing | `graph/workflow.py` |
| **Tool calling** | ✅ `execute_python`, `route_to_market_researcher`, `finalize_hypothesis` | `agents/lead_strategist.py` |
| **Non-trivial dataset** | ✅ MTA turnstile (~600K rows/run), 311 (~2,500), OSM (~1,000 POIs), ZCTA shapefiles | `tools/nyc_data_tools.py` |
| **Multi-agent pattern** | ✅ Orchestrator-worker: Lead Strategist orchestrates Data Engineer + Market Researcher | `graph/workflow.py` |
| **Deployed** | ✅ [your-deployment-url-here] | — |
| **README** | ✅ This document | `README.md` |

### Grab-Bag (≥ 2 required, 5 pts)

| Feature | Status | Location |
|---------|--------|----------|
| **Code execution** | ✅ Lead Strategist writes & runs pandas/numpy/matplotlib EDA code at runtime | `tools/python_executor.py` · `agents/lead_strategist.py → _handle_analyzing()` |
| **Second data retrieval** | ✅ API (MTA/311/OSM) + RAG (FAISS vector store over zoning documents) | `tools/rag_tools.py` · `data/zoning_knowledge.py` |
| **Data visualization** | ✅ Matplotlib scatter plot generated dynamically per query, displayed in UI | `agents/lead_strategist.py → _generate_eda_chart()` |
| **Iterative refinement loop** | ✅ Agents loop with retry logic, score floor guard, and max-iteration fallback | `graph/workflow.py` · `agents/lead_strategist.py → lead_strategist_node()` |
| **Structured output** | ✅ `finalize_hypothesis` tool enforces JSON schema for the memo; zoning verdicts are structured dicts | `agents/lead_strategist.py → _TOOLS` · `agents/market_researcher.py` |

---

## 🗺️ Codebase Structure

```
Urban Retail Strategy Copilot/
│
├── app.py                     # Streamlit UI — entry point
├── config.py                  # Constants, env vars, ZIP → neighborhood mappings
│
├── graph/
│   ├── state.py               # AgentState TypedDict — shared memory across agents
│   └── workflow.py            # LangGraph StateGraph + conditional router
│
├── agents/
│   ├── lead_strategist.py     # Orchestrator: EDA loop, routing, memo generation
│   ├── data_engineer.py       # Data collection: MTA, 311, OSM Overpass, spatial join
│   └── market_researcher.py   # Zoning RAG retrieval + LLM verdict per ZIP
│
├── tools/
│   ├── llm_client.py          # Unified LLM wrapper (Anthropic / Gemini / Groq / Ollama)
│   ├── nyc_data_tools.py      # SODA API fetchers + Overpass + geopandas spatial join
│   ├── python_executor.py     # exec() sandbox with pre-loaded pandas/numpy/matplotlib
│   └── rag_tools.py           # FAISS vector store + keyword fallback search
│
└── data/
    └── zoning_knowledge.py    # Hand-curated NYC zoning profiles (RAG corpus)
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- At least one LLM provider (see below)

### Install

```bash
git clone <repo-url>
cd "Urban Retail Strategy Copilot"
uv sync          # installs from uv.lock — exact reproducible environment
# or: pip install -r requirements.txt
```

### Configure an LLM provider (pick one)

| Provider | Cost | Setup |
|----------|------|-------|
| **Vertex AI (Gemini 2.0 Flash)** | Free | `gcloud auth application-default login` — no API key needed |
| **Google Gemini API** | Free tier | Get key at [aistudio.google.com](https://aistudio.google.com), set `GOOGLE_API_KEY=...` in `.env` |
| **Groq** | Free tier | Get key at [console.groq.com](https://console.groq.com), set `GROQ_API_KEY=...` in `.env` |
| **Anthropic** | Paid | Set `ANTHROPIC_API_KEY=...` in `.env` |
| **Ollama** | Local / free | Install [ollama.com](https://ollama.com) → `ollama pull llama3.2` |

Copy `.env.example` to `.env` and fill in your key, or use Vertex AI ADC (no key needed).

### Run

```bash
.venv/bin/streamlit run app.py
# or after activating venv:
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## 💡 Example Queries

| Business concept | Recommended output |
|-----------------|-------------------|
| `premium quiet reading café` | ZIP 10023 or 10025 (UWS) — daytime traffic, low noise, library synergy, C4-2 PASS |
| `kpop club` | ZIP 10001 (Chelsea/Koreatown) — Thu–Sat night traffic, Korean POI synergy, C6-4 PASS |
| `american comics bookstore` | ZIP 10023 (UWS / Lincoln Square) — daytime traffic, low noise, bookshop synergy |
| `luxury boutique retail` | ZIP 10021/10028 (UES) — 7-day traffic, low saturation, attraction synergy |

---

## ⚠️ Limitations

- **7–60 day rolling window** — short-term trends only; seasonal variation not captured
- **No rent data** — location quality score does not account for lease costs
- **Zoning verdicts are not legal advice** — on-site verification required before signing a lease
- **Overpass API rate limits** — competitor POI query may time out under heavy load; the UI displays a warning and prompts re-run
- **38 Manhattan ZIPs only** — does not cover Brooklyn, Queens, or other boroughs

---

## 🔄 Why this generalizes

The same pipeline handles fundamentally different business objectives by reparameterizing the objective function:

| Component | Quiet Café | K-pop Club |
|-----------|-----------|-----------|
| MTA window | Mon–Sat daytime | Thu–Sat 8PM–4AM |
| Noise interpretation | Strong penalty | Moderate — commercial tolerance |
| Synergy POIs | Libraries, bookshops | Korean restaurants, karaoke |
| Zoning gate | `explicit_deny: [nightclub]` | `explicit_allow: [bar, cabaret]` |
| Recommendation | High traffic / low noise quadrant | High synergy / commercially tolerant |

Same agents. Same tools. Same scoring formula. Different strategy.

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Gemini 2.0 Flash (Vertex AI ADC) / Anthropic / Groq / Ollama |
| Data processing | pandas, geopandas, numpy, scipy |
| Spatial join | geopandas + ZCTA shapefiles |
| APIs | NYC Open Data (SODA), OpenStreetMap Overpass |
| RAG | FAISS + sentence-transformers (`all-MiniLM-L6-v2`) |
| Visualization | matplotlib |
| Frontend | Streamlit |
| Dependency management | uv |
