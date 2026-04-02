"""
Data Engineer Agent Node

Responsibilities:
  1. Fetch MTA subway ridership data (NYC Open Data SODA API)
  2. Fetch 311 noise/construction complaints (NYC Open Data SODA API)
  3. Load NYC ZCTA shapefile for spatial joins
  4. Perform spatial join (geopandas) to produce ZIP-level panel data
  5. Return enriched panel DataFrame to the Lead Strategist
"""
from __future__ import annotations

import config
from graph.state import AgentState
from tools.nyc_data_tools import (
    fetch_mta_ridership,
    fetch_311_complaints,
    load_zcta_shapefile,
    perform_spatial_join,
    fetch_poi_data,
    compute_poi_metrics,
)


def data_engineer_node(state: AgentState) -> dict:
    """
    LangGraph node: collects and spatially joins NYC data.
    Returns state updates for the Lead Strategist to consume.
    """
    logs = list(state.get("status_logs", []))
    business_type = state.get("business_type", "quiet_cafe")

    # ── Adapt data collection to business type ────────────────────────────
    if business_type == "bar":
        time_of_day = "nightlife"
        mta_window_label = "Thu–Sat 8PM–4AM"
        complaint_types = [
            "Noise - Commercial",
            "Noise - Street/Sidewalk",
            "Noise - Residential",
        ]
        # OSM tags: what counts as a competitor vs. a synergy anchor
        competitor_tags = [
            {"amenity": "bar"},
            {"amenity": "nightclub"},
            {"amenity": "music_venue"},
        ]
        synergy_tags = [
            {"cuisine~": "korean"},    # regex on indexed tag: fast, matches korean/korean_bbq/etc.
            {"amenity": "karaoke_box"},
            {"amenity": "karaoke"},
        ]
        # NOTE: name~ searches are unindexed in Overpass and stall for minutes — do not add them
        poi_label = "bars/clubs (competitors) + Korean venues (synergy)"

    elif business_type == "quiet_cafe":
        time_of_day = "daytime"
        mta_window_label = "Mon–Sat daytime"
        complaint_types = [
            "Noise - Commercial",
            "Noise - Residential",
            "Noise - Street/Sidewalk",
            "Construction",
        ]
        competitor_tags = [
            {"amenity": "cafe"},
            {"amenity": "coffee_shop"},
        ]
        synergy_tags = [
            {"amenity": "library"},
            {"amenity": "bookshop"},
            {"shop": "books"},
            {"amenity": "coworking_space"},
        ]
        poi_label = "cafés (competitors) + libraries/bookshops (synergy)"

    else:  # retail
        time_of_day = "all"
        mta_window_label = f"{config.ROLLING_DAYS}-day"
        complaint_types = [
            "Noise - Commercial",
            "Noise - Residential",
            "Noise - Street/Sidewalk",
            "Construction",
            "Noise - Vehicle",
        ]
        competitor_tags = [{"shop": "*"}]
        synergy_tags = [
            {"amenity": "restaurant"},
            {"amenity": "cafe"},
            {"tourism": "attraction"},
        ]
        poi_label = "retail shops (competitors) + restaurants/attractions (synergy)"

    # ── Step 1: Fetch MTA ridership ────────────────────────────────────────
    logs.append(
        f"👷 [Data Engineer] Querying MTA Subway Ridership API "
        f"({mta_window_label} window, NYC Open Data)…"
    )
    try:
        mta_df = fetch_mta_ridership(days_back=config.ROLLING_DAYS, time_of_day=time_of_day)
        logs.append(
            f"   ✅ MTA data: {len(mta_df):,} station records "
            f"({int(mta_df['total_exits'].sum()):,} total exits, {mta_window_label})"
        )
    except Exception as exc:
        logs.append(f"   ⚠️  MTA API error: {exc}. Using fallback data.")
        from tools.nyc_data_tools import _fallback_mta_data
        mta_df = _fallback_mta_data()

    # ── Step 2: Fetch 311 complaints ───────────────────────────────────────
    logs.append("👷 [Data Engineer] Querying 311 Complaint API (NYC Open Data)…")
    try:
        complaints_df = fetch_311_complaints(
            days_back=config.ROLLING_DAYS, complaint_types=complaint_types
        )
        logs.append(
            f"   ✅ 311 data: {len(complaints_df):,} complaints "
            f"({', '.join(complaints_df['complaint_type'].value_counts().head(2).index.tolist())} top types)"
        )
    except Exception as exc:
        logs.append(f"   ⚠️  311 API error: {exc}. Using fallback data.")
        from tools.nyc_data_tools import _fallback_complaints_data
        complaints_df = _fallback_complaints_data(config.ROLLING_DAYS)

    # ── Step 3: Load ZCTA shapefile ────────────────────────────────────────
    logs.append("👷 [Data Engineer] Loading NYC ZCTA shapefile for spatial join…")
    try:
        zcta_gdf = load_zcta_shapefile()
        logs.append(f"   ✅ ZCTA shapefile: {len(zcta_gdf)} ZIP polygons loaded.")
    except Exception as exc:
        logs.append(f"   ⚠️  Shapefile error: {exc}. Using centroid polygons.")
        from tools.nyc_data_tools import _centroid_gdf
        zcta_gdf = _centroid_gdf()

    # ── Step 4: Spatial join via geopandas ─────────────────────────────────
    logs.append("👷 [Data Engineer] Performing geopandas spatial join (stations → ZCTAs)…")
    try:
        panel_df = perform_spatial_join(mta_df, complaints_df, zcta_gdf)
        logs.append(
            f"   ✅ Panel built: {len(panel_df)} ZIP codes | "
            f"{int(panel_df['station_count'].sum())} stations mapped | "
            f"{int(panel_df['total_complaints'].sum()):,} complaints aggregated."
        )
    except Exception as exc:
        logs.append(f"   ❌ Spatial join error: {exc}")
        return {
            "status_logs": logs,
            "next": "lead_strategist",
            "phase": "analyzing",
            "error": f"Spatial join failed: {exc}",
        }

    # ── Step 5: POI cultural mapping (Overpass/OpenStreetMap) ─────────────
    logs.append(f"👷 [Data Engineer] Fetching POI data: {poi_label}…")
    try:
        competitor_gdf = fetch_poi_data(competitor_tags)
        synergy_gdf = fetch_poi_data(synergy_tags)
        poi_metrics = compute_poi_metrics(
            competitor_gdf, synergy_gdf, zcta_gdf, list(panel_df["zip_code"].astype(str))
        )
        poi_metrics["zip_code"] = poi_metrics["zip_code"].astype(str)
        panel_df["zip_code"] = panel_df["zip_code"].astype(str)
        panel_df = panel_df.merge(poi_metrics, on="zip_code", how="left")
        panel_df["competitor_count"] = panel_df["competitor_count"].fillna(0)
        panel_df["synergy_count"] = panel_df["synergy_count"].fillna(0)
        panel_df["competitor_density"] = panel_df["competitor_density"].fillna(0)
        panel_df["synergy_score"] = panel_df["synergy_score"].fillna(0)
        panel_df["lq_competitor"] = panel_df["lq_competitor"].fillna(0)
        if len(competitor_gdf) == 0 and len(synergy_gdf) > 0:
            logs.append(
                f"   ⚠️  POI data: competitor query returned 0 results (Overpass timeout/rate-limit). "
                f"Synergy anchors OK ({len(synergy_gdf)}). "
                f"Saturation/LQ metrics will be zero — scores are computed on 3 weights."
            )
            poi_warning = (
                "The OpenStreetMap Overpass API timed out while fetching competitor venue data. "
                "The saturation/LQ weight was dropped from scoring this run. "
                "**Please re-run the analysis** — Overpass rate limits reset within ~60 seconds."
            )
        else:
            logs.append(
                f"   ✅ POI data: {len(competitor_gdf)} competitors, "
                f"{len(synergy_gdf)} synergy anchors mapped to {len(panel_df)} ZIPs."
            )
            poi_warning = ""
    except Exception as exc:
        logs.append(f"   ⚠️  POI fetch failed: {exc}. Proceeding without POI metrics.")
        panel_df["competitor_count"] = 0
        panel_df["synergy_count"] = 0
        panel_df["competitor_density"] = 0.0
        panel_df["synergy_score"] = 0.0
        panel_df["lq_competitor"] = 0.0
        poi_warning = (
            "The OpenStreetMap Overpass API failed to fetch POI data. "
            "Scores were computed without saturation and synergy weights. "
            "**Please re-run the analysis** — Overpass rate limits reset within ~60 seconds."
        )

    # ── Step 6: Serialize & summarize ─────────────────────────────────────
    panel_json = panel_df.to_json(orient="records")

    top_traffic = panel_df.nlargest(3, "total_exits")[["zip_code", "neighborhood", "total_exits"]].to_dict("records")
    top_noisy = panel_df.nlargest(3, "total_complaints")[["zip_code", "neighborhood", "total_complaints"]].to_dict("records")

    data_summary = (
        f"Panel: {len(panel_df)} Manhattan ZIPs | "
        f"MTA window: {mta_window_label} | "
        f"Total exits: {int(panel_df['total_exits'].sum()):,} | "
        f"Total complaints: {int(panel_df['total_complaints'].sum()):,} | "
        f"Top traffic ZIPs: {', '.join(r['zip_code'] for r in top_traffic)} | "
        f"Top noise ZIPs: {', '.join(r['zip_code'] for r in top_noisy)} | "
        f"Low-confidence ZIPs: {panel_df['low_confidence'].sum()}"
    )

    logs.append(f"👷 [Data Engineer] ✅ Done. {data_summary}")
    logs.append("   → Handing off to Lead Strategist for EDA.")

    result = {
        "status_logs": logs,
        "panel_df_json": panel_json,
        "mta_records": len(mta_df),
        "complaints_records": len(complaints_df),
        "data_summary": data_summary,
        "next": "lead_strategist",
        "phase": "analyzing",
    }
    if poi_warning:
        result["poi_warning"] = poi_warning
    return result
