"""
NYC Open Data tools: MTA turnstile ridership, 311 complaints, ZCTA shapefiles.
All functions include fallback data so the app works even if APIs are unreachable.
"""
from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime, timedelta
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

import config


# ── MTA Ridership ─────────────────────────────────────────────────────────────

def fetch_mta_ridership(
    days_back: int = config.ROLLING_DAYS,
    app_token: str = config.NYC_APP_TOKEN,
    time_of_day: str = "all",  # "all" | "nightlife" | "daytime"
) -> pd.DataFrame:
    """
    Fetches MTA Subway Hourly Ridership from NYC Open Data (SODA).

    time_of_day filters:
      "nightlife" — Thu–Sat 8 PM–4 AM (bar/club foot traffic)
      "daytime"   — Mon–Fri 7 AM–8 PM (café/retail foot traffic)
      "all"       — no time filter (7-day rolling total)

    Returns aggregated total exits per station complex.
    Falls back to realistic synthetic data if the API is unreachable.
    """
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=max(days_back, 14))  # wider window for time-sliced data

    # Base date filter
    date_filter = (
        f"transit_timestamp >= '{start_dt.strftime('%Y-%m-%dT00:00:00')}' "
        f"AND transit_timestamp <= '{end_dt.strftime('%Y-%m-%dT23:59:59')}'"
    )

    # Time-of-day filter using Socrata date functions
    # dow: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
    if time_of_day == "nightlife":
        # Thu(4), Fri(5), Sat(6) — hours 20-23 or 0-3
        time_filter = (
            "(date_extract_dow(transit_timestamp) IN (4, 5, 6)) "
            "AND (date_extract_hh(transit_timestamp) >= 20 "
            "OR date_extract_hh(transit_timestamp) < 4)"
        )
        where = f"({date_filter}) AND ({time_filter})"
    elif time_of_day == "daytime":
        # Mon(1)–Sat(6) — hours 7-20
        time_filter = (
            "date_extract_dow(transit_timestamp) IN (1, 2, 3, 4, 5, 6) "
            "AND date_extract_hh(transit_timestamp) >= 7 "
            "AND date_extract_hh(transit_timestamp) < 20"
        )
        where = f"({date_filter}) AND ({time_filter})"
    else:
        where = date_filter

    params = {
        "$where": where,
        "$select": (
            "transit_timestamp,station_complex_id,station_complex,"
            "latitude,longitude,ridership,transfers"
        ),
        "$limit": 50000,
        "$order": "transit_timestamp DESC",
    }
    headers = {"X-App-Token": app_token} if app_token else {}

    try:
        resp = requests.get(
            config.MTA_RIDERSHIP_URL, params=params, headers=headers, timeout=30
        )
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            raise ValueError("Empty API response")

        df = pd.DataFrame(raw)
        df["ridership"] = pd.to_numeric(df.get("ridership", 0), errors="coerce").fillna(0)
        df["transfers"] = pd.to_numeric(df.get("transfers", 0), errors="coerce").fillna(0)
        df["exits"] = df["ridership"] + df["transfers"]
        df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
        df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")

        # Aggregate by station complex
        agg = (
            df.groupby(
                ["station_complex_id", "station_complex", "latitude", "longitude"],
                dropna=False,
            )
            .agg(total_exits=("exits", "sum"), days_observed=("transit_timestamp", "nunique"))
            .reset_index()
        )

        # Filter to Manhattan bounding box
        agg = agg[
            agg["latitude"].between(40.70, 40.88)
            & agg["longitude"].between(-74.03, -73.90)
        ].copy()

        if len(agg) < 5:
            raise ValueError("Too few Manhattan stations returned – using fallback")

        return agg.reset_index(drop=True)

    except Exception as exc:
        print(f"[MTA API] {exc} → using fallback data")
        return _fallback_mta_data(time_of_day=time_of_day)


# ── 311 Complaints ────────────────────────────────────────────────────────────

def fetch_311_complaints(
    days_back: int = config.ROLLING_DAYS,
    complaint_types: Optional[list[str]] = None,
    app_token: str = config.NYC_APP_TOKEN,
) -> pd.DataFrame:
    """
    Fetches 311 Service Requests (noise + construction) for Manhattan from SODA.
    Falls back to realistic synthetic data if the API is unreachable.
    """
    if complaint_types is None:
        complaint_types = [
            "Noise - Commercial",
            "Noise - Residential",
            "Noise - Street/Sidewalk",
            "Noise - Vehicle",
            "Construction",
        ]

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)

    type_clause = " OR ".join(f"complaint_type='{t}'" for t in complaint_types)
    where = (
        f"created_date >= '{start_dt.strftime('%Y-%m-%dT00:00:00')}' "
        f"AND created_date <= '{end_dt.strftime('%Y-%m-%dT23:59:59')}' "
        f"AND borough = 'MANHATTAN' "
        f"AND ({type_clause})"
    )

    params = {
        "$where": where,
        "$select": "created_date,complaint_type,descriptor,incident_zip,latitude,longitude,community_board",
        "$limit": 50000,
    }
    headers = {"X-App-Token": app_token} if app_token else {}

    try:
        resp = requests.get(
            config.COMPLAINTS_311_URL, params=params, headers=headers, timeout=30
        )
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            raise ValueError("Empty API response")

        df = pd.DataFrame(raw)
        df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
        df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
        df["incident_zip"] = df.get("incident_zip", "").astype(str).str[:5]

        df = df[df["incident_zip"].isin(config.MANHATTAN_ZIPS)].copy()
        if len(df) < 10:
            raise ValueError("Too few records returned – using fallback")

        return df.reset_index(drop=True)

    except Exception as exc:
        print(f"[311 API] {exc} → using fallback data")
        return _fallback_complaints_data(days_back)


# ── ZCTA Shapefile ────────────────────────────────────────────────────────────

def load_zcta_shapefile() -> gpd.GeoDataFrame:
    """
    Loads NYC ZCTA (ZIP Code Tabulation Area) shapefile.
    Tries the local cache first, then downloads from NYC Open Data.
    Falls back to centroid-buffer polygons if download fails.
    """
    os.makedirs(config.SHAPEFILE_CACHE_DIR, exist_ok=True)

    # Check cache
    cached = [
        f for f in os.listdir(config.SHAPEFILE_CACHE_DIR) if f.endswith(".shp")
    ]
    if cached:
        try:
            gdf = gpd.read_file(
                os.path.join(config.SHAPEFILE_CACHE_DIR, cached[0])
            )
            return gdf
        except Exception:
            pass

    # Try downloading GeoJSON
    try:
        print("[Shapefile] Downloading NYC ZCTA GeoJSON …")
        resp = requests.get(config.ZCTA_SHAPEFILE_URL, timeout=60)
        resp.raise_for_status()
        cache_path = os.path.join(config.SHAPEFILE_CACHE_DIR, "zcta.geojson")
        with open(cache_path, "wb") as f:
            f.write(resp.content)
        gdf = gpd.read_file(cache_path)
        if not gdf.empty:
            return gdf
    except Exception as exc:
        print(f"[Shapefile] Download failed: {exc} → using centroid fallback")

    return _centroid_gdf()


# ── Spatial Join ──────────────────────────────────────────────────────────────

def perform_spatial_join(
    mta_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    zcta_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Spatial join: assigns ZIP codes to MTA stations via ZCTA polygons,
    then merges with complaint aggregates to produce a ZIP-level panel.

    Returns columns:
        zip_code, total_exits, station_count, total_complaints,
        noise_complaints, construction_complaints, community_district, neighborhood
    """
    # --- Step 1: Assign ZIP to each MTA station via spatial join ---
    mta_valid = mta_df.dropna(subset=["latitude", "longitude"]).copy()

    if len(mta_valid) > 0:
        mta_gdf = gpd.GeoDataFrame(
            mta_valid,
            geometry=gpd.points_from_xy(mta_valid["longitude"], mta_valid["latitude"]),
            crs="EPSG:4326",
        )
        if zcta_gdf.crs.to_epsg() != 4326:
            zcta_gdf = zcta_gdf.to_crs("EPSG:4326")

        zip_col = _find_zip_column(zcta_gdf)

        joined = gpd.sjoin(
            mta_gdf,
            zcta_gdf[[zip_col, "geometry"]],
            how="left",
            predicate="within",
        )
        joined = joined.rename(columns={zip_col: "zip_code"})

        zip_traffic = (
            joined.groupby("zip_code")
            .agg(total_exits=("total_exits", "sum"), station_count=("station_complex_id", "nunique"))
            .reset_index()
        )
    else:
        # Coordinate-free fallback: aggregate by known ZIP centroids
        zip_traffic = pd.DataFrame(
            {"zip_code": config.MANHATTAN_ZIPS, "total_exits": 0.0, "station_count": 0}
        )

    # --- Step 2: Aggregate complaints by ZIP ---
    zip_noise = (
        complaints_df.groupby("incident_zip")
        .agg(
            total_complaints=("complaint_type", "count"),
            noise_complaints=(
                "complaint_type",
                lambda x: x.str.contains("Noise", na=False).sum(),
            ),
            construction_complaints=(
                "complaint_type",
                lambda x: (x == "Construction").sum(),
            ),
        )
        .reset_index()
        .rename(columns={"incident_zip": "zip_code"})
    )

    # --- Step 3: Merge ---
    panel = zip_traffic.merge(zip_noise, on="zip_code", how="outer")
    panel = panel[panel["zip_code"].isin(config.MANHATTAN_ZIPS)].copy()
    panel["total_exits"] = panel["total_exits"].fillna(0)
    panel["total_complaints"] = panel["total_complaints"].fillna(0)
    panel["noise_complaints"] = panel["noise_complaints"].fillna(0)
    panel["construction_complaints"] = panel["construction_complaints"].fillna(0)
    panel["station_count"] = panel["station_count"].fillna(0)
    panel["community_district"] = panel["zip_code"].map(config.ZIP_TO_CD)
    panel["neighborhood"] = panel["zip_code"].map(config.ZIP_NAMES)

    # Flag low-confidence rows (sparse data)
    panel["low_confidence"] = (panel["station_count"] < 1) | (
        panel["total_complaints"] < 5
    )

    return panel.reset_index(drop=True)


# ── POI / Cultural Mapping ────────────────────────────────────────────────────

_MANHATTAN_BBOX = "40.70,-74.03,40.88,-73.90"  # south,west,north,east (Overpass format)
# Mirror endpoints tried in order — overpass-api.de rate-limits aggressively
_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def fetch_poi_data(
    tags: list[dict],
    bbox: str = _MANHATTAN_BBOX,
    timeout: int = 45,
) -> gpd.GeoDataFrame:
    """
    Query OpenStreetMap via the Overpass API for points of interest.

    Args:
        tags: list of dicts like [{"amenity": "bar"}, {"cuisine": "korean"}]
              Each dict is one OR-clause (any tag match returns the POI).
        bbox: Overpass bounding box string "south,west,north,east"

    Returns:
        GeoDataFrame with columns: osm_id, name, lat, lon, tags, geometry
        Falls back to empty GeoDataFrame on failure.
    """
    # Build Overpass QL query — union of all tag filters
    # Key suffix "~" enables regex matching: {"cuisine~": "korean"} → ["cuisine"~"korean",i]
    node_clauses = []
    way_clauses = []
    for tag_dict in tags:
        for k, v in tag_dict.items():
            if k.endswith("~"):
                real_k = k[:-1]
                node_clauses.append(f'node["{real_k}"~"{v}",i]({bbox});')
                way_clauses.append(f'way["{real_k}"~"{v}",i]({bbox});')
            elif v == "*":
                node_clauses.append(f'node["{k}"]({bbox});')
                way_clauses.append(f'way["{k}"]({bbox});')
            else:
                node_clauses.append(f'node["{k}"="{v}"]({bbox});')
                way_clauses.append(f'way["{k}"="{v}"]({bbox});')

    query = (
        f"[out:json][timeout:{timeout}];\n"
        f"(\n"
        + "\n".join(node_clauses + way_clauses)
        + "\n);\nout center;"
    )

    elements = []
    last_exc = None
    for url in _OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, timeout=timeout + 10)
            resp.raise_for_status()
            body = resp.json()
            remark = body.get("remark", "")
            elements = body.get("elements", [])
            if remark:
                print(f"[POI] Overpass remark (likely timeout): {remark[:120]}")
            if elements:
                print(f"[POI] Overpass OK via {url} — {len(elements)} elements")
                break
            # Got 200 but zero elements — could be server-side timeout or rate limit.
            # Try next mirror before giving up.
            print(f"[POI] {url} returned 0 elements — trying next mirror")
        except Exception as exc:
            last_exc = exc
            print(f"[POI] {url} failed: {exc} — trying next mirror")
            continue

    # If all mirrors returned 0 elements, retry once with only node queries
    # (dropping way queries halves the result set size and usually fits in timeout).
    if not elements and node_clauses:
        fallback_query = (
            f"[out:json][timeout:{timeout}];\n(\n"
            + "\n".join(node_clauses)
            + "\n);\nout center;"
        )
        print("[POI] Retrying with node-only query (no way clauses) …")
        for url in _OVERPASS_URLS:
            try:
                resp = requests.post(url, data={"data": fallback_query}, timeout=timeout + 10)
                resp.raise_for_status()
                elements = resp.json().get("elements", [])
                if elements:
                    print(f"[POI] Node-only retry OK — {len(elements)} elements")
                    break
            except Exception:
                continue

    if not elements:
        print(f"[POI] All Overpass attempts returned 0 elements → returning empty GeoDataFrame")
        return gpd.GeoDataFrame()

    rows = []
    for el in elements:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat and lon:
            rows.append({
                "osm_id": el.get("id"),
                "name": el.get("tags", {}).get("name", ""),
                "lat": float(lat),
                "lon": float(lon),
                "osm_tags": str(el.get("tags", {})),
            })

    if not rows:
        print("[POI] Elements returned but none had lat/lon — returning empty GeoDataFrame")
        return gpd.GeoDataFrame()

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    print(f"[POI] Fetched {len(gdf)} POIs from Overpass")
    return gdf


def compute_poi_metrics(
    competitor_gdf: gpd.GeoDataFrame,
    synergy_gdf: gpd.GeoDataFrame,
    zcta_gdf: gpd.GeoDataFrame,
    manhattan_zips: list[str],
) -> pd.DataFrame:
    """
    Spatial join POIs to ZCTA polygons and compute per-ZIP metrics:
        competitor_count  — raw count of competitor POIs in ZIP
        synergy_count     — raw count of synergy POIs in ZIP
        competitor_density — competitors per km²
        synergy_score      — synergy POIs per km²
        location_quotient  — LQ > 1 = cluster/hub, < 1 = underserved

    Returns a DataFrame indexed by zip_code.
    """
    if zcta_gdf.crs.to_epsg() != 4326:
        zcta_gdf = zcta_gdf.to_crs("EPSG:4326")

    zip_col = _find_zip_column(zcta_gdf)

    # Compute area in km² (reproject to a metre-based CRS)
    zcta_proj = zcta_gdf.to_crs("EPSG:3857")
    zcta_gdf = zcta_gdf.copy()
    zcta_gdf["area_km2"] = zcta_proj.geometry.area / 1e6

    result_rows = {z: {"competitor_count": 0, "synergy_count": 0, "area_km2": 1.0}
                   for z in manhattan_zips}

    def _count_per_zip(poi_gdf: gpd.GeoDataFrame, count_col: str):
        if poi_gdf is None or poi_gdf.empty or "geometry" not in poi_gdf.columns:
            return
        joined = gpd.sjoin(
            poi_gdf[["geometry"]],
            zcta_gdf[[zip_col, "geometry", "area_km2"]],
            how="left",
            predicate="within",
        )
        for _, row in joined.iterrows():
            z = str(row.get(zip_col, ""))
            if z in result_rows:
                result_rows[z][count_col] = result_rows[z].get(count_col, 0) + 1
                result_rows[z]["area_km2"] = row.get("area_km2", 1.0)

    _count_per_zip(competitor_gdf, "competitor_count")
    _count_per_zip(synergy_gdf, "synergy_count")

    df = pd.DataFrame.from_dict(result_rows, orient="index").reset_index()
    df = df.rename(columns={"index": "zip_code"})

    # Density metrics
    df["competitor_density"] = df["competitor_count"] / df["area_km2"].clip(lower=0.01)
    df["synergy_score"] = df["synergy_count"] / df["area_km2"].clip(lower=0.01)

    # Location Quotient: ZIP share of category / ZIP share of all POIs
    total_competitors = df["competitor_count"].sum() or 1
    total_synergy = df["synergy_count"].sum() or 1
    total_poi = (total_competitors + total_synergy) or 1
    df["total_poi"] = df["competitor_count"] + df["synergy_count"]

    df["lq_competitor"] = (
        (df["competitor_count"] / total_competitors)
        / ((df["total_poi"] / total_poi).clip(lower=1e-6))
    ).fillna(0)

    return df[["zip_code", "competitor_count", "synergy_count",
               "competitor_density", "synergy_score", "lq_competitor"]].copy()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_zip_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = ["ZCTA5CE20", "ZCTA5CE10", "ZIPCODE", "ZIP", "MODZCTA", "zip_code"]
    for col in candidates:
        if col in gdf.columns:
            return col
    for col in gdf.columns:
        if "zip" in col.lower() or "zcta" in col.lower():
            return col
    raise ValueError(f"Cannot find ZIP column. Available: {gdf.columns.tolist()}")


def _centroid_gdf() -> gpd.GeoDataFrame:
    """Returns a GeoDataFrame with buffered-point polygons for each Manhattan ZIP."""
    centroids = {
        "10001": (40.7484, -73.9967), "10002": (40.7157, -73.9863),
        "10003": (40.7281, -73.9893), "10004": (40.7003, -74.0397),
        "10005": (40.7076, -74.0093), "10006": (40.7082, -74.0133),
        "10007": (40.7135, -74.0078), "10009": (40.7267, -73.9808),
        "10010": (40.7391, -73.9844), "10011": (40.7436, -74.0007),
        "10012": (40.7261, -73.9981), "10013": (40.7196, -74.0067),
        "10014": (40.7338, -74.0049), "10016": (40.7454, -73.9768),
        "10017": (40.7543, -73.9755), "10018": (40.7547, -74.0000),
        "10019": (40.7652, -73.9865), "10020": (40.7584, -73.9785),
        "10021": (40.7716, -73.9566), "10022": (40.7587, -73.9667),
        "10023": (40.7760, -73.9817), "10024": (40.7829, -73.9773),
        "10025": (40.7972, -73.9676), "10026": (40.8040, -73.9534),
        "10027": (40.8138, -73.9522), "10028": (40.7762, -73.9550),
        "10029": (40.7918, -73.9446), "10030": (40.8183, -73.9373),
        "10031": (40.8234, -73.9499), "10032": (40.8384, -73.9416),
        "10033": (40.8504, -73.9303), "10034": (40.8679, -73.9233),
        "10035": (40.7973, -73.9339), "10036": (40.7589, -73.9913),
        "10037": (40.8119, -73.9362), "10038": (40.7101, -74.0018),
        "10039": (40.8302, -73.9417), "10040": (40.8587, -73.9302),
        "10044": (40.7614, -73.9496),
    }
    rows = []
    for zip_code, (lat, lon) in centroids.items():
        pt = Point(lon, lat)
        rows.append({"ZIPCODE": zip_code, "geometry": pt.buffer(0.008)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ── Fallback data ─────────────────────────────────────────────────────────────

def _fallback_mta_data(time_of_day: str = "all") -> pd.DataFrame:
    """
    Realistic Manhattan station exit counts based on known ridership patterns.
    time_of_day controls which traffic profile is used:
      "nightlife" — Thu–Sat 8PM–4AM: LES/East Village/Hell's Kitchen dominate
      "daytime"   — Mon–Fri 7AM–8PM: commuter hubs dominate
      "all"       — 7-day totals (commuter-weighted)
    """
    if time_of_day == "nightlife":
        # Late-night/weekend ridership — nightlife neighborhoods dominate
        # Sources: MTA NTD, known NYC nightlife geography
        stations = [
            # (id, name, lat, lon, thu-sat-night exits over 3 nights)
            ("501", "Times Sq-42 St (N/Q/R/W/S/1/2/3/7/A/C/E)", 40.7560, -73.9872, 48000),  # Hell's Kitchen/Times Sq
            ("628", "Union Sq-14 St (4/5/6/L/N/Q/R/W)",          40.7351, -73.9906, 42000),  # gateway to LES/EV
            ("218", "14 St (A/C/E)",                              40.7391, -74.0069, 38000),  # Chelsea/Meatpacking
            ("625", "Canal St (A/C/E)",                           40.7187, -74.0001, 35000),  # LES/Chinatown
            ("302", "Spring St (C/E)",                            40.7261, -74.0031, 32000),  # SoHo/LES
            ("303", "Houston St (1)",                             40.7281, -74.0061, 30000),  # West Village/LES
            ("217", "23 St (C/E)",                                40.7454, -74.0018, 28000),  # Chelsea
            ("401", "59 St-Columbus Circle (1/A/B/C/D)",          40.7681, -73.9819, 26000),  # UWS gateway
            ("402", "66 St-Lincoln Center (1)",                   40.7741, -73.9827, 18000),
            ("601", "Lexington Av-53 St (E/M/6)",                 40.7575, -73.9721, 22000),
            ("611", "Grand Central-42 St (4/5/6/7/S)",            40.7527, -73.9772, 25000),  # much lower at night
            ("224", "34 St-Penn Station (1/2/3)",                 40.7506, -73.9971, 20000),  # lower at night
            ("403", "72 St (1/2/3)",                              40.7776, -73.9815, 14000),
            ("405", "86 St (1)",                                  40.7886, -73.9764, 11000),
            ("406", "96 St (1/2/3)",                              40.7942, -73.9720, 12000),
            ("418", "72 St (B/C)",                                40.7782, -73.9812, 16000),
            ("419", "86 St (B/C)",                                40.7853, -73.9739, 13000),
            ("420", "96 St (B/C)",                                40.7935, -73.9680, 12000),
            ("631", "Fulton St (A/C/2/3/4/5/J/Z)",               40.7101, -74.0079, 19000),
            ("635", "Wall St (4/5)",                              40.7074, -74.0113, 6000),   # dead at night
            ("636", "Bowling Green (4/5)",                        40.7046, -74.0141, 5000),
            ("404", "79 St (1)",                                  40.7836, -73.9798, 9000),
            ("407", "103 St (1)",                                 40.7999, -73.9686, 8000),
            ("408", "Cathedral Pkwy-110 St (1)",                  40.8034, -73.9650, 7000),
            ("409", "116 St (1)",                                 40.8079, -73.9638, 8000),
            ("410", "125 St (1)",                                 40.8159, -73.9578, 14000),
            ("421", "103 St (B/C)",                              40.7999, -73.9616, 11000),
            ("422", "110 St-Cathedral Pkwy (B/C)",               40.8026, -73.9588, 8000),
            ("120", "116 St-Columbia Univ (1)",                  40.8079, -73.9638, 10000),
            ("602", "51 St (6)",                                  40.7568, -73.9682, 8000),
            ("603", "68 St-Hunter (6)",                          40.7683, -73.9645, 9000),
            ("604", "77 St (6)",                                  40.7763, -73.9606, 7000),
            ("605", "86 St (4/5/6)",                             40.7773, -73.9554, 12000),
            ("606", "96 St (6)",                                  40.7841, -73.9477, 10000),
            ("607", "103 St (6)",                                 40.7904, -73.9473, 7000),
            ("608", "110 St (6)",                                 40.7953, -73.9417, 6000),
            ("609", "116 St (6)",                                 40.7997, -73.9393, 6000),
            ("610", "125 St (4/5/6)",                            40.8045, -73.9375, 13000),
        ]
        days_obs = 3
    elif time_of_day == "daytime":
        # Daytime/commuter — Grand Central and Penn Station dominate
        stations = [
            ("611", "Grand Central-42 St (4/5/6/7/S)",            40.7527, -73.9772, 220000),
            ("224", "34 St-Penn Station (1/2/3)",                  40.7506, -73.9971, 175000),
            ("501", "Times Sq-42 St (N/Q/R/W/S/1/2/3/7/A/C/E)",   40.7560, -73.9872, 195000),
            ("628", "Union Sq-14 St (4/5/6/L/N/Q/R/W)",           40.7351, -73.9906, 150000),
            ("601", "Lexington Av-53 St (E/M/6)",                  40.7575, -73.9721, 75000),
            ("401", "59 St-Columbus Circle (1/A/B/C/D)",           40.7681, -73.9819, 78000),
            ("403", "72 St (1/2/3)",                               40.7776, -73.9815, 62000),
            ("405", "86 St (1)",                                   40.7886, -73.9764, 55000),
            ("406", "96 St (1/2/3)",                               40.7942, -73.9720, 58000),
            ("407", "103 St (1)",                                  40.7999, -73.9686, 42000),
            ("408", "Cathedral Pkwy-110 St (1)",                   40.8034, -73.9650, 36000),
            ("409", "116 St (1)",                                  40.8079, -73.9638, 38000),
            ("410", "125 St (1)",                                  40.8159, -73.9578, 55000),
            ("418", "72 St (B/C)",                                 40.7782, -73.9812, 78000),
            ("419", "86 St (B/C)",                                 40.7853, -73.9739, 65000),
            ("420", "96 St (B/C)",                                 40.7935, -73.9680, 62000),
            ("421", "103 St (B/C)",                               40.7999, -73.9616, 58000),
            ("422", "110 St-Cathedral Pkwy (B/C)",                 40.8026, -73.9588, 38000),
            ("120", "116 St-Columbia Univ (1)",                   40.8079, -73.9638, 52000),
            ("625", "Canal St (A/C/E)",                            40.7187, -74.0001, 45000),
            ("631", "Fulton St (A/C/2/3/4/5/J/Z)",               40.7101, -74.0079, 82000),
            ("635", "Wall St (4/5)",                               40.7074, -74.0113, 44000),
            ("636", "Bowling Green (4/5)",                         40.7046, -74.0141, 32000),
            ("217", "23 St (C/E)",                                 40.7454, -74.0018, 46000),
            ("218", "14 St (A/C/E)",                               40.7391, -74.0069, 58000),
            ("402", "66 St-Lincoln Center (1)",                    40.7741, -73.9827, 50000),
            ("404", "79 St (1)",                                   40.7836, -73.9798, 46000),
            ("302", "Spring St (C/E)",                             40.7261, -74.0031, 34000),
            ("303", "Houston St (1)",                              40.7281, -74.0061, 28000),
            ("602", "51 St (6)",                                   40.7568, -73.9682, 35000),
            ("603", "68 St-Hunter (6)",                           40.7683, -73.9645, 40000),
            ("604", "77 St (6)",                                   40.7763, -73.9606, 35000),
            ("605", "86 St (4/5/6)",                              40.7773, -73.9554, 60000),
            ("606", "96 St (6)",                                   40.7841, -73.9477, 48000),
            ("607", "103 St (6)",                                  40.7904, -73.9473, 35000),
            ("608", "110 St (6)",                                  40.7953, -73.9417, 28000),
            ("609", "116 St (6)",                                  40.7997, -73.9393, 30000),
            ("610", "125 St (4/5/6)",                             40.8045, -73.9375, 60000),
        ]
        days_obs = 5
    else:
        # 7-day rolling totals (commuter-weighted)
        stations = [
            ("501", "Times Sq-42 St (N/Q/R/W/S/1/2/3/7/A/C/E)", 40.7560, -73.9872, 295000),
            ("611", "Grand Central-42 St (4/5/6/7/S)",           40.7527, -73.9772, 248000),
            ("224", "34 St-Penn Station (1/2/3)",                 40.7506, -73.9971, 195000),
            ("628", "Union Sq-14 St (4/5/6/L/N/Q/R/W)",          40.7351, -73.9906, 180000),
            ("401", "59 St-Columbus Circle (1/A/B/C/D)",          40.7681, -73.9819, 92000),
            ("403", "72 St (1/2/3)",                              40.7776, -73.9815, 74000),
            ("405", "86 St (1)",                                  40.7886, -73.9764, 65000),
            ("406", "96 St (1/2/3)",                              40.7942, -73.9720, 68000),
            ("407", "103 St (1)",                                 40.7999, -73.9686, 50000),
            ("408", "Cathedral Pkwy-110 St (1)",                  40.8034, -73.9650, 44000),
            ("409", "116 St (1)",                                 40.8079, -73.9638, 46000),
            ("410", "125 St (1)",                                 40.8159, -73.9578, 64000),
            ("418", "72 St (B/C)",                                40.7782, -73.9812, 91000),
            ("419", "86 St (B/C)",                                40.7853, -73.9739, 78000),
            ("420", "96 St (B/C)",                                40.7935, -73.9680, 74000),
            ("421", "103 St (B/C)",                              40.7999, -73.9616, 70000),
            ("422", "110 St-Cathedral Pkwy (B/C)",               40.8026, -73.9588, 46000),
            ("120", "116 St-Columbia Univ (1)",                  40.8079, -73.9638, 63000),
            ("625", "Canal St (A/C/E)",                           40.7187, -74.0001, 57000),
            ("631", "Fulton St (A/C/2/3/4/5/J/Z)",              40.7101, -74.0079, 98000),
            ("635", "Wall St (4/5)",                              40.7074, -74.0113, 50000),
            ("636", "Bowling Green (4/5)",                        40.7046, -74.0141, 39000),
            ("217", "23 St (C/E)",                                40.7454, -74.0018, 54000),
            ("218", "14 St (A/C/E)",                              40.7391, -74.0069, 69000),
            ("402", "66 St-Lincoln Center (1)",                   40.7741, -73.9827, 60000),
            ("404", "79 St (1)",                                  40.7836, -73.9798, 56000),
            ("302", "Spring St (C/E)",                            40.7261, -74.0031, 42000),
            ("303", "Houston St (1)",                             40.7281, -74.0061, 35000),
            ("601", "Lexington Av-53 St (E/M/6)",                 40.7575, -73.9721, 85000),
            ("602", "51 St (6)",                                  40.7568, -73.9682, 42000),
            ("603", "68 St-Hunter (6)",                          40.7683, -73.9645, 48000),
            ("604", "77 St (6)",                                  40.7763, -73.9606, 42000),
            ("605", "86 St (4/5/6)",                             40.7773, -73.9554, 72000),
            ("606", "96 St (6)",                                  40.7841, -73.9477, 58000),
            ("607", "103 St (6)",                                 40.7904, -73.9473, 42000),
            ("608", "110 St (6)",                                 40.7953, -73.9417, 35000),
            ("609", "116 St (6)",                                 40.7997, -73.9393, 38000),
            ("610", "125 St (4/5/6)",                            40.8045, -73.9375, 72000),
        ]
        days_obs = 7

    rows = []
    for sid, name, lat, lon, exits in stations:
        rows.append({
            "station_complex_id": sid,
            "station_complex": name,
            "latitude": lat,
            "longitude": lon,
            "total_exits": float(exits),
            "days_observed": days_obs,
        })
    return pd.DataFrame(rows)


def _fallback_complaints_data(days_back: int = 7) -> pd.DataFrame:
    """
    Realistic 311 complaint distribution across Manhattan ZIPs.
    Used when the 311 API is unavailable.
    """
    rng = np.random.default_rng(seed=42)

    # Approximate complaint counts per ZIP for a 7-day period
    zip_rates = {
        "10036": 420, "10001": 280, "10011": 185, "10014": 175,
        "10002": 155, "10003": 150, "10026": 148, "10027": 142,
        "10031": 140, "10030": 135, "10032": 165, "10033": 158,
        "10034": 128, "10040": 122, "10029": 138, "10035": 130,
        "10019": 170, "10018": 158, "10016": 125, "10022": 118,
        "10017": 112, "10010": 98,  "10012": 105, "10013": 88,
        "10021": 72,  "10022": 85,  "10028": 68,  "10023": 80,
        "10024": 75,  "10025": 82,  "10009": 142, "10039": 118,
        "10037": 125, "10007": 38,  "10004": 45,  "10005": 42,
        "10006": 40,  "10038": 55,  "10020": 35,  "10044": 22,
    }
    scale = days_back / 7.0

    types = [
        "Noise - Commercial",
        "Noise - Residential",
        "Noise - Street/Sidewalk",
        "Construction",
        "Noise - Vehicle",
    ]
    weights = [0.30, 0.28, 0.18, 0.15, 0.09]

    base = datetime.now()
    rows = []
    for zip_code, rate in zip_rates.items():
        if zip_code not in config.MANHATTAN_ZIPS:
            continue
        count = int(rate * scale)
        cd = config.ZIP_TO_CD.get(zip_code, "0")
        for _ in range(count):
            rows.append(
                {
                    "incident_zip": zip_code,
                    "complaint_type": rng.choice(types, p=weights),
                    "created_date": (
                        base - timedelta(days=float(rng.integers(0, days_back)))
                    ).isoformat(),
                    "latitude": None,
                    "longitude": None,
                    "community_board": f"MANHATTAN 0{cd}" if len(cd) == 1 else f"MANHATTAN {cd}",
                    "descriptor": rng.choice(["Loud Music/Party", "Construction/Demolition", "Banging/Pounding"]),
                }
            )

    return pd.DataFrame(rows)
