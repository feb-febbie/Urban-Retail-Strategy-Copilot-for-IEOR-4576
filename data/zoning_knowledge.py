"""
Embedded NYC zoning & community district knowledge for the RAG knowledge base.

This module serves as the primary qualitative data source. In production,
you can supplement or replace this with actual PDF documents in data/pdfs/.
The content is drawn from the NYC Zoning Resolution and Community District Profiles.
"""

ZONING_DOCUMENTS = [
    # ── Community District 7 (Upper West Side / Morningside Heights) ──────────
    {
        "id": "cd7_profile",
        "title": "Manhattan Community District 7 Profile – Upper West Side / Morningside Heights",
        "zip_codes": ["10023", "10024", "10025"],
        "explicit_allow": ["cafe", "retail", "restaurant", "bookstore", "gallery"],
        "explicit_deny": ["nightclub", "cabaret", "late_night_bar", "bar_with_dancing", "entertainment_venue_past_midnight"],
        "content": """
Community District 7 encompasses the Upper West Side and Morningside Heights neighborhoods,
spanning ZIP codes 10023, 10024, and 10025.

DEMOGRAPHICS & INCOME:
- Median household income: approximately $110,000–$135,000 (well above citywide median).
- Population: ~210,000 residents. Highly educated, professional demographic.
- Target consumer profile: book-buying, café-going, culturally engaged residents and families.

COMMERCIAL ZONING:
- C4-2 and C4-3 commercial districts run along Broadway from 72nd Street north through 110th Street.
- C4 districts permit: retail stores, restaurants, cafés, bookstores, specialty food shops,
  personal service establishments, cultural facilities, and offices.
- Maximum FAR (Floor Area Ratio) for commercial uses: 3.0–4.0 depending on specific sub-district.
- No restrictions on daytime café/retail operations.
- Commercial Overlay C1-5 and C2-5 on Columbus and Amsterdam Avenues.

RESIDENTIAL ZONING:
- Interior side streets are zoned R8 and R9 (high-density residential).
- Retail uses on residential side streets require special permit.
- No late-night entertainment venues (past midnight) permitted in residential zones.

NOISE ENVIRONMENT:
- Complaint data consistently places CD7 in the lower 15–20% of Manhattan noise intensity
  when normalized by foot traffic.
- Primary noise sources: domestic (residential), not commercial or construction.
- No large-scale construction projects anticipated in planning documents for 10025.

FOOT TRAFFIC & TRANSIT:
- 1/2/3 subway lines serve 72nd, 79th, 86th, 96th, 103rd, 110th, 116th St stations.
- B/C lines serve parallel stops on CPW.
- Steady, commuter-driven foot traffic (local residents + Columbia University).
- Traffic is less tourist-driven compared to Midtown, creating higher conversion for local retail.

STRATEGIC ASSESSMENT:
ZIP 10025 (Upper West Side / Morningside Heights) is commercially viable for a premium
quiet retail or café concept. The C4-2 zoning along Broadway explicitly permits café use.
The demographic profile (income, education, cultural orientation) matches a premium reading
café target customer. Noise levels are low relative to foot traffic. Columbia University
proximity generates consistent intellectual/literary consumer demand.
        """,
    },

    # ── Community District 6 (Gramercy / Murray Hill / Kips Bay) ─────────────
    {
        "id": "cd6_profile",
        "title": "Manhattan Community District 6 Profile – Gramercy / Murray Hill",
        "zip_codes": ["10010", "10016", "10017", "10022"],
        "explicit_allow": ["cafe", "retail", "restaurant", "bar", "lounge"],
        "explicit_deny": ["nightclub", "cabaret", "heavy_manufacturing"],
        "content": """
Community District 6 covers Gramercy Park, Murray Hill, Kips Bay, and Turtle Bay.

DEMOGRAPHICS & INCOME:
- Median household income: approximately $95,000–$120,000.
- Population: ~110,000. Mix of young professionals and established residents.
- Gramercy Park is one of Manhattan's most prestigious residential addresses.

COMMERCIAL ZONING:
- C6-3 and C5 zones run along Lexington and Third Avenues.
- C5 permits: retail, restaurants, offices, hotels. Full commercial use.
- Park Avenue South and Third Avenue corridors: strong retail activity.
- Zoning supports mid-to-high-end retail and food service concepts.

NOISE ENVIRONMENT:
- Moderate noise levels. Midtown East proximity means some commercial noise.
- Murray Hill 10016 has higher complaint density (bars, restaurants on Third Ave).
- 10022 (Sutton Place / Midtown East) is quieter due to residential character.
- Overall: medium noise intensity after normalization.

TRANSIT:
- 4/5/6, N/Q/R/W, L, 7 lines offer good coverage across the district.
- Commuter-heavy foot traffic during peak hours.
        """,
    },

    # ── Community District 4 (Hell's Kitchen / Midtown West) ─────────────────
    {
        "id": "cd4_profile",
        "title": "Manhattan Community District 4 Profile – Hell's Kitchen / Midtown West",
        "zip_codes": ["10018", "10019", "10020", "10036"],
        "explicit_allow": ["bar", "nightclub", "cabaret", "tavern", "entertainment_venue", "restaurant", "retail"],
        "explicit_deny": ["low_impact_quiet_retail", "heavy_manufacturing"],
        "content": """
Community District 4 covers the Hell's Kitchen / Clinton and Midtown West areas,
including ZIP codes 10018, 10019, 10036.

CRITICAL FLAG – ZIP 10036 (Times Square):
Times Square is a HIGH-TRAFFIC / HIGH-INTERFERENCE zone. It is NOT suitable for
premium quiet retail or café concepts. Analysis of 311 data consistently places
10036 in the top quintile of noise complaints citywide. Tourist-driven foot traffic
is high in absolute volume but creates poor conversion for experience-driven retail.
Noise - Commercial complaints are among the highest in Manhattan.

COMMERCIAL ZONING:
- C6-4 and C6-7 (highest intensity commercial) zones dominate 10036.
- These zones allow virtually all commercial uses including entertainment.
- The C6 zoning that makes Times Square commercially viable ALSO permits
  the dense entertainment, signage, and noise that makes it unsuitable for
  quiet premium retail.
- 10018 (Garment District): C6-2 and M1-6 zones; some manufacturing legacy.

NOISE ENVIRONMENT:
- 10036: Highest noise complaint density in CD4, typically top 5% citywide.
- Construction noise: Ongoing Midtown development projects.
- Commercial noise: Bars, venues, theaters, tourist activity.

STRATEGIC ASSESSMENT:
ZIP 10036 represents the canonical "High Traffic / High Noise" trap.
While absolute exit volumes from Times Square stations are the highest in Manhattan,
this metric is misleading for premium quiet retail.
Recommendation: REJECT 10036 for quiet café/retail concept due to chronic high noise.
        """,
    },

    # ── Community District 2 (Greenwich Village / SoHo / TriBeCa) ────────────
    {
        "id": "cd2_profile",
        "title": "Manhattan Community District 2 Profile – Greenwich Village / SoHo / TriBeCa",
        "zip_codes": ["10012", "10013", "10014"],
        "explicit_allow": ["cafe", "retail", "restaurant", "small_bar", "lounge"],
        "explicit_deny": ["large_format_nightclub", "cabaret_with_dancing", "manufacturing"],
        "content": """
Community District 2 covers Greenwich Village, SoHo, TriBeCa, and Little Italy.

DEMOGRAPHICS & INCOME:
- Median household income: $130,000–$170,000 (one of highest in Manhattan).
- Strong arts, media, and finance professional demographic.
- High density of independent retail, galleries, boutique cafés already present.

COMMERCIAL ZONING:
- SoHo 10012: M1-5A and M1-5B manufacturing zones (SoHo Cast Iron District).
  IMPORTANT: Commercial uses in SoHo require Landmark Preservation approval.
  Ground-floor retail is permitted but subject to strict design guidelines.
  New food service establishments may require Board of Standards and Appeals variance.
- TriBeCa 10013: C6-2A and C6-3A zones. Commercial use freely permitted.
- West Village 10014: C2-6 and C1-6 overlays on residential streets. Cafés permitted
  but scale limitations apply (typically neighborhood retail scale, not large format).

NOISE ENVIRONMENT:
- Moderate noise levels, driven by nightlife (West Village 10014) and tourism (SoHo).
- TriBeCa 10013 is notably quieter—primarily daytime residential character.
- SoHo weekend tourist crowds generate significant ambient noise complaints.

STRATEGIC NOTES:
- SoHo has zoning complexity (manufacturing historic district) that creates regulatory risk.
- TriBeCa is viable but daytime foot traffic is lower than Upper West Side.
- West Village is viable but space is limited and rents are very high.
        """,
    },

    # ── Community District 8 (Upper East Side) ────────────────────────────────
    {
        "id": "cd8_profile",
        "title": "Manhattan Community District 8 Profile – Upper East Side",
        "zip_codes": ["10021", "10028", "10044"],
        "explicit_allow": ["cafe", "retail", "restaurant", "gallery", "bookstore"],
        "explicit_deny": ["nightclub", "cabaret", "late_night_bar", "bar_with_dancing"],
        "content": """
Community District 8 covers the Upper East Side (UES), including Carnegie Hill,
Yorkville, and Roosevelt Island.

DEMOGRAPHICS & INCOME:
- Median household income: $120,000–$160,000. One of the wealthiest districts in NYC.
- Strong luxury retail presence. Conservative, established residential community.
- High density of museums (Met, Guggenheim, Whitney area museums).

COMMERCIAL ZONING:
- C2-5 and C1-5 overlays on Lexington and Third Avenue.
- Second Avenue Subway (Q train) corridor: C2-5 zone, café and retail freely permitted.
- Madison Avenue: C2-5 with strong luxury retail tradition. New café concepts viable.
- Park Avenue: Primarily residential with no commercial activity permitted (R10 zone).

NOISE ENVIRONMENT:
- Among the quietest commercial districts in Manhattan.
- UES 10021 and 10028 have lower noise complaint rates than UWS.
- Second Avenue Subway construction complete—complaint spike has normalized.
- Museum Mile creates cultural foot traffic, lower noise intensity.

TRANSIT:
- 4/5/6 on Lex; Q train on Second Avenue (completed 2017).
- Moderate but steady transit ridership—local resident focus, not tourist-heavy.

STRATEGIC NOTES:
- Strong demographic match for premium quiet retail.
- Lower absolute traffic volume than UWS (10025).
- Q train extension has improved connectivity and may be driving foot traffic growth.
        """,
    },

    # ── NYC Zoning Resolution – Commercial District Rules ─────────────────────
    {
        "id": "zoning_commercial_districts",
        "title": "NYC Zoning Resolution – Commercial District (C1–C6) Use Groups",
        "zip_codes": [],
        "explicit_allow": [],
        "explicit_deny": [],
        "content": """
NYC ZONING RESOLUTION – COMMERCIAL DISTRICTS: KEY RULES FOR RETAIL & CAFÉ USE

C1 DISTRICTS (Local retail):
- Permitted uses: Grocery stores, pharmacies, personal services, small restaurants.
- Cafés and coffee shops: Permitted (Use Group 6).
- Bars/taverns: Generally NOT permitted in C1.
- Floor area limit for individual establishments: 10,000 SF.

C2 DISTRICTS (Local service):
- All C1 uses plus: larger restaurants, copy centers, car services.
- Cafés: Permitted (Use Group 6).
- Small bars: Permitted with conditions.
- Common as overlays in residential neighborhoods (C2-5, C2-6).

C4 DISTRICTS (Regional commercial centers):
- Major shopping streets and regional commercial hubs.
- Permitted uses: Retail stores of any size, restaurants, cafés, offices, hotels,
  entertainment (movies, live performance), health clubs.
- Cafés and reading rooms: Explicitly permitted (Use Group 6, 10).
- C4-2: Common along Broadway (UWS). FAR 3.0 commercial.
- C4-3: Higher intensity. FAR 4.0 commercial.
- Bars and nightlife: Permitted subject to State Liquor Authority licensing.

C5 DISTRICTS (Restricted central commercial):
- Used in high-value central areas (Park Ave, portions of Fifth Ave).
- Offices and luxury retail preferred. Cafés permitted.

C6 DISTRICTS (General central commercial):
- Highest-intensity commercial district.
- All commercial uses permitted.
- C6-4 through C6-9: Midtown Manhattan core. Permits massive office buildings.
- Cafés and retail permitted but rents are extremely high.
- Note: C6-4 in Times Square area also permits adult entertainment; not suitable
  for premium family-oriented café.

PROHIBITED USES IN ALL COMMERCIAL DISTRICTS:
- Heavy manufacturing (M-zone uses)
- Waste transfer stations

USE GROUP REFERENCE:
- Use Group 6: Retail stores, personal service. Includes cafés, coffee shops, bookstores.
- Use Group 10: Amusement, recreation. Includes reading rooms, galleries.
- Use Group 12: Eating/drinking. Includes restaurants, bars (requires SLA license for alcohol).
        """,
    },

    # ── NYC Zoning Resolution – Special Districts ─────────────────────────────
    {
        "id": "zoning_special_districts",
        "title": "NYC Zoning Resolution – Special Purpose Districts in Manhattan",
        "zip_codes": [],
        "explicit_allow": [],
        "explicit_deny": [],
        "content": """
SPECIAL PURPOSE ZONING DISTRICTS IN MANHATTAN (relevant to retail planning)

SPECIAL SOHO-CAST IRON HISTORIC DISTRICT (SoHo, ZIP 10012):
- Manufacturing legacy district. Ground-floor retail permitted only on designated
  "street frontage" blocks.
- Artists' live-work lofts: Key community character. New purely commercial tenants
  face community opposition and regulatory scrutiny.
- Landmark review required for exterior changes.
- CAUTION: Regulatory complexity adds risk and permitting delays for new retail entrants.

SPECIAL MIDTOWN DISTRICT (42nd–60th St, ZIPs 10018, 10019, 10022, 10036):
- Special rules to manage density and pedestrian flow.
- Theater Subdistrict (around 10036): Requires contribution to theater preservation fund
  for major developments.
- Fifth Avenue Subdistrict: Design controls on retail storefronts.
- Overall: High regulatory overhead in Special Midtown District.

SPECIAL TRANSIT LAND USE DISTRICT:
- Applies near major transit hubs. Encourages ground-floor retail.
- Benefits: Reduced parking requirements. Easier approval for retail uses near transit.

SPECIAL NATURAL AREA DISTRICT (Inwood, parts of 10034, 10040):
- Strict environmental controls. Limited commercial development permitted.
- NOT suitable for new commercial retail expansion.

SPECIAL HARLEM RIVER WATERFRONT DISTRICT:
- Mixed-use development zone along Harlem River. Emerging retail corridor.

LANDMARK PRESERVATION:
- Landmarked buildings and historic districts require additional approval from
  Landmarks Preservation Commission (LPC) for exterior changes.
- Affects portions of Greenwich Village, TriBeCa, SoHo, NoHo, and the Upper West Side
  (Upper West Side/Central Park West Historic District).
- Interior retail build-out: Generally NOT subject to LPC review unless exterior is altered.
        """,
    },

    # ── Strategic Framework Document ──────────────────────────────────────────
    {
        "id": "strategy_framework",
        "title": "Urban Retail Location Strategy Framework – Premium Quiet Retail",
        "zip_codes": [],
        "explicit_allow": [],
        "explicit_deny": [],
        "content": """
STRATEGIC FRAMEWORK: OPTIMAL LOCATION FOR PREMIUM QUIET RETAIL (e.g., High-End Reading Café)

OBJECTIVE FUNCTION:
Maximize: (Foot Traffic Score × Demographic Match) − (Noise Penalty × Regulatory Risk)
Subject to: Commercial zoning permits café use, noise level in bottom quartile after normalization.

DECISION CRITERIA HIERARCHY:
1. PRIMARY: Is the location in the "High Traffic / Low Normalized Noise" quadrant?
2. SECONDARY: Does zoning explicitly permit café/retail use without variance?
3. TERTIARY: Does the demographic profile match premium positioning?
4. QUATERNARY: Is regulatory risk (special districts, landmarking) manageable?

REJECTION CRITERIA:
- Location in "High Traffic / High Noise" quadrant (e.g., Times Square 10036).
- Strict residential zoning with no commercial overlay.
- Special district status creating prohibitive regulatory complexity.
- Complaint growth trend showing deteriorating noise environment.

CONFIDENCE SCORING:
- HIGH CONFIDENCE: Strong quantitative signal + clean zoning + demographic match.
- MEDIUM CONFIDENCE: Good quantitative signal but limited station coverage or sparse
  complaint data for the ZIP code (low observation count).
- LOW CONFIDENCE: Zip code has < 2 subway stations mapped or < 30 complaints recorded.
  Result flagged but preserved in ranking for human review.

TEMPORAL VALIDITY:
- Recommendations based on 7-day rolling window are valid for near-term tactical decisions.
- For long-term lease commitments, extend analysis window to 90 days.
- Seasonal adjustment: Summer months show tourist spikes in Midtown; analyze Q1/Q4 data
  for stable resident-traffic signal.

COMPETITIVE LANDSCAPE NOTE:
This framework optimizes for location quality, NOT competitive gap analysis.
Presence of existing successful cafés in the target ZIP is treated as a positive
demand signal, not a barrier. (Market validation, not saturation concern, given
premium differentiation strategy.)
        """,
    },
]

# Build a flat lookup: zip_code → list of relevant document IDs
ZIP_TO_DOC_IDS: dict[str, list[str]] = {}
for doc in ZONING_DOCUMENTS:
    for z in doc["zip_codes"]:
        ZIP_TO_DOC_IDS.setdefault(z, []).append(doc["id"])


def get_all_text_chunks() -> list[dict]:
    """Return list of chunk dicts for RAG ingestion.

    Each chunk carries the parent document's metadata arrays so that
    ``format_context`` can surface them to the LLM as a deterministic gate.
    """
    chunks = []
    for doc in ZONING_DOCUMENTS:
        # Split large documents into ~800-char chunks with overlap
        text = doc["content"].strip()
        chunk_size = 800
        overlap = 100
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if len(chunk) < 100:
                continue
            chunks.append(
                {
                    "id": f"{doc['id']}_chunk_{i}",
                    "title": doc["title"],
                    "zip_codes": doc["zip_codes"],
                    "explicit_allow": doc.get("explicit_allow", []),
                    "explicit_deny": doc.get("explicit_deny", []),
                    "content": chunk,
                }
            )
    return chunks
