"""
Central configuration for the Urban Retail Strategy Copilot.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM provider keys (the app tries them in this order) ─────────────────────
# 1. Anthropic (paid)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# 2. Groq (free tier) — console.groq.com
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# 3. Google Gemini (free tier) — aistudio.google.com
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

# 4. Ollama (local) — no key needed, auto-detected via localhost:11434

# ── NYC Open Data ─────────────────────────────────────────────────────────────
NYC_APP_TOKEN = os.environ.get("NYC_OPEN_DATA_APP_TOKEN", "")

# SODA API endpoints
MTA_RIDERSHIP_URL = "https://data.ny.gov/resource/wujg-7c2s.json"
COMPLAINTS_311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# NYC ZCTA shapefile (ZIP Code Tabulation Areas)
# GeoJSON endpoint — more reliable than the Shapefile export
ZCTA_SHAPEFILE_URL = (
    "https://data.cityofnewyork.us/resource/pri4-ifjk.geojson?$limit=300"
)

# ── Analysis parameters ───────────────────────────────────────────────────────
ROLLING_DAYS = 7          # Days of data to pull
WINSORIZE_PCT = 0.95      # Winsorization upper percentile
TRAFFIC_QUARTILE = 0.75   # Upper quartile threshold for "high traffic"
NOISE_QUARTILE = 0.25     # Lower quartile threshold for "low noise"
MAX_ITERATIONS = 4        # Max Lead Strategist iterations before stopping
MAX_TOOL_ROUNDS = 6       # Max tool-use rounds inside Lead Strategist
MAX_CANDIDATES_TO_CHECK = 5  # Top N candidates to verify with zoning

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHAPEFILE_CACHE_DIR = os.path.join(DATA_DIR, "shapefiles")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vectorstore")

# ── Manhattan ZIP codes ───────────────────────────────────────────────────────
MANHATTAN_ZIPS = [
    "10001", "10002", "10003", "10004", "10005", "10006", "10007",
    "10009", "10010", "10011", "10012", "10013", "10014", "10016",
    "10017", "10018", "10019", "10020", "10021", "10022", "10023",
    "10024", "10025", "10026", "10027", "10028", "10029", "10030",
    "10031", "10032", "10033", "10034", "10035", "10036", "10037",
    "10038", "10039", "10040", "10044",
]

# ZIP → Community District mapping (Manhattan)
ZIP_TO_CD = {
    "10001": "5",  "10002": "3",  "10003": "3",  "10004": "1",
    "10005": "1",  "10006": "1",  "10007": "1",  "10009": "3",
    "10010": "6",  "10011": "4",  "10012": "2",  "10013": "2",
    "10014": "2",  "10016": "6",  "10017": "6",  "10018": "4",
    "10019": "4",  "10020": "4",  "10021": "8",  "10022": "6",
    "10023": "7",  "10024": "7",  "10025": "7",  "10026": "10",
    "10027": "10", "10028": "8",  "10029": "11", "10030": "10",
    "10031": "9",  "10032": "12", "10033": "12", "10034": "12",
    "10035": "11", "10036": "4",  "10037": "10", "10038": "1",
    "10039": "9",  "10040": "12", "10044": "8",
}

# ZIP → neighborhood name (for display)
ZIP_NAMES = {
    "10001": "Chelsea/Garment District",
    "10002": "Lower East Side",
    "10003": "East Village / Noho",
    "10004": "Financial District",
    "10005": "Financial District",
    "10006": "Financial District",
    "10007": "Civic Center / TriBeCa",
    "10009": "East Village / Alphabet City",
    "10010": "Gramercy Park / Flatiron",
    "10011": "Chelsea",
    "10012": "SoHo / Little Italy",
    "10013": "TriBeCa / SoHo",
    "10014": "West Village / Greenwich Village",
    "10016": "Murray Hill / Kips Bay",
    "10017": "Midtown East / Turtle Bay",
    "10018": "Garment District / Hell's Kitchen",
    "10019": "Midtown West / Hell's Kitchen",
    "10020": "Rockefeller Center",
    "10021": "Upper East Side",
    "10022": "Midtown East / Sutton Place",
    "10023": "Upper West Side / Lincoln Square",
    "10024": "Upper West Side",
    "10025": "Upper West Side / Morningside Heights",
    "10026": "Central Harlem",
    "10027": "Central Harlem / Morningside Heights",
    "10028": "Upper East Side / Yorkville",
    "10029": "East Harlem / Yorkville",
    "10030": "Central Harlem",
    "10031": "Hamilton Heights / Harlem",
    "10032": "Washington Heights",
    "10033": "Washington Heights",
    "10034": "Inwood / Washington Heights",
    "10035": "East Harlem",
    "10036": "Midtown / Times Square / Hell's Kitchen",
    "10037": "Central Harlem",
    "10038": "Financial District / South Street Seaport",
    "10039": "Harlem / Sugar Hill",
    "10040": "Washington Heights / Inwood",
    "10044": "Roosevelt Island",
}
