FROM python:3.12-slim

# Copy uv from the official image (same pattern as your previous project)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ── System packages ────────────────────────────────────────────────────────────
# libgomp1  → required by faiss-cpu (OpenMP)
# libgl1    → required by some matplotlib backends on slim
# We do NOT need libgdal-dev: shapely/pyogrio/pyproj all ship bundled native libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
COPY pyproject.toml ./

# If you have generated a uv.lock (run `uv lock` locally first), copy it too
# and switch to --frozen for reproducible builds:
#   COPY uv.lock ./
#   RUN uv sync --frozen --no-dev
RUN uv sync --no-dev

# ── Pre-download sentence-transformers model ───────────────────────────────────
# Baking the model into the image avoids a slow cold-start download at runtime.
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model cached successfully.')"

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# ── Runtime ───────────────────────────────────────────────────────────────────
# .streamlit/config.toml sets port=8080, headless=true, address=0.0.0.0
CMD ["uv", "run", "streamlit", "run", "app.py"]
