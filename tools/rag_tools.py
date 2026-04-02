"""
RAG (Retrieval-Augmented Generation) knowledge base for NYC zoning laws
and community district profiles.

Uses FAISS + sentence-transformers for local vector search.
Falls back to BM25-style keyword search if sentence-transformers is unavailable.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np

import config
from data.zoning_knowledge import ZONING_DOCUMENTS, get_all_text_chunks

# Optional imports – graceful fallback if not installed
try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class ZoningRAG:
    """
    Local vector knowledge base for NYC zoning laws and district profiles.

    On first init, builds a FAISS index over embedded zoning text chunks.
    Subsequent inits load from the persisted index if available.
    """

    def __init__(self):
        self._chunks = get_all_text_chunks()
        self._index = None
        self._embedder = None
        self._embeddings: Optional[np.ndarray] = None

        if _ST_AVAILABLE and _FAISS_AVAILABLE:
            self._init_vector_store()
        # else: keyword fallback is used automatically in retrieve()

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_vector_store(self):
        """Build or load the FAISS index."""
        os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
        index_path = os.path.join(config.VECTOR_STORE_DIR, "zoning.index")
        emb_path = os.path.join(config.VECTOR_STORE_DIR, "embeddings.npy")

        # Try loading persisted index
        if os.path.exists(index_path) and os.path.exists(emb_path):
            try:
                self._index = faiss.read_index(index_path)
                self._embeddings = np.load(emb_path)
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                return
            except Exception:
                pass

        # Build fresh index
        print("[RAG] Building FAISS index …")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c["content"] for c in self._chunks]
        self._embeddings = self._embedder.encode(texts, show_progress_bar=False)
        self._embeddings = self._embeddings.astype(np.float32)

        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(self._embeddings)

        faiss.write_index(self._index, index_path)
        np.save(emb_path, self._embeddings)
        print(f"[RAG] Index built: {len(texts)} chunks.")

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        """
        Retrieve the top-k most relevant zoning knowledge chunks for a query.
        Uses FAISS vector search when available, keyword search as fallback.
        """
        if self._index is not None and self._embedder is not None:
            return self._vector_search(query, k)
        return self._keyword_search(query, k)

    def retrieve_for_zip(self, zip_code: str, query: str = "", k: int = 4) -> list[dict]:
        """
        Retrieve knowledge specific to a ZIP code, optionally refined by query.
        Always includes documents tagged for that ZIP.
        """
        # Start with ZIP-tagged documents
        tagged = [
            c for c in self._chunks
            if zip_code in c.get("zip_codes", [])
        ]

        # If we have enough tagged, use them; otherwise augment with vector/keyword search
        if len(tagged) >= k:
            return tagged[:k]

        extra_query = f"{zip_code} zoning commercial {query}"
        extra = self.retrieve(extra_query, k=k - len(tagged))
        seen_ids = {c["id"] for c in tagged}
        for c in extra:
            if c["id"] not in seen_ids:
                tagged.append(c)
                seen_ids.add(c["id"])
            if len(tagged) >= k:
                break

        return tagged[:k]

    def retrieve_for_business(self, business_type: str, k: int = 4) -> list[dict]:
        """Retrieve zoning documents relevant to a specific business type."""
        query_map = {
            "quiet_cafe": "café coffee shop quiet reading commercial zoning C4 retail permitted",
            "bar": "bar nightlife entertainment alcohol license SLA late night commercial",
            "retail": "retail store commercial zoning permitted use group 6",
        }
        query = query_map.get(business_type, business_type)
        return self.retrieve(query, k=k)

    # ── Search implementations ─────────────────────────────────────────────

    def _vector_search(self, query: str, k: int) -> list[dict]:
        q_emb = self._embedder.encode([query], show_progress_bar=False).astype(np.float32)
        distances, indices = self._index.search(q_emb, min(k, len(self._chunks)))
        return [
            {**self._chunks[i], "score": float(distances[0][rank])}
            for rank, i in enumerate(indices[0])
            if i < len(self._chunks)
        ]

    def _keyword_search(self, query: str, k: int) -> list[dict]:
        """Simple TF-based keyword relevance scoring."""
        keywords = re.findall(r"\w+", query.lower())
        scores = []
        for chunk in self._chunks:
            text_lower = chunk["content"].lower()
            score = sum(text_lower.count(kw) for kw in keywords)
            scores.append(score)

        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._chunks[i] for i in ranked_indices[:k]]

    # ── Formatted outputs for LLM consumption ─────────────────────────────

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a readable context block for the LLM.

        Prepends a deterministic [ZONING METADATA] section aggregated from all
        retrieved chunks so the LLM sees hard allow/deny constraints before
        reading the semantic text.
        """
        # Aggregate metadata across all chunks (deduplicated)
        all_allow: set[str] = set()
        all_deny: set[str] = set()
        for chunk in chunks:
            all_allow.update(chunk.get("explicit_allow", []))
            all_deny.update(chunk.get("explicit_deny", []))

        parts: list[str] = []

        if all_allow or all_deny:
            meta_lines = ["[ZONING METADATA — check this FIRST before reading text below]"]
            if all_allow:
                meta_lines.append(f"explicit_allow: {sorted(all_allow)}")
            if all_deny:
                meta_lines.append(f"explicit_deny:  {sorted(all_deny)}")
            meta_lines.append(
                "RULE: If the requested business type appears in explicit_deny, verdict = CAUTION or FAIL.\n"
                "      If it appears ONLY in explicit_allow (not in deny), it may qualify for PASS.\n"
                "      If neither list has it, rely on the semantic text below."
            )
            parts.append("\n".join(meta_lines))

        for chunk in chunks:
            parts.append(f"[SOURCE: {chunk['title']}]\n{chunk['content'].strip()}")

        return "\n\n---\n\n".join(parts)


# ── Module-level singleton ────────────────────────────────────────────────────

_rag_instance: Optional[ZoningRAG] = None


def get_rag() -> ZoningRAG:
    """Return the module-level singleton ZoningRAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ZoningRAG()
    return _rag_instance


def query_zoning(zip_code: str, business_type: str = "quiet_cafe") -> str:
    """
    Convenience function: retrieve and format zoning context for a ZIP code.
    Returns a formatted string ready for LLM consumption.
    """
    rag = get_rag()
    chunks = rag.retrieve_for_zip(zip_code, query=business_type, k=4)
    if not chunks:
        return f"No specific zoning information found for ZIP {zip_code}."
    return rag.format_context(chunks)
