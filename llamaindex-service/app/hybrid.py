"""Hybrid recall channel (C2): BM25 sparse vectors next to the dense ones.

Dense cosine stays the primary signal — every downstream mechanism
(RECALL_MIN_SCORE, recency decay, importance weighting, project boost) is
calibrated to its scale. That is why this module does NOT use Qdrant's
server-side RRF fusion: rank-fusion scores (~1/60 per rank) would silently
break all of them. Instead the sparse channel can only RESCUE keyword
matches the dense model under-ranks (exact tokens like "10MB", error codes,
identifiers): a hit's similarity becomes
max(dense_cosine, RECALL_BM25_WEIGHT * bm25_ratio), where bm25_ratio is the
hit's BM25 score normalized by the query's best BM25 score. Dense-ranked
results are never demoted.

The query side is gated to identifier-like terms only (tokens with digits,
snake_case, camelCase, ACRONYMS, dotted.paths, quoted spans). Measured on
the real 450-chunk corpus: feeding the whole natural-language question to
BM25 lets its common words ("tài liệu", "nhắc đến", ...) outscore the one
rare token — IDF alone doesn't save it because fastembed has no
Vietnamese/Japanese stopwords — and the rescue never reaches top-k. A
prompt with no identifier-like token skips the sparse channel entirely.
Stored texts still get full-text sparse vectors; only queries are filtered.

Sparse vectors live under the named vector BM25_VECTOR_NAME in the same
collections (the dense vector stays unnamed, so all existing dense code is
untouched). Qdrant applies IDF itself (Modifier.IDF on the sparse schema);
fastembed's Bm25 supplies the term frequencies. Collections without the
sparse schema (pre-migration) and environments where the BM25 model can't
load degrade to dense-only — recall must never fail because of the extra
channel.
"""

import logging
import re

from qdrant_client.http import models as qmodels

from app import config

logger = logging.getLogger("uvicorn")

_model = None
_model_failed = False
# (id(client), collection) -> whether the collection carries the sparse
# vector. Cached because it is consulted on every write and every recall.
_collection_support: dict = {}


def _get_model():
    global _model, _model_failed
    if not config.HYBRID_BM25 or _model_failed:
        return None
    if _model is None:
        try:
            from fastembed import SparseTextEmbedding

            _model = SparseTextEmbedding(config.BM25_MODEL)
        except Exception as exc:
            _model_failed = True
            logger.warning(
                "hybrid: BM25 model %s unavailable (%s) — recall stays dense-only",
                config.BM25_MODEL, exc,
            )
            return None
    return _model


def collection_enabled(client, collection: str) -> bool:
    if not config.HYBRID_BM25:
        return False
    key = (id(client), collection)
    if key not in _collection_support:
        try:
            params = client.get_collection(collection).config.params
            _collection_support[key] = config.BM25_VECTOR_NAME in (params.sparse_vectors or {})
        except Exception:
            return False  # collection missing — don't cache the answer
    return _collection_support[key]


def _to_sparse(embedding) -> qmodels.SparseVector | None:
    if embedding is None or not len(embedding.indices):
        return None
    return qmodels.SparseVector(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )


def text_vector(text: str) -> qmodels.SparseVector | None:
    model = _get_model()
    if model is None or not (text or "").strip():
        return None
    return _to_sparse(next(iter(model.embed([text])), None))


# One alternation per identifier shape; quoted spans are taken verbatim.
_EXACT_TERM_RE = re.compile(
    r"`([^`\n]{2,80})`|\"([^\"\n]{2,80})\"|'([^'\n]{2,80})'|「([^」\n]{2,80})」"
    r"|\b([A-Za-z]\w*\d\w*|\d+[A-Za-z]\w*)\b"      # letter/digit mixes: 10MB, sha256, m3
    r"|\b(\w+_\w+(?:_\w+)*)\b"                     # snake_case / SCREAMING_SNAKE
    r"|\b([A-Z]{2,}[a-z]\w+|[a-z]+[A-Z]\w+)\b"     # CamelCase / mixedCase
    r"|\b([A-Z]{3,})\b"                            # bare acronyms
    r"|\b(\w+\.\w+(?:\.\w+)+)\b"                   # dotted.paths
)


def exact_terms(text: str) -> list[str]:
    """The identifier-like terms of a prompt — the only thing the sparse
    channel is allowed to search for."""
    terms = []
    for groups in _EXACT_TERM_RE.findall(text or ""):
        term = next((g for g in groups if g), "")
        if len(term) >= 2 and term not in terms:
            terms.append(term)
    return terms


def query_vector(text: str) -> qmodels.SparseVector | None:
    model = _get_model()
    if model is None:
        return None
    terms = exact_terms(text)
    if not terms:
        return None
    return _to_sparse(next(iter(model.query_embed(" ".join(terms))), None))


def point_vector(client, collection: str, dense: list, text: str):
    """Vector payload for an upsert: {"": dense, "bm25": sparse} when the
    collection and model support it, the plain dense list otherwise."""
    if collection_enabled(client, collection):
        sparse = text_vector(text)
        if sparse is not None:
            return {"": dense, config.BM25_VECTOR_NAME: sparse}
    return dense


def search(client, collection: str, query_text: str, query_filter, limit: int,
           with_payload=True) -> list:
    """BM25 candidates for a query; [] whenever the channel is unavailable."""
    if not collection_enabled(client, collection):
        return []
    sparse = query_vector(query_text)
    if sparse is None:
        return []
    try:
        return client.query_points(
            collection_name=collection,
            query=sparse,
            using=config.BM25_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
        ).points
    except Exception as exc:
        logger.warning("hybrid: sparse search failed on %s (%s)", collection, exc)
        return []


def fuse(dense_hits: list, sparse_hits: list) -> list[dict]:
    """Merge both channels into [{"id", "similarity", "payload"}], sorted by
    similarity desc. Dense hits keep their cosine; a sparse hit contributes
    RECALL_BM25_WEIGHT * (score / best score), taken only when higher."""
    best_bm25 = max((h.score for h in sparse_hits), default=0.0)
    merged: dict = {}
    for h in dense_hits:
        merged[h.id] = {"id": h.id, "similarity": h.score, "payload": h.payload or {}}
    for h in sparse_hits:
        rescued = config.RECALL_BM25_WEIGHT * (h.score / best_bm25) if best_bm25 else 0.0
        entry = merged.get(h.id)
        if entry is None:
            merged[h.id] = {"id": h.id, "similarity": rescued, "payload": h.payload or {}}
        elif rescued > entry["similarity"]:
            entry["similarity"] = rescued
    return sorted(merged.values(), key=lambda e: e["similarity"], reverse=True)
