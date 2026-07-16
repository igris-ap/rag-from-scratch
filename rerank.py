"""
rerank.py — Cross-encoder reranking.

Hybrid search (vector_store.hybrid_search) is fast but approximate: it
scores each chunk against the query independently (bi-encoder embeddings)
or by keyword overlap (BM25). Neither actually reads the query and the
chunk together.

A cross-encoder does. It takes (query, chunk) as a single input pair and
outputs one relevance score for that specific pair — much more accurate,
much slower. Too slow to run over an entire corpus, but cheap enough to
run over the ~20-30 candidates hybrid search already narrowed things down
to. That's the standard two-stage retrieve-then-rerank pattern:

    broad + cheap (hybrid_search)  →  narrow + accurate (rerank)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on the MS MARCO passage ranking dataset — general-purpose,
  works well out of the box for "is this passage relevant to this
  query" without fine-tuning.
- ~80MB, runs fine on CPU for tens of candidates.
"""

from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Load the cross-encoder once at import time (same pattern as
# vector_store.py's embedding model — load once, reuse for every call).
# ---------------------------------------------------------------------------

print(f"Loading reranker model: {RERANK_MODEL_NAME} ...")
_reranker = CrossEncoder(RERANK_MODEL_NAME)
print("Reranker model ready.")


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Re-score and re-sort a list of candidate chunks against the query
    using a cross-encoder, then keep the top_k.

    Args:
        query: The user's question as plain text.
        candidates: List of chunk dicts (from hybrid_search or search).
                    Each must have a "content" key.
        top_k: How many reranked results to keep.

    Returns:
        List of dicts — same shape as the input dicts, plus a
        "rerank_score" key — sorted by rerank_score descending,
        truncated to top_k. If candidates is empty, returns [].
    """
    if not candidates:
        return []

    # Cross-encoder input is a list of (query, passage) pairs.
    pairs = [(query, c["content"]) for c in candidates]

    # predict() returns one relevance logit per pair. Higher = more
    # relevant. Unlike cosine similarity these are NOT bounded to
    # [0, 1] or [-1, 1] — treat them as ranking scores, not probabilities.
    scores = _reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from vector_store import hybrid_search

    print("=== Reranker test ===\n")

    test_query = "What is retrieval augmented generation?"
    candidates = hybrid_search(test_query, top_k=15)

    if not candidates:
        print("No candidates from hybrid_search — index some documents first.")
        exit()

    print(f"Query: {test_query}")
    print(f"\n--- Before rerank (hybrid RRF order, top 5) ---")
    for c in candidates[:5]:
        print(f"  rrf={c['rrf_score']:.4f} | {c['content'][:80]}...")

    reranked = rerank(test_query, candidates, top_k=5)

    print(f"\n--- After rerank (cross-encoder order, top 5) ---")
    for c in reranked:
        print(f"  rerank={c['rerank_score']:.4f} | {c['content'][:80]}...")
