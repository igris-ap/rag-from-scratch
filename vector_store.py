"""
vector_store.py — Stage 3 (revised): Embeddings + Postgres/pgvector.
Now with hybrid search: dense vector search + Postgres full-text (BM25-style)
search, fused with Reciprocal Rank Fusion (RRF).

Embedding model: sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional vectors
- 80MB — downloads fast, loads fast
- Good quality for English RAG
- No special query prefix needed (unlike BGE models)
- Runs fully in-process — no HTTP calls, no server
"""

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "rag",
    "user": "rag",
    "password": "rag",
}

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors

DEFAULT_TOP_K = 7
SCORE_THRESHOLD = 0.3

# How many candidates each retrieval leg pulls before fusion. Wider than
# top_k so RRF has enough overlap between the two rankings to actually
# matter — if you only pulled top_k from each leg you'd lose recall.
FUSION_CANDIDATE_MULTIPLIER = 4

# RRF constant. 60 is the standard value from the original RRF paper
# (Cormack et al.) — it de-emphasizes rank position 1 vs 2 vs 3 less
# aggressively than a smaller k would. Rarely needs tuning.
RRF_K = 60

# ---------------------------------------------------------------------------
# Load the embedding model once at import time
# ---------------------------------------------------------------------------
# The model loads into memory when this module is first imported.
# Every embed() call after that is fast — no network, no server round-trip.

print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
_model = SentenceTransformer(EMBED_MODEL_NAME)
print("Embedding model ready.")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    """
    Convert a string into a 384-dimensional vector.

    Args:
        text: The text to embed (a query or a document chunk).

    Returns:
        List of 384 floats representing the text's meaning.
        normalize_embeddings=True means all vectors are scaled to length 1.
        This makes cosine similarity equivalent to a dot product, which is
        slightly faster and is what pgvector's <=> operator expects.
    """
    vector = _model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in one go.

    sentence-transformers processes multiple texts in parallel on CPU.
    For indexing hundreds of chunks this is significantly faster than
    calling embed() in a loop — one call instead of N calls.

    Args:
        texts: List of strings to embed.

    Returns:
        List of vectors, one per input text, in the same order.
    """
    vectors = _model.encode(
        texts,
        batch_size=32,          # process 32 texts at a time
        normalize_embeddings=True,
        show_progress_bar=True,  # prints progress during indexing
    )
    return vectors.tolist()


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_connection():
    """Open and return a Postgres connection."""
    return psycopg2.connect(**DB_CONFIG)


def setup_db():
    """
    Create the pgvector extension and child_chunks table if they don't exist.

    vector(384) matches all-MiniLM-L6-v2's output dimension.
    If you switch models, drop this table and recreate it with the
    correct dimension — Postgres will reject inserts if they don't match.

    Also adds a generated tsvector column + GIN index for full-text
    (BM25-style) search, used by the hybrid search leg.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Enable pgvector extension (safe to call multiple times)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create table with 384-dim vector column, plus a generated tsvector
    # column for full-text search. GENERATED ALWAYS ... STORED means
    # Postgres keeps content_tsv in sync with content automatically —
    # no separate update step needed when you insert a row.
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS child_chunks (
            id SERIAL PRIMARY KEY,
            parent_id TEXT NOT NULL,
            source TEXT NOT NULL,
            child_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector({VECTOR_DIM}),
            content_tsv tsvector GENERATED ALWAYS AS
                (to_tsvector('english', content)) STORED
        );
    """)

    # HNSW index — makes nearest-neighbour search fast at scale.
    # Without this, Postgres scans every row for every query (slow).
    cur.execute("""
        CREATE INDEX IF NOT EXISTS child_chunks_embedding_idx
        ON child_chunks
        USING hnsw (embedding vector_cosine_ops);
    """)

    # GIN index on the tsvector column — makes full-text search fast.
    # Same idea as the HNSW index above, just for keyword search instead
    # of vector search.
    cur.execute("""
        CREATE INDEX IF NOT EXISTS child_chunks_tsv_idx
        ON child_chunks
        USING gin (content_tsv);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database ready.")


def migrate_add_fulltext_search():
    """
    One-time migration for an EXISTING database created before hybrid
    search was added. Adds the content_tsv column + GIN index to a
    child_chunks table that already has data in it.

    Safe to run multiple times (IF NOT EXISTS everywhere). If your table
    was created by an older version of setup_db(), run this once instead
    of dropping and re-indexing everything.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        ALTER TABLE child_chunks
        ADD COLUMN IF NOT EXISTS content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS child_chunks_tsv_idx
        ON child_chunks
        USING gin (content_tsv);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Migration complete: content_tsv column + GIN index added.")


# ---------------------------------------------------------------------------
# Storing child chunks
# ---------------------------------------------------------------------------

def clear_chunks():
    """Delete all rows — call before re-indexing to avoid duplicates."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM child_chunks;")
    conn.commit()
    cur.close()
    conn.close()
    print("Cleared existing chunks.")


def store_children(child_chunks: list[dict]):
    """
    Embed all child chunks in batch and insert into Postgres.

    Uses embed_batch() so all chunks are embedded in one call
    rather than one call per chunk. content_tsv is populated
    automatically by Postgres (it's a generated column).

    Args:
        child_chunks: List of child chunk dicts from chunker.py.
                       Each dict: { parent_id, source, child_index, content }
    """
    if not child_chunks:
        print("No chunks to store.")
        return

    total = len(child_chunks)
    print(f"\nEmbedding {total} child chunks in batch...")

    # Extract text for batch embedding
    texts = [c["content"] for c in child_chunks]
    vectors = embed_batch(texts)

    print(f"Inserting {total} rows into Postgres...")
    conn = get_connection()
    cur = conn.cursor()

    # Build rows for batch insert
    rows = []
    for chunk, vector in zip(child_chunks, vectors):
        # pgvector expects the vector as a string: "[0.1, 0.2, ...]"
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        rows.append((
            chunk["parent_id"],
            chunk["source"],
            chunk["child_index"],
            chunk["content"],
            vector_str,
        ))

    # Single batch insert — one round-trip to Postgres
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO child_chunks
            (parent_id, source, child_index, content, embedding)
        VALUES %s
        """,
        rows,
        page_size=100,
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done. Stored {total} chunks in Postgres.")


# ---------------------------------------------------------------------------
# Searching — dense (vector) leg
# ---------------------------------------------------------------------------

def _vector_search_ranked(query_vector_str: str, limit: int) -> list[dict]:
    """
    Raw dense-vector nearest-neighbour search, no score threshold applied.

    Used internally by both search() (threshold applied after) and
    hybrid_search() (threshold doesn't matter for RRF — only rank does).

    Returns:
        List of dicts: { parent_id, source, child_index, content, score }
        Sorted by cosine similarity descending.
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """
        SELECT
            parent_id,
            source,
            child_index,
            content,
            1 - (embedding <=> %s::vector) AS score
        FROM child_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vector_str, query_vector_str, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "parent_id": row["parent_id"],
            "source": row["source"],
            "child_index": row["child_index"],
            "content": row["content"],
            "score": round(float(row["score"]), 4),
        }
        for row in rows
    ]


def search(query: str, top_k: int = DEFAULT_TOP_K,
           score_threshold: float = SCORE_THRESHOLD) -> list[dict]:
    """
    Find the top-K child chunks most similar to the query, using dense
    vector search only. Kept for backwards compatibility / debugging —
    prefer hybrid_search() for actual retrieval.

    Steps:
        1. Embed the query into a 384-dim vector
        2. Run SQL with pgvector's <=> (cosine distance) operator
        3. Filter by score_threshold
        4. Return matching chunks sorted best-first

    Args:
        query: The user's question as plain text.
        top_k: Max number of results to return.
        score_threshold: Min similarity score (0-1) to include a result.
                          0.3 = at least 30% similar to the query.

    Returns:
        List of dicts: { parent_id, source, child_index, content, score }
        Sorted by score descending (best match first).
    """
    query_vector = embed(query)
    query_vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    ranked = _vector_search_ranked(query_vector_str, limit=top_k)
    return [r for r in ranked if r["score"] >= score_threshold]


# ---------------------------------------------------------------------------
# Searching — sparse (BM25-style full-text) leg
# ---------------------------------------------------------------------------

def bm25_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Keyword search using Postgres full-text search.

    ts_rank_cd behaves like BM25/TF-IDF: it rewards term frequency and
    proximity of matched terms, and normalizes for document length. This
    is what catches exact matches dense vector search misses — product
    codes, acronyms, names, numbers — because it matches literal tokens
    rather than semantic meaning.

    plainto_tsquery() turns the raw query string into a tsquery,
    automatically ANDing the significant words together (it also strips
    stopwords, same as the indexing side).

    Args:
        query: The user's question as plain text.
        top_k: Max number of results to return.

    Returns:
        List of dicts: { parent_id, source, child_index, content, score }
        Sorted by ts_rank_cd descending (best match first). Rows with no
        keyword overlap at all are excluded (rank 0 / no match).
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """
        SELECT
            parent_id,
            source,
            child_index,
            content,
            ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS score
        FROM child_chunks
        WHERE content_tsv @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, query, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "parent_id": row["parent_id"],
            "source": row["source"],
            "child_index": row["child_index"],
            "content": row["content"],
            "score": round(float(row["score"]), 4),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Hybrid search — fuse dense + sparse with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _chunk_key(row: dict) -> tuple:
    """Unique identity for a child chunk, used to merge the two rankings."""
    return (row["parent_id"], row["child_index"])


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Combine dense vector search and sparse full-text (BM25-style) search
    using Reciprocal Rank Fusion (RRF).

    Why RRF instead of averaging the two scores directly?
    Cosine similarity (0-1) and ts_rank_cd (an unbounded, corpus-dependent
    number) live on totally different scales — averaging them directly
    would let whichever score happens to be numerically bigger dominate.
    RRF sidesteps this by only looking at RANK POSITION in each list, not
    the raw score:

        RRF_score(doc) = sum over each ranking r that contains doc of
                          1 / (RRF_K + rank_of_doc_in_r)

    A doc that appears near the top of *either* list gets a high score.
    A doc that appears near the top of *both* lists gets an even higher
    one — that's the "hybrid" benefit in practice.

    Args:
        query: The user's question as plain text.
        top_k: Max number of fused results to return.

    Returns:
        List of dicts: { parent_id, source, child_index, content,
                          vector_score, bm25_score, rrf_score }
        Sorted by rrf_score descending (best match first).
    """
    candidate_limit = top_k * FUSION_CANDIDATE_MULTIPLIER

    query_vector = embed(query)
    query_vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    vector_results = _vector_search_ranked(query_vector_str, limit=candidate_limit)
    bm25_results = bm25_search(query, top_k=candidate_limit)

    # rrf_scores[key] accumulates the fused score; rows[key] keeps the
    # actual chunk data (content etc.) the first time we see it.
    rrf_scores: dict[tuple, float] = {}
    rows: dict[tuple, dict] = {}
    vector_score_by_key: dict[tuple, float] = {}
    bm25_score_by_key: dict[tuple, float] = {}

    for rank, row in enumerate(vector_results, start=1):
        key = _chunk_key(row)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        rows.setdefault(key, row)
        vector_score_by_key[key] = row["score"]

    for rank, row in enumerate(bm25_results, start=1):
        key = _chunk_key(row)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        rows.setdefault(key, row)
        bm25_score_by_key[key] = row["score"]

    fused = []
    for key, rrf_score in rrf_scores.items():
        row = rows[key]
        fused.append({
            "parent_id": row["parent_id"],
            "source": row["source"],
            "child_index": row["child_index"],
            "content": row["content"],
            "vector_score": vector_score_by_key.get(key),
            "bm25_score": bm25_score_by_key.get(key),
            "rrf_score": round(rrf_score, 5),
        })

    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from chunker import index_all_documents, convert_all_pdfs

    print("=== Stage 3: Vector store test ===\n")

    setup_db()
    convert_all_pdfs()
    children = index_all_documents()

    if not children:
        print("No chunks to store. Add PDFs to docs/ or run chunker.py first.")
        exit()

    clear_chunks()
    store_children(children)

    print("\n--- Testing hybrid search ---")
    test_queries = [
        "What is retrieval augmented generation?",
        "How does chunking work?",
        "What is cosine similarity?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = hybrid_search(query, top_k=3)
        if results:
            for r in results:
                print(f"  rrf={r['rrf_score']:.4f}  "
                      f"vec={r['vector_score']}  bm25={r['bm25_score']}  "
                      f"| {r['source']} | {r['content'][:80]}...")
        else:
            print("  No results.")

    print("\nStage 3 complete.")
