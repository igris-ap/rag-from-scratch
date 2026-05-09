"""
vector_store.py — Stage 3 (revised): Embeddings + Postgres/pgvector.

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
    "host":     "localhost",
    "port":     5432,
    "dbname":   "rag",
    "user":     "rag",
    "password": "rag",
}

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM       = 384    # all-MiniLM-L6-v2 produces 384-dimensional vectors
DEFAULT_TOP_K    = 7
SCORE_THRESHOLD  = 0.3

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
        batch_size=32,               # process 32 texts at a time
        normalize_embeddings=True,
        show_progress_bar=True,      # prints progress during indexing
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
    """
    conn = get_connection()
    cur  = conn.cursor()

    # Enable pgvector extension (safe to call multiple times)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create table with 384-dim vector column
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS child_chunks (
            id          SERIAL PRIMARY KEY,
            parent_id   TEXT NOT NULL,
            source      TEXT NOT NULL,
            child_index INTEGER NOT NULL,
            content     TEXT NOT NULL,
            embedding   vector({VECTOR_DIM})
        );
    """)

    # HNSW index — makes nearest-neighbour search fast at scale
    # Without this, Postgres scans every row for every query (slow)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS child_chunks_embedding_idx
        ON child_chunks
        USING hnsw (embedding vector_cosine_ops);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database ready.")


# ---------------------------------------------------------------------------
# Storing child chunks
# ---------------------------------------------------------------------------

def clear_chunks():
    """Delete all rows — call before re-indexing to avoid duplicates."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM child_chunks;")
    conn.commit()
    cur.close()
    conn.close()
    print("Cleared existing chunks.")


def store_children(child_chunks: list[dict]):
    """
    Embed all child chunks in batch and insert into Postgres.

    Uses embed_batch() so all chunks are embedded in one call
    rather than one call per chunk.

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
    texts   = [c["content"] for c in child_chunks]
    vectors = embed_batch(texts)

    print(f"Inserting {total} rows into Postgres...")

    conn = get_connection()
    cur  = conn.cursor()

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
# Searching
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = DEFAULT_TOP_K,
           score_threshold: float = SCORE_THRESHOLD) -> list[dict]:
    """
    Find the top-K child chunks most similar to the query.

    Steps:
      1. Embed the query into a 384-dim vector
      2. Run SQL with pgvector's <=> (cosine distance) operator
      3. Filter by score_threshold
      4. Return matching chunks sorted best-first

    Args:
        query:           The user's question as plain text.
        top_k:           Max number of results to return.
        score_threshold: Min similarity score (0-1) to include a result.
                         0.3 = at least 30% similar to the query.
                         Raise this if you get too many irrelevant results.
                         Lower it if you get too few results.

    Returns:
        List of dicts: { parent_id, source, child_index, content, score }
        Sorted by score descending (best match first).
    """
    # Embed the query — same model, same normalization as stored chunks
    query_vector     = embed(query)
    query_vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    conn = get_connection()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # <=> = cosine distance (lower = more similar)
    # 1 - distance = cosine similarity (higher = more similar)
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
        (query_vector_str, query_vector_str, top_k),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Filter and convert to plain dicts
    results = []
    for row in rows:
        if row["score"] >= score_threshold:
            results.append({
                "parent_id":   row["parent_id"],
                "source":      row["source"],
                "child_index": row["child_index"],
                "content":     row["content"],
                "score":       round(float(row["score"]), 4),
            })

    return results


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

    print("\n--- Testing search ---")
    test_queries = [
        "What is retrieval augmented generation?",
        "How does chunking work?",
        "What is cosine similarity?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search(query, top_k=3)
        if results:
            for r in results:
                print(f"  score={r['score']:.3f} | {r['source']} | "
                      f"{r['content'][:80]}...")
        else:
            print("  No results above threshold.")

    print("\nStage 3 complete.")
