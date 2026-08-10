"""
tools.py — Tool registry for the agentic RAG system.

What is a tool registry?
  An agent needs to know WHAT tools exist, WHAT each one does, and HOW
  to call it. The registry is that catalogue. The agent (in agent.py)
  reads the registry description, picks a tool by name, and we dispatch
  the call here.

  This is the same pattern used by OpenAI function calling, LangChain
  tools, and every serious agent framework — we're just doing it without
  a framework.

Why four tools and not just one?
  Vector search (semantic) and keyword search (BM25) retrieve different
  things. Semantic search finds conceptually similar chunks even with
  different wording. BM25 finds exact keyword matches — better for
  technical terms, proper nouns, version numbers. hybrid_search fuses
  both signals (via RRF) in one call for when either alone might miss —
  the agent can reach for it when it's not confident which single
  signal will win.

  The agent deciding which one to use (or both) is the point. That
  decision-making is what makes it agentic.

Tools defined here:
  1. vector_search   — semantic similarity search via pgvector (existing)
  2. keyword_search  — BM25 term-frequency keyword search (existing)
  3. recall_memory   — search conversation history for relevant past context (existing)
  4. hybrid_search    — fused vector + Postgres full-text search via RRF (new)

Reranking note:
  Candidate pools below are intentionally wider than what gets returned
  to generation (agent.py reranks the returned parent chunks down with a
  cross-encoder before the reflection step). A tool returning only 3
  candidates gives the reranker nothing to actually re-order — so the
  parent caps here were bumped from 3 to 8. agent.py trims back down
  after reranking.
"""

import math
import json
import re
from collections import Counter

from vector_store import search, hybrid_search as vector_store_hybrid_search
from chunker import parents_from_child_results


# ---------------------------------------------------------------------------
# Tool registry — what the agent sees when choosing a tool
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "vector_search": {
        "description": (
            "Semantic similarity search over the indexed documents. "
            "Best for conceptual questions, paraphrased queries, and "
            "questions where the exact wording may differ from the document. "
            "Use this for most questions."
        ),
        "input": "query string",
    },
    "keyword_search": {
        "description": (
            "Exact keyword / BM25 search over the indexed documents. "
            "Best for specific technical terms, proper nouns, acronyms, "
            "version numbers, or when you need exact string matches. "
            "Use this when vector_search might miss specific terminology."
        ),
        "input": "query string",
    },
    "recall_memory": {
        "description": (
            "Search the current conversation history for relevant context "
            "that was already discussed. Use this when the question refers "
            "to something the user or assistant said earlier in the conversation, "
            "or when a prior answer might already contain the needed information."
        ),
        "input": "query string",
    },
    "hybrid_search": {
        "description": (
            "Combined semantic + keyword search over the indexed documents, "
            "fusing both signals into one ranking. Best when the question mixes "
            "a conceptual idea with a specific term (e.g. 'how does the CHUNK_SIZE "
            "constant affect retrieval quality?'), or when you're not confident "
            "vector_search or keyword_search alone would find the right chunk."
        ),
        "input": "query string",
    },
}


def describe_tools() -> str:
    """
    Return a formatted string describing all available tools.
    This is passed to the LLM so it knows what tools exist.
    """
    lines = ["Available tools:"]
    for name, info in TOOL_REGISTRY.items():
        lines.append(f'\n  "{name}": {info["description"]}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 1: Vector search (semantic)
# ---------------------------------------------------------------------------

def vector_search(query: str, top_k: int = 15, score_threshold: float = 0.3) -> list[dict]:
    """
    Semantic search via pgvector — the existing retrieval pipeline.

    Searches child chunks by cosine similarity, then loads parent chunks
    for full context. This is exactly what rag.py's retrieve() does,
    exposed here as a named tool.

    top_k was bumped from 7 → 15 and the parent cap from 3 → 8 so that
    agent.py's rerank step has a real pool of candidates to re-order
    instead of just re-sorting 3 items.

    Args:
        query:           The search query string.
        top_k:           Max child chunks to retrieve.
        score_threshold: Minimum similarity score (0–1).

    Returns:
        List of parent chunk dicts: [{parent_id, source, content}, ...]
        Empty list if nothing relevant found.
    """
    child_results = search(query, top_k=top_k, score_threshold=score_threshold)
    return parents_from_child_results(child_results, limit=8)


# ---------------------------------------------------------------------------
# Tool 2: BM25 keyword search
# ---------------------------------------------------------------------------
# BM25 (Best Match 25) is the industry standard keyword ranking function.
# It's what Elasticsearch and most search engines use under the hood.
#
# How it works:
#   For each document, BM25 computes a score for a query:
#
#   score(d, q) = Σ_term  IDF(term) × (tf × (k1+1)) / (tf + k1×(1 - b + b×|d|/avgdl))
#
#   where:
#     tf       = term frequency in this document
#     |d|      = document length (words)
#     avgdl    = average document length in corpus
#     k1       = term saturation parameter (default 1.5)
#              — controls how much repeated terms boost the score
#              — higher k1 = repeated terms matter more
#     b        = length normalisation parameter (default 0.75)
#              — b=1 = full length normalisation
#              — b=0 = no length normalisation
#     IDF(t)   = log((N - df + 0.5) / (df + 0.5) + 1)
#              — how rare is this term across all documents?
#
# Why BM25 over TF-IDF for search?
#   TF-IDF has a problem: a document with 100 occurrences of "python"
#   scores linearly higher than one with 10. BM25 saturates — beyond a
#   certain frequency, extra occurrences contribute diminishing returns.
#   This makes BM25 more robust for long documents.
#
# Implementation note:
#   We build the BM25 index from parent_store/ JSON files on first call
#   and cache it in memory for subsequent calls.

_bm25_index: dict | None = None  # cached index

# Words are runs of letters/digits, optionally joined by internal hyphens so
# compound technical terms survive as one token ("parent-child",
# "all-minilm-l6-v2"). Everything else — Markdown emphasis, trailing
# punctuation, brackets — is a separator.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """
    Split text into lowercase BM25 terms.

    This replaces a plain `.lower().split()`, which left Markdown and
    punctuation fused to the words and silently broke matching: the corpus
    indexed "**rag" and "turns**" and "reply." as distinct terms, while a
    question ending in "?" produced the term "generation?" that matched
    nothing in the index. Query and documents must be tokenized the same
    way for term matching to work at all, so both sides call this.

    Hyphens inside a word are kept, since splitting "parent-child" into two
    terms would lose the compound the user actually searched for.
    """
    return _TOKEN_RE.findall(text.lower())


def _build_bm25_index() -> dict:
    """
    Build a BM25 index from all parent chunks stored on disk.

    Reads every JSON file from parent_store/, tokenises the content
    (simple whitespace split — fast, no preprocessing needed for BM25),
    and computes the statistics needed for BM25 scoring.

    Returns:
        index dict with:
          documents   — list of {parent_id, source, content, tokens}
          df          — Counter: term → document frequency
          avgdl       — average document length in tokens
          N           — total number of documents
    """
    import os
    from pathlib import Path

    parent_store_dir = Path("parent_store")
    if not parent_store_dir.exists():
        return {"documents": [], "df": Counter(), "avgdl": 0, "N": 0}

    documents = []
    for filepath in parent_store_dir.glob("*.json"):
        try:
            with open(filepath) as f:
                chunk = json.load(f)
            tokens = _tokenize(chunk.get("content", ""))
            documents.append({
                "parent_id": chunk.get("parent_id", filepath.stem),
                "source": chunk.get("source", "unknown"),
                "content": chunk.get("content", ""),
                "tokens": tokens,
            })
        except Exception:
            continue

    if not documents:
        return {"documents": [], "df": Counter(), "avgdl": 0, "N": 0}

    # Document frequency: how many documents contain each term
    df: Counter = Counter()
    for doc in documents:
        for term in set(doc["tokens"]):  # set = count each term once per doc
            df[term] += 1

    avgdl = sum(len(d["tokens"]) for d in documents) / len(documents)

    return {
        "documents": documents,
        "df": df,
        "avgdl": avgdl,
        "N": len(documents),
    }


def _bm25_score(
    query_terms: list[str],
    doc_tokens: list[str],
    df: Counter,
    N: int,
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a single document given query terms."""
    tf_counter = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0

    for term in query_terms:
        if term not in tf_counter:
            continue
        tf = tf_counter[term]
        df_t = df.get(term, 0)

        # IDF component
        idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

        # TF normalisation component
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))

        score += idf * tf_norm

    return score


def keyword_search(query: str, top_k: int = 8) -> list[dict]:
    """
    BM25 keyword search over all indexed parent chunks.

    top_k was bumped from 3 → 8 for the same reason as vector_search:
    the reranker downstream needs a real candidate pool.

    Args:
        query: The search query. Works best with specific terms.
        top_k: Number of top-scoring documents to return.

    Returns:
        List of parent chunk dicts sorted by BM25 score (best first).
        Empty list if no documents are indexed or no terms match.
    """
    global _bm25_index

    # Build index on first call, reuse on subsequent calls
    if _bm25_index is None:
        _bm25_index = _build_bm25_index()

    index = _bm25_index
    if not index["documents"]:
        return []

    query_terms = _tokenize(query)

    # Score every document
    scored = []
    for doc in index["documents"]:
        score = _bm25_score(
            query_terms,
            doc["tokens"],
            index["df"],
            index["N"],
            index["avgdl"],
        )
        if score > 0:
            scored.append((score, doc))

    # Sort by score descending, return top_k
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {"parent_id": d["parent_id"], "source": d["source"], "content": d["content"]}
        for _, d in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# Tool 3: Memory recall
# ---------------------------------------------------------------------------

def recall_memory(query: str, conversation_history: list[dict], top_k: int = 3) -> list[dict]:
    """
    Search the conversation history for turns relevant to the query.

    Uses simple TF overlap scoring (no embeddings needed — history is
    usually short enough that exact term matching works well).

    Why is this useful?
      If the user asked "how does chunking work?" two turns ago and got
      a detailed answer, and now asks "remind me of the chunk sizes",
      the agent should recall that prior answer rather than re-retrieving
      from documents. This avoids redundant retrieval and lets the agent
      build on what it already said.

    Args:
        query:                The current query to match against history.
        conversation_history: List of {"role": ..., "content": ...} dicts.
        top_k:                Max number of relevant turns to return.

    Returns:
        List of dicts: [{role, content, relevance_score}, ...]
        Sorted by relevance, most relevant first.
        Empty list if history is empty or nothing matches.
    """
    if not conversation_history:
        return []

    query_terms = set(_tokenize(query))
    if not query_terms:
        return []

    scored = []
    for turn in conversation_history:
        if turn["role"] not in ("user", "assistant"):
            continue
        content = turn["content"]
        content_terms = set(_tokenize(content))

        # Jaccard-like overlap: |query ∩ content| / |query|
        overlap = len(query_terms & content_terms)
        if overlap > 0:
            score = overlap / len(query_terms)
            scored.append((score, turn))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {"role": t["role"], "content": t["content"], "relevance_score": round(s, 3)}
        for s, t in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# Tool 4: Hybrid search (vector + BM25, fused via RRF)
# ---------------------------------------------------------------------------

def hybrid_search(query: str, top_k: int = 8) -> list[dict]:
    """
    Fused semantic + keyword search, via vector_store.hybrid_search().

    Unlike vector_search/keyword_search above (which query independently
    and let the agent's tool-selection loop pick one or retry with the
    other), this tool runs BOTH signals in one call and fuses their
    rankings with Reciprocal Rank Fusion (RRF) — see vector_store.py for
    the fusion details. Useful when the agent isn't confident a single
    signal will win, or the query mixes a concept with a specific term.

    Args:
        query: The search query string.
        top_k: Max number of fused child-chunk results to consider before
               loading parents.

    Returns:
        List of parent chunk dicts: [{parent_id, source, content}, ...]
        Empty list if nothing relevant found.
    """
    child_results = vector_store_hybrid_search(query, top_k=top_k)
    return parents_from_child_results(child_results, limit=8)


# ---------------------------------------------------------------------------
# Dispatcher — call a tool by name
# ---------------------------------------------------------------------------

def call_tool(
    tool_name: str,
    query: str,
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """
    Dispatch a tool call by name and return results.

    This is what agent.py calls after the LLM selects a tool.
    Single entry point — agent.py doesn't need to know the internals
    of each tool.

    Args:
        tool_name:            One of: "vector_search", "keyword_search",
                               "recall_memory", "hybrid_search"
        query:                The search query.
        conversation_history: Required for recall_memory; ignored by others.

    Returns:
        List of result dicts. Format varies slightly by tool but all
        contain at least a "content" key for use in prompts.
    """
    if tool_name == "vector_search":
        return vector_search(query)

    elif tool_name == "keyword_search":
        return keyword_search(query)

    elif tool_name == "recall_memory":
        return recall_memory(query, conversation_history or [])

    elif tool_name == "hybrid_search":
        return hybrid_search(query)

    else:
        # Unknown tool — return empty rather than crashing
        print(f"[tools] Warning: unknown tool '{tool_name}', returning empty results")
        return []
