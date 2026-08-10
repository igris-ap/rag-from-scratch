# RAG from Scratch

A fully working **Agentic Retrieval-Augmented Generation (RAG)** system built from scratch in Python — no LangChain, no LangGraph, no vector database SDKs. Just Python, Postgres, and Ollama.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Postgres](https://img.shields.io/badge/Postgres-pgvector-336791)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What this is

Most RAG tutorials use LangChain or LlamaIndex — frameworks that hide the actual mechanics behind layers of abstraction. This project implements every component manually so you can see exactly how agentic RAG works:

- How PDFs become searchable chunks
- How text becomes vectors and gets stored in a database
- How an LLM selects retrieval tools at runtime based on the query
- How the system evaluates its own retrieval and retries if context is insufficient
- How the LLM critiques and revises its own answers before returning them
- How conversation history and query rewriting resolve references across turns

---

## Features

- **Agentic tool selection** — LLM picks between vector search, BM25 keyword search, hybrid search, and memory recall based on the query type
- **Retrieval reflection loop** — after retrieval, the LLM evaluates whether context is sufficient; if not, it rewrites the query and retries with the next untried tool (up to 3 attempts)
- **Cross-encoder reranking** — a second-stage model re-scores each (query, chunk) pair together and keeps only the best few, which a bi-encoder embedding cannot do
- **Hybrid search with RRF** — dense vector and Postgres full-text results fused by Reciprocal Rank Fusion, combining by rank position rather than incomparable raw scores
- **Answer self-critique** — generated answers are evaluated by a second LLM call and revised if they miss the question or contradict the context
- **BM25 keyword search** — exact-match retrieval for technical terms, constants, and proper nouns; implemented from scratch without Elasticsearch
- **Parent-child chunking** — search small chunks for precision, generate from large chunks for context
- **Hybrid storage** — child chunks as vectors in Postgres/pgvector, parent chunks as JSON on disk
- **Query intelligence** — detects vague questions, rewrites for clarity, splits multi-part questions
- **Conversation memory** — summarizes history across turns to resolve pronouns and references
- **Sub-question answering** — splits complex queries and synthesises the answers
- **Gradio UI** — browser-based chat with PDF upload, re-indexing, and verbose mode
- **Zero frameworks** — no LangChain, no LangGraph, no vector database SDK

---

## Architecture

```
User question
      │
      ▼
process_turn()  [main.py]  →  rag.answer()  [rag.py]
      │
      ▼
summarize_conversation()      ← compress history into 1-2 sentences
      │
      ▼
analyze_query()               ← rewrite, detect unclear, split multi-part
      │
      ├── unclear → ask for clarification
      │
      └── clear → for each sub-question:
                        │
                        ▼
                  [AGENT LOOP]  run_agent()  [agent.py]
                        │
                        ▼
                  select_tool()         ← LLM picks tool based on query type
                        │
                        ▼
                  call_tool()           ← vector_search | keyword_search |
                        │                  hybrid_search | recall_memory
                        ▼
                  rerank()              ← cross-encoder narrows the candidate
                        │                  pool (no LLM call — local model)
                        ▼
                  reflect_on_retrieval() ── SUFFICIENT? ──► generate_answer()
                        │                                          │
                        └── NO → rewrite query + next untried      ▼
                                 tool → retry (×3)          critique_and_revise()
                                                                   │
                                                           PASS? ──► return answer
                                                           FAIL? ──► revise → return
                        │
                        ▼
                  _synthesise()         ← merge if multiple sub-questions
                        │
                        ▼
                    Final answer
```

Both the CLI (`main.py`) and the Gradio UI (`app.py`) call the same `process_turn()`,
which delegates entirely to `rag.answer()` — there is one code path, not two.

**LLM calls per query:**
- Fast path (clear query, sufficient retrieval, answer passes critique): **5 calls**
  — analyze, select_tool, reflect, generate, critique
- Worst path (history summary, 3 retries, revised answer, multi-part synthesis): **~10 calls**

Reranking adds no LLM call — it runs a local cross-encoder.

### File structure

```
.
├── app.py                  # Gradio web UI
├── main.py                 # CLI entry point + pipeline wiring
├── rag.py                  # Top-level answer() — routes through agent loop
├── agent.py                # Agentic core: tool selection, reflection, critique
├── tools.py                # Tool registry: vector, BM25, hybrid, memory recall
├── rerank.py               # Cross-encoder reranking (second-stage retrieval)
├── llm.py                  # Ollama HTTP client (no SDK)
├── chunker.py              # PDF → parent + child chunks
├── vector_store.py         # Embeddings + Postgres/pgvector + hybrid RRF search
├── query_intelligence.py   # Query rewriting + conversation summary
├── eval.py                 # Retrieval quality evaluation script
├── eval_questions.json     # Evaluation question set (targets the reference PDF)
├── docs/                   # Put your PDFs here (ships with the reference PDF)
├── markdown/               # Auto-generated markdown from PDFs
├── parent_store/           # Auto-generated parent chunk JSON files
├── requirements.txt
└── README.md
```

Prompts live inline in the module that uses them — `agent.py` (tool selection,
reflection, generation, critique) and `query_intelligence.py` (analysis,
summarization) — rather than in a central prompts file.

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| LLM | Ollama (llama3.2) | Local, free, no API key |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | In-process, fast batch embedding |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Scores (query, chunk) jointly — far more accurate than embeddings |
| Vector store | Postgres + pgvector (HNSW) | Production-grade, one DB for everything |
| Keyword search | BM25 (from scratch) + Postgres full-text (GIN) | Exact-match retrieval without Elasticsearch |
| Fusion | Reciprocal Rank Fusion | Merges rankings on incomparable score scales |
| PDF parsing | pymupdf / pymupdf4llm | Preserves heading structure for chunking |
| UI | Gradio | Minimal code, works out of the box |
| HTTP | Python urllib (stdlib) | No dependencies for LLM calls |

---

## Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- [Docker](https://docker.com) for Postgres

> **Run Ollama on a GPU if you possibly can.** The pipeline makes 5–10 LLM calls per
> question, so latency is dominated by prompt processing. On a CPU-only setup a single
> question took ~185s; on a laptop RTX 4050 the identical question took ~5.5s — a 33×
> difference from the same code. Check with `ollama ps`: `PROCESSOR` should read `GPU`,
> not `100% CPU`.

### 1. Clone the repo

```bash
git clone https://github.com/igris-ap/rag-from-scratch.git
cd rag-from-scratch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the Ollama model

```bash
ollama pull llama3.2
```

### 4. Start Postgres with pgvector

```bash
docker run -d \
  --name rag-postgres \
  --restart unless-stopped \
  -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

`--restart unless-stopped` means the container comes back after a reboot. Without it
you will hit `connection refused` on the next boot and need `docker start rag-postgres`.

### 5. Add your PDFs

```bash
cp your_documents.pdf docs/
```

### 6. Run the app

```bash
python3 app.py
```

Open **http://localhost:7860** in your browser.

---

## How each component works

### Agent loop (`agent.py`)

The core of the system. For each query, the agent runs four LLM calls plus a local rerank pass:

**1. Tool selection** — the LLM reads the query and picks the best retrieval tool from the registry. Conceptual questions go to vector search. Technical terms and exact names go to BM25 keyword search. Questions mixing a concept with a specific term go to hybrid search. References to prior conversation go to memory recall.

**2. Reranking** — the selected tool deliberately returns a *wide* candidate pool, and a cross-encoder re-scores each (query, chunk) pair and keeps the top few. This is not an LLM call — see `rerank.py` below.

**3. Retrieval reflection** — the LLM evaluates whether the reranked context is sufficient to answer the question. If not, it rewrites the query and retries with the next *untried* tool. Up to 3 attempts before falling back.

**4. Answer generation** — the LLM generates an answer using only the retrieved context. If no context was retrieved, it returns an explicit "I don't have information about this in the provided documents" rather than hallucinating from training data. That exact sentence is defined once as `agent.NO_CONTEXT_ANSWER` and shared with `eval.py`, so the eval can detect a decline reliably.

**5. Self-critique** — a second LLM call evaluates the generated answer. If the answer is vague, misses key aspects, or contradicts the context, it is revised before being returned to the user.

### Tool registry (`tools.py`)

Four tools the agent can select from:

**`vector_search`** — semantic similarity search via pgvector. Finds conceptually similar chunks even when wording differs. Best for most questions.

**`keyword_search`** — BM25 ranking over all parent chunks, implemented from scratch. Finds exact term matches — better for specific constants, version numbers, and proper nouns. BM25 saturates repeated term frequency, making it more robust than raw TF-IDF for long documents.

**`hybrid_search`** — runs both signals in one call and fuses their rankings with RRF. Best when the question mixes a conceptual idea with a specific term, or when neither signal alone is clearly right.

**`recall_memory`** — searches conversation history for relevant prior turns using term overlap scoring. Used when the user references something discussed earlier.

Each tool returns a wider pool than reaches the LLM (parent caps of 8 rather than 3), specifically so the reranker has something meaningful to reorder.

### Reranking (`rerank.py`)

Embedding search is a *bi-encoder*: query and chunk are encoded separately and compared by cosine distance, so the model never sees them together. A **cross-encoder** takes `(query, chunk)` as a single input and outputs one relevance score for that pair — much more accurate, far too slow to run over a whole corpus, but cheap over the handful of candidates retrieval already narrowed to. That is the standard two-stage pattern:

```
broad + cheap (vector / BM25 / hybrid)  →  narrow + accurate (cross-encoder rerank)
```

The model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB) loads once at import and is reused.

### Chunking (`chunker.py`)

PDFs are converted to Markdown using `pymupdf4llm`, which preserves heading structure. The text is then split at H1/H2/H3 headers into **parent chunks** (2,000–10,000 characters). Each parent is further split into **child chunks** (500 characters, 100-character overlap).

Parent chunks are saved as JSON files. Child chunks are embedded and stored in Postgres. When a child chunk matches a query, its parent is loaded to give the LLM full context.

### Vector search (`vector_store.py`)

Uses `sentence-transformers/all-MiniLM-L6-v2` loaded directly in-process — no HTTP calls. All chunks are embedded in a single batch call at index time. Vectors are stored in Postgres using the `pgvector` extension with an HNSW index for fast approximate nearest-neighbour search. Search uses cosine similarity via pgvector's `<=>` operator.

The same table also carries a generated `tsvector` column with a GIN index, so one Postgres table serves both retrieval legs:

- `search()` — dense cosine similarity only (kept for debugging).
- `bm25_search()` — Postgres full-text ranking via `ts_rank_cd`.
- `hybrid_search()` — fuses both with **Reciprocal Rank Fusion**.

RRF combines results by *rank position*, not raw score:

```
RRF_score(doc) = Σ  1 / (RRF_K + rank_of_doc_in_that_ranking)
```

This matters because cosine similarity (0–1) and `ts_rank_cd` (unbounded, corpus-dependent) live on incomparable scales — averaging them directly would let whichever number happens to be larger dominate. A chunk ranked highly by *either* leg scores well; one ranked highly by *both* scores best.

### Query intelligence (`query_intelligence.py`)

Before the agent loop runs, every query is analysed:

1. **Clarity check** — is the question specific enough to search for?
2. **Rewriting** — replace pronouns, remove filler, make self-contained
3. **Splitting** — break multi-topic questions into up to 3 sub-questions

The LLM returns structured JSON which is parsed and used to route the query into the agent loop.

### Evaluation (`eval.py`)

Runs a set of questions through the agent pipeline and reports:
- **Retrieval coverage** — did the retriever find any chunks?
- **Answer informativeness** — did the LLM answer, or decline?

No labelled ground truth needed — runs entirely with your local Postgres and Ollama stack.

`eval_questions.json` targets `docs/rag_system_reference.pdf`, which ships with the repo so the eval is reproducible on a fresh clone. Each question maps to one section of that document.

Informativeness is detected via `agent.NO_CONTEXT_ANSWER` anchored at the *start* of the answer, not by searching for the phrase anywhere — otherwise a correct answer that happens to *describe* the fallback behaviour (as the hallucination question does) would be scored as a failure.

```bash
python3 eval.py                        # runs eval_questions.json
python3 eval.py --verbose              # also prints each answer
python3 eval.py --questions my_qs.json # custom question file
```

---

## Configuration

Key constants you can tune:

Config lives as constants in the module that uses it — there is no central settings file.

| File | Constant | Default | Effect |
|---|---|---|---|
| `agent.py` | `MAX_RETRIES` | 3 | Max retrieval attempts before giving up |
| `agent.py` | `MIN_CONTEXT_CHARS` | 100 | Below this, reflection fails without an LLM call |
| `agent.py` | `RERANK_KEEP` | 3 | Candidates kept per attempt after reranking |
| `vector_store.py` | `SCORE_THRESHOLD` | 0.3 | Minimum similarity to include a chunk |
| `vector_store.py` | `DEFAULT_TOP_K` | 7 | How many chunks to retrieve |
| `vector_store.py` | `RRF_K` | 60 | RRF constant (standard value from the original paper) |
| `vector_store.py` | `FUSION_CANDIDATE_MULTIPLIER` | 4 | How wide each leg pulls before fusion |
| `rerank.py` | `RERANK_MODEL_NAME` | ms-marco-MiniLM-L-6-v2 | Cross-encoder model |
| `chunker.py` | `CHILD_CHUNK_SIZE` | 500 | Characters per child chunk |
| `chunker.py` | `CHILD_CHUNK_OVERLAP` | 100 | Overlap between child chunks |
| `chunker.py` | `MIN_PARENT_SIZE` / `MAX_PARENT_SIZE` | 2000 / 10000 | Parent chunk merge / split thresholds |
| `main.py` | `MAX_HISTORY` | 10 | Messages kept in memory |

`env.example` documents these as environment variables, but nothing reads them yet — all config is hardcoded.

---

## Possible improvements

- **Streaming** — stream LLM tokens to the Gradio UI as they generate
- **Persistent sessions** — replace in-memory history with Postgres-backed sessions
- **Multi-user** — add authentication and per-user document namespaces
- **RAGAS evaluation** — add ground-truth labelled eval for answer quality scoring
- **Multi-step planning** — ReAct-style planning for complex multi-hop questions
- **Config via env** — wire up `env.example` so the constants are actually configurable

---

## License

MIT
