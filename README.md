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

- **Agentic tool selection** — LLM picks between vector search, BM25 keyword search, and memory recall based on the query type
- **Retrieval reflection loop** — after retrieval, the LLM evaluates whether context is sufficient; if not, it rewrites the query and retries (up to 3 attempts)
- **Answer self-critique** — generated answers are evaluated by a second LLM call and revised if they miss the question or contradict the context
- **BM25 keyword search** — exact-match retrieval for technical terms, constants, and proper nouns; implemented from scratch without Elasticsearch
- **Parent-child chunking** — search small chunks for precision, generate from large chunks for context
- **Hybrid storage** — child chunks as vectors in Postgres/pgvector, parent chunks as JSON on disk
- **Query intelligence** — detects vague questions, rewrites for clarity, splits multi-part questions
- **Conversation memory** — summarizes history across turns to resolve pronouns and references
- **Parallel sub-question answering** — splits complex queries and aggregates answers
- **Gradio UI** — browser-based chat with PDF upload, re-indexing, and verbose mode
- **Zero frameworks** — no LangChain, no LangGraph, no vector database SDK

---

## Architecture

```
User question
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
                  [AGENT LOOP]
                        │
                        ▼
                  select_tool()         ← LLM picks tool based on query type
                        │
                        ▼
                  call_tool()           ← vector_search | keyword_search | recall_memory
                        │
                        ▼
                  reflect_on_retrieval() ── SUFFICIENT? ──► generate_answer()
                        │                                          │
                        └── NO → rewrite query → retry (×3)       ▼
                                                           critique_and_revise()
                                                                   │
                                                           PASS? ──► return answer
                                                           FAIL? ──► revise → return
                        │
                        ▼
                  aggregate_answers()   ← merge if multiple sub-questions
                        │
                        ▼
                    Final answer
```

**LLM calls per query:**
- Fast path (clear query, sufficient retrieval, answer passes critique): **5 calls**
- Worst path (vague query, 3 retries, answer revised): **up to 10 calls**

### File structure

```
.
├── app.py                  # Gradio web UI
├── main.py                 # CLI entry point + pipeline wiring
├── rag.py                  # Top-level answer() — routes through agent loop
├── agent.py                # Agentic core: tool selection, reflection, critique
├── tools.py                # Tool registry: vector search, BM25, memory recall
├── prompts.py              # All LLM prompts in one place
├── llm.py                  # Ollama HTTP client (no SDK)
├── chunker.py              # PDF → parent + child chunks
├── vector_store.py         # Embeddings + Postgres/pgvector
├── query_intelligence.py   # Query rewriting + conversation summary
├── eval.py                 # Retrieval quality evaluation script
├── eval_questions.json     # Sample evaluation question set
├── docs/                   # Put your PDFs here
├── markdown/               # Auto-generated markdown from PDFs
├── parent_store/           # Auto-generated parent chunk JSON files
├── requirements.txt
└── README.md
```

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| LLM | Ollama (llama3.2) | Local, free, no API key |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | In-process, fast batch embedding |
| Vector store | Postgres + pgvector | Production-grade, one DB for everything |
| Keyword search | BM25 (from scratch) | Exact-match retrieval without Elasticsearch |
| PDF parsing | pymupdf / pymupdf4llm | Preserves heading structure for chunking |
| UI | Gradio | Minimal code, works out of the box |
| HTTP | Python urllib (stdlib) | No dependencies for LLM calls |

---

## Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- [Docker](https://docker.com) for Postgres

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
  -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

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

The core of the system. For each query, the agent runs four sequential LLM calls:

**1. Tool selection** — the LLM reads the query and picks the best retrieval tool from the registry. Conceptual questions go to vector search. Technical terms and exact names go to BM25 keyword search. References to prior conversation go to memory recall.

**2. Retrieval reflection** — after the tool returns results, the LLM evaluates whether the context is sufficient to answer the question. If not, it rewrites the query with different keywords and retries with the next tool. Up to 3 attempts before falling back.

**3. Answer generation** — the LLM generates an answer using only the retrieved context. If no context was retrieved, it returns an explicit "I don't have information about this" rather than hallucinating from training data.

**4. Self-critique** — a second LLM call evaluates the generated answer. If the answer is vague, misses key aspects, or contradicts the context, it is revised before being returned to the user.

### Tool registry (`tools.py`)

Three tools the agent can select from:

**`vector_search`** — semantic similarity search via pgvector. Finds conceptually similar chunks even when wording differs. Best for most questions.

**`keyword_search`** — BM25 ranking over all parent chunks, implemented from scratch. Finds exact term matches — better for specific constants, version numbers, and proper nouns. BM25 saturates repeated term frequency, making it more robust than raw TF-IDF for long documents.

**`recall_memory`** — searches conversation history for relevant prior turns using term overlap scoring. Used when the user references something discussed earlier.

### Chunking (`chunker.py`)

PDFs are converted to Markdown using `pymupdf4llm`, which preserves heading structure. The text is then split at H1/H2/H3 headers into **parent chunks** (2,000–10,000 characters). Each parent is further split into **child chunks** (500 characters, 100-character overlap).

Parent chunks are saved as JSON files. Child chunks are embedded and stored in Postgres. When a child chunk matches a query, its parent is loaded to give the LLM full context.

### Vector search (`vector_store.py`)

Uses `sentence-transformers/all-MiniLM-L6-v2` loaded directly in-process — no HTTP calls. All chunks are embedded in a single batch call at index time. Vectors are stored in Postgres using the `pgvector` extension with an HNSW index for fast approximate nearest-neighbour search. Search uses cosine similarity via pgvector's `<=>` operator.

### Query intelligence (`query_intelligence.py`)

Before the agent loop runs, every query is analysed:

1. **Clarity check** — is the question specific enough to search for?
2. **Rewriting** — replace pronouns, remove filler, make self-contained
3. **Splitting** — break multi-topic questions into up to 3 sub-questions

The LLM returns structured JSON which is parsed and used to route the query into the agent loop.

### Prompts (`prompts.py`)

All LLM prompts are centralised in one file. This makes tuning behaviour straightforward — changing a prompt once affects the whole system. The file also documents how many LLM calls happen per query and why each one exists.

### Evaluation (`eval.py`)

Runs a set of questions through the retrieval pipeline and reports:
- **Retrieval coverage** — did the retriever find any chunks?
- **Answer informativeness** — did the LLM answer, or fall back to "I don't know"?

No labelled ground truth needed — runs entirely with your local Postgres and Ollama stack.

```bash
python3 eval.py                        # runs eval_questions.json
python3 eval.py --verbose              # also prints each answer
python3 eval.py --questions my_qs.json # custom question file
```

---

## Configuration

Key constants you can tune:

| File | Constant | Default | Effect |
|---|---|---|---|
| `agent.py` | `MAX_RETRIES` | 3 | Max retrieval attempts before giving up |
| `agent.py` | `MIN_CONTEXT_CHARS` | 100 | Minimum context length to consider sufficient |
| `vector_store.py` | `SCORE_THRESHOLD` | 0.3 | Minimum similarity to include a chunk |
| `vector_store.py` | `DEFAULT_TOP_K` | 7 | How many chunks to retrieve |
| `rag.py` | `MAX_PARENTS` | 3 | Max parent chunks sent to LLM |
| `chunker.py` | `CHILD_CHUNK_SIZE` | 500 | Characters per child chunk |
| `chunker.py` | `CHILD_CHUNK_OVERLAP` | 100 | Overlap between child chunks |
| `main.py` | `MAX_HISTORY` | 10 | Messages kept in memory |

---

## Possible improvements

- **Reranking** — add a cross-encoder reranker after retrieval for better precision
- **Streaming** — stream LLM tokens to the Gradio UI as they generate
- **Persistent sessions** — replace in-memory history with Postgres-backed sessions
- **Multi-user** — add authentication and per-user document namespaces
- **RAGAS evaluation** — add ground-truth labelled eval for answer quality scoring
- **Multi-step planning** — ReAct-style planning for complex multi-hop questions

---

## License

MIT
