# RAG from Scratch

A fully working **Retrieval-Augmented Generation (RAG)** system built from scratch in Python — no LangChain, no LangGraph, no vector database SDKs. Just Python, Postgres, and Ollama.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Postgres](https://img.shields.io/badge/Postgres-pgvector-336791)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What this is

Most RAG tutorials use LangChain or LlamaIndex — frameworks that hide the actual mechanics behind layers of abstraction. This project implements every component manually so you can see exactly how RAG works:

- How PDFs become searchable chunks
- How text becomes vectors and gets stored in a database
- How a query finds the right chunks
- How the LLM generates a grounded answer
- How conversation history and query rewriting make it agentic

---

## Features

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
summarize_conversation()     ← compress history into 1-2 sentences
      │
      ▼
analyze_query()              ← rewrite, detect unclear, split multi-part
      │
      ├── unclear → ask for clarification
      │
      └── clear → for each sub-question:
                    retrieve()        ← search child chunks in pgvector
                    load_parent()     ← fetch full context from disk
                    build_prompt()    ← assemble RAG prompt
                    chat()            ← call Ollama LLM
                         │
                         ▼
                  aggregate_answers() ← merge if multiple sub-questions
                         │
                         ▼
                    Final answer
```

### File structure

```
.
├── app.py                  # Gradio web UI
├── main.py                 # CLI entry point + pipeline wiring
├── llm.py                  # Ollama HTTP client (no SDK)
├── chunker.py              # PDF → parent + child chunks
├── vector_store.py         # Embeddings + Postgres/pgvector
├── rag.py                  # Retrieve → prompt → generate
├── query_intelligence.py   # Query rewriting + conversation summary
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

### Chunking (`chunker.py`)

PDFs are converted to Markdown using `pymupdf4llm`, which preserves heading structure. The text is then split at H1/H2/H3 headers into **parent chunks** (2,000–10,000 characters). Each parent is further split into **child chunks** (500 characters, 100-character overlap).

Parent chunks are saved as JSON files. Child chunks are embedded and stored in Postgres. When a child chunk matches a query, its parent is loaded to give the LLM full context.

### Vector search (`vector_store.py`)

Uses `sentence-transformers/all-MiniLM-L6-v2` loaded directly in-process — no HTTP calls. All chunks are embedded in a single batch call at index time. Vectors are stored in Postgres using the `pgvector` extension with an HNSW index for fast approximate nearest-neighbour search.

Search uses cosine similarity via pgvector's `<=>` operator.

### Query intelligence (`query_intelligence.py`)

Before retrieval, every query is analyzed by the LLM:

1. **Clarity check** — is the question specific enough to search for?
2. **Rewriting** — replace pronouns, remove filler, make self-contained
3. **Splitting** — break multi-topic questions into up to 3 sub-questions

The LLM returns structured JSON which is parsed and used to route the query.

### Conversation memory (`main.py`)

The last 10 messages are kept in a rolling window. Before each turn, they are summarized into 1-2 sentences. This summary is passed to the query analyzer to resolve references like "it", "that model", or "the second approach".

---

## Configuration

Key constants you can tune:

| File | Constant | Default | Effect |
|---|---|---|---|
| `vector_store.py` | `SCORE_THRESHOLD` | 0.3 | Minimum similarity to include a chunk |
| `vector_store.py` | `DEFAULT_TOP_K` | 7 | How many chunks to retrieve |
| `rag.py` | `MAX_PARENTS` | 3 | Max parent chunks sent to LLM |
| `chunker.py` | `CHILD_CHUNK_SIZE` | 500 | Characters per child chunk |
| `chunker.py` | `CHILD_CHUNK_OVERLAP` | 100 | Overlap between child chunks |
| `main.py` | `MAX_HISTORY` | 10 | Messages kept in memory |

---

## Possible improvements

- **Reranking** — add a cross-encoder reranker after retrieval for better precision
- **Hybrid search** — add BM25 sparse embeddings alongside dense for keyword matching
- **Persistent sessions** — replace in-memory history with Postgres-backed sessions
- **Evaluation** — add RAGAS to measure retrieval and answer quality
- **Streaming** — stream LLM responses token by token for better UX
- **Multi-user** — add authentication and per-user document namespaces

---

## License

MIT
