"""
chunker.py — Stage 2: PDF → text → parent chunks → child chunks.

No LangChain. Just pymupdf for PDF reading and plain Python for splitting.

The strategy (same as the original project):
  - Parent chunks: large sections split on Markdown headers (H1/H2/H3).
    Stored on disk as JSON. Used for answer generation — lots of context.
  - Child chunks: small fixed-size pieces cut from each parent.
    Stored in the vector database. Used for search — precise matching.

Why two levels?
  Searching small chunks = better precision (less noise in the vector match).
  Generating from large chunks = better answers (more context for the LLM).
"""

import os
import re
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Install check — friendly message if pymupdf is missing
# ---------------------------------------------------------------------------
try:
    import pymupdf          # the library name after `pip install pymupdf`
    import pymupdf4llm
except ImportError:
    raise ImportError(
        "pymupdf is not installed.\n"
        "Run: pip install pymupdf\n"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCS_DIR         = "docs"           # put your PDFs here
MARKDOWN_DIR     = "markdown"       # converted markdown files land here
PARENT_STORE_DIR = "parent_store"   # one JSON file per parent chunk

CHILD_CHUNK_SIZE    = 500   # characters per child chunk
CHILD_CHUNK_OVERLAP = 100   # characters of overlap between adjacent children
MIN_PARENT_SIZE     = 2000  # merge parents smaller than this
MAX_PARENT_SIZE     = 10000 # split parents larger than this


# ---------------------------------------------------------------------------
# Step 1 of 3: PDF → Markdown
# ---------------------------------------------------------------------------

def pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a single PDF file to a Markdown string.

    pymupdf4llm reads the PDF's structure (headings, paragraphs, tables)
    and produces Markdown. This preserves heading levels (# ## ###) which
    we'll use in Step 2 to split into parent chunks.

    Args:
        pdf_path: Path to the .pdf file.

    Returns:
        The full document as a Markdown string.
    """
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(
        doc,
        page_separators=True,   # adds a separator between pages
        ignore_images=True,     # skip embedded images
    )
    # Clean up any encoding oddities
    md = md.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    return md


def convert_all_pdfs(docs_dir: str = DOCS_DIR, markdown_dir: str = MARKDOWN_DIR):
    """
    Convert every PDF in docs_dir to a .md file in markdown_dir.
    Skips files that already have a corresponding .md (unless you delete them).
    """
    os.makedirs(markdown_dir, exist_ok=True)
    pdf_files = list(Path(docs_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in '{docs_dir}/'. Add some PDFs and try again.")
        return

    for pdf_path in pdf_files:
        md_path = Path(markdown_dir) / (pdf_path.stem + ".md")
        if md_path.exists():
            print(f"  Skipping {pdf_path.name} — already converted.")
            continue

        print(f"  Converting {pdf_path.name} ...")
        md_text = pdf_to_markdown(str(pdf_path))
        md_path.write_text(md_text, encoding="utf-8")
        print(f"  Saved → {md_path}")


# ---------------------------------------------------------------------------
# Step 2 of 3: Markdown → Parent chunks
# ---------------------------------------------------------------------------

def split_on_headers(md_text: str, source_name: str) -> list[dict]:
    """
    Split a Markdown string into sections based on H1/H2/H3 headers.

    Each section becomes a "parent chunk" — a dict with:
        {
            "parent_id": "doc_stem_parent_0",
            "source":    "filename.pdf",
            "content":   "## Section heading\n\nAll the text...",
        }

    How it works:
        We scan line by line. Every time we hit a line starting with
        #, ##, or ###, we start a new section. The text between two
        headers = one parent chunk.

    Args:
        md_text:     Full Markdown string.
        source_name: e.g. "javascript" (the PDF stem, no extension).

    Returns:
        List of parent chunk dicts.
    """
    # Regex: line starting with one to three # characters
    header_pattern = re.compile(r"^#{1,3}\s", re.MULTILINE)

    # Find all positions where headers start
    header_positions = [m.start() for m in header_pattern.finditer(md_text)]

    # If no headers found, treat the whole document as one chunk
    if not header_positions:
        return [{
            "parent_id": f"{source_name}_parent_0",
            "source":    source_name + ".pdf",
            "content":   md_text.strip(),
        }]

    # Slice the text between consecutive header positions
    sections = []
    for i, start in enumerate(header_positions):
        end = header_positions[i + 1] if i + 1 < len(header_positions) else len(md_text)
        section_text = md_text[start:end].strip()
        if section_text:
            sections.append(section_text)

    # Assign IDs and source
    chunks = []
    for i, text in enumerate(sections):
        chunks.append({
            "parent_id": f"{source_name}_parent_{i}",
            "source":    source_name + ".pdf",
            "content":   text,
        })

    return chunks


def merge_small_parents(chunks: list[dict], min_size: int = MIN_PARENT_SIZE) -> list[dict]:
    """
    Merge consecutive chunks that are too small.

    Why: A heading like "## Introduction" followed by one sentence is useless
    as a standalone chunk. We glue it to the next section until the combined
    text is at least min_size characters.

    Args:
        chunks:   List of parent chunk dicts from split_on_headers().
        min_size: Minimum character count to keep a chunk alone.

    Returns:
        New list of parent chunks, with small ones merged into their neighbours.
    """
    if not chunks:
        return []

    merged = []
    buffer = chunks[0].copy()   # start accumulating into buffer

    for chunk in chunks[1:]:
        if len(buffer["content"]) < min_size:
            # Buffer is still too small — append this chunk's content to it
            buffer["content"] += "\n\n" + chunk["content"]
        else:
            # Buffer is big enough — save it, start a new buffer
            merged.append(buffer)
            buffer = chunk.copy()

    # Don't forget the last buffer
    merged.append(buffer)
    return merged


def split_large_parents(chunks: list[dict], max_size: int = MAX_PARENT_SIZE) -> list[dict]:
    """
    Split any chunk that exceeds max_size characters.

    Why: A 50,000-character chunk sent to the LLM as context would eat the
    entire context window. We cap parent size so the LLM always gets
    manageable input.

    Splitting strategy: divide into equal-sized pieces so no piece exceeds max_size.

    Args:
        chunks:   List of parent chunk dicts.
        max_size: Maximum character count per chunk.

    Returns:
        New list with oversized chunks split into smaller ones.
    """
    result = []
    for chunk in chunks:
        text = chunk["content"]
        if len(text) <= max_size:
            result.append(chunk)
            continue

        # How many pieces do we need?
        n_pieces = math.ceil(len(text) / max_size)
        piece_size = len(text) // n_pieces

        for i in range(n_pieces):
            start = i * piece_size
            end   = start + piece_size if i < n_pieces - 1 else len(text)
            result.append({
                "parent_id": f"{chunk['parent_id']}_sub{i}",
                "source":    chunk["source"],
                "content":   text[start:end].strip(),
            })

    return result


def build_parent_chunks(md_text: str, source_name: str) -> list[dict]:
    """
    Full parent-chunk pipeline for one document:
      split on headers → merge small → split large

    Returns the final list of parent chunk dicts.
    """
    chunks = split_on_headers(md_text, source_name)
    chunks = merge_small_parents(chunks)
    chunks = split_large_parents(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Step 3 of 3: Parent chunks → Child chunks
# ---------------------------------------------------------------------------

def split_into_children(parent: dict, chunk_size: int = CHILD_CHUNK_SIZE,
                         overlap: int = CHILD_CHUNK_OVERLAP) -> list[dict]:
    """
    Cut one parent chunk into many small child chunks.

    Strategy: sliding window over the text.
      - Move forward by (chunk_size - overlap) each step.
      - This means consecutive children share `overlap` characters.
      - Overlap prevents a sentence from being cut in half with no context
        in either neighbour.

    Each child chunk is a dict:
        {
            "parent_id":  "doc_parent_0",   ← links back to its parent
            "source":     "file.pdf",
            "content":    "small piece of text...",
            "child_index": 0,               ← position within the parent
        }

    Args:
        parent:     A parent chunk dict.
        chunk_size: Characters per child.
        overlap:    Characters of overlap between adjacent children.

    Returns:
        List of child chunk dicts.
    """
    text     = parent["content"]
    step     = chunk_size - overlap    # how far to advance each iteration
    children = []
    i        = 0
    idx      = 0

    while i < len(text):
        piece = text[i : i + chunk_size]
        if piece.strip():              # skip empty/whitespace-only pieces
            children.append({
                "parent_id":   parent["parent_id"],
                "source":      parent["source"],
                "content":     piece,
                "child_index": idx,
            })
            idx += 1
        i += step

    return children


# ---------------------------------------------------------------------------
# Save / load parent chunks (disk storage)
# ---------------------------------------------------------------------------

def save_parent_chunks(parents: list[dict], store_dir: str = PARENT_STORE_DIR):
    """
    Save each parent chunk as its own JSON file.

    File name = parent_id + ".json"
    This makes lookup O(1): given a parent_id, just open that file.

    In the original project this is exactly what retrieve_parent_chunks() does.
    """
    os.makedirs(store_dir, exist_ok=True)

    # Clear old files so stale chunks don't persist after re-indexing
    for old_file in Path(store_dir).glob("*.json"):
        old_file.unlink()

    for parent in parents:
        file_path = Path(store_dir) / f"{parent['parent_id']}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(parent, f, ensure_ascii=False, indent=2)


def load_parent_chunk(parent_id: str, store_dir: str = PARENT_STORE_DIR) -> dict | None:
    """
    Load one parent chunk by its ID.

    Returns the dict, or None if not found.
    This is what the RAG pipeline calls after finding a relevant child chunk.
    """
    file_path = Path(store_dir) / f"{parent_id}.json"
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parents_from_child_results(child_results: list[dict], limit: int) -> list[dict]:
    """
    Turn ranked child-chunk search results into their parent chunks.

    Dedupes by parent_id (keeping first-seen order, i.e. best rank first),
    then loads up to `limit` parents from disk. Shared by every retrieval
    path — rag.retrieve() and tools.vector_search()/hybrid_search() — so
    the child-to-parent resolution logic lives in exactly one place.

    Args:
        child_results: List of child chunk dicts, each with a "parent_id" key.
        limit:         Max number of parent chunks to load.

    Returns:
        List of parent chunk dicts: [{parent_id, source, content}, ...]
    """
    if not child_results:
        return []

    seen_parent_ids = []
    for child in child_results:
        pid = child["parent_id"]
        if pid not in seen_parent_ids:
            seen_parent_ids.append(pid)

    parents = []
    for parent_id in seen_parent_ids[:limit]:
        parent = load_parent_chunk(parent_id)
        if parent:
            parents.append(parent)

    return parents


# ---------------------------------------------------------------------------
# Master function: run the full pipeline for all documents
# ---------------------------------------------------------------------------

def index_all_documents(
    markdown_dir: str = MARKDOWN_DIR,
    parent_store_dir: str = PARENT_STORE_DIR,
) -> list[dict]:
    """
    Process every .md file in markdown_dir through the full pipeline:
        Markdown → parent chunks → save to disk
        parent chunks → child chunks → return list

    Returns:
        All child chunks across all documents (to be embedded in Stage 3).
    """
    md_files = list(Path(markdown_dir).glob("*.md"))
    if not md_files:
        print(f"No markdown files in '{markdown_dir}/'. Run convert_all_pdfs() first.")
        return []

    all_parents  = []
    all_children = []

    for md_file in md_files:
        source_name = md_file.stem
        print(f"\nProcessing: {md_file.name}")

        md_text = md_file.read_text(encoding="utf-8")
        parents = build_parent_chunks(md_text, source_name)

        print(f"  → {len(parents)} parent chunks")

        children = []
        for parent in parents:
            children.extend(split_into_children(parent))

        print(f"  → {len(children)} child chunks")

        all_parents.extend(parents)
        all_children.extend(children)

    save_parent_chunks(all_parents, parent_store_dir)
    print(f"\nSaved {len(all_parents)} parent chunks to '{parent_store_dir}/'")
    print(f"Total child chunks ready to embed: {len(all_children)}")

    return all_children


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Stage 2: Chunker test ===\n")

    # Make the required folders
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(MARKDOWN_DIR, exist_ok=True)

    # If there are no PDFs yet, demonstrate with synthetic text
    if not list(Path(DOCS_DIR).glob("*.pdf")):
        print("No PDFs found — running demo with synthetic Markdown.\n")

        sample_md = """# Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that combines
information retrieval with language model generation. Instead of relying
solely on the model's trained knowledge, RAG fetches relevant documents
at query time and uses them as context for the answer.

This makes the system more accurate, more up-to-date, and more
transparent — you can always point to the source document that
produced a given answer.

## How Retrieval Works

The retrieval step converts both documents and queries into vector
embeddings — numerical representations of meaning. Documents that are
semantically similar to the query will have embeddings that are close
in vector space, measured by cosine similarity.

A vector database stores these embeddings and can find the top-K most
similar documents to any query in milliseconds, even across millions
of documents.

## How Generation Works

Once relevant chunks are retrieved, they are inserted into the LLM
prompt as context. The model is instructed to answer using only the
provided context, which grounds the response in real source material
and reduces hallucination.

## Parent-Child Chunking

A key design decision in production RAG systems is chunk size.
Small chunks give precise retrieval — the embedding captures a narrow,
specific meaning. Large chunks give the LLM enough context to write
a coherent, complete answer.

The parent-child strategy solves this by storing two versions of every
section: a small child chunk for retrieval, and a large parent chunk
for generation. When a child chunk matches a query, we fetch its parent
to get the full context.
"""
        # Write sample md so we can test the pipeline
        Path(MARKDOWN_DIR + "/sample.md").write_text(sample_md, encoding="utf-8")

    # Convert any PDFs that exist
    convert_all_pdfs()

    # Run full pipeline
    children = index_all_documents()

    # Print some examples
    if children:
        print("\n--- Example child chunk ---")
        c = children[0]
        print(f"parent_id:   {c['parent_id']}")
        print(f"source:      {c['source']}")
        print(f"child_index: {c['child_index']}")
        print(f"length:      {len(c['content'])} chars")
        print(f"content:     {c['content'][:200]}...")

        print("\n--- Loading its parent ---")
        parent = load_parent_chunk(c["parent_id"])
        if parent:
            print(f"parent length: {len(parent['content'])} chars")
            print(f"parent start:  {parent['content'][:200]}...")

    print("\nStage 2 complete.")
