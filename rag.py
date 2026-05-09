"""
rag.py — Stage 4: The core RAG loop.

This is where everything comes together:
  1. User asks a question
  2. We search for relevant child chunks in Postgres
  3. We load their parent chunks from disk (more context)
  4. We build a prompt with the retrieved context
  5. We send it to Ollama and return the answer

No LangChain. Just the three modules we already built:
  - llm.py          → chat()
  - chunker.py      → load_parent_chunk()
  - vector_store.py → search()
"""

from llm import chat
from chunker import load_parent_chunk
from vector_store import search

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_K             = 7     # how many child chunks to retrieve
SCORE_THRESHOLD   = 0.3   # minimum similarity score to include a chunk
MAX_PARENTS       = 3     # max parent chunks to load (avoid huge prompts)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# This is the instruction we give the LLM before every answer.
# It tells the model to:
#   - Only use retrieved context, not its own training knowledge
#   - Cite which source the answer came from
#   - Say "I don't know" if the context doesn't cover the question
#
# Being explicit here is critical — without it, the LLM will happily
# hallucinate answers from its training data instead of your documents.

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions
strictly based on the provided context documents.

Rules:
1. Only use information from the CONTEXT section below to answer.
2. If the context does not contain enough information, say:
   "I don't have enough information in the provided documents to answer this."
3. Always mention which source document your answer comes from.
4. Be concise and direct. Do not repeat the question back.
5. Do not make up information or use outside knowledge.
"""


# ---------------------------------------------------------------------------
# Step 1: Retrieve relevant context
# ---------------------------------------------------------------------------

def retrieve(query: str) -> list[dict]:
    """
    Find the most relevant chunks for a query.

    Strategy (same as the original project):
      - Search for top-K child chunks (small, precise vectors)
      - For each unique parent_id found, load the full parent chunk
      - Return parents instead of children as context for the LLM

    Why load parents?
      The child chunk that matched the query might be just one sentence.
      Its parent contains the full section — headings, surrounding paragraphs,
      complete explanation. The LLM needs that full context to answer well.

    Why deduplicate by parent_id?
      Multiple child chunks from the same parent might all match the query.
      We only need the parent once — loading it twice wastes prompt space.

    Returns:
        List of parent chunk dicts, sorted best-match first.
        Each dict: { parent_id, source, content }
    """
    # Search Postgres for the most similar child chunks
    child_results = search(query, top_k=TOP_K, score_threshold=SCORE_THRESHOLD)

    if not child_results:
        return []

    # Collect unique parent IDs, preserving order (best match first)
    seen_parent_ids = []
    for child in child_results:
        pid = child["parent_id"]
        if pid not in seen_parent_ids:
            seen_parent_ids.append(pid)

    # Load parent chunks from disk, up to MAX_PARENTS
    parents = []
    for parent_id in seen_parent_ids[:MAX_PARENTS]:
        parent = load_parent_chunk(parent_id)
        if parent:
            parents.append(parent)

    return parents


# ---------------------------------------------------------------------------
# Step 2: Build the prompt
# ---------------------------------------------------------------------------

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Assemble the user message that gets sent to the LLM.

    Format:
        CONTEXT:
        [Source: file.pdf]
        ... full parent text ...

        [Source: file2.pdf]
        ... full parent text ...

        QUESTION:
        What is ...?

    The LLM sees the system prompt (RAG_SYSTEM_PROMPT) separately.
    This function only builds the user-turn message.

    Args:
        query:          The user's question.
        context_chunks: List of parent chunk dicts from retrieve().

    Returns:
        A formatted string ready to send as the user message.
    """
    if not context_chunks:
        # No context found — tell the LLM explicitly
        return (
            f"CONTEXT:\nNo relevant documents were found.\n\n"
            f"QUESTION:\n{query}"
        )

    # Format each parent chunk with its source label
    context_blocks = []
    for chunk in context_chunks:
        block = f"[Source: {chunk['source']}]\n{chunk['content']}"
        context_blocks.append(block)

    context_text = "\n\n---\n\n".join(context_blocks)

    return (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{query}"
    )


# ---------------------------------------------------------------------------
# Step 3: Generate the answer
# ---------------------------------------------------------------------------

def answer(query: str, verbose: bool = False) -> str:
    """
    Full RAG pipeline: retrieve → build prompt → generate answer.

    Args:
        query:   The user's question as a plain string.
        verbose: If True, print retrieved sources before the answer.
                 Useful for debugging — lets you see what context
                 the LLM is working with.

    Returns:
        The LLM's answer as a plain string.
    """
    # Step 1: Retrieve relevant parent chunks
    context_chunks = retrieve(query)

    if verbose:
        if context_chunks:
            print(f"\n[Retrieved {len(context_chunks)} context chunks]")
            for c in context_chunks:
                print(f"  - {c['source']} (parent: {c['parent_id']})")
        else:
            print("\n[No relevant chunks found]")

    # Step 2: Build the prompt
    user_message = build_prompt(query, context_chunks)

    # Step 3: Call the LLM
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    response = chat(messages)

    return response


# ---------------------------------------------------------------------------
# Simple interactive loop
# ---------------------------------------------------------------------------

def run_interactive():
    """
    A simple terminal chat loop.

    Type a question, get an answer, repeat.
    Type 'quit' or 'exit' to stop.
    Type 'verbose' to toggle showing retrieved sources.
    """
    print("\n=== RAG Question Answering System ===")
    print("Type your question and press Enter.")
    print("Commands: 'verbose' = toggle source display | 'quit' = exit\n")

    verbose = False

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        if query.lower() == "verbose":
            verbose = not verbose
            print(f"[Verbose mode: {'ON' if verbose else 'OFF'}]\n")
            continue

        # Generate and print the answer
        print("\nAssistant: ", end="", flush=True)
        reply = answer(query, verbose=verbose)
        print(reply)
        print()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Stage 4: RAG loop test ===\n")

    # Single test question — change this to match your documents
    test_query = "What is retrieval augmented generation?"

    print(f"Query: {test_query}\n")
    reply = answer(test_query, verbose=True)
    print(f"\nAnswer:\n{reply}")

    print("\n" + "="*50)
    print("Basic test done. Starting interactive mode...\n")

    # Drop into the interactive loop
    run_interactive()
