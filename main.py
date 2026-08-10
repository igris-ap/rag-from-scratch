"""
main.py — CLI entry point + pipeline wiring.

process_turn() delegates to rag.answer(), which handles conversation
summarization, query analysis, the full agent loop (tool selection →
rerank → reflection → generation → critique) per sub-question, and
multi-sub-question synthesis. app.py (Gradio) calls process_turn() too,
so the CLI and the UI always share one code path.

Wires together:
  chunker.py       → index_all_documents(), convert_all_pdfs()
  vector_store.py  → setup_db(), store_children()
  rag.py           → answer()   (routes through the agent loop)

Conversation memory:
  We keep a plain Python list of {"role": ..., "content": ...} dicts.
  That's all memory is — a growing list that we pass around.
  We trim it to the last MAX_HISTORY messages to avoid huge prompts.
  rag.answer() internally summarizes recent history for context before
  each agent run.
"""

from chunker import convert_all_pdfs, index_all_documents
from vector_store import setup_db, clear_chunks, store_children
from rag import answer as rag_answer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_HISTORY = 10   # keep last N messages in the rolling window
                   # older context is captured by the summarizer


# ---------------------------------------------------------------------------
# Single turn: process one user message
# ---------------------------------------------------------------------------

def process_turn(
    user_input: str, history: list[dict], verbose: bool = False
) -> tuple[str, list[dict], dict]:
    """
    Process one user message through the full agentic pipeline.

    Args:
        user_input: The raw text the user typed.
        history:    The conversation history so far (list of message dicts).
                    Modified in-place and returned.
        verbose:    If True, prints agent step logs (tool selection, rerank,
                    reflection, generation, critique) to the console.

    Returns:
        (assistant_reply, updated_history, analysis_info)
        analysis_info is rag.answer()'s query-analysis result — callers
        that want to display it (verbose CLI, Gradio "show query analysis")
        should use this instead of calling analyze_query()/
        summarize_conversation() again themselves.

    rag.answer() internally handles conversation summarization, query
    analysis/rewriting, clarification requests for unclear queries, the
    full agent loop (tool selection, rerank, retrieval reflection,
    generation, self-critique) per sub-question, and synthesis if the
    query was split into multiple sub-questions.
    """
    reply, analysis = rag_answer(user_input, conversation_history=history, verbose=verbose)

    # Update history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})

    # Trim history to MAX_HISTORY messages
    # The summarizer (inside rag.answer()) captures older context so we
    # don't lose it when trimming.
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    return reply, history, analysis


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_documents():
    """
    Run the full indexing pipeline:
      PDFs → Markdown → parent + child chunks → embed → store in Postgres

    Call this once before starting the chat loop, or any time you add
    new documents to the docs/ folder.
    """
    print("\nIndexing documents...")
    convert_all_pdfs()
    children = index_all_documents()

    if not children:
        print("No documents found. Add PDFs to the docs/ folder.")
        return False

    clear_chunks()
    store_children(children)
    print(f"Indexed {len(children)} chunks. Ready to chat.\n")
    return True


# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

def run():
    """
    Main entry point — sets up the database and runs the chat loop.

    Commands during chat:
      'reindex'  → re-run the full indexing pipeline
      'history'  → show the current conversation history
      'clear'    → clear conversation history and start fresh
      'verbose'  → toggle showing query analysis + agent step logs
      'quit'     → exit
    """
    print("=" * 55)
    print("  RAG from Scratch — no LangChain, no LangGraph")
    print("=" * 55)

    # Setup database
    print("\nSetting up database...")
    setup_db()

    # Index documents
    index_documents()

    # Conversation history — plain list of message dicts
    history: list[dict] = []
    verbose = False

    print("Type your question below.")
    print("Commands: reindex | history | clear | verbose | quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # --- Handle commands ---
        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        if user_input.lower() == "reindex":
            index_documents()
            continue

        if user_input.lower() == "history":
            if not history:
                print("[No history yet]\n")
            else:
                print("\n--- Conversation history ---")
                for msg in history:
                    role = msg["role"].upper()
                    content = msg["content"][:200]
                    print(f"{role}: {content}{'...' if len(msg['content']) > 200 else ''}")
                print()
            continue

        if user_input.lower() == "clear":
            history = []
            print("[History cleared]\n")
            continue

        if user_input.lower() == "verbose":
            verbose = not verbose
            print(f"[Verbose: {'ON — showing query analysis + agent steps' if verbose else 'OFF'}]\n")
            continue

        # --- Process the turn ---
        reply, history, analysis = process_turn(user_input, history, verbose=verbose)

        if verbose:
            print(f"\n[Query analysis]")
            print(f"  is_clear:  {analysis['is_clear']}")
            print(f"  questions: {analysis['questions']}")
            if analysis["summary"]:
                print(f"  summary:   {analysis['summary'][:100]}...")
            print()

        print(f"\nAssistant: {reply}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
