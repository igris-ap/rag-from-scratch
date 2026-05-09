"""
main.py — Stage 6: Conversation memory + full pipeline.

This is the entry point. It wires together all previous stages:

  llm.py               → chat()
  chunker.py           → index_all_documents(), convert_all_pdfs()
  vector_store.py      → setup_db(), store_children(), search()
  rag.py               → retrieve(), build_prompt(), answer()
  query_intelligence.py → analyze_query(), summarize_conversation()

The full flow for each user turn:

  1. summarize_conversation()  — compress history into 1-2 sentences
  2. analyze_query()           — rewrite query, detect if unclear, split if multi-part
  3. If unclear → ask for clarification, wait for next input
  4. If clear   → for each sub-question:
                    retrieve() → build_prompt() → answer()
  5. If multiple sub-questions → aggregate answers into one response
  6. Append user + assistant messages to history
  7. Loop

Conversation memory:
  We keep a plain Python list of {"role": ..., "content": ...} dicts.
  That's all memory is — a growing list that we pass around.
  We trim it to the last MAX_HISTORY messages to avoid huge prompts.
  The summarizer compresses older context into a short string so
  nothing important is lost when we trim.
"""

from llm import chat
from chunker import convert_all_pdfs, index_all_documents
from vector_store import setup_db, clear_chunks, store_children
from rag import retrieve, build_prompt, RAG_SYSTEM_PROMPT
from query_intelligence import analyze_query, summarize_conversation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_HISTORY = 10   # keep last N messages in the rolling window
                   # older context is captured by the summarizer


# ---------------------------------------------------------------------------
# Multi-answer aggregation
# ---------------------------------------------------------------------------

AGGREGATE_SYSTEM_PROMPT = """You are a helpful assistant.
You have been given multiple answers to different parts of a user's question.
Combine them into a single, coherent, well-structured response.

Rules:
- Do not repeat the same information twice
- Keep the combined answer concise and direct
- Maintain all source citations from the individual answers
- Use natural connecting language between sections
- Do not say "Answer 1" or "Answer 2" — just flow naturally
"""


def aggregate_answers(original_query: str, answers: list[tuple[str, str]]) -> str:
    """
    Merge multiple sub-question answers into one coherent response.

    Called only when the query was split into multiple sub-questions.
    Each answer is independently retrieved and generated — this step
    weaves them together so the user gets one clean response.

    Args:
        original_query: The user's original (unsplit) question.
        answers:        List of (sub_question, answer) tuples.

    Returns:
        A single merged answer string.
    """
    # Build the message showing all sub-answers
    parts = []
    for i, (question, answer) in enumerate(answers, 1):
        parts.append(f"Sub-question {i}: {question}\nAnswer {i}: {answer}")

    combined = "\n\n".join(parts)

    messages = [
        {"role": "system", "content": AGGREGATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original question: {original_query}\n\n"
                f"Individual answers:\n{combined}\n\n"
                f"Please combine these into one clear response."
            ),
        },
    ]

    return chat(messages)


# ---------------------------------------------------------------------------
# Single turn: process one user message
# ---------------------------------------------------------------------------

def process_turn(user_input: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    Process one user message through the full pipeline.

    Args:
        user_input: The raw text the user typed.
        history:    The conversation history so far (list of message dicts).
                    Modified in-place and returned.

    Returns:
        (assistant_reply, updated_history)

    The history is a list of {"role": ..., "content": ...} dicts.
    We append the user message and assistant reply to it each turn.
    """

    # --- Step 1: Summarize recent conversation for context ---
    # Pass only the last MAX_HISTORY messages to the summarizer
    # so it doesn't get overwhelmed by a long conversation
    recent = history[-MAX_HISTORY:]
    conversation_summary = summarize_conversation(recent)

    # --- Step 2: Analyze the query ---
    analysis = analyze_query(user_input, conversation_summary)

    # --- Step 3: Handle unclear queries ---
    if not analysis["is_clear"]:
        clarification = analysis["clarification_needed"]
        if not clarification:
            clarification = "Could you please clarify your question?"

        # Add to history and return clarification
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": clarification})
        return clarification, history

    # --- Step 4: Answer each sub-question ---
    questions = analysis["questions"]
    answers   = []

    for question in questions:
        # Retrieve relevant parent chunks
        context_chunks = retrieve(question)

        # Build the RAG prompt
        user_message = build_prompt(question, context_chunks)

        # Generate the answer
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            # Inject conversation summary so the LLM has context
            # about what was discussed before
            *(
                [{"role": "user", "content": f"[Conversation so far: {conversation_summary}]"},
                 {"role": "assistant", "content": "Understood, I have context from our conversation."}]
                if conversation_summary else []
            ),
            {"role": "user", "content": user_message},
        ]
        reply = chat(messages)
        answers.append((question, reply))

    # --- Step 5: Aggregate if multiple sub-questions ---
    if len(answers) == 1:
        # Single question — use the answer directly
        final_reply = answers[0][1]
    else:
        # Multiple questions — merge into one response
        final_reply = aggregate_answers(user_input, answers)

    # --- Step 6: Update history ---
    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": final_reply})

    # Trim history to MAX_HISTORY messages
    # The summarizer captures older context so we don't lose it
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    return final_reply, history


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
      'verbose'  → toggle showing retrieved sources
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
            print(f"[Verbose: {'ON — showing retrieved sources' if verbose else 'OFF'}]\n")
            continue

        # --- Process the turn ---
        if verbose:
            # Show what the query analyzer decided
            recent  = history[-MAX_HISTORY:]
            summary = summarize_conversation(recent)
            analysis = analyze_query(user_input, summary)
            print(f"\n[Query analysis]")
            print(f"  is_clear:  {analysis['is_clear']}")
            print(f"  questions: {analysis['questions']}")
            if summary:
                print(f"  summary:   {summary[:100]}...")
            print()

        reply, history = process_turn(user_input, history)

        print(f"\nAssistant: {reply}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
