"""
rag.py — Top-level entry points for the agentic RAG pipeline.

  retrieve(query)             — direct hybrid search (vector + BM25 via RRF),
                                 used by eval.py and anywhere that needs
                                 retrieval without the full agent loop.
  answer(query, history, ...) — analyze_query() → agent.run_agent()
                                 (tool select → reflect → generate → critique)
                                 for each sub-question, then synthesise.
  stream_answer(...)          — Gradio streaming helper.
"""

from query_intelligence import analyze_query, summarize_conversation
from agent import run_agent
from vector_store import hybrid_search
from chunker import parents_from_child_results


# ---------------------------------------------------------------------------
# Retrieval — unchanged public API, now backed by hybrid search
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 7) -> list[dict]:
    """
    Direct hybrid search (vector + BM25, fused via RRF) — used by eval.py
    and anywhere that needs retrieval without the full agent loop.

    This used to call vector_store.search() (dense-only). It now calls
    vector_store.hybrid_search() so eval.py measures the same retrieval
    backend the agent's vector_search/hybrid_search tools use — keeping
    the two paths in sync.

    Args:
        query: The search query string.
        top_k: Max child chunks to consider before loading parents.

    Returns:
        List of parent chunk dicts: [{parent_id, source, content}, ...]
    """
    child_results = hybrid_search(query, top_k=top_k)
    return parents_from_child_results(child_results, limit=3)


# ---------------------------------------------------------------------------
# Answer — routes through the agent (unchanged)
# ---------------------------------------------------------------------------

def answer(
    user_query: str,
    conversation_history: list[dict] | None = None,
    verbose: bool = False,
) -> tuple[str, dict]:
    """
    Generate an answer for the user's query using the agentic RAG loop.

    Flow:
      1. Summarise conversation history (for pronoun resolution)
      2. Analyse + rewrite the query via query_intelligence
      3. For each sub-question, run the full agent loop:
           tool selection → retrieval → rerank → reflection → generation → critique
      4. If multiple sub-questions, synthesise into one final answer

    Args:
        user_query:           Raw user input.
        conversation_history: List of {"role": ..., "content": ...} dicts.
        verbose:              If True, print agent step logs to console.

    Returns:
        (final_answer, analysis_info)
        analysis_info: {"is_clear": bool, "questions": list[str], "summary": str}
        — the query-analysis result, returned so callers that want to
          display it (verbose CLI, Gradio "show query analysis") don't
          need to call summarize_conversation()/analyze_query() a second
          time themselves.
    """
    history = conversation_history or []

    # Step 1: Summarise recent history for context
    summary = ""
    if len(history) >= 2:
        summary = summarize_conversation(history[-6:])

    # Step 2: Analyse and rewrite the query
    analysis = analyze_query(user_query, conversation_summary=summary)
    analysis_info = {
        "is_clear": analysis["is_clear"],
        "questions": analysis["questions"],
        "summary": summary,
    }

    # If unclear, return the clarification request immediately
    if not analysis["is_clear"]:
        return analysis["clarification_needed"], analysis_info

    questions = analysis["questions"]

    # Step 3: Run agent loop for each sub-question
    sub_answers = []

    for q in questions:
        if verbose:
            print(f"\n[rag] Running agent for: '{q}'")

        result = run_agent(q, conversation_history=history, verbose=verbose)
        sub_answers.append(result["answer"])

        if verbose:
            print(f"[rag] Tool used: {result['tool_used']} ({result['retrieval_attempts']} attempt(s))")
            print(f"[rag] Answer revised by critique: {result['was_revised']}")

    # Step 4: If only one sub-question, return its answer directly
    if len(sub_answers) == 1:
        return sub_answers[0], analysis_info

    # Step 5: Synthesise multiple sub-answers into one coherent response
    return _synthesise(user_query, questions, sub_answers), analysis_info


def _synthesise(
    original_query: str,
    sub_questions: list[str],
    sub_answers: list[str],
) -> str:
    """
    Combine multiple sub-question answers into one coherent response.

    Used when query_intelligence splits a question into multiple parts.
    Calls the LLM once more to write a unified answer.
    """
    from llm import chat

    parts = []
    for q, a in zip(sub_questions, sub_answers):
        parts.append(f"Sub-question: {q}\nAnswer: {a}")
    combined = "\n\n".join(parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a synthesis assistant. The user asked a multi-part question. "
                "You have been given answers to each part. "
                "Write a single, coherent, well-structured response that integrates all the answers. "
                "Do not repeat information unnecessarily. Be concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question: {original_query}\n\n"
                f"Sub-answers:\n{combined}\n\n"
                f"Write a unified answer:"
            ),
        },
    ]

    return chat(messages, temperature=0.0)


# ---------------------------------------------------------------------------
# Gradio streaming helper — unchanged
# ---------------------------------------------------------------------------

def stream_answer(
    user_query: str,
    conversation_history: list[dict] | None = None,
):
    """
    Generator that yields the answer word by word for Gradio streaming.

    Since the agent loop involves multiple LLM calls (not streamable end-to-end),
    we run the full agent first, then stream the final answer token by token
    for a smooth UI experience.

    Usage (in Gradio):
        for chunk in stream_answer(query, history):
            yield chunk
    """
    full_answer, _ = answer(user_query, conversation_history)

    # Stream word by word
    words = full_answer.split(" ")
    accumulated = ""
    for word in words:
        accumulated += word + " "
        yield accumulated.rstrip()
