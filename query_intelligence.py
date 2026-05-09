"""
query_intelligence.py — Stage 5: Query analysis before retrieval.

This is what separates a basic RAG from an agentic one.
Before we even search, we analyze the user's question and decide:

  1. Is the question clear enough to answer?
     → If not, ask for clarification instead of guessing.

  2. Does the question contain pronouns or references to prior context?
     → Rewrite it to be self-contained so the vector search works well.
     e.g. "What does it do?" → "What does the MolMo model do?"

  3. Is the question actually multiple questions in one?
     → Split into sub-questions and answer each independently.
     e.g. "What is RAG and how does chunking work?" → two questions

Why this matters for retrieval:
  Vector search works by matching the MEANING of your query to stored chunks.
  A vague query like "tell me more" or "what about it?" has no clear meaning
  to embed — the search returns garbage. A rewritten, specific query
  like "What are the architectural components of the MolMo model?" retrieves
  the right chunks.

How it works:
  We send the query + recent conversation history to the LLM and ask it
  to return a structured JSON response with three fields:
    - is_clear:            bool
    - questions:           list of rewritten sub-questions (or [rewritten query])
    - clarification_needed: message to show user if is_clear is False
"""

import json
from llm import chat


# ---------------------------------------------------------------------------
# The analysis prompt
# ---------------------------------------------------------------------------
# We instruct the LLM to return ONLY JSON — no preamble, no markdown fences.
# This makes it reliable to parse with json.loads().

ANALYSIS_SYSTEM_PROMPT = """You are a query analysis assistant for a RAG system.
Your job is to analyze a user's question and return a JSON object.

You must return ONLY valid JSON — no explanation, no markdown, no code fences.

JSON format:
{
  "is_clear": true or false,
  "questions": ["rewritten question 1", "rewritten question 2"],
  "clarification_needed": "message to ask user, or empty string if clear"
}

Rules for is_clear:
  - Set false if the question is too vague to search for (e.g. "tell me more",
    "what about it?", "explain", "go on")
  - Set false if the question refers to something with no context to resolve it
  - Set true for everything else, even imperfect questions

Rules for questions:
  - Always return at least one question
  - Rewrite the question to be fully self-contained:
      * Replace pronouns (it, they, this, that) with the actual subject
      * Remove filler words ("can you", "please", "I want to know")
      * Make it a direct, specific question
  - If the question contains multiple distinct topics, split into separate questions
  - Maximum 3 sub-questions — do not over-split
  - If is_clear is false, still return your best attempt at ["original question"]

Rules for clarification_needed:
  - If is_clear is true: return empty string ""
  - If is_clear is false: write a short, friendly question asking what they mean
"""


def analyze_query(
    user_query: str,
    conversation_summary: str = "",
) -> dict:
    """
    Analyze a user query and return structured analysis.

    Args:
        user_query:           The raw question from the user.
        conversation_summary: A short summary of recent conversation turns.
                              Used to resolve pronouns and references.
                              Empty string if this is the first message.

    Returns:
        A dict with three keys:
          - is_clear (bool)
          - questions (list of strings)
          - clarification_needed (str)

        Example — clear single question:
          {
            "is_clear": True,
            "questions": ["What are the main components of the MolMo model?"],
            "clarification_needed": ""
          }

        Example — unclear question:
          {
            "is_clear": False,
            "questions": ["tell me more"],
            "clarification_needed": "Could you clarify what you'd like to know more about?"
          }

        Example — multi-part question:
          {
            "is_clear": True,
            "questions": [
              "What is retrieval augmented generation?",
              "How does the chunking process work in RAG systems?"
            ],
            "clarification_needed": ""
          }
    """
    # Build the user message — include conversation summary if available
    if conversation_summary:
        user_message = (
            f"Recent conversation context:\n{conversation_summary}\n\n"
            f"User's new question: {user_query}"
        )
    else:
        user_message = f"User's question: {user_query}"

    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    raw_response = chat(messages, temperature=0.0)

    # Parse the JSON response
    # The LLM might occasionally add stray characters — we clean them up
    return _parse_json_response(raw_response, user_query)


def _parse_json_response(raw: str, original_query: str) -> dict:
    """
    Parse the LLM's JSON response safely.

    If parsing fails for any reason, we fall back to a safe default
    that treats the original query as clear and passes it through unchanged.
    This means the system degrades gracefully — a JSON parse failure
    doesn't crash the whole pipeline, it just skips the rewriting step.

    Args:
        raw:            The raw string response from the LLM.
        original_query: The original user query (used in fallback).

    Returns:
        Parsed dict with is_clear, questions, clarification_needed.
    """
    # Strip any accidental markdown fences the LLM might add
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove ```json ... ``` wrapping
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)

        # Validate the expected keys are present
        is_clear   = bool(data.get("is_clear", True))
        questions  = data.get("questions", [original_query])
        clarify    = data.get("clarification_needed", "")

        # Ensure questions is a non-empty list of strings
        if not isinstance(questions, list) or not questions:
            questions = [original_query]
        questions = [str(q).strip() for q in questions if str(q).strip()]
        if not questions:
            questions = [original_query]

        return {
            "is_clear":             is_clear,
            "questions":            questions,
            "clarification_needed": str(clarify),
        }

    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback — treat as clear, pass query through unchanged
        return {
            "is_clear":             True,
            "questions":            [original_query],
            "clarification_needed": "",
        }


# ---------------------------------------------------------------------------
# Conversation summarizer
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """You are a conversation summarizer.
Given a conversation history, write a 1-2 sentence summary covering:
  - The main topics discussed
  - Any key entities, names, or documents mentioned
  - The most recent thing the user asked about

Keep it factual and brief. This summary will be used to resolve
pronouns and references in the user's next question.

If the conversation is too short to summarize meaningfully, return an empty string.
"""


def summarize_conversation(messages: list[dict]) -> str:
    """
    Summarize recent conversation turns into a short context string.

    This summary is passed to analyze_query() so the LLM can resolve
    references like "it", "that model", "the second approach" etc.

    Args:
        messages: List of conversation message dicts.
                  Each dict: { "role": "user"/"assistant", "content": "..." }
                  Pass the last 6 messages for best results.

    Returns:
        A 1-2 sentence summary string, or "" if conversation is too short.
    """
    # Need at least 2 turns to summarize meaningfully
    # Filter out system messages — only user/assistant turns
    turns = [m for m in messages if m["role"] in ("user", "assistant")]

    if len(turns) < 2:
        return ""

    # Format the conversation for the LLM
    formatted = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"  # cap at 300 chars per turn
        for m in turns[-6:]  # last 6 turns max
    )

    summary_messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Conversation:\n{formatted}"},
    ]

    summary = chat(summary_messages, temperature=0.0)
    return summary.strip()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Stage 5: Query intelligence test ===\n")

    test_cases = [
        # (query, conversation_summary, description)
        (
            "What is RAG and how does chunking work?",
            "",
            "Multi-part question — should split into 2"
        ),
        (
            "What does it do?",
            "The user was asking about the MolMo vision-language model.",
            "Pronoun reference — should rewrite using context"
        ),
        (
            "tell me more",
            "We discussed how pgvector stores embeddings in Postgres.",
            "Vague query — should ask for clarification"
        ),
        (
            "What are the key architectural components of the MolMo model?",
            "",
            "Clear specific question — should pass through cleanly"
        ),
        (
            "explain",
            "",
            "Too vague — should ask for clarification"
        ),
    ]

    for query, summary, description in test_cases:
        print(f"Test: {description}")
        print(f"  Input:   '{query}'")
        if summary:
            print(f"  Context: '{summary}'")

        result = analyze_query(query, summary)

        print(f"  is_clear:  {result['is_clear']}")
        print(f"  questions: {result['questions']}")
        if result["clarification_needed"]:
            print(f"  clarify:   {result['clarification_needed']}")
        print()

    # Test the summarizer
    print("--- Summarizer test ---")
    fake_history = [
        {"role": "user",      "content": "What is the MolMo model?"},
        {"role": "assistant", "content": "MolMo is a vision-language model developed by AllenAI that can process both images and text."},
        {"role": "user",      "content": "How does it handle image inputs?"},
        {"role": "assistant", "content": "It uses a visual encoder to convert images into token embeddings that the language model can process."},
    ]

    summary = summarize_conversation(fake_history)
    print(f"Summary: {summary}")

    print("\nStage 5 complete.")
