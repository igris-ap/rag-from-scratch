"""
prompts.py — Central registry for all LLM prompts in the system.

Why centralise prompts?
  When prompts are scattered inline across multiple files, changing
  behaviour means hunting through the codebase. Centralising them here
  means:

  1. Single place to tune — change a prompt once, affects the whole system
  2. Easy to compare versions — git diff on one file shows all prompt changes
  3. Explicit about what each LLM call is responsible for
  4. Makes it obvious how many LLM calls the system makes per query

How many LLM calls happen per query in this system?
  Minimum (fast path — clear query, sufficient retrieval, answer passes):
    1. query_intelligence.py  → analyze_query()         — query analysis
    2. agent.py               → select_tool()            — tool selection
    3. agent.py               → reflect_on_retrieval()   — retrieval reflection
    4. agent.py               → generate_answer()        — answer generation
    5. agent.py               → critique_and_revise()    — self-critique
    Total: 5 LLM calls

  Maximum (worst path — vague query, 3 retrieval retries, answer revised):
    1. query_intelligence.py  → summarize_conversation() — conversation summary
    2. query_intelligence.py  → analyze_query()          — query analysis
    3-5. agent.py             → select_tool() × 1        — tool selection
    6-8. agent.py             → reflect_on_retrieval() × 3 — 3 retry loops
    9.   agent.py             → generate_answer()         — answer generation
    10.  agent.py             → critique_and_revise()     — self-critique
    Total: up to 10 LLM calls

  This is the cost of being agentic. Each call is cheap with a local
  Ollama model, but it's worth knowing the upper bound.

Prompt design principles used here:
  1. Return ONLY JSON for structured outputs — no preamble, no fences
  2. Explicit format specification — show the exact JSON shape
  3. Rules section — enumerate edge cases explicitly
  4. Examples — show the LLM what good output looks like
  5. temperature=0.0 for all structured calls — we want determinism
"""

# ---------------------------------------------------------------------------
# query_intelligence.py prompts
# ---------------------------------------------------------------------------

QUERY_ANALYSIS_SYSTEM = """You are a query analysis assistant for a RAG system.
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
- Rewrite the question to be fully self-contained
- Replace pronouns (it, they, this, that) with the actual subject
- If the question contains multiple distinct topics, split into separate questions
- Maximum 3 sub-questions

Rules for clarification_needed:
- If is_clear is true: return empty string ""
- If is_clear is false: write a short, friendly clarification question"""

CONVERSATION_SUMMARY_SYSTEM = """You are a conversation summarizer.
Given a conversation history, write a 1-2 sentence summary covering:
- The main topics discussed
- Any key entities, names, or documents mentioned
- The most recent thing the user asked about

Keep it factual and brief. Return empty string if conversation is too short."""

# ---------------------------------------------------------------------------
# agent.py prompts
# ---------------------------------------------------------------------------

TOOL_SELECTION_SYSTEM = """You are the tool-selection component of an agentic RAG system.
Your job: read the user's question and select the best tool to retrieve relevant context.

{tool_descriptions}

Rules:
- Return ONLY a JSON object — no explanation, no markdown, no code fences.
- Format: {{"tool": "<tool_name>", "reason": "<one sentence why>"}}
- tool must be exactly one of: {tool_names}
- Default to "vector_search" when uncertain.

Examples:
  Question: "What is the architecture of the system?"
  → {{"tool": "vector_search", "reason": "Conceptual question suits semantic search."}}

  Question: "What is the exact value of CHUNK_SIZE?"
  → {{"tool": "keyword_search", "reason": "Exact constant name needs keyword match."}}

  Question: "What did you say earlier about embeddings?"
  → {{"tool": "recall_memory", "reason": "User referring to earlier conversation turn."}}"""

REFLECTION_SYSTEM = """You are the retrieval-reflection component of an agentic RAG system.
Your job: evaluate whether retrieved context is sufficient to answer the question.

Question: {question}

Retrieved context:
{context}

Rules:
- Return ONLY a JSON object — no explanation, no markdown, no code fences.
- Format: {{"sufficient": true/false, "reason": "<one sentence>", "rewritten_query": "<query or empty>"}}
- sufficient: true if context contains enough information to answer the question.
- sufficient: false if context is empty, irrelevant, too short, or misses key aspects.
- rewritten_query: if not sufficient, a rewritten query with different keywords or angle.
  If sufficient, return empty string "".

Be strict: partial answers → set sufficient to false."""

GENERATION_SYSTEM = """You are a helpful assistant answering questions based on retrieved context.
Use ONLY the information in the context below to answer the question.
If the context does not contain enough information, say so clearly.
Do not make up information.

Context:
{context}"""

CRITIQUE_SYSTEM = """You are the quality-control component of an agentic RAG system.
Your job: evaluate whether the generated answer adequately addresses the question.

Question: {question}
Generated answer: {answer}
Context used: {context_summary}

Rules:
- Return ONLY a JSON object — no explanation, no markdown, no code fences.
- Format: {{"passes": true/false, "issues": "<description or empty>", "revised_answer": "<revised or empty>"}}
- passes: true if the answer directly addresses the question and is grounded in context.
- passes: false if the answer is vague, misses key aspects, or contradicts context.
- revised_answer: if passes is false, write a better answer. If true, return "".

Be constructive — revise rather than reject."""

# ---------------------------------------------------------------------------
# Prompt builder helpers
# ---------------------------------------------------------------------------

def build_reflection_prompt(question: str, context_chunks: list[dict]) -> str:
    """Format the reflection system prompt with actual context."""
    if not context_chunks:
        context_str = "[No context retrieved]"
    else:
        parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")[:800]
            parts.append(f"[Chunk {i} — source: {source}]\n{content}")
        context_str = "\n\n".join(parts)

    return REFLECTION_SYSTEM.format(question=question, context=context_str)


def build_generation_prompt(context_chunks: list[dict]) -> str:
    """Format the generation system prompt with retrieved context."""
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        parts.append(f"[Source {i}: {source}]\n{content}")
    context_str = "\n\n".join(parts) if parts else "[No context available]"
    return GENERATION_SYSTEM.format(context=context_str)


def build_critique_prompt(
    question: str, answer: str, context_chunks: list[dict]
) -> str:
    """Format the critique system prompt."""
    sources = list({c.get("source", "unknown") for c in context_chunks})
    context_summary = f"{len(context_chunks)} chunk(s) from: {', '.join(sources)}"
    return CRITIQUE_SYSTEM.format(
        question=question,
        answer=answer,
        context_summary=context_summary,
    )
