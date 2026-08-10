"""
agent.py — The agentic core of the RAG system.

This is the file that makes the system genuinely agentic.

What makes something an "agent"?
  A fixed pipeline always does the same steps in the same order.
  An agent OBSERVES intermediate results and DECIDES what to do next.
  It has a feedback loop.

This agent has three feedback loops:

  Loop 1 — Tool selection
    The LLM reads the query and a description of available tools,
    then decides which tool to call. It's not hardcoded — the LLM
    makes a judgment call based on what the question needs.

  Loop 2 — Retrieval reflection
    After tool call returns results, the LLM asks itself:
    "Is this context actually sufficient to answer the question?"
    If NO → rewrite the query and try a different tool (up to MAX_RETRIES).
    If YES → proceed to generation.
    This is the core feedback loop that separates pipeline from agent.

  Loop 3 — Answer self-critique
    After generating an answer, the LLM critiques its own output:
    "Does this answer actually address the question?
     Is anything missing or potentially wrong?"
    If critique flags issues → revise the answer before returning.
    If answer passes → return as-is.

The full flow:

  query
    │
    ▼
  [select_tool]         LLM picks: vector_search / keyword_search /
    │                                hybrid_search / recall_memory
    ▼
  [call_tool]           Execute the selected tool
    │
    ▼
  [rerank]               Cross-encoder re-scores + trims the candidate
    │                     pool the tool returned (see rerank.py)
    ▼
  [reflect_on_retrieval] ──── SUFFICIENT? ────► [generate_answer]
    │                                                    │
    └── INSUFFICIENT? ──► rewrite query ──┐             ▼
         (up to MAX_RETRIES)              │         [critique_answer]
              │                           │              │
              └───────────────────────────┘     PASS? ──► return answer
                                                FAIL? ──► revise → return

Each step is a SEPARATE LLM call with a focused prompt (except rerank,
which is a local cross-encoder pass — no LLM call needed).
Single responsibility at every stage — easier to debug, easier to explain.
"""

import json
from llm import chat
from tools import call_tool, describe_tools, TOOL_REGISTRY
from rerank import rerank


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_RETRIES = 3          # max retrieval attempts before giving up
MIN_CONTEXT_CHARS = 100  # minimum context length to consider "sufficient"
RERANK_KEEP = 3           # how many candidates survive reranking per attempt

# The exact sentence the agent emits when it cannot answer from context.
# Defined once and injected into the prompts below (and returned directly
# when retrieval found nothing) so the generation prompt, the critique
# prompt, the no-context return, and eval.py's fallback detection can
# never drift apart. eval.py imports this — don't inline the string.
NO_CONTEXT_ANSWER = "I don't have information about this in the provided documents."


# ---------------------------------------------------------------------------
# Step 1: Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_PROMPT = """You are the tool-selection component of an agentic RAG system.
Your job: read the user's question and select the best tool to retrieve relevant context.

{tool_descriptions}

Rules:
- Return ONLY a JSON object — no explanation, no markdown, no code fences.
- Format: {{"tool": "<tool_name>", "reason": "<one sentence why>"}}
- tool must be exactly one of: {tool_names}
- Default to "vector_search" when uncertain.
- If conversation history is empty or this is the first question, NEVER choose recall_memory.
  recall_memory is only useful when the user explicitly references something said earlier.

Examples:
  Question: "What is the architecture of the system?"
  → {{"tool": "vector_search", "reason": "Conceptual question suits semantic search."}}

  Question: "What is the exact value of the CHUNK_SIZE constant?"
  → {{"tool": "keyword_search", "reason": "Exact term match needed for a specific constant."}}

  Question: "What did you say earlier about embeddings?"
  → {{"tool": "recall_memory", "reason": "User is referring to a previous conversation turn."}}

  Question: "What is parent-child chunking?"
  → {{"tool": "vector_search", "reason": "Knowledge question with no prior history — use semantic search."}}

  Question: "How does the SCORE_THRESHOLD setting affect what counts as similar enough?"
  → {{"tool": "hybrid_search", "reason": "Mixes a conceptual question with a specific constant name — combine both signals."}}
"""


def select_tool(query: str) -> tuple[str, str]:
    """
    Ask the LLM which tool to use for this query.

    Args:
        query: The user's question (already rewritten by query_intelligence).

    Returns:
        (tool_name, reason) — tool_name is one of the keys in TOOL_REGISTRY.
        Falls back to "vector_search" if LLM response can't be parsed.
    """
    tool_names = list(TOOL_REGISTRY.keys())

    system_prompt = TOOL_SELECTION_PROMPT.format(
        tool_descriptions=describe_tools(),
        tool_names=str(tool_names),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}"},
    ]

    raw = chat(messages, temperature=0.0)

    # Parse JSON response
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)
        tool_name = data.get("tool", "vector_search")
        reason = data.get("reason", "")

        # Validate — if LLM hallucinated a tool name, fall back
        if tool_name not in tool_names:
            tool_name = "vector_search"
            reason = "Fallback: LLM returned unknown tool name."

        return tool_name, reason

    except (json.JSONDecodeError, KeyError):
        return "vector_search", "Fallback: JSON parse failed."


# ---------------------------------------------------------------------------
# Step 2: Retrieval reflection
# ---------------------------------------------------------------------------

REFLECTION_PROMPT = """You are the retrieval-reflection component of an agentic RAG system.
Your job: evaluate whether retrieved context is sufficient to answer the question.

Question: {question}

Retrieved context:
{context}

Rules:
- Return ONLY a JSON object — no explanation, no markdown, no code fences.
- Format: {{"sufficient": true/false, "reason": "<one sentence, in your own words>", "rewritten_query": "<query or empty>"}}
- sufficient: true if the context discusses, defines, or explains the SAME concept/entity the question
  asks about — even using different wording, synonyms, or paraphrasing. The context does NOT need to
  repeat the question's exact words or phrase. Judge by meaning, not by string matching.
- sufficient: false if the context is about a genuinely different subject with no discussion — direct
  or paraphrased — of what's being asked. A context about a different named thing (a different constant,
  a different model, a different section) than the one in the question is NOT sufficient, even if it's
  in the same general domain.
- Do NOT set sufficient to false just because the wording differs from the question, or because the
  answer is incomplete — partial context about the RIGHT subject, described in different words, is enough.
- Write "reason" as your own one-sentence judgment about THIS specific context — do not reuse stock phrases.
- rewritten_query: only if sufficient is false, provide a rewritten query. Otherwise return "".

Examples (for illustration only — write your own reasoning for the actual case, don't copy this wording):

  Question: "What is retrieval augmented generation?"
  Context: explains a method that retrieves relevant documents and feeds them to a language model to
  ground its output, without using the words "retrieval augmented generation" verbatim.
  → {{"sufficient": true, "rewritten_query": ""}}  (same concept, different words — this counts)

  Question: "What is the exact value of the SCORE_THRESHOLD constant?"
  Context: describes an unrelated evaluation policy with no mention, direct or paraphrased, of any
  similarity threshold or scoring constant.
  → {{"sufficient": false, "rewritten_query": "SCORE_THRESHOLD value configuration"}}  (genuinely different subject)
"""


def reflect_on_retrieval(query: str, context_chunks: list[dict]) -> tuple[bool, str, str]:
    """
    Ask the LLM whether the retrieved context is sufficient.

    Args:
        query:          The question being answered.
        context_chunks: List of retrieved chunk dicts (from call_tool, post-rerank).

    Returns:
        (sufficient, reason, rewritten_query)
        sufficient:      True if context is good enough to answer.
        reason:          Why the LLM made this decision.
        rewritten_query: A better query to try if not sufficient. "" if sufficient.
    """
    # Format context for the prompt
    if not context_chunks:
        context_str = "[No context retrieved]"
    else:
        parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")[:800]  # cap at 800 chars per chunk
            parts.append(f"[Chunk {i} — source: {source}]\n{content}")
        context_str = "\n\n".join(parts)

    # Hard check: if context is too short, skip LLM call and fail fast
    total_chars = sum(len(c.get("content", "")) for c in context_chunks)
    if total_chars < MIN_CONTEXT_CHARS:
        return False, "Context too short or empty.", query + " (more detail)"

    messages = [
        {
            "role": "system",
            "content": REFLECTION_PROMPT.format(
                question=query,
                context=context_str,
            ),
        },
        {"role": "user", "content": "Is this context sufficient?"},
    ]

    raw = chat(messages, temperature=0.0)

    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)
        sufficient = bool(data.get("sufficient", False))
        reason = data.get("reason", "")
        rewritten = data.get("rewritten_query", "")
        return sufficient, reason, rewritten

    except (json.JSONDecodeError, KeyError):
        # Parse failed — assume sufficient to avoid infinite loops
        return True, "Reflection parse failed — proceeding.", ""


# ---------------------------------------------------------------------------
# Step 3: Answer generation
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """You are a question-answering assistant with access ONLY to the retrieved context below.
You have no other knowledge. If the context does not contain information relevant to the question, 
you MUST respond with exactly: "{fallback}"
Do not use any knowledge from your training. Only use what is in the context.

Context:
{context}
"""

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Generate an answer from the retrieved context.

    Args:
        query:          The user's question.
        context_chunks: The retrieved, reranked, and reflection-approved context.

    Returns:
        The generated answer as a plain string.
    """
    # Format context
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        parts.append(f"[Source {i}: {source}]\n{content}")
    context_str = "\n\n".join(parts) if parts else "[No context available]"

    messages = [
        {
            "role": "system",
            "content": GENERATION_PROMPT.format(
                context=context_str, fallback=NO_CONTEXT_ANSWER
            ),
        },
        {"role": "user", "content": query},
    ]

    return chat(messages, temperature=0.0)


# ---------------------------------------------------------------------------
# Step 4: Self-critique and revision
# ---------------------------------------------------------------------------

CRITIQUE_PROMPT = """You are the quality-control component of an agentic RAG system.

Your job: evaluate whether the generated answer adequately addresses the question.

Question: {question}

Generated answer: {answer}

Context used: {context_summary}

Rules:
- Return ONLY a JSON object — no explanation, no markdown, no code fences.
- Format: {{"passes": true/false, "issues": "<description or empty>", "revised_answer": "<revised or empty>"}}
- passes: true if the answer directly addresses the question, is grounded in context, and has no obvious gaps.
- passes: false if the answer is vague, misses key aspects, says "I don't know" when context is available,
  contradicts the context, or is too short to be useful.
- passes: false if the context used doesn't actually cover the specific subject of the question — e.g.
  the question asks about a specific named thing (a constant, a section, a claim) and the context talks
  about something else in the same general area instead of that specific thing.
- issues: if passes is false, briefly describe what's wrong.
- revised_answer: if passes is false and a better answer CAN be written from the given context, write it.
  If the context genuinely does not cover the question's subject (topic mismatch), set revised_answer to
  exactly: "{fallback}"
  If passes is true, return empty string
- passes: false if the answer contains information that does not appear anywhere in the context summary
  (this means the model used training knowledge instead of the retrieved context). "".

Be constructive — revise rather than reject where possible, but don't paper over a genuine topic mismatch.
"""


def critique_and_revise(
    query: str,
    answer: str,
    context_chunks: list[dict],
) -> tuple[str, bool, str]:
    """
    Critique the generated answer and revise if necessary.

    Args:
        query:          The original question.
        answer:         The generated answer to evaluate.
        context_chunks: The context that was used for generation.

    Returns:
        (final_answer, was_revised, critique_notes)
        final_answer:   The original or revised answer.
        was_revised:    True if the answer was revised.
        critique_notes: What the critique found (for logging/transparency).
    """
    # Brief context summary for the critique prompt (don't repeat full context)
    sources = list({c.get("source", "unknown") for c in context_chunks})
    context_summary = f"{len(context_chunks)} chunk(s) from: {', '.join(sources)}"

    messages = [
        {
            "role": "system",
            "content": CRITIQUE_PROMPT.format(
                question=query,
                answer=answer,
                context_summary=context_summary,
                fallback=NO_CONTEXT_ANSWER,
            ),
        },
        {"role": "user", "content": "Evaluate this answer."},
    ]

    raw = chat(messages, temperature=0.0)

    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)
        passes = bool(data.get("passes", True))
        issues = data.get("issues", "")
        revised = data.get("revised_answer", "").strip()

        if not passes and revised:
            return revised, True, issues
        else:
            return answer, False, issues

    except (json.JSONDecodeError, KeyError):
        # Parse failed — return original answer unchanged
        return answer, False, "Critique parse failed."


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(
    query: str,
    conversation_history: list[dict] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run the full agentic RAG loop for a single query.

    This is the entry point called by rag.py (or main.py / Gradio).

    Args:
        query:                The user's question (pre-processed by query_intelligence).
        conversation_history: Full conversation history for recall_memory tool.
        verbose:              If True, print each agent step to console.

    Returns:
        A dict with:
          answer          — the final answer string
          tool_used       — which tool was selected
          tool_reason     — why that tool was selected
          retrieval_attempts — how many retrieval attempts were made
          was_revised     — whether the answer was revised by critique
          critique_notes  — what the critique flagged (empty string if passed)
          context_chunks  — the chunks used for final generation
    """
    history = conversation_history or []

    def log(msg: str):
        if verbose:
            print(f"[agent] {msg}")

    # ------------------------------------------------------------------
    # Step 1: Tool selection
    # ------------------------------------------------------------------
    tool_name, tool_reason = select_tool(query)
    log(f"Tool selected: {tool_name} — {tool_reason}")

    # ------------------------------------------------------------------
    # Step 2: Retrieval loop with rerank + reflection
    # ------------------------------------------------------------------
    current_query = query
    context_chunks = []
    attempts = 0
    sufficient = False

    # Track which tools we've tried to avoid repeating the same one
    tried_tools = []

    while attempts < MAX_RETRIES and not sufficient:
        attempts += 1
        log(f"Retrieval attempt {attempts}/{MAX_RETRIES} — tool: {tool_name}, query: '{current_query}'")

        # Call the selected tool
        results = call_tool(tool_name, current_query, conversation_history=history)
        log(f"  Retrieved {len(results)} candidate(s)")

        # Rerank: cross-encoder re-scores the candidate pool against the
        # current query and keeps the top RERANK_KEEP. This runs
        # regardless of which tool was used — vector_search and
        # hybrid_search return a wider pool specifically so this step
        # has something to work with; recall_memory / keyword_search
        # results get reranked too (rerank() only needs a "content" key).
        if results:
            results = rerank(current_query, results, top_k=RERANK_KEEP)
            log(f"  Reranked to top {len(results)}")

        # Reflect: is this sufficient?
        sufficient, reason, rewritten_query = reflect_on_retrieval(current_query, results)
        log(f"  Reflection: sufficient={sufficient} — {reason}")

        if sufficient:
            context_chunks = results
            break

        # Not sufficient — prepare for next attempt
        tried_tools.append(tool_name)

        # Retry strategy: always try vector_search first if not yet tried,
        # then keyword_search, then hybrid_search, then recall_memory as
        # last resort. hybrid_search sits before recall_memory since it
        # combines both document-retrieval signals — worth trying before
        # falling back to conversation history.
        retry_order = ["vector_search", "keyword_search", "hybrid_search", "recall_memory"]
        next_tool = None
        for t in retry_order:
            if t not in tried_tools:
                next_tool = t
                break
        if next_tool is None:
            log("All tools exhausted — using best available context")
            context_chunks = results
            break
        tool_name = next_tool

        # Use the rewritten query if reflection provided one
        if rewritten_query and rewritten_query.strip():
            current_query = rewritten_query

    if not context_chunks and results:
        context_chunks = results

    # ------------------------------------------------------------------
    # Step 3: Generate answer
    # ------------------------------------------------------------------
    log(f"Generating answer from {len(context_chunks)} chunk(s)...")

    if not context_chunks:
        return {
            "answer": NO_CONTEXT_ANSWER,
            "tool_used": tool_name,
            "tool_reason": tool_reason,
            "retrieval_attempts": attempts,
            "was_revised": False,
            "critique_notes": "No context retrieved — skipped generation.",
            "context_chunks": [],
        }

    answer = generate_answer(query, context_chunks)
    log(f"  Answer generated ({len(answer)} chars)")

    # ------------------------------------------------------------------
    # Step 4: Self-critique and revision
    # ------------------------------------------------------------------
    log("Running self-critique...")
    final_answer, was_revised, critique_notes = critique_and_revise(
        query, answer, context_chunks
    )

    if was_revised:
        log(f"  Answer revised — issues: {critique_notes}")
    else:
        log(f"  Answer passed critique")

    return {
        "answer": final_answer,
        "tool_used": tool_name,
        "tool_reason": tool_reason,
        "retrieval_attempts": attempts,
        "was_revised": was_revised,
        "critique_notes": critique_notes,
        "context_chunks": context_chunks,
    }
