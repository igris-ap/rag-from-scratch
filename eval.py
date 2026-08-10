"""
eval.py — Retrieval quality evaluation for the RAG system.

Measures two things without needing labelled ground-truth answers:

1. RETRIEVAL COVERAGE
   For each question, did the retriever find any chunks at all?
   And how many unique sources were returned?

2. ANSWER FAITHFULNESS (heuristic)
   Does the LLM answer actually reference the retrieved context,
   or does it fall back to "I don't have enough information"?

Why no RAGAS / BERTScore?
Those require either ground-truth answers or an external grading model.
This eval runs entirely with what you already have: Postgres + Ollama.
It gives you a real signal about whether the retrieval pipeline is working.

Usage:
    python3 eval.py                          # runs default eval_questions.json
    python3 eval.py --questions my_qs.json   # custom question file
    python3 eval.py --verbose                # also prints each answer

Output:
    eval_results.json   — full per-question results
    Console summary     — pass/fail counts and retrieval stats
"""

import json
import time
import argparse
from pathlib import Path

from agent import run_agent, NO_CONTEXT_ANSWER

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS_FILE = "eval_questions.json"
OUTPUT_FILE = "eval_results.json"

# A question "passes" retrieval if at least this many chunks are found
MIN_CHUNKS_FOR_PASS = 1

# An answer that merely *mentions* the fallback sentence is not a fallback.
# Below this length, a full-text match is still treated as one (covers the
# LLM prefixing the sentinel with something like "Answer: ").
SHORT_ANSWER_CHARS = 200

# Match on the sentinel's opening clause rather than the whole sentence, so
# a decline still registers if the LLM truncates or lightly rewords the tail
# ("...about this" / "...about that topic"). Derived from the constant so it
# tracks any future edit to NO_CONTEXT_ANSWER.
_SENTINEL_PREFIX = " ".join(NO_CONTEXT_ANSWER.lower().split()[:4])


def is_fallback_answer(answer: str) -> bool:
    """
    Did the agent decline to answer, rather than answer the question?

    Detects the sentinel that agent.py emits (NO_CONTEXT_ANSWER) when no
    usable context was retrieved.

    Why not a plain substring check?
      The previous version tested `"don't have information" in answer`.
      That misfires on any answer that legitimately *quotes* the fallback
      while explaining it. Concretely: "How does the system avoid
      hallucination?" produces a fully correct, well-grounded answer that
      describes the self-critique step "...replaced with an honest
      'I don't have information about this' response" — and the substring
      check scored that correct answer as a failure.

      Anchoring on the *start* of the answer fixes it: a real fallback
      opens with the sentinel, whereas a descriptive mention is buried
      mid-text.
    """
    normalized = answer.strip().lstrip("\"'").lower()

    if normalized.startswith(_SENTINEL_PREFIX):
        return True

    # Very short answers containing the sentinel anywhere are declines too —
    # nothing substantive can be wrapped around it at that length.
    return len(answer) < SHORT_ANSWER_CHARS and _SENTINEL_PREFIX in normalized

# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def evaluate_question(question: str, verbose: bool = False) -> dict:
    """
    Run a single question through the agentic RAG pipeline and collect metrics.

    NOTE: This calls agent.run_agent() directly, ONCE, rather than calling
    rag.retrieve() and rag.answer() separately. The previous version called
    both independently — retrieve() always ran a direct hybrid_search, while
    answer() routed through the full agent loop where select_tool() could
    pick a different tool entirely. That meant the "sources" reported here
    could describe a completely different retrieval than the one that
    actually produced the printed answer. Calling run_agent() once and
    reading its context_chunks guarantees the sources shown are exactly
    what generated the answer.

    Trade-off: this bypasses rag.answer()'s query-analysis layer (clarity
    check, multi-question splitting/synthesis) and evaluates each question
    as a single agent run. Fine for this question set, since none of them
    are multi-part — but if you add a multi-part question to
    eval_questions.json, this won't exercise the splitting/synthesis path.

    Returns a dict with:
        question        — the input question
        chunks_retrieved — number of parent chunks used for generation
        sources         — list of unique source filenames actually used
        tool_used       — which tool the agent selected
        retrieval_attempts — how many retrieval attempts were made
        retrieval_pass  — True if >= MIN_CHUNKS_FOR_PASS chunks found
        answer_has_info — True if the answer is not a fallback "I don't have"
        answer          — the full LLM response
        latency_s       — end-to-end time in seconds
    """
    start = time.time()

    agent_result = run_agent(question, verbose=False)

    chunks = agent_result["context_chunks"]
    sources = list({c.get("source", "unknown") for c in chunks})
    llm_answer = agent_result["answer"]

    elapsed = round(time.time() - start, 2)

    retrieval_pass = len(chunks) >= MIN_CHUNKS_FOR_PASS
    answer_has_info = not is_fallback_answer(llm_answer)

    result = {
        "question": question,
        "chunks_retrieved": len(chunks),
        "sources": sources,
        "tool_used": agent_result["tool_used"],
        "retrieval_attempts": agent_result["retrieval_attempts"],
        "retrieval_pass": retrieval_pass,
        "answer_has_info": answer_has_info,
        "answer": llm_answer,
        "latency_s": elapsed,
    }

    if verbose:
        status = "PASS" if retrieval_pass else "FAIL"
        print(f"\n[{status}] {question}")
        print(f"  Tool: {agent_result['tool_used']} ({agent_result['retrieval_attempts']} attempt(s))")
        print(f"  Chunks: {len(chunks)} | Sources: {sources} | {elapsed}s")
        print(f"  Answer: {llm_answer[:200]}{'...' if len(llm_answer) > 200 else ''}")

    return result


def run_eval(questions_file: str, verbose: bool = False):
    """
    Run the full evaluation suite and print a summary.
    """
    questions_path = Path(questions_file)
    if not questions_path.exists():
        print(f"Error: questions file not found: {questions_file}")
        print("Create one with a list of question strings, or use eval_questions.json")
        return

    with open(questions_path) as f:
        data = json.load(f)

    # Accept either a plain list or {"questions": [...]}
    if isinstance(data, list):
        questions = data
    else:
        questions = data.get("questions", [])

    if not questions:
        print("No questions found in the file.")
        return

    print(f"\n=== RAG Eval — {len(questions)} questions ===\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q[:80]}...", end="\r", flush=True)
        result = evaluate_question(q, verbose=verbose)
        results.append(result)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total = len(results)
    retrieval_passes = sum(1 for r in results if r["retrieval_pass"])
    answer_passes = sum(1 for r in results if r["answer_has_info"])
    avg_chunks = round(sum(r["chunks_retrieved"] for r in results) / total, 2)
    avg_latency = round(sum(r["latency_s"] for r in results) / total, 2)

    print("\n" + "=" * 50)
    print(f"RETRIEVAL COVERAGE : {retrieval_passes}/{total} questions returned >= {MIN_CHUNKS_FOR_PASS} chunk(s)")
    print(f"ANSWER INFORMATIVENESS : {answer_passes}/{total} answers had content (not fallback)")
    print(f"AVG CHUNKS PER QUERY   : {avg_chunks}")
    print(f"AVG LATENCY            : {avg_latency}s")
    print("=" * 50)

    # Questions where retrieval failed
    failed = [r for r in results if not r["retrieval_pass"]]
    if failed:
        print(f"\nRetrieval failures ({len(failed)}):")
        for r in failed:
            print(f"  - {r['question']}")

    # Save full results
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"summary": {
            "total": total,
            "retrieval_pass_rate": f"{retrieval_passes}/{total}",
            "answer_informativeness_rate": f"{answer_passes}/{total}",
            "avg_chunks_retrieved": avg_chunks,
            "avg_latency_s": avg_latency,
        }, "results": results}, f, indent=2)

    print(f"\nFull results saved to: {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS_FILE,
        help="Path to JSON file with evaluation questions (default: eval_questions.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each question's answer and retrieved sources",
    )
    args = parser.parse_args()

    run_eval(args.questions, verbose=args.verbose)
