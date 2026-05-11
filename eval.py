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

from rag import retrieve, answer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS_FILE = "eval_questions.json"
OUTPUT_FILE = "eval_results.json"

# A question "passes" retrieval if at least this many chunks are found
MIN_CHUNKS_FOR_PASS = 1

# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def evaluate_question(question: str, verbose: bool = False) -> dict:
    """
    Run a single question through the RAG pipeline and collect metrics.

    Returns a dict with:
        question        — the input question
        chunks_retrieved — number of parent chunks returned
        sources         — list of unique source filenames
        retrieval_pass  — True if >= MIN_CHUNKS_FOR_PASS chunks found
        answer_has_info — True if the answer is not the fallback "I don't have"
        answer          — the full LLM response
        latency_s       — end-to-end time in seconds
    """
    start = time.time()

    # Step 1: retrieval only (to measure without LLM noise)
    chunks = retrieve(question)
    sources = list({c["source"] for c in chunks})

    # Step 2: full answer
    llm_answer = answer(question, verbose=False)

    elapsed = round(time.time() - start, 2)

    retrieval_pass = len(chunks) >= MIN_CHUNKS_FOR_PASS
    answer_has_info = "i don't have enough information" not in llm_answer.lower()

    result = {
        "question": question,
        "chunks_retrieved": len(chunks),
        "sources": sources,
        "retrieval_pass": retrieval_pass,
        "answer_has_info": answer_has_info,
        "answer": llm_answer,
        "latency_s": elapsed,
    }

    if verbose:
        status = "PASS" if retrieval_pass else "FAIL"
        print(f"\n[{status}] {question}")
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
