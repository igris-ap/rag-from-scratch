"""
Tests for BM25 keyword search and the agent's retrieval resilience.

These run against the committed fixture document
(docs/rag_system_reference.pdf and its parent chunks in parent_store/),
so they work on a fresh clone without needing Postgres or Ollama —
keyword_search reads parent_store/*.json straight off disk.

Run from the repository root:
    python3 -m unittest discover tests
"""

import unittest
from unittest.mock import patch

from agent import run_agent
from tools import _tokenize, keyword_search

FIXTURE_SOURCE = "rag_system_reference.pdf"


class TokenizationTests(unittest.TestCase):
    """
    The corpus is Markdown converted from PDF, so terms arrive wrapped in
    emphasis and trailing punctuation. Documents and queries must tokenize
    identically or BM25 matching silently fails.
    """

    def test_markdown_emphasis_is_stripped_from_terms(self):
        self.assertIn("rag", _tokenize("**RAG** is a retrieval technique."))
        self.assertIn("turns", _tokenize("## **How History Is Handled Across Turns**"))

    def test_trailing_question_mark_does_not_change_the_term(self):
        # Regression: `.lower().split()` produced the term "generation?",
        # which matched nothing in the index.
        self.assertEqual(
            _tokenize("What is retrieval augmented generation?"),
            ["what", "is", "retrieval", "augmented", "generation"],
        )

    def test_query_and_document_forms_produce_the_same_term(self):
        from_doc = _tokenize("Retrieval Augmented Generation, or RAG, is a technique.")
        from_query = _tokenize("what is rag?")
        self.assertIn("rag", from_doc)
        self.assertIn("rag", from_query)

    def test_preserves_hyphenated_technical_terms(self):
        self.assertEqual(_tokenize("parent-child chunking"), ["parent-child", "chunking"])
        self.assertEqual(_tokenize("all-MiniLM-L6-v2"), ["all-minilm-l6-v2"])

    def test_ignores_punctuation_only_input(self):
        self.assertEqual(_tokenize("—  ***  ??"), [])


class KeywordSearchTests(unittest.TestCase):
    """End-to-end BM25 search over the committed fixture document."""

    def test_question_retrieves_the_fixture_document(self):
        results = keyword_search("What is retrieval augmented generation?")
        self.assertTrue(results, "expected at least one BM25 hit")
        self.assertEqual(results[0]["source"], FIXTURE_SOURCE)

    def test_punctuation_does_not_change_the_top_result(self):
        with_mark = keyword_search("What is parent-child chunking?")
        without = keyword_search("what is parent child chunking")
        self.assertTrue(with_mark)
        self.assertTrue(without)
        self.assertEqual(with_mark[0]["parent_id"], without[0]["parent_id"])

    def test_returns_empty_for_terms_absent_from_the_corpus(self):
        self.assertEqual(keyword_search("zzzqqq nonexistentterm"), [])


class AgentRetrievalResilienceTests(unittest.TestCase):
    """
    A retrieval tool can fail for reasons unrelated to the question — if
    Postgres is down, vector_search and hybrid_search raise while
    keyword_search (which reads from disk) still works. The agent should
    fall through to the next tool rather than failing the whole answer.
    """

    def test_agent_retries_next_tool_after_the_first_one_raises(self):
        keyword_results = keyword_search("What is retrieval augmented generation?")
        self.assertTrue(keyword_results, "fixture must be indexed on disk")

        with (
            patch("agent.select_tool", return_value=("vector_search", "test")),
            patch(
                "agent.call_tool",
                side_effect=[RuntimeError("database down"), keyword_results],
            ),
            patch("agent.rerank", side_effect=lambda _q, results, top_k: results[:top_k]),
            patch(
                "agent.reflect_on_retrieval",
                side_effect=lambda _q, chunks: (bool(chunks), "test", ""),
            ),
            patch("agent.generate_answer", return_value="an answer"),
            patch("agent.critique_and_revise", return_value=("an answer", False, "")),
        ):
            result = run_agent("What is retrieval augmented generation?")

        self.assertEqual(result["retrieval_attempts"], 2)
        self.assertEqual(result["context_chunks"][0]["source"], FIXTURE_SOURCE)
        self.assertEqual(result["answer"], "an answer")

    def test_agent_returns_fallback_when_every_tool_fails(self):
        from agent import NO_CONTEXT_ANSWER

        with (
            patch("agent.select_tool", return_value=("vector_search", "test")),
            patch("agent.call_tool", side_effect=RuntimeError("everything is down")),
            patch(
                "agent.reflect_on_retrieval",
                side_effect=lambda _q, chunks: (bool(chunks), "test", ""),
            ),
        ):
            result = run_agent("What is retrieval augmented generation?")

        self.assertEqual(result["answer"], NO_CONTEXT_ANSWER)
        self.assertEqual(result["context_chunks"], [])


if __name__ == "__main__":
    unittest.main()
