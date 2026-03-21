"""
RAG Evaluator — LLM-as-Judge scoring for BizIntel.

Computes 6 metrics per query:
  1. Context Relevancy  — Are retrieved docs relevant to the query? (0–1)
  2. Groundedness        — Is every claim in the answer backed by sources? (0–1)
  3. Answer Relevancy    — Does the answer address the user's question? (0–1)
  4. Precision@K         — % of retrieved docs matching expected domains (0–1)
  5. Structure Score     — Does the answer contain expected sections? (0–1)
  6. Bad Result Check    — Did any known-bad results appear? (1 = clean, 0 = contaminated)

Also captures:
  - latency_seconds     — end-to-end time for the full RAG pipeline
  - total_tokens        — total token usage from the OpenAI response
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from dotenv import load_dotenv

# .env lives at project root — 4 levels up from src/bizintel/evaluation/evaluator.py
_env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_path)

from bizintel.config.llm_client import get_llm_client  # noqa: E402
from bizintel.config.settings import LLM_MODEL          # noqa: E402

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Scores a single RAG result using LLM-as-Judge + deterministic checks."""

    def __init__(self, model: str | None = None) -> None:
        self._client = get_llm_client()
        self._model = model or LLM_MODEL

    # ── LLM Judge helper ─────────────────────────────────────────────

    def _judge(self, prompt: str) -> float:
        """
        Send a scoring prompt to the LLM and extract a float score 0–1.
        Falls back to 0.0 if parsing fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            text = response.choices[0].message.content.strip()
            # Extract the first decimal number from the response
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)  # clamp 0–1
            return 0.0
        except Exception as e:
            logger.warning("Judge call failed: %s", e)
            return 0.0

    # ── Metric 1: Context Relevancy (LLM-as-Judge) ──────────────────

    def score_context_relevancy(
        self, query: str, retrieved_docs: list[str],
    ) -> float:
        """
        Are the retrieved documents relevant to the user's query?
        Score 0–1 (1 = all docs are highly relevant).
        """
        docs_text = "\n---\n".join(retrieved_docs[:5])
        prompt = (
            "You are an impartial relevance judge. Given a user query and a set of "
            "retrieved startup documents, rate how relevant the documents are to the "
            "query on a scale of 0.0 to 1.0.\n\n"
            "- 1.0 = All documents are directly relevant to the query topic\n"
            "- 0.5 = Some documents are relevant, some are off-topic\n"
            "- 0.0 = None of the documents are relevant\n\n"
            "Output ONLY a single number between 0.0 and 1.0, nothing else.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved Documents:\n{docs_text}\n\n"
            "Score:"
        )
        return self._judge(prompt)

    # ── Metric 2: Groundedness / Faithfulness (LLM-as-Judge) ────────

    def score_groundedness(
        self, answer: str, retrieved_docs: list[str],
    ) -> float:
        """
        Is every claim in the answer supported by the retrieved documents?
        Score 0–1 (1 = fully grounded, no hallucination).
        """
        docs_text = "\n---\n".join(retrieved_docs[:5])
        prompt = (
            "You are a faithfulness judge. Given an AI-generated answer and the source "
            "documents it was based on, rate how well the answer is grounded in the "
            "sources on a scale of 0.0 to 1.0.\n\n"
            "- 1.0 = Every claim in the answer is supported by the source documents\n"
            "- 0.5 = Some claims are supported, some are not found in sources\n"
            "- 0.0 = The answer is mostly fabricated / not supported by sources\n\n"
            "Output ONLY a single number between 0.0 and 1.0, nothing else.\n\n"
            f"Source Documents:\n{docs_text}\n\n"
            f"AI Answer:\n{answer}\n\n"
            "Score:"
        )
        return self._judge(prompt)

    # ── Metric 3: Answer Relevancy (LLM-as-Judge) ───────────────────

    def score_answer_relevancy(
        self, query: str, answer: str, analysis_type: str,
    ) -> float:
        """
        Does the answer actually address the user's question?
        Score 0–1 (1 = perfectly addresses the query in the right format).
        """
        prompt = (
            "You are an answer quality judge. Given a user query, the requested "
            f"analysis type ('{analysis_type}'), and the AI-generated answer, rate "
            "how well the answer addresses the query on a scale of 0.0 to 1.0.\n\n"
            "- 1.0 = Answer directly addresses the query in the correct format\n"
            "- 0.5 = Answer is partially relevant or in the wrong format\n"
            "- 0.0 = Answer does not address the query at all\n\n"
            "Output ONLY a single number between 0.0 and 1.0, nothing else.\n\n"
            f"Query: {query}\n"
            f"Requested Format: {analysis_type}\n\n"
            f"AI Answer:\n{answer}\n\n"
            "Score:"
        )
        return self._judge(prompt)

    # ── Metric 4: Precision@K (deterministic) ───────────────────────

    @staticmethod
    def score_precision_at_k(
        retrieved_docs: list[str],
        expected_domains: list[str],
    ) -> float:
        """
        What fraction of retrieved docs contain at least one expected domain keyword?
        Score 0–1 (1 = all docs match expected domain).
        """
        if not retrieved_docs or not expected_domains:
            return 0.0

        hits = 0
        for doc in retrieved_docs:
            doc_lower = doc.lower()
            if any(domain.lower() in doc_lower for domain in expected_domains):
                hits += 1

        return hits / len(retrieved_docs)

    # ── Metric 5: Structure Score (deterministic) ───────────────────

    @staticmethod
    def score_structure(
        answer: str,
        expected_sections: list[str],
    ) -> float:
        """
        Does the answer contain the expected section headings?
        Score 0–1 (1 = all expected sections present).
        Returns 1.0 if no sections are expected (e.g., 'similar' or 'auto').
        """
        if not expected_sections:
            return 1.0  # no structural requirement

        answer_lower = answer.lower()
        hits = sum(
            1 for section in expected_sections
            if section.lower() in answer_lower
        )
        return hits / len(expected_sections)

    # ── Metric 6: Bad Result Check (deterministic) ──────────────────

    @staticmethod
    def score_bad_result_check(
        retrieved_docs: list[str],
        bad_results: list[str],
    ) -> float:
        """
        Did any known-bad results appear in retrieved docs?
        Score 1.0 = clean (no bad results), 0.0 = contaminated.
        """
        if not bad_results:
            return 1.0  # no known bad results to check

        combined = " ".join(retrieved_docs).lower()
        for bad in bad_results:
            if bad.lower() in combined:
                return 0.0  # found a bad result
        return 1.0

    # ── Aggregate scorer ─────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        analysis_type: str,
        answer: str,
        retrieved_docs: list[str],
        expected_domains: list[str],
        expected_sections: list[str],
        bad_results: list[str],
        latency: float,
        total_tokens: int,
    ) -> dict:
        """
        Run all 6 metrics on a single query result.

        Returns a dict with all scores + metadata.
        """
        context_relevancy = self.score_context_relevancy(query, retrieved_docs)
        groundedness = self.score_groundedness(answer, retrieved_docs)
        answer_relevancy = self.score_answer_relevancy(query, answer, analysis_type)

        precision_at_k = self.score_precision_at_k(retrieved_docs, expected_domains)
        structure = self.score_structure(answer, expected_sections)
        bad_check = self.score_bad_result_check(retrieved_docs, bad_results)

        return {
            "context_relevancy": round(context_relevancy, 3),
            "groundedness": round(groundedness, 3),
            "answer_relevancy": round(answer_relevancy, 3),
            "precision_at_k": round(precision_at_k, 3),
            "structure_score": round(structure, 3),
            "bad_result_check": round(bad_check, 3),
            "latency_seconds": round(latency, 2),
            "total_tokens": total_tokens,
        }
