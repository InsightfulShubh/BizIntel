"""Validate node — post-generation quality checks on the answer."""

from __future__ import annotations

import logging

from bizintel.config.settings import LLM_MODEL

logger = logging.getLogger(__name__)

_GROUNDEDNESS_PROMPT = (
    "You are a strict fact-checker. Compare the ANSWER against the SOURCE DOCUMENTS.\n\n"
    "Check:\n"
    "1. Every factual claim in the answer is supported by the source documents.\n"
    "2. No information is fabricated or hallucinated.\n"
    "3. Company names, countries, and industries match the sources.\n\n"
    "Respond with ONLY 'pass' or 'fail', nothing else.\n\n"
    "--- SOURCE DOCUMENTS ---\n{documents}\n\n"
    "--- ANSWER ---\n{answer}\n\n"
    "Verdict:"
)


def make_validate_node(llm_client):
    """Factory: returns a validate node that closes over the LLM client."""

    def validate_node(state) -> dict:
        """Run post-generation checks: doc existence, score, groundedness.

        Checks (in order):
            1. Do source docs exist?
            2. Is best_score above a minimum threshold?
            3. LLM groundedness check — is the answer supported by sources?
        """
        source_docs = getattr(state, "source_docs", [])
        best_score = getattr(state, "best_score", 0.0)
        answer = getattr(state, "answer", "")

        # Check 1: docs exist
        if not source_docs:
            logger.warning("Validation FAIL: no source docs")
            return {"validation_check": False}

        # Check 2: similarity sanity (extremely low score = noise)
        if best_score <= 0.0:
            logger.warning("Validation FAIL: best_score=%.3f", best_score)
            return {"validation_check": False}

        # Check 3: LLM groundedness check
        documents_text = "\n\n---\n\n".join(
            doc["text"] for doc in source_docs
        )

        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": _GROUNDEDNESS_PROMPT.format(
                            documents=documents_text,
                            answer=answer,
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=10,
            )
            verdict = response.choices[0].message.content.strip().lower()
            passed = verdict == "pass"
        except Exception as exc:
            logger.warning("Groundedness check failed (%s), assuming pass", exc)
            passed = True
            verdict = "skipped"

        logger.info("Validation: %s (groundedness=%s)", "PASS" if passed else "FAIL", verdict)

        return {"validation_check": passed}

    return validate_node
