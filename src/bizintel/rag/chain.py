"""
RAG Chain — orchestrates retrieval + LLM call to produce AI analysis.

Flow:
  1. User query → Retriever → top-K documents
  2. Documents + query → prompt template → final prompt
  3. Final prompt → LLM (OpenAI / Groq) → structured response
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (if it exists)
_env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_path)

from bizintel.config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    DEFAULT_ANALYSIS_TYPE,
    TOP_K,
    GUARDRAILS_ENABLED,
    CONFIDENCE_THRESHOLD_SOFT,
    CONFIDENCE_THRESHOLD_HARD,
)
from bizintel.config.llm_client import get_llm_client
from bizintel.rag.retriever import StartupRetriever, RetrievalResult
from bizintel.rag.prompt_templates import get_prompt
from bizintel.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


class BizIntelChain:
    """
    End-to-end RAG chain: query → retrieve → guardrail → prompt → LLM → answer.

    Uses dependency injection — receives a retriever, not raw embedder/store.
    """

    # ── Guardrail messages ───────────────────────────────────────────

    _REFUSAL_MSG = (
        "🚫 **I don't have enough information to answer this reliably.**\n\n"
        "The retrieved context doesn't appear relevant to your query. "
        "This can happen when:\n"
        "- The topic is outside the startup database's coverage\n"
        "- The query is too vague or ambiguous\n"
        "- No matching startups exist in our 134K dataset\n\n"
        "*Try rephrasing your query or broadening your search terms.*"
    )

    _LOW_CONFIDENCE_DISCLAIMER = (
        "⚠️ **Low confidence** — the retrieved context may not fully match "
        "your query. Results below should be reviewed critically.\n\n---\n\n"
    )

    def __init__(
        self,
        retriever: StartupRetriever,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> None:
        self._retriever = retriever
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Initialise LLM client (auto-selects OpenAI or Groq based on settings)
        self._client = get_llm_client()

    # ── Query expansion ──────────────────────────────────────────────

    _EXPAND_PROMPT = (
        "You are a search query optimizer for a startup database. "
        "The user will give a query that may mention a company name or be vague. "
        "Rewrite it into a rich, descriptive search query that captures the "
        "BUSINESS DOMAIN, INDUSTRY, KEY PRODUCTS, and TECHNOLOGY of what the "
        "user is looking for. Do NOT include instructions like 'find' or 'search'. "
        "Output ONLY the rewritten query, nothing else.\n\n"
        "Examples:\n"
        "  User: Find startups similar to Stripe\n"
        "  Rewritten: online payment processing fintech developer API billing "
        "infrastructure internet commerce\n\n"
        "  User: competitors of Airbnb\n"
        "  Rewritten: short-term rental marketplace vacation home booking "
        "hospitality travel accommodation platform\n\n"
        "  User: AI healthcare startups in India\n"
        "  Rewritten: artificial intelligence healthcare medical technology "
        "health-tech diagnosis India\n\n"
        "User: {query}\nRewritten:"
    )

    def _expand_query(self, query: str) -> str:
        """
        Use the LLM to expand a vague or name-based query into a rich
        semantic description that the embedding model can match on.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": self._EXPAND_PROMPT.format(query=query)},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            expanded = response.choices[0].message.content.strip()
            logger.info("Query expanded: '%s' → '%s'", query[:60], expanded[:80])
            return expanded
        except Exception as e:
            logger.warning("Query expansion failed (%s), using original query", e)
            return query

    def analyze(
        self,
        query: str,
        analysis_type: str = DEFAULT_ANALYSIS_TYPE,
        top_k: int = TOP_K,
        where: dict | None = None,
    ) -> dict:
        """
        Run the full RAG pipeline: retrieve → prompt → LLM → response.

        Parameters
        ----------
        query : str
            User's natural language question.
        analysis_type : str
            "auto", "similar", "swot", "competitor", "comparison", "ecosystem".
        top_k : int
            Number of documents to retrieve.
        where : dict | None
            Optional metadata filter (e.g. {"source": "YC"}).

        Returns
        -------
        dict
            {
                "answer": str,          # LLM-generated analysis
                "sources": list[dict],  # retrieved documents used
                "analysis_type": str,
                "model": str,
                "confidence": str,      # "high" | "low" | "none"
                "best_score": float,    # top reranker score
                "mean_score": float,    # mean reranker score
            }
        """
        # ── Step 1: Expand + Retrieve ────────────────────────────────
        logger.info(
            "Analyzing: '%s' [type=%s, top_k=%d]",
            query[:80], analysis_type, top_k,
        )

        # Expand vague/name-based queries into rich semantic descriptions
        search_query = self._expand_query(query)

        retrieval: RetrievalResult = self._retriever.retrieve(
            query=search_query, top_k=top_k, where=where,
        )

        results = retrieval.documents
        confidence = retrieval.confidence
        best_score = retrieval.best_score
        mean_score = retrieval.mean_score

        # ── Step 2: Guardrail check (only when GUARDRAILS_ENABLED) ────
        if not results:
            return {
                "answer": "No relevant startups found for your query. Try broadening your search.",
                "sources": [],
                "analysis_type": analysis_type,
                "model": self._model,
                "confidence": "none",
                "best_score": 0.0,
                "mean_score": 0.0,
            }

        if GUARDRAILS_ENABLED and confidence == "none":
            logger.warning(
                "GUARDRAIL: refusing answer — best_score=%.3f < hard=%.3f",
                best_score, CONFIDENCE_THRESHOLD_HARD,
            )
            # Package sources even for refusals so the user can inspect them
            sources = [
                {
                    "doc_id": r.doc_id,
                    "text": r.text,
                    "metadata": r.metadata,
                    "distance": r.distance,
                }
                for r in results
            ]
            return {
                "answer": self._REFUSAL_MSG,
                "sources": sources,
                "analysis_type": analysis_type,
                "model": self._model,
                "confidence": confidence,
                "best_score": best_score,
                "mean_score": mean_score,
            }

        # ── Step 3: Build prompt ─────────────────────────────────────
        documents_text = "\n\n---\n\n".join(r.text for r in results)
        prompt = get_prompt(
            analysis_type=analysis_type,
            query=query,
            documents=documents_text,
        )

        # ── Step 4: Call LLM ─────────────────────────────────────────
        logger.info("Calling %s (temp=%.1f)", self._model, self._temperature)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        answer = response.choices[0].message.content

        # Prepend low-confidence disclaimer if needed
        if GUARDRAILS_ENABLED and confidence == "low":
            answer = self._LOW_CONFIDENCE_DISCLAIMER + answer

        logger.info(
            "LLM response: %d chars, %d tokens used",
            len(answer),
            response.usage.total_tokens if response.usage else 0,
        )

        # ── Step 5: Package response ─────────────────────────────────
        sources = [
            {
                "doc_id": r.doc_id,
                "text": r.text,
                "metadata": r.metadata,
                "distance": r.distance,
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "analysis_type": analysis_type,
            "model": self._model,
            "confidence": confidence,
            "best_score": best_score,
            "mean_score": mean_score,
        }
