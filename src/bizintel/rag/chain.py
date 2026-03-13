"""
RAG Chain — orchestrates retrieval + LLM call to produce AI analysis.

Flow:
  1. User query → Retriever → top-K documents
  2. Documents + query → prompt template → final prompt
  3. Final prompt → OpenAI LLM → structured response
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root (if it exists)
_env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_path)

from bizintel.config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    DEFAULT_ANALYSIS_TYPE,
    TOP_K,
)
from bizintel.rag.retriever import StartupRetriever
from bizintel.rag.prompt_templates import get_prompt
from bizintel.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


class BizIntelChain:
    """
    End-to-end RAG chain: query → retrieve → prompt → LLM → answer.

    Uses dependency injection — receives a retriever, not raw embedder/store.
    """

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

        # Initialise OpenAI client (reads OPENAI_API_KEY from env)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set — LLM calls will fail. "
                "Set it via: $env:OPENAI_API_KEY = 'sk-...'"
            )
        self._client = OpenAI(api_key=api_key)

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
            }
        """
        # ── Step 1: Expand + Retrieve ────────────────────────────────
        logger.info(
            "Analyzing: '%s' [type=%s, top_k=%d]",
            query[:80], analysis_type, top_k,
        )

        # Expand vague/name-based queries into rich semantic descriptions
        search_query = self._expand_query(query)

        results: list[SearchResult] = self._retriever.retrieve(
            query=search_query, top_k=top_k, where=where,
        )

        if not results:
            return {
                "answer": "No relevant startups found for your query. Try broadening your search.",
                "sources": [],
                "analysis_type": analysis_type,
                "model": self._model,
            }

        # ── Step 2: Build prompt ─────────────────────────────────────
        documents_text = "\n\n---\n\n".join(r.text for r in results)
        prompt = get_prompt(
            analysis_type=analysis_type,
            query=query,
            documents=documents_text,
        )

        # ── Step 3: Call LLM ─────────────────────────────────────────
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

        logger.info(
            "LLM response: %d chars, %d tokens used",
            len(answer),
            response.usage.total_tokens if response.usage else 0,
        )

        # ── Step 4: Package response ─────────────────────────────────
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
        }
