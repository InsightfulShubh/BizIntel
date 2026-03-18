"""
Unified LLM client factory for BizIntel.

Returns an OpenAI-compatible client configured for the selected provider.
Both OpenAI and Groq expose the same chat.completions.create() API,
so callers don't need to know which backend is active.

Usage:
    from bizintel.config.llm_client import get_llm_client
    client = get_llm_client()          # uses LLM_PROVIDER from settings
    client.chat.completions.create(...)
"""

from __future__ import annotations

import logging
import os

from openai import OpenAI

from bizintel.config.settings import (
    LLM_PROVIDER,
    OPENAI_BASE_URL,
    GROQ_BASE_URL,
)

logger = logging.getLogger(__name__)

# ── Environment variable names per provider ──────────────────────────────

_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}

_BASE_URLS: dict[str, str | None] = {
    "openai": OPENAI_BASE_URL,       # None → default OpenAI endpoint
    "groq": GROQ_BASE_URL,           # "https://api.groq.com/openai/v1"
}


def get_llm_client(provider: str | None = None) -> OpenAI:
    """
    Build and return an OpenAI-compatible client for the active LLM provider.

    Parameters
    ----------
    provider : str | None
        Override the provider ("openai" or "groq").
        Defaults to ``LLM_PROVIDER`` from settings.

    Returns
    -------
    OpenAI
        A ready-to-use client with the correct ``base_url`` and ``api_key``.

    Raises
    ------
    ValueError
        If the provider is not recognised.
    RuntimeError
        If the required API key env-var is not set.
    """
    provider = (provider or LLM_PROVIDER).lower()

    if provider not in _API_KEY_ENV:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Supported: {list(_API_KEY_ENV.keys())}"
        )

    env_var = _API_KEY_ENV[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(
            f"{env_var} not set — required for provider '{provider}'. "
            f"Set it via:  $env:{env_var} = 'your-key'"
        )

    base_url = _BASE_URLS[provider]

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    logger.info("LLM client: provider=%s, base_url=%s", provider, base_url or "default")

    return OpenAI(**kwargs)
