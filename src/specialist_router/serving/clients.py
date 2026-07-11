"""A thin OpenAI-compatible chat client used by the live agents (Phase 4; not used in CI).

Both arms — the vLLM-served local specialist and the frontier API — speak the OpenAI
``/chat/completions`` contract, so a single client parameterised by an
:class:`~specialist_router.config.EndpointConfig` serves both. ``httpx`` is imported lazily so
importing this module never requires the optional ``serving`` extra to be installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from specialist_router.config import EndpointConfig


class ChatClientError(Exception):
    """Raised when a chat completion request fails or returns an unexpected shape."""


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """The text and token usage of one chat completion."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class ChatClient:
    """Minimal blocking chat client for an OpenAI-compatible endpoint."""

    def __init__(self, endpoint: EndpointConfig) -> None:
        """Bind the client to an endpoint (resolving its API key from the environment)."""
        self._endpoint = endpoint
        self._api_key = (
            os.environ.get(endpoint.api_key_env) if endpoint.api_key_env is not None else None
        )

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 512) -> ChatResponse:
        """Request a single chat completion and return its text and usage.

        Raises:
            ChatClientError: If the request fails or the response is malformed.
        """
        import httpx  # lazy: optional 'serving' extra

        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._endpoint.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        try:
            response = httpx.post(
                f"{self._endpoint.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._endpoint.timeout_s,
            )
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ChatClientError(f"chat completion failed: {exc}") from exc
        return ChatResponse(
            text=str(text),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
