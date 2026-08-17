"""The frontier-API arm: a provider-agnostic chat agent over an OpenAI-compatible endpoint.

Thin wiring only — the behaviour lives in
:class:`~specialist_router.agents.chat_agent.ChatToolAgent` and
:class:`~specialist_router.serving.clients.ChatClient`. Used on the ``real`` serving backend
(Phase 4); the CPU/CI path uses the stub simulator instead.
"""

from __future__ import annotations

from specialist_router.agents.chat_agent import ChatToolAgent
from specialist_router.config import EndpointConfig
from specialist_router.serving.clients import ChatClient, ChatResponse


def build_api_agent(endpoint: EndpointConfig) -> ChatToolAgent:
    """Build a fresh frontier-API agent bound to ``endpoint`` (one instance per episode)."""
    client = ChatClient(endpoint)

    def complete(messages: list[dict[str, str]], max_tokens: int) -> ChatResponse:
        return client.complete(messages, max_tokens)

    return ChatToolAgent(name="api", complete_fn=complete)
