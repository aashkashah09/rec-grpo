"""The local-specialist arm: a chat agent over a vLLM OpenAI-compatible endpoint.

Identical wiring to :mod:`specialist_router.agents.api_agent` but pointed at the locally-served
small model (base model in Phase 2, the GRPO specialist checkpoint from Phase 4 onward). Used on
the ``real`` serving backend; the CPU/CI path uses the stub simulator.
"""

from __future__ import annotations

from specialist_router.agents.chat_agent import ChatToolAgent
from specialist_router.config import EndpointConfig
from specialist_router.serving.clients import ChatClient, ChatResponse


def build_local_agent(endpoint: EndpointConfig) -> ChatToolAgent:
    """Build a fresh local-specialist agent bound to ``endpoint`` (one instance per episode)."""
    client = ChatClient(endpoint)

    def complete(messages: list[dict[str, str]], max_tokens: int) -> ChatResponse:
        return client.complete(messages, max_tokens)

    return ChatToolAgent(name="local", complete_fn=complete)
