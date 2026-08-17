"""Token-usage accounting and cost computation for live agents.

Kept separate from the agents themselves so the cost model is a pure, testable function of a
usage record and an endpoint's prices. The stub backend does not use this (its cost is sampled
from config); this is the real-backend path exercised in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

from specialist_router.config import EndpointConfig


@dataclass(slots=True)
class Usage:
    """Accumulated token usage across an episode's model calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Accumulate one model call's token counts."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens


def endpoint_cost(usage: Usage, endpoint: EndpointConfig) -> float:
    """Compute USD cost from token usage and an endpoint's per-1k-token prices."""
    return (
        usage.prompt_tokens / 1000.0 * endpoint.price_prompt_per_1k_usd
        + usage.completion_tokens / 1000.0 * endpoint.price_completion_per_1k_usd
    )
