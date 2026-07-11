# 005 — Agents and the CPU stub backend

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 2 (Router + serving + logging + OPE)

## Context

Phase 2 needs the two routing arms — `api_agent` (frontier) and `local_agent` (vLLM-served small
model) — but must be fully buildable, testable, and reproducible **without a GPU or any API spend**
(`PROJECT_PLAN.md` §3: "or an API-served small model as CPU-friendly fallback"). CI may not call
external APIs or require a GPU (`CLAUDE.md`).

## Decisions

1. **A stub simulator is the CPU/CI backend, and the source of every Phase-2 number.**
   `SimulatedArmRunner` (`serving/service.py`) draws quality from a seeded Bernoulli whose success
   probability decays with the *true* difficulty (the simulator may see it; the router may not),
   then scores it **through the real verifier** by submitting the ground-truth or a wrong answer —
   so `quality` is a genuine verifier verdict, not a shortcut. Cost/latency are seeded clipped
   Gaussians. Draws are keyed by `(seed, task_id, arm)`, so traffic is reproducible and
   order-independent. Every artifact it produces is labeled stub/CPU-simulator; real-model numbers
   arrive in Phase 4.
2. **Real agents share one provider-agnostic implementation.** `ChatToolAgent`
   (`agents/chat_agent.py`) turns the episode `Observation` into chat messages, requests one JSON
   action, and parses it into the episode loop's `Action`. `api_agent`/`local_agent` are thin
   builders binding it to different OpenAI-compatible endpoints via `ChatClient`
   (`serving/clients.py`). `httpx` is imported lazily, so these modules import cleanly in CI
   without the `serving` extra.
3. **Two arm runners, one interface.** `SimulatedArmRunner` (no episode, no DB — scalable traffic)
   and `EpisodeArmRunner` (a live agent through the sandboxed episode loop, measuring wall-clock
   latency and token cost) both satisfy the `ArmRunner` protocol, so `RouterService` is backend-
   agnostic. The metered episode path is integration-tested on CPU with the Phase-1 `OracleAgent`.

## New modules (beyond `PROJECT_PLAN.md` §2)

`agents/chat_agent.py` (shared tool-calling agent), `agents/metering.py` (token→USD cost),
`serving/service.py` (the `RouterService` core + arm runners). `serving/clients.py` and
`serving/app.py` are in the planned layout. These are recorded here per `CLAUDE.md`.

## Consequences

- The full phase (traffic → OPE → replay) runs on CPU with zero external calls; real agents drop in
  behind the same `ArmRunner`/`Agent` interfaces in Phase 4 by flipping `serving.backend: real`.
- The stub's quality/cost/latency profiles live in `configs/serving.yaml`; the routing story is a
  property of those config numbers, made explicit and swappable rather than hidden in code.
