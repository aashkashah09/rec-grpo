# 004 — Modules added beyond the PROJECT_PLAN layout

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 1 (Environment, tasks, verifier)

## Context

`CONVENTIONS.md` requires an ADR for anything not in `PROJECT_PLAN.md` §2. Phase 1 adds three small
modules that the conventions imply but the layout does not list explicitly.

## Decisions

1. **`src/specialist_router/config.py`** — the single typed config module `CONVENTIONS.md` mandates
   ("everything in `configs/*.yaml`, loaded through one typed config module"). Pydantic models
   with `extra="forbid"` so config typos fail loudly; `load_config(path, seed_override)` is the
   one entry point, honouring the `--config`/`--seed` contract.
2. **`src/specialist_router/env/records.py`** — the versioned JSONL schema module `CONVENTIONS.md`
   requires ("JSONL logs must have a versioned schema module"). Holds `SCHEMA_VERSION` and the
   `Task`, `Trajectory`, `ToolCall`, `Verdict` pydantic models and the `AnswerType` enum.
3. **`src/specialist_router/env/reference_agents.py`** + the `Agent` protocol in
   `agents/base.py` — the scripted `OracleAgent`/`TrivialAgent` are required by the Phase-1
   done-condition (oracle 100%, trivial ~0%). They are deterministic and contain no LLM; the
   real `local_agent.py`/`api_agent.py` remain Phase 2. `agents/base.py` (the protocol) is
   listed in the layout and is needed now for the episode loop.

Also: **`configs/env.mini.yaml`** — a small profile (same schema as `env.yaml`) for CI, tests,
and fast local runs. **`numpy`** added as the one new runtime dependency (seeded generation);
heavy ML deps remain deferred (ADR-000).

## Consequences

- The additions are small, conventions-driven, and keep experiment constants out of code.
- `EnvIndex` (in `env/tasks.py`) precomputes lookups so pure-Python ground truth is O(n) once
  rather than per task; it is an internal helper, not a new public subsystem.
