# 000 — Phase 0 scaffolding choices

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 0 (Scaffolding)

## Context

`PROJECT_PLAN.md` §3 Phase 0 requires standing up the repository skeleton, tooling, and CI
so later phases drop into a working, linted, typed, tested project. "Done when: CI green on
the empty skeleton; `make test lint` pass locally." Several implementation details were
under-specified; this ADR records the decisions taken so reviewers can see the reasoning.

## Decisions

1. **`uv` for dependency management, installed via the official installer.** Mandated by
   `CONVENTIONS.md`. `pyproject.toml` is the single source of deps (no `requirements.txt`); a
   `uv.lock` is committed for reproducible CI.

2. **Python floor 3.11.** The minimum `CONVENTIONS.md` allows, and the widest-compatible target
   for the GPU/cloud images used in Phase 3. ruff and mypy target `py311`.

3. **Heavy ML deps are deferred, not installed in Phase 0.** `torch`, `trl`, `peft`,
   `transformers`, `vllm`, `wandb` (Phase 3) and `sentence-transformers` / `scikit-learn`
   / `fastapi` (Phase 2) are declared as optional extras (`training`, `serving`) but left
   commented until the phase that needs them. This keeps CPU CI fast and honest — CI never
   pretends to exercise training. *Consequence:* the extras must be fleshed out (and pinned)
   at the start of Phases 2 and 3.

4. **Empty-but-documented package skeleton.** Every sub-package ships an `__init__.py` with a
   Google-style docstring naming its future responsibility and phase — but **no stub
   functions or classes**. This gives `mypy --strict` and `pytest` valid targets with zero
   `type: ignore`, and avoids implementing ahead of the current phase (`CONVENTIONS.md` rule #4).

5. **Honest Makefile stubs for future targets.** `setup`, `lint`, `test` are implemented.
   `generate-tasks`, `serve`, `traffic`, `ope`, `repro-phase2`, `eval`, `report` exist for
   discoverability but `echo` a "not implemented until Phase N" message and exit non-zero, so
   nothing silently appears to work before its phase lands.

6. **Config files are placeholders (`schema_version: 0`).** The five `configs/*.yaml` files
   exist with a header and version only. Real keys — notably the reward weights λ (cost) and
   μ (latency) — are intentionally omitted; `CONVENTIONS.md` says to decide those with the user
   rather than guess.

7. **Single Phase-0 smoke test.** `tests/unit/test_smoke.py` asserts the package imports and
   exposes a well-formed dotted-numeric `__version__`. It is deliberately the only test:
   enough to keep `make test` green without asserting any (non-existent) experimental result.

8. **Git identity is repo-local only.** Commits are authored as the user
   (`user.name aashkashah09`, `user.email aashkapshah@gmail.com`) via repo-local
   `git config`; the machine's global git config is left untouched. Commits carry no
   tool-generated authorship trailer.

## Consequences

- CI runs ruff + `ruff format --check` + `mypy src` + `pytest` on CPU for every push/PR.
- The repo is immediately lint-clean and type-clean, so Phase 1 starts from green.
- Deferring heavy deps means a reviewer cloning today gets a small, fast install; the
  training/serving extras are the first thing to pin in their respective phases.
