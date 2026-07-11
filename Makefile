# Makefile — repro targets per CLAUDE.md. Tooling runs through `uv run`.
# Phase 0 implements: setup, lint, test. Later-phase targets are honest stubs
# that fail loudly rather than pretend to work (no fabricated capability).

.PHONY: setup lint test \
        generate-tasks serve traffic ope repro-phase2 eval report

# ---- Phase 0 (implemented) -------------------------------------------------

setup:  ## Install deps into .venv and wire up pre-commit hooks.
	uv sync
	uv run pre-commit install

lint:  ## ruff lint + format check + mypy --strict on src/.
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

test:  ## Run the pytest suite (CPU-only; gpu/external markers skipped).
	uv run pytest

# ---- Phases 1-5 (not yet implemented) --------------------------------------
# Each stub exits non-zero so nothing silently "passes" before its phase lands.

generate-tasks:  ## [Phase 1] Emit tasks + ground truth.
	@echo "generate-tasks: not implemented until Phase 1" && exit 1

serve:  ## [Phase 2] Launch the FastAPI /solve service.
	@echo "serve: not implemented until Phase 2" && exit 1

traffic:  ## [Phase 2] Send logging-policy traffic through the service.
	@echo "traffic: not implemented until Phase 2" && exit 1

ope:  ## [Phase 2] Run IPS/SNIPS/DR estimators over logged decisions.
	@echo "ope: not implemented until Phase 2" && exit 1

repro-phase2:  ## [Phase 2] traffic -> OPE -> replay from committed logs.
	@echo "repro-phase2: not implemented until Phase 2" && exit 1

eval:  ## [Phase 2/4] Held-out task-success + frontier curve.
	@echo "eval: not implemented until Phase 2" && exit 1

report:  ## [Phase 5] Generate report tables/figures from artifacts.
	@echo "report: not implemented until Phase 5" && exit 1
