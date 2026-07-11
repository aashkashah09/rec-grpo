# Makefile — repro targets per CONVENTIONS.md. Tooling runs through `uv run`.
# Phase 0 implements: setup, lint, test. Later-phase targets are honest stubs
# that fail loudly rather than pretend to work (no fabricated capability).

.PHONY: setup lint test \
        generate-tasks serve traffic ope repro-phase2 eval report \
        grpo-dryrun grpo-dryrun-mock train-grpo sft-warmup

# Defaults for generate-tasks (override on the command line, e.g. `make generate-tasks CONFIG=configs/env.yaml`).
CONFIG ?= configs/env.mini.yaml
SEED   ?= 20260711
OUT    ?= build/tasks.jsonl
DB     ?= build/env.sqlite

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

generate-tasks:  ## [Phase 1] Emit tasks + ground truth as JSONL (and the backing SQLite DB).
	uv run python scripts/generate_tasks.py --config $(CONFIG) --seed $(SEED) --out $(OUT) --db $(DB)

# Phase-2 defaults (override on the command line, e.g. `make traffic N=5000`).
ENV_CONFIG     ?= configs/env.mini.yaml
ROUTER_CONFIG  ?= configs/router.yaml
OPE_CONFIG     ?= configs/ope.yaml
SERVING_CONFIG ?= configs/serving.yaml
N              ?= 2000
DECISIONS      ?= build/phase2/decisions.jsonl

serve:  ## [Phase 2] Launch the FastAPI /solve service (needs the 'serving' extra).
	uv run python scripts/serve.py --env-config $(ENV_CONFIG) --router-config $(ROUTER_CONFIG) \
		--serving-config $(SERVING_CONFIG)

traffic:  ## [Phase 2] Generate Uniform-logging-policy traffic (stub backend) as decisions JSONL.
	uv run python scripts/run_traffic.py --env-config $(ENV_CONFIG) --router-config $(ROUTER_CONFIG) \
		--serving-config $(SERVING_CONFIG) --n $(N) --out $(DECISIONS)

ope:  ## [Phase 2] Run IPS/SNIPS/DR/DM estimators (+ bootstrap CIs) over logged decisions.
	uv run python scripts/run_ope.py --decisions $(DECISIONS) --router-config $(ROUTER_CONFIG) \
		--ope-config $(OPE_CONFIG)

repro-phase2:  ## [Phase 2] Full CPU repro: traffic -> OPE -> replay-A/B -> breakage -> tables/figures.
	uv run python scripts/repro_phase2.py --env-config $(ENV_CONFIG) --router-config $(ROUTER_CONFIG) \
		--ope-config $(OPE_CONFIG) --serving-config $(SERVING_CONFIG)

eval:  ## [Phase 2/4] Regenerate the cost/quality frontier figure from committed artifacts.
	uv run python -c "import json; from specialist_router.analysis import plots; \
		plots.plot_frontier(json.load(open('artifacts/phase2/frontier.json')), 'artifacts/phase2/frontier.png'); \
		print('wrote artifacts/phase2/frontier.png')"

report:  ## [Phase 5] Generate report tables/figures from artifacts.
	@echo "report: not implemented until Phase 5" && exit 1

# ---- Phase 3 (GRPO specialist training) ------------------------------------
# grpo-dryrun runs a real tiny GRPOTrainer step on CPU and needs the macOS-installable 'training-cpu'
# extra (uv sync --extra training-cpu); grpo-dryrun-mock needs no heavy deps. train-grpo and
# sft-warmup need the full 'training' extra (GPU) / a frontier-API budget respectively. See ADR-014.
GRPO_CONFIG ?= configs/grpo.yaml

grpo-dryrun:  ## [Phase 3] Real tiny GRPOTrainer step on CPU (needs 'training-cpu'; macOS-friendly).
	uv run python scripts/grpo_dryrun.py --config $(GRPO_CONFIG) --seed $(SEED) --real

grpo-dryrun-mock:  ## [Phase 3] Dependency-free mock dry-run (no torch/trl; runs anywhere incl. CI).
	uv run python scripts/grpo_dryrun.py --config $(GRPO_CONFIG) --seed $(SEED) --mock

train-grpo:  ## [Phase 3] Launch GRPO training (GPU; needs the 'training' extra). Add ARGS=--dry-run for CPU.
	uv run python scripts/train_grpo.py --config $(GRPO_CONFIG) --seed $(SEED) $(ARGS)

sft-warmup:  ## [Phase 3] SFT demo generation (frontier API). Prints cost; needs ARGS=--confirm-spend to spend.
	uv run python scripts/sft_warmup.py --config $(GRPO_CONFIG) --serving-config $(SERVING_CONFIG) $(ARGS)
