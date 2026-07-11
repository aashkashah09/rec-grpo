# Specialist + Router

**RL-trained SQL agent behind an OPE-validated learned cascade.**

A GRPO-trained small language model (the "specialist") for SQL analytics tasks, deployed
behind a contextual-bandit router that chooses per query between the specialist and a
frontier API — with propensity-logged decisions and off-policy evaluation (IPS/SNIPS/DR)
used to select the routing policy, validated by replay A/B.

> **Status: Phase 2 — router, serving, propensity logging, and OPE.** The contextual-bandit
> router, off-policy estimators (IPS/SNIPS/DR + direct method), bootstrap CIs, replay-A/B, and the
> estimator-breakage study are implemented and tested, driven on CPU by a **stub-agent simulator**.
> Every Phase-2 number below is stub/CPU-simulator data — real-model numbers arrive in Phase 4.
> See [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
make setup    # uv sync + install pre-commit hooks
make lint     # ruff check + ruff format --check + mypy --strict src
make test     # pytest (CPU-only; gpu/external markers skipped)
```

## Repository map

```
configs/                 # per-phase YAML configs (env, router, ope, grpo, serving)
src/specialist_router/
  env/                   # Phase 1: seeded SQLite DB, tasks, sandboxed tools, verifier
  agents/                # Phase 2/4: local specialist + frontier-API agents
  router/                # Phase 2: featurizer, bandit policies, reward, propensity logger
  ope/                   # Phase 2: IPS/SNIPS/DR estimators, bootstrap CIs, replay A/B
  training/              # Phase 3: GRPO wiring, rollout, optional SFT warmup (GPU only)
  serving/               # Phase 2: FastAPI /solve endpoint
  evaluation/            # Phase 2/4: held-out success harness, frontier curve
  analysis/              # generates all report tables/figures from artifacts
scripts/                 # thin CLI entrypoints (per pipeline stage)
tests/                   # unit / property / integration
artifacts/               # committed small results (append-only)
docs/                    # architecture, decisions (ADRs), report
```

## Phases

Work proceeds one phase at a time (see [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §3). Each phase
ends with tests green, CI green, and a decision record for any spec deviation.

### Phase 0 — Scaffolding ✅

Repository skeleton, `uv`-managed `pyproject.toml`, ruff + mypy (strict) + pytest, a
pre-commit config, a CPU-only CI workflow, the `Makefile` repro targets, `.env.example`
(key names only), and empty-but-documented packages. `make test` and `make lint` pass on
the empty skeleton. No experiment code and no results — those begin in Phase 1.

### Phase 1 — Environment, tasks, verifier ✅

A seeded e-commerce SQLite database (6 tables, money as integer cents, ISO dates), 8
procedurally-generated task templates with dual ground truth, a hard-sandboxed tool layer, a
multi-turn episode loop, and a deterministic value-comparison verifier.

- **Database** ([`env/database.py`](src/specialist_router/env/database.py),
  [`schema.sql`](src/specialist_router/env/schema.sql)) — `customers`, `products`, `orders`,
  `order_items`, `refunds`, `returns`. Fully determined by `(config, seed)` via NumPy.
- **8 task templates** ([`env/tasks.py`](src/specialist_router/env/tasks.py)) —
  `revenue_by_segment`, `refund_rate_cohort`, `topk_categories`, `mom_growth`,
  `customer_ltv`, `return_rate_anomaly`, `category_order_ratio`, `null_discount_edge`. Each
  question states its **exact formula** and **expected answer format**. Ground truth is
  computed in pure Python **and** as reference SQL; tests assert they agree.
- **Sandboxed tools** ([`env/tools.py`](src/specialist_router/env/tools.py)) — `inspect_schema`
  (columns with types + foreign keys), read-only `run_sql` (authorizer + query_only + single
  statement + row/opcode budgets), and `python_calc` (AST-whitelist; never `eval`/`exec`).
- **Verifier** ([`env/verifier.py`](src/specialist_router/env/verifier.py)) — typed value
  comparison with per-type tolerance (money to $0.01, ratios/pp relative); rates are decimal
  fractions, never percentages. Verdicts are `{correct, reason, extracted, expected, ...}`.
- **Episode loop + reference agents** — max 12 turns and a tool budget enforced centrally;
  a scripted `OracleAgent` scores **100%** and a `TrivialAgent` scores **0%** on the mini
  environment (see `tests/integration/`).

Generate a task set with ground truth:

```bash
make generate-tasks                 # mini profile -> build/tasks.jsonl + build/env.sqlite
# or a full run:
uv run python scripts/generate_tasks.py --config configs/env.yaml --seed 42 \
    --out build/tasks.jsonl --db build/env.sqlite
```

A committed 16-task sample lives at
[`artifacts/samples/phase1_tasks_sample.jsonl`](artifacts/samples/phase1_tasks_sample.jsonl).
Design rationale is in [`docs/decisions/`](docs/decisions/) (ADRs 001–004).

### Phase 2 — Router, serving, propensity logging & OPE ✅

A per-query contextual-bandit router chooses between a cheap **local** arm and an expensive
**api** arm; decisions are propensity-logged and evaluated off-policy, then validated by replay.
Driven on CPU by a stub simulator (no GPU, no API), so the whole pipeline reproduces from config.

- **Featurizer** ([`router/features.py`](src/specialist_router/router/features.py)) — bias +
  difficulty-proxy heuristics (length, entity/numeric/date counts, question-type one-hot) + a
  deterministic hashing embedding. The hidden `difficulty`/`template_id` never enter the vector.
- **Reward** ([`router/reward.py`](src/specialist_router/router/reward.py)) —
  `quality − λ·cost_norm − μ·latency_norm` with **λ = 0.3, μ = 0.1** and *fixed* reference scales
  (stationary across logs). `quality` is the deterministic verifier verdict — no LLM-judge
  (see [ADR-007](docs/decisions/007-reward-and-no-judge.md)).
- **Policies** ([`router/policies.py`](src/specialist_router/router/policies.py)) — Uniform
  (logging), EpsilonGreedy, LinUCB, ThompsonLogistic. Every decision logs context, action,
  propensity, policy version, and reward components as versioned JSONL
  ([`RouterDecision`](src/specialist_router/env/records.py), schema v2).
- **OPE** ([`ope/`](src/specialist_router/ope/)) — IPS, SNIPS, direct method, and cross-fitted DR,
  with bootstrap CIs and ESS diagnostics — applied **only** to the single routing decision, never
  to agent trajectories ([`CLAUDE.md`](CLAUDE.md) rule #2).
- **Replay-A/B** ([`ope/replay.py`](src/specialist_router/ope/replay.py)) — deploy each candidate
  on fresh traffic and compare realized value to the OPE prediction; realized and predicted rewards
  are asserted to be on the same λ/μ scale.

Reproduce everything on CPU (writes tables + figures under [`artifacts/phase2/`](artifacts/phase2/)):

```bash
make repro-phase2          # traffic -> OPE -> replay-A/B -> breakage -> tables/figures
make traffic ope           # or the individual stages
```

**OPE — value of each policy** (stub/CPU-simulator; 2000 Uniform-logged decisions):

| policy | IPS | SNIPS | DM | DR | DR 95% CI | ESS frac |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 0.508 | 0.508 | 0.511 | 0.511 | [0.492, 0.531] | 1.00 |
| epsilon_greedy | 0.570 | 0.584 | 0.580 | 0.584 | [0.564, 0.605] | 0.54 |
| linucb | 0.538 | 0.545 | 0.545 | 0.547 | [0.530, 0.565] | 0.89 |
| thompson_logistic | 0.565 | 0.583 | 0.581 | 0.583 | [0.564, 0.603] | 0.59 |
| always_local | 0.451 | 0.441 | 0.442 | 0.443 | [0.411, 0.476] | 0.51 |
| always_api | 0.566 | 0.580 | 0.579 | 0.579 | [0.559, 0.598] | 0.49 |

The learned routers beat Uniform and always-local; **replay-A/B lands 5/6 policies inside their DR
CI** (epsilon_greedy is the honest miss — OPE was slightly optimistic for the most-exploiting
policy). On the realized cost/quality frontier, `epsilon_greedy` reaches **quality 0.81 at
$0.014/decision** vs `always_api`'s **0.88 at $0.018** — near-frontier quality routing ~20% of
traffic to the cheap arm. With these particular stub cost/latency scales the *reward*-optimal point
sits close to always-frontier; the λ/μ sweep is what surfaces where the router wins on reward too.

**λ/μ reward-weight sweep** (re-scoring the same 2000 logs; does a learned router beat always_api
on reward?). The learned router wins in **11/24 cells, starting at λ ≥ 0.30**; the margin grows
with the cost weight (up to +0.481 at λ=1.0). The default (λ=0.3, μ=0.1) sits right at the
crossover — which is exactly why the frontier, not raw reward, is where routing pays at low λ:

| λ | μ | always_api DR | best learned DR | winner | margin |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.10 | 0.836 | 0.834 | thompson_logistic | −0.001 |
| 0.30 | 0.10 | 0.579 | 0.584 | epsilon_greedy | +0.005 |
| 0.50 | 0.10 | 0.408 | 0.482 | epsilon_greedy | +0.074 |
| 1.00 | 0.20 | −0.076 | 0.406 | thompson_logistic | +0.481 |

Full grid + heatmap: [`artifacts/phase2/lambda_mu_sweep_table.md`](artifacts/phase2/lambda_mu_sweep_table.md),
[`lambda_mu_sweep.png`](artifacts/phase2/lambda_mu_sweep.png). Reproduce with
`uv run python scripts/sweep_lambda_mu.py`.

**When estimators break** (known-truth tabular simulator, shrinking logging overlap):

| min π₀ | IPS std | SNIPS std | DR std | ESS frac |
| --- | --- | --- | --- | --- |
| 0.50 | 0.009 | 0.009 | 0.010 | 0.70 |
| 0.05 | 0.028 | 0.025 | 0.025 | 0.13 |
| 0.01 | 0.063 | 0.047 | 0.051 | 0.03 |

As overlap collapses, IPS variance explodes while SNIPS/DR stay steadier and ESS falls toward zero —
the reason the logging policy is Uniform by design. Full tables and figures:
[`artifacts/phase2/`](artifacts/phase2/). Design rationale:
[ADRs 005–009](docs/decisions/).

### Phase 3 — GRPO specialist training 🚧 (code + config + CPU dry-run; training run pending)

GRPO fine-tunes Qwen3-4B-Instruct into the **local** SQL specialist against the *same* environment
and the *same* deterministic verifier as the rest of the project, so Phase 4 is a checkpoint swap.
This session ships the code, config, and a CPU dry-run; the training run happens on rented GPU
hardware and its metrics are **TBD (pending run)** — no training numbers are reported until they
trace to a real W&B run.

- **Seam = TRL `rollout_func`** ([`training/rollout.py`](src/specialist_router/training/rollout.py),
  [ADR-010](docs/decisions/010-grpo-rollout-seam.md)) — we own the multi-turn generation loop, so
  the Phase-1/2 JSON tool protocol, `verify()`, and the eval harness are reused byte-for-byte.
  `EnvRollout` drives the real [`run_episode`](src/specialist_router/env/episode.py) loop and records
  the token stream with an **assistant-only loss mask** (tool-result tokens masked out).
- **Reward** ([`training/reward.py`](src/specialist_router/training/reward.py),
  [ADR-011](docs/decisions/011-training-reward.md)) — `R = 1.0·correct + 0.15·format_score`.
  Correctness is the deterministic verifier verdict; the small format term keeps a gradient alive
  when a GRPO group is all-correct/all-wrong (binary-reward collapse) but is too small to beat a
  correct answer. GRPO turns `R` into a group-relative advantage — a training objective, **not**
  trajectory-level OPE ([`CLAUDE.md`](CLAUDE.md) rule #2).
- **Env-coupled data** ([`training/data.py`](src/specialist_router/training/data.py)) — training
  tasks are generated from the environment, split into a deterministic template-balanced train /
  held-out partition (a task is on exactly one side), with a pass-rate curriculum that skips tasks
  the model always fails or always solves.
- **Held-out eval** ([`evaluation/harness.py`](src/specialist_router/evaluation/harness.py)) —
  every K steps on a fixed held-out sample: overall + per-template success, top-3 checkpoints by
  held-out success, and a **format-rate drift guard** (format-rate falling while reward rises warns
  of protocol drift / verifier gaming). W&B logging, resume-from-checkpoint, and SIGTERM
  checkpoint-flush for spot instances live in
  [`training/callbacks.py`](src/specialist_router/training/callbacks.py).
- **Optional SFT warmup** ([`training/sft_warmup.py`](src/specialist_router/training/sft_warmup.py),
  [ADR-013](docs/decisions/013-sft-warmup.md)) — only if base tool-format compliance < 60%; demos
  come from the Phase-2 frontier `api_agent` (verifier-correct episodes only), kept strictly separate
  from the RL run. It **prints an estimated API cost and refuses to spend without `--confirm-spend`**.

Validate the whole pipeline on CPU before renting a GPU (no torch/trl needed):

```bash
make grpo-dryrun     # env rollout -> verify -> reward -> group advantage -> trainer-input contract
```

Then, on a GPU box (`uv sync --extra training`):

```bash
uv run pytest -m training          # verify config wiring against the pinned TRL
make sft-warmup ARGS=--confirm-spend   # optional; prints cost first
make train-grpo                    # GRPO training (reads configs/grpo.yaml)
```

**Hardware** ([`configs/grpo.yaml`](configs/grpo.yaml) is the ~40–80 GB, group-size-8 default):
QLoRA (4-bit NF4) on Qwen3-4B fits a **24 GB** card (RTX 4090 / A6000-tier) with reduced group size
(2–4), vLLM sleep-mode, and tighter tool-output caps; **40–48 GB** is comfortable for group size 6–8;
80 GB is not required. **GPU-hours are TBD (pending run)** — multi-turn 4B rollout is
generation-bound, so wall-clock will be measured on the first short run and extrapolated rather than
guessed (the plan's 30–60 GPU-hr estimate is a starting point, not a claim).

_Later phases (Integration, Report) are described in `PROJECT_PLAN.md` and will be filled in here as
they complete._
