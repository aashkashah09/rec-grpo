# Specialist + Router: RL-Trained SQL Agent Behind an OPE-Validated Learned Cascade

**One-line summary:** A GRPO-trained small language model ("specialist") for SQL analytics
tasks, deployed behind a contextual-bandit router that chooses per-query between the
specialist and a frontier API — with propensity-logged decisions and off-policy evaluation
(IPS/SNIPS/DR) used to select routing policies, validated by replay A/B.

**Headline result targeted:** "A ~$200 fine-tuned 4B model handles the majority of traffic
at near-frontier quality and a fraction of the cost, routed per-query by an OPE-validated
learned policy."

---

## 1. Non-negotiable principles

1. **No fabricated results.** Every number in the README/report must trace to a committed
   run artifact (JSONL log, W&B run ID, or results JSON). If an experiment hasn't run,
   the doc says "TBD".
2. **Honest OPE scope.** Off-policy evaluation claims apply ONLY to the single routing
   decision (a true contextual bandit: one context, one action, one reward). The trained
   specialist is evaluated by held-out task success on the environment — never by
   trajectory-level importance sampling. This distinction must appear in the report.
3. **Deterministic verification.** Task ground truth is computed programmatically from the
   database by the task generator. Verifiers compare values (with numeric tolerance),
   never string-match model output.
4. **Reproducibility.** Every experiment is a YAML config + a seed. `make repro-phase2`
   must regenerate the router/OPE results on CPU from committed logs.
5. **Small, reviewed increments.** Work proceeds phase by phase. Each phase ends with all
   tests passing, CI green, and a short entry in `docs/decisions/`.

## 2. Repository layout

```
specialist-router/
├── CONVENTIONS.md                  # Engineering conventions and guardrails
├── PROJECT_PLAN.md            # This file
├── README.md                  # User-facing; written incrementally, finalized Phase 5
├── pyproject.toml             # uv-managed; single source of deps
├── Makefile                   # repro targets: test, lint, serve, repro-phase2, ...
├── Dockerfile                 # CPU repro image (env + router + OPE, no training)
├── .github/workflows/ci.yml   # ruff + mypy + pytest on every push
├── configs/
│   ├── env.yaml               # DB seed, task template mix, difficulty knobs
│   ├── router.yaml            # bandit algo, features, reward weights (λ cost, μ latency)
│   ├── ope.yaml               # estimator settings, bootstrap CI params
│   ├── grpo.yaml              # model, LoRA, TRL GRPOTrainer hyperparams
│   └── serving.yaml           # vLLM + API endpoints, timeouts, budgets
├── src/specialist_router/
│   ├── env/
│   │   ├── database.py        # seeded SQLite e-commerce DB builder
│   │   ├── schema.sql
│   │   ├── tasks.py           # 8 task templates, procedural generation, ground truth
│   │   ├── tools.py           # inspect_schema, run_sql, python_calc (sandboxed)
│   │   ├── episode.py         # multi-turn episode loop, max-turns, tool budget
│   │   └── verifier.py        # exact/tolerant value comparison, structured verdicts
│   ├── agents/
│   │   ├── base.py            # Agent protocol: (task) -> trajectory, answer
│   │   ├── local_agent.py     # vLLM-served small model (base or specialist checkpoint)
│   │   └── api_agent.py       # frontier API agent (provider-agnostic client)
│   ├── router/
│   │   ├── features.py        # context featurizer (embeddings + cheap heuristics)
│   │   ├── policies.py        # Uniform, EpsilonGreedy, LinUCB, ThompsonLogistic
│   │   ├── reward.py          # composite reward: quality − λ·cost − μ·latency
│   │   └── logger.py          # propensity-logged decision records (JSONL, schema'd)
│   ├── ope/
│   │   ├── estimators.py      # IPS, SNIPS, DoublyRobust (+ direct method baseline)
│   │   ├── ci.py              # bootstrap confidence intervals, ESS diagnostics
│   │   └── replay.py          # replay A/B evaluation to validate OPE predictions
│   ├── training/
│   │   ├── grpo_run.py        # TRL GRPOTrainer wiring, LoRA, reward fn from verifier
│   │   ├── rollout.py         # env-coupled generation (vLLM) for training prompts
│   │   └── sft_warmup.py      # optional small SFT on tool-format demonstrations
│   ├── serving/
│   │   ├── app.py             # FastAPI: /solve endpoint -> router -> agent -> log
│   │   └── clients.py
│   ├── evaluation/
│   │   ├── harness.py         # held-out task success, per-template breakdown
│   │   └── frontier_curve.py  # cost/quality/latency frontier computation + plot
│   └── analysis/
│       └── report_tables.py   # generates all report tables/figures from artifacts
├── scripts/                   # thin CLI entrypoints (one per pipeline stage)
├── tests/
│   ├── unit/                  # verifier, tasks, policies, estimators, reward
│   ├── property/              # OPE estimators vs known-truth simulator (hypothesis)
│   └── integration/           # end-to-end episode on tiny DB; router loop on stub agents
├── artifacts/                 # committed small results (logs samples, eval JSONs, plots)
└── docs/
    ├── architecture.md        # diagrams + data flow
    ├── decisions/             # ADR-style: one file per major choice
    └── report.md              # final technical report (Phase 5)
```

## 3. Phases

### Phase 0 — Scaffolding (~1 session)
- Init repo: pyproject (uv), ruff, mypy (strict on src/), pytest, pre-commit, CI workflow,
  Makefile, directory skeleton, `.env.example` (OPENAI/ANTHROPIC-style key names,
  HF_TOKEN, WANDB_API_KEY).
- **Done when:** CI green on empty skeleton; `make test lint` pass locally.

### Phase 1 — Environment, tasks, verifier (~2 sessions)
- Seeded SQLite e-commerce DB (~6 tables, 50–200k rows generated from config seed).
- 8 task templates (examples): revenue-by-segment aggregation; refund-rate cohort
  comparison; top-K with tie-break rules; time-window growth rate; join-heavy customer
  LTV; anomaly lookup (which product's return rate exceeded X%); multi-step ratio;
  null/edge-case handling question. Each template: parameter sampler + ground-truth
  computation + difficulty tag (easy/med/hard) + canonical answer type.
- Tool layer with hard sandboxing: `run_sql` read-only connection, statement whitelist,
  row/time limits; `python_calc` restricted eval on numeric expressions only.
- Episode loop: system prompt with tool schema, max 12 turns, tool budget, structured
  trajectory record.
- Verifier: typed answer parsing, numeric tolerance, verdict object
  {correct: bool, reason, extracted, expected}.
- Tests: unit tests for every template's ground truth on a frozen mini-DB; verifier
  edge cases (formatting, units, rounding); sandbox escape attempts.
- **Done when:** `scripts/generate_tasks.py` emits N tasks with ground truth;
  a scripted "oracle agent" (executes the reference SQL) scores 100%; a trivial
  agent scores ~0%; all tests pass.

### Phase 2 — Router + serving + logging + OPE (~3 sessions) ⭐ standalone milestone
- Agents: `api_agent` (frontier model) and `local_agent` (off-the-shelf Qwen3-4B-Instruct
  via vLLM, or an API-served small model as CPU-friendly fallback for development).
- Featurizer: sentence-embedding of question (small local model), plus heuristics
  (template difficulty tag hidden at inference — use proxies: question length, #entities,
  date-window presence).
- Reward: quality ∈ {0,1} from verifier (or calibrated LLM-judge fallback where verifier
  N/A) minus λ·normalized_cost minus μ·normalized_latency; λ, μ in configs with a
  sensitivity sweep.
- Policies: Uniform (logging policy), EpsilonGreedy, LinUCB, ThompsonLogistic. Every
  decision logged with: context features, action, propensity, policy version, reward
  components, model versions, timestamps. JSONL schema versioned + validated.
- Traffic generation: run M tasks (start 2k) through the service under the Uniform
  logging policy to build the logged dataset.
- OPE: IPS, SNIPS, DR (reward model = gradient-boosted or ridge on context×action),
  bootstrap CIs, effective-sample-size diagnostics, and a "when estimators break"
  mini-study (shrinking overlap, deterministic logging).
- Validation: replay A/B — deploy candidate policy on fresh tasks, compare realized value
  vs OPE prediction; report calibration table.
- Property tests: simulator with known true policy values; assert estimator bias/variance
  behavior (IPS unbiased-in-expectation, SNIPS lower variance, DR best under model
  misspecification patterns).
- **Done when:** `make repro-phase2` runs traffic→OPE→replay on CPU from config and
  reproduces committed tables; frontier plot (cost vs quality) renders; README section
  "Router & OPE" written with real numbers.

### Phase 3 — GRPO specialist training (~2 sessions + GPU time)
- This phase adds: prompt/format spec, reward fn (verifier-backed, plus small format
  reward), TRL GRPOTrainer config (Qwen3-4B, QLoRA r=16–32, group size 8, KL/clip per
  config), rollout wiring against the environment, W&B logging, checkpoint/eval cadence,
  resume-from-checkpoint.
- Optional SFT warmup (500–1500 synthetic tool-format demos generated via frontier API)
  if base model's tool-format compliance <60%.
- You run: single A100 80GB or RTX 4090/A6000 tier (LoRA 4B fits well under 80GB;
  4090 feasible with QLoRA + reduced group size). Budget: 30–60 GPU-hours ≈ $60–200
  on spot/community clouds. Watch reward curves; expected trajectory: format reward
  saturates first, accuracy climbs on easy templates, hard templates lag.
- Guardrails: eval on held-out tasks every K steps; early-stop on divergence; keep top-3
  checkpoints by held-out success.
- **Done when:** specialist beats base model by a clear margin on held-out tasks
  (target: base ~30–50% → specialist ≥70% overall; report per-template) with W&B run
  linked in README.
- **Bonus chapter (only if observed):** reward-hacking case study — any trajectory where
  the model games the verifier; document, fix verifier, show before/after.

### Phase 4 — Integration: specialist into the cascade (~1 session)
- Swap `local_agent` to the trained checkpoint; regenerate logged traffic under Uniform;
  rerun OPE to select the best router; replay-A/B validate; recompute frontier curve
  (now three points/curves: frontier-only, base-cascade, specialist-cascade).
- **Done when:** final headline metrics computed from artifacts: % traffic to specialist,
  relative quality vs frontier-only, relative cost, OPE-predicted vs realized value.

### Phase 5 — Report, polish, demo (~2 sessions)
- `docs/report.md`: problem → system → results → OPE honesty section → estimator-breakage
  study → training curves → limitations. Engineering-blog tone, every figure generated by
  `analysis/report_tables.py`.
- README final pass: 90-second pitch at top, architecture diagram, quickstart
  (`docker run` for CPU demo), results table, repo map, "what I'd do next".
- 2-minute demo recording (you), linked in README.
- **Done when:** a stranger can run the CPU demo in <10 minutes and every claim links to
  an artifact.

## 4. Compute & accounts checklist (user-provided)
- HF token; W&B account; one frontier-API key (budget cap ~$50–150 for judging,
  SFT-demo generation, and api_agent traffic).
- GPU rental for Phase 3 only (Lambda/RunPod/Vast-tier, single GPU). Everything else CPU.

## 5. Risks and fallbacks
- **GRPO unstable / gains modest:** ship Phase 2 as the complete project (it already is);
  report the training attempt honestly with curves. Fallback specialist = SFT-only model.
- **Base model can't use tools at all:** add SFT warmup (planned optional step).
- **OPE overlap too thin:** logging policy is Uniform by design → propensities bounded;
  ESS diagnostic in every report table.
- **Scope creep:** any feature not in this plan requires a decision record first.
