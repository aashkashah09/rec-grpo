# Architecture

Data flow for the Specialist + Router system (from [`PROJECT_PLAN.md`](../PROJECT_PLAN.md)).
Components are implemented phase by phase. **Implemented so far:** the entire `env/` block
below (Phase 1) — seeded DB, task templates + ground truth, sandboxed tools, episode loop, and
verifier. The `agents/`, `router/`, and `ope/`/`evaluation/` blocks are Phase 2+ and this
diagram runs ahead of the code there until Phase 4.

```mermaid
flowchart TD
    subgraph ENV["env/ (Phase 1)"]
        DB[(Seeded SQLite DB)]
        TASKS[Task templates + ground truth]
        TOOLS[Sandboxed tools:<br/>inspect_schema, run_sql, python_calc]
        EPISODE[Episode loop<br/>max turns, tool budget]
        VERIFIER[Verifier<br/>value comparison + tolerance]
        DB --> TASKS
        TASKS --> EPISODE
        TOOLS --> EPISODE
        EPISODE --> VERIFIER
    end

    subgraph AGENTS["agents/ (Phase 2/4)"]
        LOCAL[local_agent<br/>base or specialist checkpoint]
        API[api_agent<br/>frontier API]
    end

    subgraph ROUTER["router/ (Phase 2)"]
        FEATURES[Featurizer<br/>embeddings + heuristics]
        POLICY[Bandit policy<br/>Uniform / EpsilonGreedy / LinUCB / ThompsonLogistic]
        REWARD[Reward<br/>quality - lambda*cost - mu*latency]
        LOGGER[Propensity logger<br/>versioned JSONL]
    end

    subgraph OFFLINE["ope/ + evaluation/ (Phase 2)"]
        OPE[IPS / SNIPS / DR<br/>+ bootstrap CIs, ESS]
        REPLAY[Replay A/B validation]
        FRONTIER[Cost/quality/latency frontier]
    end

    TASKS --> FEATURES
    FEATURES --> POLICY
    POLICY -->|action| LOCAL
    POLICY -->|action| API
    LOCAL --> EPISODE
    API --> EPISODE
    VERIFIER --> REWARD
    POLICY -->|propensity| LOGGER
    REWARD --> LOGGER
    LOGGER --> OPE
    OPE --> REPLAY
    OPE --> FRONTIER
```

Trust-critical, deterministic components — the verifier, task ground truth, and the OPE
estimators — are kept as small pure functions so they are trivially testable. Off-policy
evaluation applies **only** to the single routing decision, never to agent trajectories
(see [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) §1.2).
