# Architecture

Data flow for the Specialist + Router system (from [`PROJECT_PLAN.md`](../PROJECT_PLAN.md)).
Components are implemented phase by phase. **Implemented so far:** the entire `env/` block
(Phase 1) and the `router/` + `ope/` blocks (Phase 2), driven on CPU by a **stub simulator**
backend in place of live agents. The real `local_agent`/`api_agent` arms exist behind the same
`ArmRunner`/`Agent` interfaces but are exercised only in Phase 4 (`serving.backend: real`); until
then the diagram's `agents/` box runs ahead of what CI drives.

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

    subgraph AGENTS["agents/ + serving/ (Phase 2 stub → Phase 4 real)"]
        STUB[SimulatedArmRunner<br/>seeded quality/cost/latency — CPU/CI]
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
    POLICY -->|action| STUB
    POLICY -->|action| LOCAL
    POLICY -->|action| API
    STUB -->|submits GT/wrong| VERIFIER
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
