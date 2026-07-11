# 006 — Context featurizer and the hashing / MiniLM split

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 2 (Router + serving + logging + OPE)

## Context

The router must decide per query **without** the ground-truth `difficulty` tag or `template_id`
(`PROJECT_PLAN.md` §3) — only from signals a real deployment would have. The featurizer needs a
semantic component, but CI must stay fast, network-free, and torch-free (ADR-000 defers torch to
Phase 3).

## Decisions

1. **Difficulty proxies, not the tag.** The vector is `[bias] + heuristics + embedding`. Heuristics
   are transparent difficulty proxies: question length (chars/tokens), counts of numeric literals,
   schema/vocab entity mentions and date tokens, surface-cue flags (top-K/ratio/date-window), and a
   coarse question-type one-hot (aggregation/comparison/ranking/edge). The hidden `difficulty` and
   `template_id` are logged for analysis only and never enter the vector (asserted by a test).
2. **Hashing embedding is the default and the only locked path.** `Featurizer` with
   `embedding: hashing` computes a deterministic signed char-n-gram hashing embedding (numpy +
   `hashlib`, L2-normalized). It needs no model download and is stable across processes, so it is
   used in CI and `repro-phase2`.
3. **MiniLM is optional and *not* a locked dependency.** `embedding: minilm` lazily imports
   `sentence-transformers`. We deliberately did **not** add it to a locked extra: it pulls torch,
   which ADR-000 defers to Phase 3, and would bloat the lock and slow CI. It is a documented manual
   install for the real-embedding run; a fixed seeded random projection maps its 384-d output to
   the configured `embed_dim` so the feature width matches the hashing path.

## Consequences

- CI exercises the exact featurizer used for the committed numbers (hashing), with no network.
- Swapping to MiniLM later changes only `configs/router.yaml` + a manual `pip install`, and keeps
  the same vector width and downstream schema.
- Entity vocabulary is derived from `DbConfig`, so it tracks whatever the environment was generated
  with rather than being hard-coded.
