# 003 — Verifier, answer conventions, and dual ground truth

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 1 (Environment, tasks, verifier)

## Context

`CONVENTIONS.md` hard rule #3: verification is deterministic value comparison with tolerance, never
prose string-matching. The templates need unambiguous answer units, and the ground truth needs
to be trustworthy. Two conventions were confirmed with the user.

## Decisions

1. **Structured final answer.** Agents submit via a `final_answer` tool call carrying a typed
   value (number / int / list of strings). The verifier compares that value to ground truth —
   it never parses model prose. An unparseable/absent answer is an incorrect verdict, not a
   crash.
2. **Answer units (confirmed with user):**
   - rates / growth / ratios → **decimal fractions** (`0.123`, not `12.3%`);
   - cohort gaps → **percentage points** as plain numbers (`2.5` = 2.5 pp);
   - money → **USD dollars** with an absolute **$0.01** tolerance;
   - counts / ids → exact integers (`5` == `5.0`);
   - rankings → ordered `list[str]`, compared exactly after `strip().lower()`.
   Every rendered question **states its formula and its expected format** explicitly (a unit
   test asserts the format marker is present), so units are never ambiguous.
3. **Tolerance table** (`configs/*.yaml → verifier`): money abs $0.01; ratio/growth
   `max(1e-6, 1e-3·|expected|)`; percentage-points `max(1e-4, 1e-3·|expected|)`; integer/list
   exact.
4. **Dual ground truth (confirmed with user).** Each template has a pure-Python reference
   (source of truth) **and** a separate reference SQL (what the oracle runs through the
   sandbox). Tests assert the two agree on both the frozen fixture and generated data — a bug
   in either path is caught. The two revenue notions are named distinctly in code:
   `net_of_discounts_cents` vs `net_of_refunds_cents`.
5. **Trust-critical error policy.** The verifier raises on malformed *expected* values (a
   generator bug) but returns an incorrect verdict for malformed *submitted* values (the
   agent's fault). No silent failure.

## Consequences

- The verifier is a small pure function, near-completely unit-tested (units, rounding,
  fraction-vs-percent traps, list order/case, NULL/empty, malformed inputs).
- Answer-unit choices are baked into question text; changing them means re-rendering questions
  and updating the tolerance table.
