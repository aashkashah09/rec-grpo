"""CLI: SFT warmup demo generation (frontier API) — gated behind --confirm-spend.

Thin wrapper over training/sft_warmup.py. Prints an estimated API cost and makes NO frontier call
unless --confirm-spend is passed. See ADR-013.
"""

from __future__ import annotations

import sys

from specialist_router.training.sft_warmup import main

if __name__ == "__main__":
    sys.exit(main())
