"""CLI: build the seeded SQLite e-commerce database from a config + seed.

Thin entrypoint over :mod:`specialist_router.env.database`; every experiment stage takes
``--config`` and ``--seed`` so runs are reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from specialist_router.config import load_config
from specialist_router.env.database import build_dataset, write_sqlite_file


def main() -> None:
    """Parse arguments, build the dataset, and write it to a SQLite file."""
    parser = argparse.ArgumentParser(description="Build the seeded e-commerce SQLite database.")
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--out", required=True, help="Output SQLite file path.")
    args = parser.parse_args()

    config = load_config(args.config, seed_override=args.seed)
    dataset = build_dataset(config.db, config.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_sqlite_file(dataset, args.out)
    print(
        f"Wrote {args.out}: {len(dataset.customers)} customers, {len(dataset.products)} products, "
        f"{len(dataset.orders)} orders, {len(dataset.order_items)} items, "
        f"{len(dataset.refunds)} refunds, {len(dataset.returns)} returns (seed={config.seed})."
    )


if __name__ == "__main__":
    main()
