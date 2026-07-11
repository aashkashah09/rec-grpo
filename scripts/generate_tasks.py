"""CLI: generate tasks with programmatic ground truth and write them as JSONL.

Emits one :class:`~specialist_router.env.records.Task` per line (versioned schema). Optionally
also writes the backing SQLite database (``--db``) so an oracle can run the reference SQL.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from specialist_router.config import load_config
from specialist_router.env.database import build_dataset, write_sqlite_file
from specialist_router.env.tasks import generate_tasks


def main() -> None:
    """Parse arguments, generate tasks, and write them (and optionally the DB) to disk."""
    parser = argparse.ArgumentParser(description="Generate tasks with ground truth as JSONL.")
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--n", type=int, default=None, help="Override the number of tasks.")
    parser.add_argument("--out", required=True, help="Output JSONL path for tasks.")
    parser.add_argument("--db", default=None, help="Optional SQLite path to also write.")
    args = parser.parse_args()

    config = load_config(args.config, seed_override=args.seed)
    if args.n is not None:
        config = config.model_copy(
            update={"tasks": config.tasks.model_copy(update={"n_tasks": args.n})}
        )

    dataset = build_dataset(config.db, config.seed)
    tasks = generate_tasks(dataset, config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for task in tasks:
            handle.write(task.model_dump_json() + "\n")

    if args.db is not None:
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
        write_sqlite_file(dataset, args.db)

    by_template = Counter(task.template_id for task in tasks)
    print(f"Wrote {len(tasks)} tasks to {out_path} (seed={config.seed}).")
    for template_id, count in sorted(by_template.items()):
        print(f"  {template_id}: {count}")


if __name__ == "__main__":
    main()
