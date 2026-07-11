from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access, so cfg.train.lr works as well as cfg['train']['lr']."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - attribute error is the useful one
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, optionally applying dotted-key overrides."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = copy.deepcopy(raw)
        for key, value in overrides.items():
            node = raw
            *parents, leaf = key.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
    return Config(raw)
