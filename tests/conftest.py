"""Shared pytest fixtures: the frozen mini database, its dataset/index, and env config."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from specialist_router.config import (
    Config,
    OpeConfig,
    RouterConfig,
    ServingConfig,
    load_config,
    load_ope_config,
    load_router_config,
    load_serving_config,
)
from specialist_router.env.database import (
    Dataset,
    load_dataset_from_connection,
    schema_sql,
    write_sqlite_file,
)
from specialist_router.env.tasks import EnvIndex

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_SQL = _ROOT / "tests" / "fixtures" / "mini_db.sql"
_MINI_CONFIG = _ROOT / "configs" / "env.mini.yaml"
_CONFIGS = _ROOT / "configs"


@pytest.fixture
def env_config() -> Config:
    """The mini environment config (used for tolerances and tool limits)."""
    return load_config(_MINI_CONFIG)


@pytest.fixture
def router_config() -> RouterConfig:
    """The Phase-2 router config (reward weights, featurizer, policies)."""
    return load_router_config(_CONFIGS / "router.yaml")


@pytest.fixture
def ope_config() -> OpeConfig:
    """The Phase-2 OPE config (estimators, bootstrap, DR outcome model)."""
    return load_ope_config(_CONFIGS / "ope.yaml")


@pytest.fixture
def serving_config() -> ServingConfig:
    """The Phase-2 serving config (stub arm profiles)."""
    return load_serving_config(_CONFIGS / "serving.yaml")


@pytest.fixture
def mini_conn() -> sqlite3.Connection:
    """A writable in-memory connection loaded with the schema and the frozen fixture rows."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_sql())
    conn.executescript(_FIXTURE_SQL.read_text())
    conn.commit()
    return conn


@pytest.fixture
def mini_dataset(mini_conn: sqlite3.Connection) -> Dataset:
    """The frozen fixture as a :class:`Dataset` (the pure-Python view)."""
    return load_dataset_from_connection(mini_conn)


@pytest.fixture
def mini_index(mini_dataset: Dataset) -> EnvIndex:
    """A prebuilt :class:`EnvIndex` over the frozen fixture."""
    return EnvIndex.from_dataset(mini_dataset)


@pytest.fixture
def mini_db_file(mini_dataset: Dataset, tmp_path: Path) -> Path:
    """A read-only-capable SQLite file built from the frozen fixture (for sandbox tests)."""
    path = tmp_path / "mini.sqlite"
    write_sqlite_file(mini_dataset, path)
    return path
