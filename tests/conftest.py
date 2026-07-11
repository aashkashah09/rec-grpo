"""Shared pytest fixtures: the frozen mini database, its dataset/index, and env config."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from specialist_router.config import Config, load_config
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


@pytest.fixture
def env_config() -> Config:
    """The mini environment config (used for tolerances and tool limits)."""
    return load_config(_MINI_CONFIG)


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
