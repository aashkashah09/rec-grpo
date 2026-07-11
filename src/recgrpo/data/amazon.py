"""Loading and cleaning the Amazon Product Reviews (2014) dump.

The review file is line-delimited JSON; the metadata file is line-delimited
python-literal dicts (single quotes, occasional unescaped HTML), which is why
it is parsed with ast.literal_eval rather than json.loads.
"""

from __future__ import annotations

import ast
import gzip
import html
import json
import re
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")
_TAGS = re.compile(r"<[^>]+>")


def download(url: str, dest: str | Path) -> Path:
    dest = Path(dest)
    if dest.exists():
        logger.info("%s already present, skipping download", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def _iter_gzip_lines(path: str | Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Return a (user_id, item_id, timestamp) frame of raw string ids."""
    rows = []
    for line in _iter_gzip_lines(path):
        rec = json.loads(line)
        rows.append((rec["reviewerID"], rec["asin"], int(rec["unixReviewTime"])))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    logger.info("loaded %d raw reviews", len(df))
    return df


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in _flatten(value))
    text = html.unescape(str(value))
    text = _TAGS.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _flatten(value: Iterable) -> Iterator[str]:
    for item in value:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield str(item)


def load_metadata(path: str | Path, keep_asins: set[str] | None = None) -> dict[str, dict]:
    """Parse the metadata dump, keeping only the fields used for semantic IDs."""
    meta: dict[str, dict] = {}
    for line in _iter_gzip_lines(path):
        try:
            rec = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        asin = rec.get("asin")
        if not asin or (keep_asins is not None and asin not in keep_asins):
            continue
        meta[asin] = {
            "title": clean_text(rec.get("title")),
            "brand": clean_text(rec.get("brand")),
            "categories": clean_text(rec.get("categories")),
            "description": clean_text(rec.get("description")),
            "price": rec.get("price"),
        }
    logger.info("parsed metadata for %d items", len(meta))
    return meta


def apply_k_core(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Iteratively drop users and items with fewer than k interactions."""
    before = len(df)
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        keep_users = user_counts[user_counts >= k].index
        keep_items = item_counts[item_counts >= k].index
        filtered = df[df["user_id"].isin(keep_users) & df["item_id"].isin(keep_items)]
        if len(filtered) == len(df):
            break
        df = filtered
    logger.info("%d-core: %d -> %d interactions", k, before, len(df))
    return df.reset_index(drop=True)


def build_catalog(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Map raw string ids to dense integer indices, ordered by first appearance."""
    df = df.sort_values(["timestamp", "user_id", "item_id"], kind="mergesort")
    item_map = {asin: idx for idx, asin in enumerate(df["item_id"].drop_duplicates())}
    user_map = {uid: idx for idx, uid in enumerate(df["user_id"].drop_duplicates())}
    out = pd.DataFrame(
        {
            "user": df["user_id"].map(user_map).to_numpy(),
            "item": df["item_id"].map(item_map).to_numpy(),
            "timestamp": df["timestamp"].to_numpy(),
        }
    )
    return out.reset_index(drop=True), user_map, item_map


def build_item_text(meta: dict[str, dict], item_map: dict[str, int], fields, max_chars: int) -> list[str]:
    """One text blob per dense item index, in item-index order."""
    texts = [""] * len(item_map)
    for asin, idx in item_map.items():
        rec = meta.get(asin, {})
        parts = []
        for field in fields:
            value = rec.get(field) or ""
            if field == "description":
                value = value[:max_chars]
            if value:
                parts.append(f"{field}: {value}")
        texts[idx] = " | ".join(parts) if parts else f"title: {asin}"
    missing = sum(1 for t in texts if t.startswith("title: B0"))
    if missing:
        logger.warning("%d items have no usable metadata, falling back to asin", missing)
    return texts
