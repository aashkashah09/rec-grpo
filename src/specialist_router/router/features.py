"""Context featurizer: turn a task question into a fixed-width numeric feature vector.

The router must decide *without* seeing the ground-truth ``difficulty`` tag or ``template_id``
(``PROJECT_PLAN`` §3) — it may only use signals a real deployment would have. So the vector is:

* a **bias** term,
* cheap, transparent **heuristics** (length, counts of numeric literals / schema entities / date
  tokens, and a few surface-cue flags and a coarse question-type one-hot) that act as proxies for
  difficulty, and
* a **semantic embedding** block: a deterministic, numpy-only *hashing* embedding by default
  (used in CI and ``repro-phase2`` — no model download, no network), or a sentence-transformers
  MiniLM embedding on the optional ``minilm`` path (documented manual install; see ADR-006).

Determinism matters: the same question must always produce the same vector so logged decisions,
OPE, and replay all agree. The hashing embedding therefore uses ``hashlib`` (stable across
processes), never Python's salted builtin ``hash``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from specialist_router.config import DbConfig, FeaturesConfig

# Schema tokens the router may plausibly recognise in a question (table and salient column names).
# These are surface strings, not the hidden template id.
_SCHEMA_TOKENS: tuple[str, ...] = (
    "customer",
    "customers",
    "product",
    "products",
    "order",
    "orders",
    "order item",
    "order items",
    "refund",
    "refunds",
    "return",
    "returns",
    "segment",
    "category",
    "categories",
    "discount",
    "quantity",
    "unit_price",
    "unit price",
    "revenue",
    "price",
)

_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH_RE = re.compile(r"\bmonth\b", re.IGNORECASE)
_WORD_RE = re.compile(r"\S+")

_TOPK_MARKERS = ("top ", "rank", "highest", "list the")
_RATIO_MARKERS = ("rate", "ratio", "fraction", "growth", "per ")
_COMPARISON_MARKERS = ("cohort", "exceed", "percentage points", "growth from", "compared")
_EDGE_MARKERS = ("null", "missing", "if there are no", "treat")

# The coarse question-type buckets (mutually exclusive, one-hot). Order fixed for stable columns.
_BUCKETS: tuple[str, ...] = ("aggregation", "comparison", "ranking", "edge")


@dataclass(frozen=True, slots=True)
class EntityVocab:
    """The set of domain entity phrases the featurizer counts as 'entities' in a question.

    Built from the data-generation config (segments/countries/channels/categories) plus static
    schema tokens, so it tracks whatever vocabulary the environment was generated with.
    """

    terms: tuple[str, ...]

    @classmethod
    def from_db_config(cls, db: DbConfig) -> EntityVocab:
        """Assemble the vocabulary from config vocabularies and the static schema tokens."""
        terms: set[str] = set(_SCHEMA_TOKENS)
        for group in (
            db.segments,
            db.countries,
            db.channels,
            db.marketing_channels,
            db.categories,
        ):
            terms.update(t.lower() for t in group)
        return cls(terms=tuple(sorted(terms)))


def _clip01(value: float) -> float:
    """Clamp to ``[0, 1]`` (features are bounded so weights stay interpretable)."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


class Featurizer:
    """Map a question string to a fixed-width ``float64`` feature vector.

    The width and the (stable) column names are fixed at construction from the config, so every
    logged :class:`~specialist_router.env.records.RouterDecision` records a consistent
    ``feature_dim``/``feature_names``.
    """

    def __init__(self, config: FeaturesConfig, vocab: EntityVocab) -> None:
        """Bind the featurizer to its config and entity vocabulary."""
        self._config = config
        self._vocab = vocab
        self._heuristic_names = self._build_heuristic_names()
        self._embed_names = [f"embed_{i}" for i in range(config.embed_dim)]
        self._minilm_model: object | None = None  # lazily constructed on the minilm path

    @staticmethod
    def _build_heuristic_names() -> list[str]:
        base = [
            "bias",
            "char_len",
            "token_len",
            "n_numeric",
            "n_entities",
            "has_date_window",
            "n_date_tokens",
            "has_topk_marker",
            "has_ratio_marker",
        ]
        return base + [f"bucket_{b}" for b in _BUCKETS]

    @property
    def feature_names(self) -> list[str]:
        """Stable column names, ``len == feature_dim`` (heuristics then embedding dims)."""
        return self._heuristic_names + self._embed_names

    @property
    def feature_dim(self) -> int:
        """Total feature width (heuristic block + embedding block)."""
        return len(self._heuristic_names) + self._config.embed_dim

    def transform(self, question: str) -> npt.NDArray[np.float64]:
        """Featurize one question into a ``(feature_dim,)`` vector."""
        heuristics = self._heuristics(question)
        embedding = self._embedding(question)
        return np.concatenate([heuristics, embedding]).astype(np.float64)

    def _heuristics(self, question: str) -> npt.NDArray[np.float64]:
        cfg = self._config
        lower = question.lower()
        tokens = _WORD_RE.findall(question)
        n_numeric = len(_NUMERIC_RE.findall(question))
        n_entities = sum(lower.count(term) for term in self._vocab.terms)
        n_years = len(_YEAR_RE.findall(question))
        has_month = 1.0 if _MONTH_RE.search(question) else 0.0
        n_date_tokens = n_years + int(has_month)
        has_date_window = 1.0 if (n_years > 0 or has_month) else 0.0

        values = [
            1.0,  # bias
            _clip01(len(question) / cfg.len_norm_chars),
            _clip01(len(tokens) / cfg.len_norm_tokens),
            _clip01(n_numeric / cfg.count_norm),
            _clip01(n_entities / cfg.count_norm),
            has_date_window,
            _clip01(n_date_tokens / cfg.count_norm),
            1.0 if any(m in lower for m in _TOPK_MARKERS) else 0.0,
            1.0 if any(m in lower for m in _RATIO_MARKERS) else 0.0,
        ]
        values.extend(self._bucket_one_hot(lower))
        return np.array(values, dtype=np.float64)

    @staticmethod
    def _bucket_one_hot(lower: str) -> list[float]:
        """Assign one coarse question type by priority (ranking > edge > comparison > agg)."""
        if any(m in lower for m in _TOPK_MARKERS):
            bucket = "ranking"
        elif any(m in lower for m in _EDGE_MARKERS):
            bucket = "edge"
        elif any(m in lower for m in _COMPARISON_MARKERS):
            bucket = "comparison"
        else:
            bucket = "aggregation"
        return [1.0 if b == bucket else 0.0 for b in _BUCKETS]

    def _embedding(self, question: str) -> npt.NDArray[np.float64]:
        if self._config.embedding == "minilm":
            return self._minilm_embedding(question)
        return self._hashing_embedding(question)

    def _hashing_embedding(self, question: str) -> npt.NDArray[np.float64]:
        """Deterministic signed char-n-gram hashing embedding, L2-normalized.

        Each character n-gram is hashed (blake2b) to a bucket and a sign; collisions are benign
        at this scale. L2 normalization makes the block scale-comparable to the bounded
        heuristics. Uses ``hashlib`` so the mapping is stable across processes and runs.
        """
        dim = self._config.embed_dim
        vec = np.zeros(dim, dtype=np.float64)
        text = question.lower()
        for n in range(self._config.ngram_min, self._config.ngram_max + 1):
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                gram = text[i : i + n].encode("utf-8")
                digest = hashlib.blake2b(gram, digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % dim
                sign = 1.0 if (digest[4] & 1) else -1.0
                vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec

    def _minilm_embedding(self, question: str) -> npt.NDArray[np.float64]:
        """Sentence-transformers MiniLM embedding, projected to ``embed_dim`` (optional path).

        Not used in CI or ``repro-phase2``. Requires ``sentence-transformers`` (a manual install;
        it is intentionally not a locked extra so torch stays out of the Phase-2 lock — ADR-006).
        A fixed seeded Gaussian random projection maps the 384-d model output to ``embed_dim`` so
        the feature width matches the hashing path.
        """
        model = self._ensure_minilm()
        raw = np.asarray(model.encode([question])[0], dtype=np.float64)  # type: ignore[attr-defined]
        dim = self._config.embed_dim
        if raw.shape[0] == dim:
            projected = raw
        else:
            rng = np.random.default_rng(0)
            projection = rng.standard_normal((raw.shape[0], dim)) / np.sqrt(dim)
            projected = raw @ projection
        norm = float(np.linalg.norm(projected))
        return projected / norm if norm > 0.0 else projected

    def _ensure_minilm(self) -> object:
        if self._minilm_model is None:
            from sentence_transformers import SentenceTransformer  # lazy: optional dependency

            self._minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._minilm_model
