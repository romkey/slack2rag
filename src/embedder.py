"""
Generates dense vector embeddings via Ollama, with optional sparse encoding
for hybrid search.

Dense embeddings:
  Ollama REST API (/api/embed) — uses whatever model is pulled on the
  Ollama server.  No PyTorch or heavy ML libraries needed in this process.

Sparse encoding:
  A lightweight BM25-like tokenizer that maps terms to hashed bucket
  indices with log-TF weighting.  No external dependencies required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import urllib.request
import urllib.error
from collections import Counter
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend returns an error."""


# ── Sparse encoder ────────────────────────────────────────────────────────────

SPARSE_VOCAB_SIZE = 2**16  # 65 536 hash buckets — low collision for typical text

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "this", "that", "these", "those", "it", "its", "i",
    "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _token_hash(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % SPARSE_VOCAB_SIZE


def tokenize_text(text: str) -> List[str]:
    """Tokenize *text* into lower-case non-stopword terms (len > 1).

    Useful for both sparse encoding and term-frequency analysis.
    """
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOPWORDS and len(t) > 1]


class SparseEncoder:
    """Deterministic BM25-like sparse vector encoder (no training required)."""

    def encode(self, texts: List[str]) -> List[dict]:
        """Return a list of dicts with 'indices' and 'values' keys."""
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> dict:
        tokens = tokenize_text(text)
        if not tokens:
            return {"indices": [0], "values": [0.0]}

        counts = Counter(tokens)
        bucket_weights: dict[int, float] = {}
        for token, count in counts.items():
            idx = _token_hash(token)
            bucket_weights[idx] = bucket_weights.get(idx, 0.0) + 1.0 + math.log(count)

        pairs = sorted(bucket_weights.items())
        return {
            "indices": [p[0] for p in pairs],
            "values": [p[1] for p in pairs],
        }


# ── Ollama embedder ──────────────────────────────────────────────────────────

class Embedder:
    """Generate dense embeddings via an Ollama server."""

    def __init__(self, url: str, model: str) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._dimension: int | None = None

        logger.info("Connecting to Ollama at %s  model: %s", self._url, model)
        try:
            probe = self._embed(["dimension probe"])
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to connect to Ollama at {self._url} with model {model!r}: {exc}"
            ) from exc
        self._dimension = len(probe[0])
        logger.info("Ollama embedding dimension: %d", self._dimension)

    @property
    def dimension(self) -> int:
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of float vectors, one per input text."""
        if not texts:
            return []
        return self._embed(texts)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Call Ollama's /api/embed endpoint (supports batched input)."""
        endpoint = f"{self._url}/api/embed"
        results: List[List[float]] = []
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            payload = json.dumps({
                "model": self._model,
                "input": batch,
            }).encode()
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode(errors="replace").strip()
                except Exception:
                    pass
                raise EmbeddingError(
                    f"Ollama returned HTTP {exc.code} ({exc.reason}) "
                    f"for POST {endpoint} with model {self._model!r}"
                    + (f"\n  Response: {body}" if body else "")
                ) from exc
            except urllib.error.URLError as exc:
                raise EmbeddingError(
                    f"Could not reach Ollama at {self._url}: {exc.reason}"
                ) from exc

            results.extend(data["embeddings"])
        return results
