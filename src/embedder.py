"""
Generates dense vector embeddings via an OpenAI-compatible inference
endpoint, with optional sparse encoding for hybrid search.

Dense embeddings:
  POST {base_url}/embeddings — works with OpenAI itself, Ollama's /v1
  compatibility layer, llama.cpp's server, vLLM, LM Studio, and any
  other OpenAI-compatible provider.  No PyTorch or heavy ML libraries
  needed in this process.

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


# ── OpenAI-compatible embedder ────────────────────────────────────────────────

_CONTEXT_ERROR_FRAGMENTS = (
    "context length", "context window", "too long", "exceeds",
    "maximum context", "maximum tokens", "maximum_context",
    "context_length_exceeded",
)
_MIN_TEXT_LEN = 64


def _is_context_length_error(http_exc: urllib.error.HTTPError) -> bool:
    """Return True if the HTTP error is the server complaining about input length."""
    if http_exc.code not in (400, 413, 422):
        return False
    try:
        body = http_exc.read().decode(errors="replace").lower()
        return any(frag in body for frag in _CONTEXT_ERROR_FRAGMENTS)
    except Exception:
        return False


def _truncate_at_word(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*, cutting at a word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut


class Embedder:
    """Generate dense embeddings via an OpenAI-compatible inference server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        context_length: int = 0,
        api_key: str = "",
        input_prefix: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._input_prefix = input_prefix
        self._dimension: int | None = None
        self._max_chars = context_length * 3 if context_length > 0 else 0

        logger.info("Connecting to LLM at %s  embedding model: %s", self._base_url, model)
        if self._input_prefix:
            logger.info("  embedding input prefix: %r", self._input_prefix)
        if self._max_chars:
            logger.info("  context limit: %d tokens (~%d chars)", context_length, self._max_chars)
        try:
            probe = self._call_api(self._prepare_texts(["dimension probe"]))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to connect to LLM at {self._base_url} with model {model!r}: {exc}"
            ) from exc
        self._dimension = len(probe[0])
        logger.info("Embedding dimension: %d", self._dimension)

    @property
    def dimension(self) -> int:
        assert self._dimension is not None
        return self._dimension

    # ── public entry point ─────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of float vectors, one per input text.

        Pre-truncates texts to the configured context limit, and
        automatically retries with shorter input if the server still
        rejects any text for exceeding its context window.
        """
        if not texts:
            return []

        texts = self._prepare_texts(texts)

        try:
            return self._call_api(texts)
        except _ContextLengthError:
            logger.warning(
                "Batch of %d texts hit context-length limit — "
                "falling back to one-at-a-time with auto-truncation",
                len(texts),
            )
            return [self._embed_single(t) for t in texts]

    # ── internals ──────────────────────────────────────────────────────────

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """Apply model-specific input formatting and context truncation."""
        if self._input_prefix:
            texts = [f"{self._input_prefix}{text}" for text in texts]
        if self._max_chars:
            texts = [_truncate_at_word(t, self._max_chars) for t in texts]
        return texts

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text, halving it on each retry if it's too long."""
        attempt = 0
        current = text
        while True:
            try:
                return self._call_api([current])[0]
            except _ContextLengthError:
                attempt += 1
                new_len = len(current) // 2
                if new_len < _MIN_TEXT_LEN:
                    logger.error(
                        "Text still too long after %d truncation attempts "
                        "(%d chars).  Giving up.  Preview: %.200s",
                        attempt, len(current), text,
                    )
                    raise EmbeddingError(
                        f"Cannot embed text even at {len(current)} chars "
                        f"(model {self._model!r})"
                    )
                current = _truncate_at_word(text, new_len)
                logger.warning(
                    "Context-length error (attempt %d) — truncating to %d "
                    "chars and retrying.  Preview: %.120s",
                    attempt, len(current), text,
                )

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call the OpenAI-compatible /embeddings endpoint.

        Raises _ContextLengthError for context-window rejections,
        EmbeddingError for everything else.
        """
        endpoint = f"{self._base_url}/embeddings"
        results: List[List[float]] = []

        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            payload = json.dumps({
                "model": self._model,
                "input": batch,
            }).encode()
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers=headers,
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if _is_context_length_error(exc):
                    raise _ContextLengthError(
                        f"Input exceeds context length for model {self._model!r}"
                    ) from exc
                body = ""
                try:
                    body = exc.read().decode(errors="replace").strip()
                except Exception:
                    pass
                raise EmbeddingError(
                    f"LLM returned HTTP {exc.code} ({exc.reason}) "
                    f"for POST {endpoint} with model {self._model!r}"
                    + (f"\n  Response: {body}" if body else "")
                ) from exc
            except urllib.error.URLError as exc:
                raise EmbeddingError(
                    f"Could not reach LLM at {self._base_url}: {exc.reason}"
                ) from exc

            try:
                items = data["data"]
                vectors = [item["embedding"] for item in items]
            except (KeyError, TypeError) as exc:
                raise EmbeddingError(
                    f"Unexpected response shape from {endpoint}: "
                    f"{json.dumps(data)[:300]}"
                ) from exc
            results.extend(vectors)
        return results


class _ContextLengthError(EmbeddingError):
    """Internal: server rejected input as too long for the model's context."""
