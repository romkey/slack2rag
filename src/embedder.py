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

    # Some OpenAI-compatible servers (notably Ollama's /v1 layer) accept an
    # array input but silently return null embeddings for each item.  When
    # we detect that, we drop to one-request-per-input mode for the rest
    # of this Embedder's lifetime.
    def __init__(
        self,
        base_url: str,
        model: str,
        context_length: int = 0,
        api_key: str = "",
        input_prefix: str = "",
        request_batch_size: int = 50,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._input_prefix = input_prefix
        self._dimension: int | None = None
        self._max_chars = context_length * 3 if context_length > 0 else 0
        self._request_batch_size = max(1, request_batch_size)
        # Servers known to mishandle array inputs flip this to True after
        # the first bad response and we send one input per request afterwards.
        self._force_single_input = False

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
        results: List[List[float]] = []

        chunk_size = 1 if self._force_single_input else self._request_batch_size
        i = 0
        while i < len(texts):
            batch = texts[i : i + chunk_size]
            try:
                vectors = self._post_embeddings(batch)
            except _MalformedBatchError:
                # Server returned the right shape but null/invalid embeddings.
                # Drop to single-input mode and retry this batch one at a time.
                if len(batch) == 1:
                    raise EmbeddingError(
                        f"LLM at {self._base_url} returned an invalid embedding "
                        f"for a single input with model {self._model!r}.  This "
                        "usually means the model isn't an embedding model, the "
                        "input was empty, or the server is misconfigured."
                    )
                logger.warning(
                    "LLM returned null/invalid embeddings for a batch of %d; "
                    "falling back to one-input-per-request for the rest of "
                    "this run (some OpenAI-compatible servers, notably "
                    "Ollama's /v1, mishandle array inputs).",
                    len(batch),
                )
                self._force_single_input = True
                chunk_size = 1
                continue  # retry this same slice in single-input mode
            results.extend(vectors)
            i += len(batch)
        return results

    def _post_embeddings(self, batch: List[str]) -> List[List[float]]:
        """Send one HTTP request and return validated embeddings.

        Raises _ContextLengthError for context errors, _MalformedBatchError
        when the response shape is right but values are null/invalid, and
        EmbeddingError for anything else.
        """
        endpoint = f"{self._base_url}/embeddings"
        payload = json.dumps({
            "model": self._model,
            "input": batch,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(endpoint, data=payload, headers=headers)

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

        return self._extract_embeddings(data, expected=len(batch), endpoint=endpoint)

    def _extract_embeddings(
        self, data: object, expected: int, endpoint: str,
    ) -> List[List[float]]:
        """Pull embedding vectors out of an OpenAI-shaped response.

        Sorts items by their `index` field so callers see the same order
        as the input regardless of how the server orders the response.
        Raises _MalformedBatchError if any item is missing/null/invalid.
        """
        if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
            raise EmbeddingError(
                f"Unexpected response shape from {endpoint}: "
                f"{json.dumps(data)[:300]}"
            )
        items = data["data"]
        if len(items) != expected:
            raise EmbeddingError(
                f"{endpoint} returned {len(items)} embedding(s) for {expected} "
                f"input(s).  Response: {json.dumps(data)[:300]}"
            )

        ordered: List[List[float] | None] = [None] * expected
        for slot, item in enumerate(items):
            if not isinstance(item, dict):
                raise _MalformedBatchError()
            idx = item.get("index", slot)
            if not isinstance(idx, int) or idx < 0 or idx >= expected:
                idx = slot
            embedding = item.get("embedding")
            if not _is_valid_embedding(embedding):
                raise _MalformedBatchError()
            ordered[idx] = embedding

        if any(v is None for v in ordered):
            raise _MalformedBatchError()
        return ordered  # type: ignore[return-value]


def _is_valid_embedding(value: object) -> bool:
    """A valid embedding is a non-empty list of finite numbers."""
    if not isinstance(value, list) or not value:
        return False
    for x in value:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return False
    return True


class _ContextLengthError(EmbeddingError):
    """Internal: server rejected input as too long for the model's context."""


class _MalformedBatchError(Exception):
    """Internal: response had the right shape but invalid embedding values."""
