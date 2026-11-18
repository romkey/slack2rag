"""Chat-completions client for the OpenAI-compatible model gateway.

Every ingest-time model call (relevance check, fact extraction, exclusion
check) goes through chat_json(), which asks for a JSON object and retries
transport failures. Model names are configuration, never code.
"""

from __future__ import annotations

import contextlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The chat model could not be reached or gave no usable answer."""


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    user_content: str,
    system: str = "",
    timeout: int = 120,
) -> str | None:
    """POST /chat/completions and return the assistant's content.

    Returns None on any HTTP/transport error or unexpected response
    (already logged). The request body is never logged: it holds Slack
    text that must not be persisted.
    """
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    # urlopen also accepts file: and other schemes; only HTTP(S) makes sense.
    if urllib.parse.urlsplit(endpoint).scheme not in ("http", "https"):
        logger.error("Chat base URL must start with http:// or https://: %r", base_url)
        return None
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "stream": False,
        }
    ).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # S310: the URL is the operator's configured base URL, and the check
    # above has already refused any scheme other than http/https.
    req = urllib.request.Request(endpoint, data=payload, headers=headers)  # noqa: S310

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data: Any = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = ""
        with contextlib.suppress(Exception):
            body = exc.read().decode(errors="replace").strip()[:300]
        logger.error(
            "Chat model returned HTTP %d for %s (model %r): %s",
            exc.code,
            endpoint,
            model,
            body,
        )
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.error("Could not get an answer from %s: %s", base_url, exc)
        return None

    try:
        content: str = data["choices"][0]["message"]["content"]
    except KeyError, IndexError, TypeError:
        logger.error("Unexpected chat-completions response shape from %s", endpoint)
        return None
    return content


class ChatModel:
    """One configured chat model on the gateway."""

    def __init__(
        self, base_url: str, model: str, api_key: str = "", timeout: int = 180
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @retry(
        retry=retry_if_exception_type(LLMError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        reraise=True,
    )
    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        """Ask for a JSON object; raise LLMError if none comes back after retries."""
        content = chat_completion(
            self.base_url,
            self.api_key,
            self.model,
            user,
            system=system,
            timeout=self.timeout,
        )
        if content is None:
            raise LLMError(
                f"chat model {self.model!r} at {self.base_url} gave no answer"
            )
        try:
            result: Any = json.loads(_strip_fences(content))
        except json.JSONDecodeError:
            raise LLMError(
                f"chat model {self.model!r} did not answer in JSON"
            ) from None
        if not isinstance(result, dict):
            raise LLMError(f"chat model {self.model!r} answered with non-object JSON")
        return result


def _strip_fences(text: str) -> str:
    """Remove a ```json ... ``` wrapper some models add despite the format."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.removesuffix("```").strip()
    return text
