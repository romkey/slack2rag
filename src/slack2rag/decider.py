"""
Decision model: asks a chat-completions LLM whether a Slack message (or a
whole thread) is a useful resource worth indexing for RAG search.

Every message is still stored in PostgreSQL; only messages the decision
model accepts are embedded and stored in Qdrant.  The operator's
DECISION_PROMPT says what counts as useful; this module appends the
message and the JSON answer format, so the prompt can't break parsing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from .evaluator import _chat_completion

logger = logging.getLogger(__name__)

# The text sent for one message or thread is cut to this many characters,
# which bounds the cost of a very long thread; the start of a thread
# (the question and first answers) carries the decision.
MAX_INPUT_CHARS = 8_000

_ANSWER_FORMAT = (
    "Respond with JSON only, in exactly this form: "
    '{"useful": true or false, "reason": "<one short sentence>"}'
)

_YES = ("true", "yes", "y", "1", "useful")
_NO = ("false", "no", "n", "0", "not useful")


class DecisionError(RuntimeError):
    """The decision model could not be reached, so nothing was decided."""


@dataclass(frozen=True)
class Decision:
    useful: bool
    reason: str


def _parse_useful(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _YES:
            return True
        if text in _NO:
            return False
    return None


def parse_decision(response_text: str) -> Decision | None:
    """Parse the model's JSON answer; None when it isn't usable."""
    try:
        result: Any = json.loads(response_text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    useful = _parse_useful(result.get("useful"))
    if useful is None:
        return None
    return Decision(useful=useful, reason=str(result.get("reason", "")).strip())


class DecisionModel:
    """Decides, one message or thread at a time, what goes into Qdrant."""

    def __init__(
        self,
        base_url: str,
        model: str,
        prompt: str,
        api_key: str = "",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.prompt = prompt
        self.api_key = api_key
        self.timeout = timeout

    def build_prompt(self, text: str, channel_name: str = "") -> str:
        if len(text) > MAX_INPUT_CHARS:
            text = text[:MAX_INPUT_CHARS] + "\n[…truncated]"
        where = f"Slack channel: #{channel_name}\n\n" if channel_name else ""
        return f"{self.prompt}\n\n{where}Message:\n{text}\n\n{_ANSWER_FORMAT}"

    @retry(
        retry=retry_if_exception_type(DecisionError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        reraise=True,
    )
    def _ask(self, prompt: str) -> str:
        content = _chat_completion(
            self.base_url, self.api_key, self.model, prompt, timeout=self.timeout
        )
        if content is None:
            raise DecisionError(
                f"decision model {self.model!r} at {self.base_url} gave no answer"
            )
        return content

    def decide(self, text: str, channel_name: str = "") -> Decision:
        """Return whether *text* is worth indexing.

        Raises DecisionError when the model can't be reached (after
        retries), so the sync cycle fails and the channel cursor doesn't
        move past undecided messages.  An answer that can't be parsed
        counts as useful: indexing some noise beats losing knowledge.
        """
        content = self._ask(self.build_prompt(text, channel_name))
        decision = parse_decision(content)
        if decision is None:
            logger.warning(
                "Decision model gave an unparseable answer; indexing anyway: %.200s",
                content.strip().replace("\n", " "),
            )
            return Decision(useful=True, reason="unparseable decision; kept")
        return decision
