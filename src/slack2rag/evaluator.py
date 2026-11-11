"""
Message quality evaluator using an OpenAI-compatible chat-completions LLM.

Sends each Slack message along with a user-supplied prompt to an LLM and
expects a JSON response with ``score`` (1–10) and ``reason``.  Messages
scoring >= 7 are written to /results/good.txt; the rest go to
/results/bad.txt.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import List, Optional

from .pg_store import PgStore

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\b(\d{1,2})\b")
GOOD_THRESHOLD = 7
RESULTS_DIR = "/results"


def _chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    user_content: str,
    timeout: int = 120,
) -> Optional[str]:
    """Call /chat/completions and return the assistant's content string.

    Returns None on any HTTP/transport error (already logged).
    """
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": False,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=payload, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode(errors="replace").strip()
        except Exception:
            pass
        logger.error(
            "LLM returned HTTP %d for %s (model %r): %s",
            exc.code, endpoint, model, body,
        )
        return None
    except urllib.error.URLError as exc:
        logger.error("Could not reach LLM at %s: %s", base_url, exc.reason)
        return None

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.error(
            "Unexpected chat-completions response shape from %s: %s",
            endpoint, json.dumps(data)[:300],
        )
        return None


def score_message(
    base_url: str, model: str, prompt: str, message: str,
    api_key: str = "",
) -> tuple[int | None, str]:
    """Ask the LLM to score *message* using *prompt*.

    Returns (score, reason).  score is None if the response couldn't
    be parsed; reason may be empty.
    """
    full_prompt = (
        f"{prompt}\n\n"
        f"Message:\n{message}\n\n"
        'Respond with JSON: {"score": <1-10>, "reason": "<brief reason>"}'
    )

    response_text = _chat_completion(base_url, api_key, model, full_prompt)
    if response_text is None:
        return None, ""
    response_text = response_text.strip()

    try:
        result = json.loads(response_text)
        score = int(result["score"])
        reason = str(result.get("reason", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "JSON parse failed (%s).  Raw model response: %s",
            exc, response_text,
        )
        match = _SCORE_RE.search(response_text)
        if not match:
            logger.warning(
                "Could not extract a numeric score either — skipping",
            )
            return None, ""
        score = int(match.group(1))
        logger.info("Fell back to regex score extraction: %d", score)
        reason = ""

    if score > 10:
        logger.warning("Score %d out of range, clamping to 10", score)
        score = 10
    return score, reason


def _format_line(
    score: int,
    reason: str,
    channel_name: str,
    user_name: str,
    ts: str,
    text: str,
) -> str:
    preview = text.replace("\n", " ")
    if len(preview) > 300:
        preview = preview[:300] + "…"
    line = f"[{score:>2}] #{channel_name}  @{user_name}  ts={ts}  {preview}"
    if reason:
        line += f"\n     reason: {reason}"
    return line + "\n"


def run_eval(
    base_url: str,
    model: str,
    prompt: str,
    channels: List[dict],
    slack,
    api_key: str = "",
    pg: Optional[PgStore] = None,
) -> None:
    """Score every message in every channel and write results to files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    good_path = os.path.join(RESULTS_DIR, "good.txt")
    bad_path = os.path.join(RESULTS_DIR, "bad.txt")

    logger.info("Eval mode: scoring messages with model %r at %s", model, base_url)
    logger.info("  Prompt: %.200s", prompt)
    logger.info("  Threshold: >= %d → %s, < %d → %s",
                GOOD_THRESHOLD, good_path, GOOD_THRESHOLD, bad_path)

    total = 0
    good_count = 0
    bad_count = 0
    errors = 0

    with open(good_path, "w") as good_f, open(bad_path, "w") as bad_f:
        for ch in channels:
            channel_id = ch["id"]
            channel_name = ch.get("name", channel_id)
            logger.info("Eval: scanning #%s …", channel_name)

            if pg:
                pg.upsert_channel(ch)

            for msg in slack.get_channel_messages(channel_id, oldest_ts=None):
                if pg:
                    pg.upsert_message(channel_id, msg)
                    uid = msg.get("user")
                    if uid and isinstance(uid, str):
                        rec = slack.get_user_record(uid)
                        if rec:
                            pg.upsert_user(rec)

                raw_text = msg.get("text", "").strip()
                if not raw_text:
                    continue

                resolved_text = slack.resolve_text(raw_text)
                user_id = msg.get("user", "unknown")
                user_name = slack.get_user_name(user_id)
                ts = msg["ts"]

                score, reason = score_message(base_url, model, prompt, resolved_text, api_key=api_key)
                total += 1

                if score is None:
                    errors += 1
                    logger.warning(
                        "Eval: no score for message ts=%s in #%s — skipping.  "
                        "Preview: %.120s",
                        ts, channel_name, resolved_text,
                    )
                    continue

                line = _format_line(score, reason, channel_name, user_name, ts, resolved_text)

                if score >= GOOD_THRESHOLD:
                    good_f.write(line)
                    good_count += 1
                else:
                    bad_f.write(line)
                    bad_count += 1

                if total % 100 == 0:
                    logger.info(
                        "Eval progress: %d scored (%d good, %d bad, %d errors)",
                        total, good_count, bad_count, errors,
                    )

    logger.info(
        "Eval complete: %d messages scored — %d good (>=%d), %d bad, %d errors",
        total, good_count, GOOD_THRESHOLD, bad_count, errors,
    )
    logger.info("  Results: %s  %s", good_path, bad_path)
