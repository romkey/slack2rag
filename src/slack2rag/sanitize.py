"""Deterministic sanitization: secret scanning and personal-data scrubbing.

scan_secrets() runs on raw Slack text before any model sees it. It uses
rules in the style of gitleaks (known token formats, private keys,
credentials in URLs, password assignments) plus a Shannon-entropy check
for long random-looking strings. A hit is replaced with [REDACTED] and
reported by rule name only, never by value.

scrub_personal() runs on extracted fact text: emails, phone numbers, Slack
mentions and handles, and the names of workspace members (held in memory
only) become neutral placeholders.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

REDACTED = "[REDACTED]"

_SECRET_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "private-key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
            r"(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
            re.DOTALL,
        ),
    ),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("slack-app-token", re.compile(r"\bxapp-\d-[A-Za-z0-9-]{10,}")),
    (
        "slack-webhook",
        re.compile(r"https://hooks\.slack\.com/(?:services|workflows)/[A-Za-z0-9/_-]+"),
    ),
    (
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{40,})"),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("stripe-key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"),
    ),
    ("url-credentials", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@[^\s]+")),
    (
        "password-assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|passcode|passphrase)\b\s*(?:is|=|:)\s*\S+"
        ),
    ),
    (
        # A code-like value (it has a digit) after a word that names a secret.
        "code-assignment",
        re.compile(
            r"(?i)\b(?:pin|secret|api[_ -]?key|token|door code|gate code|lock code"
            r"|wifi key|wi-fi key|combo|combination|code)\b\s*(?:is|=|:)\s*"
            r"[\"'`]?[^\s\"'`]*\d[^\s\"'`]*"
        ),
    ),
]

_CANDIDATE = re.compile(r"[A-Za-z0-9+/=_-]{24,}")
_ENTROPY_THRESHOLD = 4.0  # bits per character


@dataclass(frozen=True)
class ScanResult:
    text: str  # the input with every hit replaced by REDACTED
    rules: tuple[str, ...]  # rule name of each hit, in order found

    @property
    def flagged(self) -> bool:
        return bool(self.rules)


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _looks_random(token: str) -> bool:
    """A long token mixing character classes with high entropy."""
    if token.startswith(("http", "www")) or token.isalpha() or token.isdigit():
        return False
    classes = sum(
        (
            any(c.islower() for c in token),
            any(c.isupper() for c in token),
            any(c.isdigit() for c in token),
        )
    )
    return classes >= 2 and shannon_entropy(token) >= _ENTROPY_THRESHOLD


def scan_secrets(text: str) -> ScanResult:
    """Redact everything that looks like a credential or secret."""
    rules: list[str] = []
    for name, pattern in _SECRET_RULES:
        text, count = pattern.subn(REDACTED, text)
        rules.extend([name] * count)

    def entropy_hit(match: re.Match[str]) -> str:
        if _looks_random(match.group(0)):
            rules.append("high-entropy")
            return REDACTED
        return match.group(0)

    text = _CANDIDATE.sub(entropy_hit, text)
    return ScanResult(text=text, rules=tuple(rules))


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)")
_SLACK_MENTION = re.compile(r"<@[A-Z0-9]+(?:\|[^>]*)?>")
_HANDLE = re.compile(r"(?<![\w/.])@[A-Za-z0-9_-][A-Za-z0-9._-]*[A-Za-z0-9_-]")
_ALLOWED_HANDLES = frozenset({"@here", "@channel", "@everyone"})


def scrub_personal(text: str, names: Iterable[str] = ()) -> str:
    """Replace personal data with placeholders.

    *names* are workspace members' display and real names, matched as whole
    words, case-insensitively, longest first.
    """
    text = _EMAIL.sub("[email]", text)
    text = _SLACK_MENTION.sub("[member]", text)
    text = _HANDLE.sub(
        lambda m: m.group(0) if m.group(0).lower() in _ALLOWED_HANDLES else "[member]",
        text,
    )
    text = _PHONE.sub("[phone]", text)
    for name in sorted({n.strip() for n in names if len(n.strip()) >= 3}, key=len)[
        ::-1
    ]:
        text = re.sub(
            rf"(?<!\w){re.escape(name)}(?:'s)?(?!\w)", "[member]", text, flags=re.I
        )
    return text


def find_personal(text: str, names: Iterable[str] = ()) -> list[str]:
    """Kinds of personal data still present in *text* (for spot checks)."""
    found = []
    if _EMAIL.search(text):
        found.append("email")
    if _PHONE.search(text):
        found.append("phone")
    if _SLACK_MENTION.search(text) or any(
        m.group(0).lower() not in _ALLOWED_HANDLES for m in _HANDLE.finditer(text)
    ):
        found.append("handle")
    lowered = text.lower()
    for name in names:
        name = name.strip()
        if len(name) >= 3 and re.search(
            rf"(?<!\w){re.escape(name.lower())}(?!\w)", lowered
        ):
            found.append("name")
            break
    return found
