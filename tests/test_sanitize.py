"""Secret scanner and personal-data scrub tests.

Every "secret" here is a made-up value in the shape of a real one.
"""

from __future__ import annotations

import pytest

from slack2rag.sanitize import (
    REDACTED,
    find_personal,
    scan_secrets,
    scrub_personal,
    shannon_entropy,
)

FAKE_SLACK = "xoxb-" + "1234567890-abcdefghij"  # split so scanners skip it
FAKE_KEY = (
    "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEow\n-----END RSA " + "PRIVATE KEY-----"
)


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (f"token {FAKE_SLACK} ok", "slack-token"),
        (FAKE_KEY, "private-key"),
        ("AKIA" + "ABCDEFGHIJKLMNOP", "aws-access-key"),
        ("ghp_" + "a" * 36, "github-token"),
        ("sk-" + "proj-" + "x" * 24, "openai-key"),
        ("https://hooks.slack.com/services/T0/B0/xyz", "slack-webhook"),
        ("postgres://admin:pa55@db/x", "url-credentials"),
        ("the password: hunter2", "password-assignment"),
        ("door code is 4471", "code-assignment"),
        ("eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.abc", "jwt"),
    ],
)
def test_rules(text: str, rule: str) -> None:
    result = scan_secrets(text)
    assert rule in result.rules
    assert result.flagged
    assert REDACTED in result.text


def test_entropy_catches_random_strings_but_not_words() -> None:
    gibberish = "Zq8vL2mX9pR4" + "tY7wB3nK6sD1"  # split so scanners skip it
    assert scan_secrets(f"key {gibberish}").rules == ("high-entropy",)
    assert not scan_secrets("internationalization_is_a_long_word").flagged
    assert not scan_secrets("https://example.org/a/very/long/path/to/a/manual").flagged
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0


def test_ordinary_text_is_untouched() -> None:
    text = "The wifi key is posted by the door. The token machine takes quarters."
    result = scan_secrets(text)
    assert result.text == text
    assert not result.flagged


def test_scrub_personal() -> None:
    text = (
        "Email bob@example.org or call (503) 555-1234; ask @carol or <@U123>,"
        " or Alice Smith's friend alice. Ping @here."
    )
    out = scrub_personal(text, ["Alice Smith", "alice", "Al"])
    assert "bob@" not in out
    assert "555" not in out
    assert "@carol" not in out
    assert "Alice" not in out
    assert "alice" not in out
    assert "@here" in out
    assert out.count("[member]") == 4


def test_find_personal() -> None:
    assert find_personal("plain text", ["Alice"]) == []
    assert find_personal("x@y.org 503-555-1234 @dave Alice", ["Alice"]) == [
        "email",
        "phone",
        "handle",
        "name",
    ]
    assert find_personal("@channel", []) == []
