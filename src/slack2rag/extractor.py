"""
Fact extraction: Slack is mined for facts, not archived.

For each conversation unit, in order:

1. Secret scan the raw text (deterministic); hits are redacted and
   reported for the quarantine log by rule name only.
2. Relevance check (model): does the unit answer a question about the
   space, state a fact about it, or point to a resource? If not, stop.
3. Extraction (model) into zero or more standalone facts, written in a
   neutral voice with no names, quotes or handles.
4. Exclusion check (model) on each fact: facts about individuals,
   incidents, credentials, admin procedures or access-path internals are
   dropped; known weaknesses stay, tagged ``weakness``.
5. Personal-data scrub (deterministic) on the fact text.

The facts, never the conversation, are what get stored and indexed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .llm import ChatModel, LLMError
from .sanitize import REDACTED, scan_secrets, scrub_personal

logger = logging.getLogger(__name__)

KINDS = (
    "location",
    "howto",
    "policy",
    "resource",
    "status",
    "announcement",
    "contact",
    "history",
)
CONFIDENCES = ("low", "medium", "high")

# How long a fact of each kind stays current, from the date it was said.
STALE_AFTER_DAYS: dict[str, int | None] = {
    "status": 7,
    "announcement": 30,
    "location": 365,
    "howto": 365,
    "contact": 365,
    "resource": None,
    "policy": None,
    "history": None,
}

MAX_INPUT_CHARS = 12_000
MAX_FACTS_PER_UNIT = 20
_URL = re.compile(r"https?://[^\s<>()\"']+")

RELEVANCE_SYSTEM = """\
You screen conversations from a hackerspace's Slack. Decide whether the \
conversation is worth mining for durable facts about the space. Answer \
relevant if it does at least one of these: answers a question about the \
space (its tools, equipment, rooms, materials, procedures, policies, events \
or services); states a fact about the space; or points to a resource (a \
link, document, supplier or channel) useful to members. Answer not relevant \
for social chat, greetings, jokes, thanks, plans for a single evening, \
questions nobody answered, and anything that only makes sense in the \
moment. Messages from "Integration" are automated announcements.
Respond with JSON only: {"relevant": true or false, "reason": "<one short sentence>"}"""

EXTRACTION_SYSTEM = """\
You turn a hackerspace Slack conversation into standalone facts for a \
knowledge base members search. Rules:
- Write every fact in a neutral voice. Never quote a message. Never name, \
describe or refer to who said something, and never include names, \
@handles, emails or phone numbers.
- Each fact must be useful to someone who never saw the conversation. If \
it isn't, don't emit it. Rewrite the question so it stands alone.
- When a thread corrects itself, use only the final, confirmed answer.
- "Ask so-and-so" becomes a contact fact naming a role or a channel, never \
a person.
- Every URL given as a resource becomes a resource fact whose answer \
describes it in one line and whose links include the URL exactly as \
written. Never invent a URL.
- Never include passwords, door or lock codes, keys, tokens or anything \
marked [REDACTED].
- Leave out anything about individual people, incidents (thefts, \
vandalism, disputes), admin procedures (changing, pairing, disabling or \
reconfiguring infrastructure) and the internals of door, gate, lock or \
RFID systems.
- confidence: high if the answer was confirmed or stated with authority, \
medium if plausible and uncontradicted, low if it was a guess or disputed.
- entities: the tools, machines, rooms, areas and events the fact is \
about, as named in the conversation.
Kinds: location (where something is), howto (how to do or use something), \
policy (a rule), resource (a link, document, supplier or channel), status \
(the current state of something), announcement (news or a change), \
contact (which role or channel to ask), history (how something came to \
be).
Respond with JSON only, in this form (an empty list when there are no facts):
{"facts": [{"kind": "...", "question": "...", "answer": "...", \
"entities": ["..."], "links": ["..."], "confidence": "low|medium|high"\
TOPICS_FIELD}]}"""

TOPICS_FIELD = ', "topics": ["..."]'
TOPICS_RULE = (
    "\n- topics: zero to three slugs from this list only, the best match first: "
)

EXCLUSION_SYSTEM = """\
You check facts before they go into a hackerspace's member knowledge base. \
For each fact decide whether it must be excluded because it: is about an \
individual person (who did, owns, said or is responsible for something, \
unless it names a publicly listed role); describes an incident (theft, \
vandalism, injury, dispute, ban); contains or hints at a credential or \
secret (passwords, codes, keys, tokens); is an admin procedure (changing, \
disabling, pairing, deleting or reconfiguring infrastructure, accounts or \
devices); or describes firmware, configuration or internals of anything in \
the access path (RFID readers, door and gate controllers, locks). Also say \
whether it describes a known weakness (something unreliable, broken or \
easy to get wrong); weaknesses are kept, not excluded.
Respond with JSON only: {"verdicts": [{"index": 0, "exclude": true or \
false, "category": "individual|incident|credential|admin|access-path|none", \
"weakness": true or false}]}"""


@dataclass(frozen=True)
class Fact:
    kind: str
    question: str
    answer: str
    confidence: str
    entities: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The fact as embedded and shown to a model."""
        return f"Q: {self.question}\nA: {self.answer}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass
class Extraction:
    relevant: bool
    reason: str
    facts: list[Fact] = field(default_factory=list)
    secret_rules: tuple[str, ...] = ()
    excluded: int = 0  # facts dropped by the exclusion check


def fact_doc_id(channel_id: str, unit_key: str, index: int) -> str:
    """The source document ID of the index-th fact of a unit."""
    return f"{channel_id}:{unit_key}:{index}"


def point_id(source: str, doc_id: str, chunk_index: int = 0) -> str:
    """uuid5(source:doc_id:chunk_index), stable so a re-sync overwrites."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{doc_id}:{chunk_index}"))


def fact_id(channel_id: str, unit_key: str, index: int) -> str:
    """Stable ID of a fact: its staging row and its Qdrant point."""
    return point_id("slack", fact_doc_id(channel_id, unit_key, index))


def stale_after(kind: str, said: datetime.date) -> datetime.datetime | None:
    days = STALE_AFTER_DAYS.get(kind)
    if days is None:
        return None
    start = datetime.datetime.combine(said, datetime.time(), tzinfo=datetime.UTC)
    return start + datetime.timedelta(days=days)


def load_topics(path: str) -> tuple[str, ...]:
    """Taxonomy slugs, one per line; blank lines and # comments ignored."""
    if not path:
        return ()
    from pathlib import Path

    slugs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            slugs.append(line)
    return tuple(slugs)


def _strings(value: Any, limit: int = 10) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out[:limit])


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "relevant")
    return bool(value) if isinstance(value, int | float) else False


class Extractor:
    def __init__(
        self,
        model: ChatModel,
        topics: Sequence[str] = (),
        member_names: Iterable[str] = (),
    ) -> None:
        self.model = model
        self.topics = tuple(topics)
        self.member_names = frozenset(member_names)
        system = EXTRACTION_SYSTEM.replace(
            "TOPICS_FIELD", TOPICS_FIELD if self.topics else ""
        )
        if self.topics:
            system = system.replace(
                "\nKinds:", TOPICS_RULE + ", ".join(self.topics) + "\nKinds:"
            )
        self.extraction_system = system

    @property
    def version(self) -> str:
        """Changes whenever a prompt, the model or the taxonomy changes.

        Units processed under another version are reprocessed.
        """
        material = json.dumps(
            [
                self.model.model,
                RELEVANCE_SYSTEM,
                self.extraction_system,
                EXCLUSION_SYSTEM,
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    # ── steps ─────────────────────────────────────────────────────────────

    def is_relevant(self, text: str, channel: str, boost: bool) -> tuple[bool, str]:
        note = (
            "\nThis was pinned or bookmarked in the channel, so members"
            " considered it important; lean towards relevant."
            if boost
            else ""
        )
        answer = self.model.chat_json(
            RELEVANCE_SYSTEM,
            f"Channel: #{channel}{note}\n\nConversation:\n{text}",
        )
        if "relevant" not in answer:
            raise LLMError("relevance answer has no 'relevant' field")
        return _truthy(answer["relevant"]), str(answer.get("reason", ""))[:300]

    def extract(self, text: str, channel: str) -> list[Fact]:
        answer = self.model.chat_json(
            self.extraction_system,
            f"Channel: #{channel}\n\nConversation:\n{text}",
        )
        raw = answer.get("facts")
        if not isinstance(raw, list):
            raise LLMError("extraction answer has no 'facts' list")
        facts = []
        for item in raw[:MAX_FACTS_PER_UNIT]:
            fact = self._validate(item, text)
            if fact is not None:
                facts.append(fact)
        return facts

    def _validate(self, item: Any, source: str) -> Fact | None:
        if not isinstance(item, dict):
            return None
        kind = str(item.get("kind", "")).strip().lower()
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if kind not in KINDS or not answer or not question:
            return None
        confidence = str(item.get("confidence", "low")).strip().lower()
        if confidence not in CONFIDENCES:
            confidence = "low"
        # A link must have been in the conversation; the model can't add one.
        links = tuple(link for link in _strings(item.get("links")) if link in source)
        topics = tuple(t for t in _strings(item.get("topics"), 3) if t in self.topics)
        return Fact(
            kind=kind,
            question=question[:500],
            answer=answer[:2000],
            confidence=confidence,
            entities=_strings(item.get("entities")),
            links=links,
            topics=topics,
        )

    def check_exclusions(self, facts: list[Fact]) -> list[Fact]:
        if not facts:
            return []
        listing = "\n".join(
            f"{i}. [{f.kind}] {f.question} — {f.answer}" for i, f in enumerate(facts)
        )
        answer = self.model.chat_json(EXCLUSION_SYSTEM, f"Facts:\n{listing}")
        verdicts = answer.get("verdicts")
        if not isinstance(verdicts, list):
            raise LLMError("exclusion answer has no 'verdicts' list")
        by_index: dict[int, dict[str, Any]] = {}
        for v in verdicts:
            if isinstance(v, dict) and isinstance(v.get("index"), int):
                by_index[v["index"]] = v
        kept = []
        for i, fact in enumerate(facts):
            verdict = by_index.get(i)
            # A fact the checker didn't rule on is dropped: fail closed.
            if verdict is None or _truthy(verdict.get("exclude", True)):
                continue
            if _truthy(verdict.get("weakness", False)):
                fact = Fact(**{**fact.__dict__, "tags": (*fact.tags, "weakness")})
            kept.append(fact)
        return kept

    def scrub(self, fact: Fact) -> Fact | None:
        """Personal-data scrub and a second secret scan on the fact itself."""
        question = scan_secrets(scrub_personal(fact.question, self.member_names))
        answer = scan_secrets(scrub_personal(fact.answer, self.member_names))
        if question.flagged or answer.flagged or REDACTED in fact.text:
            return None
        links = tuple(link for link in fact.links if not scan_secrets(link).flagged)
        return Fact(
            **{
                **fact.__dict__,
                "question": question.text,
                "answer": answer.text,
                "links": links,
                "entities": tuple(
                    scrub_personal(e, self.member_names) for e in fact.entities
                ),
            }
        )

    # ── whole pipeline ────────────────────────────────────────────────────

    def process(self, raw_text: str, channel: str, boost: bool = False) -> Extraction:
        """Run all five steps on one unit's text. Raises LLMError."""
        scan = scan_secrets(raw_text)
        text = scan.text
        if len(text) > MAX_INPUT_CHARS:
            text = text[:MAX_INPUT_CHARS] + "\n[…truncated]"
        relevant, reason = self.is_relevant(text, channel, boost)
        if not relevant:
            return Extraction(False, reason, secret_rules=scan.rules)
        extracted = self.extract(text, channel)
        kept = self.check_exclusions(extracted)
        facts = [f for f in (self.scrub(f) for f in kept) if f is not None]
        return Extraction(
            True,
            reason,
            facts=facts,
            secret_rules=scan.rules,
            excluded=len(extracted) - len(facts),
        )
