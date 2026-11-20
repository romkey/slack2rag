"""
The Slack source's sync jobs.

catchup()    hourly: new messages, edits and deletions in each channel's
             recent history, pins, bookmarks, the channel directory, units
             marked dirty by live events and units waiting to go quiet.
reconcile()  nightly: everything catchup does, plus every unit that has
             facts (or was active in the last RECONCILE_ACTIVE_DAYS, or was
             processed by an older extractor) is refetched by its stored
             timestamps and reprocessed if it changed. Every fact seen is
             stamped with the run ID; points from older runs are deleted,
             unless the run saw less than half of what the last one did.
process_dirty()  the units live events marked, right away, so a deleted
             message's facts are gone within minutes.

A channel the bot is no longer in (or that is denylisted) loses everything
derived from it on the next run of any job.
"""

from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .embedder import Embedder, SparseEncoder
from .extractor import Extractor, point_id
from .llm import LLMError
from .pg_store import SOURCE, FactChanges, FactRow, PgStore, RunCounts, UnitRow
from .sanitize import scan_secrets, scrub_personal
from .slack_client import BotIdentity, ChannelGoneError, SlackClient, SlackObject
from .units import (
    Message,
    Unit,
    bookmark_unit,
    mentions_bot,
    pin_unit,
    sessions,
    thread_unit,
    to_message,
    ts_date,
)
from .vector_store import Point, VectorStore, source_filter

logger = logging.getLogger(__name__)

AUTHORITY = 1  # a Slack fact; the wiki (4) always outranks it
RECONCILE_ACTIVE_DAYS = 30
SAFETY_FRACTION = 0.5


def _iso(value: datetime.datetime | datetime.date | None) -> str | None:
    return None if value is None else value.isoformat()


def fact_payload(row: FactRow, run_id: int | None) -> dict[str, Any]:
    """A fact as a point payload in the shared schema."""
    date = row.said_on.isoformat()
    context = (
        f"From a Slack discussion in #{row.channel_name} on {date}."
        " Slack facts are member knowledge; the wiki outranks them."
    )
    return {
        "source": SOURCE,
        "doc_id": f"{row.channel_id}:{row.unit_key}:{row.fact_index}",
        "chunk_index": 0,
        "title": row.question,
        "heading_path": [f"#{row.channel_name}"],
        "text": f"Q: {row.question}\nA: {row.answer}",
        "context": context,
        "url": row.permalink,
        "tier": "member",
        "authority": AUTHORITY,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "indexed_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "stale_after": _iso(row.stale_after),
        "superseded_by": row.superseded_by,
        "content_hash": row.content_hash,
        "sync_run_id": run_id,
        "asset_ids": [],
        "location_ids": [],
        "work_ids": [],
        "event_ids": [],
        "training_ids": [],
        "topics": row.topics,
        "section": None,
        "page": None,
        "lang": "en",
        # Slack-specific: what the linker and the bots need.
        "kind": row.kind,
        "confidence": row.confidence,
        "entities": row.entities,
        "links": row.links,
        "tags": row.tags,
        "channel_id": row.channel_id,
        "channel_name": row.channel_name,
        "said_on": date,
        "label": f"Slack discussion, {date}",
    }


def embed_text(payload: dict[str, Any]) -> str:
    """What gets embedded: the context sentence, then the text."""
    return f"{payload['context']}\n{payload['text']}"


@dataclass
class RunState:
    run_id: int
    kind: str
    counts: RunCounts = field(default_factory=RunCounts)
    done: set[tuple[str, str]] = field(default_factory=set)  # (channel, unit key)
    seen_facts: list[str] = field(default_factory=list)
    directory_ids: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        slack: SlackClient,
        store: PgStore,
        vectors: VectorStore,
        embedder: Embedder,
        extractor: Extractor,
        sparse: SparseEncoder | None = None,
        clock: Any = time.time,
    ) -> None:
        self.cfg = cfg
        self.slack = slack
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self.extractor = extractor
        self.sparse = sparse or SparseEncoder()
        self.clock = clock
        self.me: BotIdentity = slack.identify()
        self._channel_names: dict[str, str] = {}

    # ── jobs ──────────────────────────────────────────────────────────────

    def catchup(self) -> RunCounts:
        return self._run("catchup")

    def reconcile(self) -> RunCounts:
        return self._run("reconcile")

    def process_dirty(self) -> RunCounts:
        return self._run("live")

    def rebuild(self, reextract: bool = False) -> RunCounts:
        """Rebuild this source's points in Qdrant, then catch up.

        Deletes every Slack point and re-embeds every stored fact (no model
        calls, and facts from messages Slack no longer shows are kept), then
        re-indexes the channel directory. With *reextract*, every unit is
        also refetched and run through the model again, as after
        `slack2rag reprocess`.
        """
        return self._run("rebuild", reextract=reextract)

    def _run(self, kind: str, reextract: bool = False) -> RunCounts:
        run = RunState(run_id=self.store.start_run(kind), kind=kind)
        status, message = "ok", ""
        try:
            if kind == "rebuild":
                self._reindex_all(run)
                if reextract:
                    count = self.store.mark_all_dirty()
                    logger.info("Rebuild: reprocessing %d units", count)
            if kind == "live":
                members = self.store.member_channel_ids()
                self._channel_names = {
                    cid: name
                    for cid, name in self.store.channel_names().items()
                    if cid in members
                }
            else:
                for channel in self._channels(run):
                    self._sync_channel(channel, run)
            self._refetch_units(self.store.dirty_units(), run)
            if kind == "reconcile":
                since = datetime.datetime.fromtimestamp(
                    self.clock() - RECONCILE_ACTIVE_DAYS * 86400, tz=datetime.UTC
                )
                rows = [
                    r
                    for r in self.store.units(
                        active_since=since, with_facts_or_active=True
                    )
                    if r.channel_id in self._channel_names
                ]
                stale = [
                    r
                    for r in self.store.units()
                    if r.extractor_version != self.extractor.version
                    and r.channel_id in self._channel_names
                ]
                self._refetch_units([*rows, *stale], run)
                status, message = self._sweep(run)
        except Exception as exc:
            status, message = "error", f"{type(exc).__name__}: {exc}"
            run.counts.errors += 1
            self.store.finish_run(run.run_id, run.counts, status, message)
            raise
        run.counts.seen = len(set(run.seen_facts))
        self.store.finish_run(run.run_id, run.counts, status, message)
        logger.info(
            "%s run %d %s: %d facts seen, %d added, %d updated, %d deleted,"
            " %d quarantined, %d errors",
            kind,
            run.run_id,
            status,
            run.counts.seen,
            run.counts.added,
            run.counts.updated,
            run.counts.deleted,
            run.counts.quarantined,
            run.counts.errors,
        )
        return run.counts

    # ── channels ──────────────────────────────────────────────────────────

    def _channels(self, run: RunState) -> list[SlackObject]:
        """Channels to read; forgets every channel no longer among them."""
        deny = self.cfg.slack_channel_denylist
        channels = [
            ch
            for ch in self.slack.member_channels()
            if ch.get("id") not in deny and ch.get("name") not in deny
        ]
        current = {str(ch["id"]) for ch in channels}
        for gone in sorted(self.store.member_channel_ids() - current):
            self.forget_channel(gone, run)
        self._channel_names = {
            str(ch["id"]): str(ch.get("name") or ch["id"]) for ch in channels
        }
        self._index_directory(channels, run)
        return channels

    def forget_channel(self, channel_id: str, run: RunState | None = None) -> None:
        """Delete everything derived from a channel, in staging and Qdrant."""
        removed = self.store.forget_channel(channel_id)
        self.vectors.delete_where(source_filter(SOURCE, channel_id=channel_id))
        self._channel_names.pop(channel_id, None)
        if run is not None:
            run.counts.deleted += len(removed)
        logger.info(
            "Forgot channel %s: %d facts and its directory entry deleted",
            channel_id,
            len(removed),
        )

    def _index_directory(self, channels: Sequence[SlackObject], run: RunState) -> None:
        """One point per channel: name, topic and purpose ("ask in #laser")."""
        names = self.extractor.member_names
        points = []
        for ch in channels:
            cid = str(ch["id"])
            name = str(ch.get("name") or cid)
            topic = str((ch.get("topic") or {}).get("value") or "")
            purpose = str((ch.get("purpose") or {}).get("value") or "")
            topic = scrub_personal(scan_secrets(topic).text, names)
            purpose = scrub_personal(scan_secrets(purpose).text, names)
            changed = self.store.upsert_channel(cid, name, topic, purpose)
            doc_id = f"channel:{cid}"
            pid = point_id(SOURCE, doc_id)
            run.directory_ids.append(pid)
            if not changed and self.vectors.existing_ids([pid]):
                continue
            text = f"Slack channel #{name}."
            if topic:
                text += f" Topic: {topic}."
            if purpose:
                text += f" Purpose: {purpose}."
            now = datetime.datetime.now(tz=datetime.UTC).isoformat()
            payload = {
                "source": SOURCE,
                "doc_id": doc_id,
                "chunk_index": 0,
                "title": f"#{name}",
                "heading_path": ["Slack channels"],
                "text": text,
                "context": (
                    "An entry in the directory of Slack channels members can ask in."
                ),
                "url": f"{self.me.workspace_url}/archives/{cid}"
                if self.me.workspace_url
                else "",
                "tier": "member",
                "authority": AUTHORITY,
                "created_at": now,
                "updated_at": now,
                "indexed_at": now,
                "stale_after": None,
                "superseded_by": None,
                "content_hash": "",
                "sync_run_id": run.run_id,
                "asset_ids": [],
                "location_ids": [],
                "work_ids": [],
                "event_ids": [],
                "training_ids": [],
                "topics": [],
                "section": None,
                "page": None,
                "lang": "en",
                "kind": "channel",
                "channel_id": cid,
                "channel_name": name,
                "label": "Slack channel directory",
            }
            points.append((pid, payload))
        self._upsert(points)

    # ── per-channel catch-up ─────────────────────────────────────────────

    def _sync_channel(self, channel: SlackObject, run: RunState) -> None:
        cid = str(channel["id"])
        now = self.clock()
        cursor = self.store.get_cursor(cid)
        if cursor is None:
            oldest = (
                str(now - self.cfg.backfill_days * 86400)
                if self.cfg.backfill_days
                else None
            )
        else:
            oldest = str(
                min(float(cursor), now - self.cfg.catchup_lookback_hours * 3600)
            )
            covering = self.store.session_start_covering(cid, oldest)
            if covering is not None:
                oldest = str(float(covering) - 0.000001)
        try:
            pins = self.slack.pins(cid)
            pinned = {str(m.get("ts")) for m in pins}
            raw = list(self.slack.history(cid, oldest=oldest))
            self._sync_history(cid, raw, oldest, pinned, run)
            self._sync_pins(cid, pins, run)
            self._sync_bookmarks(cid, run)
        except ChannelGoneError:
            self.forget_channel(cid, run)
            return
        latest = max((float(m["ts"]) for m in raw), default=None)
        if latest is not None and (cursor is None or latest > float(cursor)):
            self.store.set_cursor(
                cid, str(max(raw, key=lambda m: float(m["ts"]))["ts"])
            )

    def _sync_history(
        self,
        cid: str,
        raw: list[SlackObject],
        oldest: str | None,
        pinned: set[str],
        run: RunState,
    ) -> None:
        """Threads and sessions in *raw*; pinned messages are pin units."""
        stored_threads = {
            r.unit_key: r for r in self.store.units(channel_id=cid, kind="thread")
        }
        loose: list[Message] = []
        parents: set[str] = set()
        for msg in raw:
            ts = str(msg["ts"])
            thread_ts = msg.get("thread_ts")
            if thread_ts and thread_ts != ts:
                continue  # a reply broadcast to the channel; its thread has it
            if int(msg.get("reply_count") or 0) > 0 or ts in stored_threads:
                parents.add(ts)
                self._maybe_sync_thread(
                    cid, msg, stored_threads.get(ts), ts in pinned, run
                )
                continue
            if ts in pinned:
                continue  # its pin unit covers it
            if mentions_bot(msg, self.me):
                continue  # a question to the bot; its answer must not come back
            m = to_message(msg, self.me, self.cfg.slack_bot_allowlist)
            if m is not None:
                loose.append(m)

        found = sessions(
            cid,
            loose,
            self.cfg.session_gap_minutes,
            self.cfg.session_max_messages,
            self.cfg.session_overlap_messages,
        )
        for unit in found:
            self.process_unit(unit, run)

        # Stored units in the fetched window that no longer exist: their
        # messages were deleted (or regrouped into another session).
        floor = float(oldest) if oldest else float("-inf")
        keys = {u.key for u in found}
        for row in self.store.units(channel_id=cid, kind="session"):
            if float(row.unit_key) > floor and row.unit_key not in keys:
                self._delete_unit(row, run)
        for key, row in stored_threads.items():
            if float(key) > floor and key not in parents and row.relevant is not None:
                # Gone from history, or its replies were all deleted.
                self._refetch_units([row], run)

    def _maybe_sync_thread(
        self,
        cid: str,
        parent: SlackObject,
        row: UnitRow | None,
        boost: bool,
        run: RunState,
    ) -> None:
        key = str(parent["ts"])
        if (
            row is not None
            and not row.dirty
            and row.relevant is not None
            and row.boost == boost
        ):
            latest = float(parent.get("latest_reply") or 0)
            edited = float((parent.get("edited") or {}).get("ts") or 0)
            replies = int(parent.get("reply_count") or 0)
            unchanged = (
                latest <= row.last_activity.timestamp() + 0.0005
                and edited <= row.last_activity.timestamp() + 0.0005
                and replies + 1 >= len(row.message_ts)
                and row.extractor_version == self.extractor.version
            )
            if unchanged:
                self._see(row, run)
                run.done.add((cid, key))
                return
        unit = thread_unit(
            cid,
            self.slack.thread(cid, key),
            self.me,
            self.cfg.slack_bot_allowlist,
            boost=boost,
        )
        if unit is None:
            if row is not None:
                self._delete_unit(row, run)
            return
        self.process_unit(unit, run)

    def _sync_pins(self, cid: str, pins: list[SlackObject], run: RunState) -> None:
        pinned_keys = set()
        for msg in pins:
            ts = str(msg.get("ts") or "")
            if not ts:
                continue
            if int(msg.get("reply_count") or 0) > 0:
                if (cid, ts) not in run.done:  # a pinned thread outside the window
                    self._maybe_sync_thread(
                        cid, msg, self.store.get_unit(cid, ts), True, run
                    )
                continue
            m = to_message(msg, self.me, self.cfg.slack_bot_allowlist)
            if m is None or mentions_bot(msg, self.me):
                continue
            unit = pin_unit(cid, m)
            pinned_keys.add(unit.key)
            self.process_unit(unit, run)
        for row in self.store.units(channel_id=cid, kind="pin"):
            if row.unit_key not in pinned_keys:
                self._delete_unit(row, run)

    def _sync_bookmarks(self, cid: str, run: RunState) -> None:
        keys = set()
        for bookmark in self.slack.bookmarks(cid):
            unit = bookmark_unit(cid, bookmark)
            if unit is None:
                continue
            keys.add(unit.key)
            self.process_unit(unit, run)
        for row in self.store.units(channel_id=cid, kind="bookmark"):
            if row.unit_key not in keys:
                self._delete_unit(row, run)

    # ── refetching stored units ──────────────────────────────────────────

    def _refetch_units(self, rows: Iterable[UnitRow], run: RunState) -> None:
        for row in rows:
            if (row.channel_id, row.unit_key) in run.done:
                continue
            if row.channel_id not in self._channel_names:
                continue  # forgetting the channel deletes it
            try:
                unit = self.refetch(row)
            except ChannelGoneError:
                self.forget_channel(row.channel_id, run)
                continue
            if unit is None:
                self._delete_unit(row, run)
                continue
            if unit.key != row.unit_key:
                # A session's first message was deleted: it has a new key.
                self._delete_unit(row, run)
            self.process_unit(unit, run)

    def refetch(self, row: UnitRow) -> Unit | None:
        """Rebuild a stored unit from Slack by its stored timestamps."""
        cid, key = row.channel_id, row.unit_key
        allow = self.cfg.slack_bot_allowlist
        if row.kind == "thread":
            return thread_unit(
                cid, self.slack.thread(cid, key), self.me, allow, row.boost
            )
        if row.kind == "pin":
            ts = key.removeprefix("pin:")
            for msg in self.slack.history(cid, oldest=ts, latest=ts, inclusive=True):
                m = to_message(msg, self.me, allow)
                if m is not None and str(msg["ts"]) == ts:
                    return pin_unit(cid, m)
            return None
        if row.kind == "bookmark":
            for bookmark in self.slack.bookmarks(cid):
                unit = bookmark_unit(cid, bookmark)
                if unit is not None and unit.key == key:
                    return unit
            return None
        # A session: its own messages and the overlap borrowed before it.
        stamps = row.message_ts or [key]
        first, last = min(stamps, key=float), max(stamps, key=float)
        messages = []
        for msg in self.slack.history(cid, oldest=first, latest=last, inclusive=True):
            reply = bool(msg.get("thread_ts")) and msg.get("thread_ts") != msg.get("ts")
            parent = int(msg.get("reply_count") or 0) > 0
            if reply or parent or mentions_bot(msg, self.me):
                continue
            m = to_message(msg, self.me, allow)
            if m is not None and m.ts in stamps:
                messages.append(m)
        messages.sort(key=lambda m: float(m.ts))
        own = [m for m in messages if float(m.ts) >= float(key)]
        if not own:
            return None
        overlap = len(messages) - len(own)
        return Unit(
            channel_id=cid,
            key=own[0].ts,
            kind="session",
            messages=messages,
            overlap=overlap,
        )

    # ── processing one unit ──────────────────────────────────────────────

    def process_unit(self, unit: Unit, run: RunState, force: bool = False) -> None:
        cid = unit.channel_id
        if (cid, unit.key) in run.done and not force:
            return
        run.done.add((cid, unit.key))
        row = self.store.get_unit(cid, unit.key)
        content_hash = unit.content_hash
        version = self.extractor.version
        if (
            row is not None
            and row.content_hash == content_hash
            and row.extractor_version == version
            and row.relevant is not None
            and row.boost == unit.boost
        ):
            if row.dirty:
                self.store.clear_dirty(cid, unit.key)  # nothing changed after all
            self._see(row, run)
            return

        now = self.clock()
        urgent = (
            row is not None and row.fact_count > 0 and self._lost_or_edited(row, unit)
        )
        if not unit.is_quiet(now, self.cfg.quiet_hours) and not urgent:
            # Not quiet yet: record it so the next run comes back to it.
            self.store.save_unit(
                channel_id=cid,
                unit_key=unit.key,
                kind=unit.kind,
                message_ts=unit.message_ts,
                content_hash=row.content_hash if row else "",
                extractor_version=row.extractor_version if row else "",
                relevant=None,
                boost=unit.boost,
                last_activity=unit.last_activity,
                facts=None,
            )
            if row is not None:
                self._see(row, run)
            return

        self._quarantine(unit, run)
        channel = self._channel_names.get(cid, cid)
        try:
            result = self.extractor.process(unit.render(), channel, unit.boost)
        except LLMError as exc:
            # Leave the stored unit as it was; the next run tries again.
            run.counts.errors += 1
            logger.error("Could not process %s/%s: %s", cid, unit.key, exc)
            if row is not None:
                self._see(row, run)
            return
        facts = result.facts if result.relevant else []
        changes = self.store.save_unit(
            channel_id=cid,
            unit_key=unit.key,
            kind=unit.kind,
            message_ts=unit.message_ts,
            content_hash=content_hash,
            extractor_version=version,
            relevant=result.relevant,
            boost=unit.boost,
            last_activity=unit.last_activity,
            facts=facts,
            channel_name=channel,
            permalink=self._permalink(unit),
            said_on=ts_date(unit.last_activity),
            sync_run_id=run.run_id,
        )
        logger.debug(
            "%s/%s %s: %d facts (%d dropped by the exclusion check)",
            cid,
            unit.key,
            "relevant" if result.relevant else "not relevant",
            len(facts),
            result.excluded,
        )
        self._apply(changes, run)

    def _lost_or_edited(self, row: UnitRow, unit: Unit) -> bool:
        """A message of a unit with facts was deleted or edited since."""
        if set(row.message_ts) - set(unit.message_ts):
            return True
        if row.dirty:
            return True
        # Slack timestamps both: the unit's activity when it was processed.
        before = row.last_activity.timestamp() + 0.0005
        return any(float(m.edited_ts or 0) > before for m in unit.messages)

    def _permalink(self, unit: Unit) -> str:
        if unit.kind == "bookmark":
            return ""
        first = unit.messages[unit.overlap] if unit.messages else None
        if first is None:
            return ""
        return self.slack.permalink(unit.channel_id, first.ts, first.thread_ts)

    def _quarantine(self, unit: Unit, run: RunState) -> None:
        if unit.kind == "bookmark":
            hit = scan_secrets(unit.extra_text)
            if hit.flagged:
                self.store.record_quarantine(unit.channel_id, unit.key, hit.rules)
                run.counts.quarantined += 1
            return
        for m in unit.messages:
            hit = scan_secrets(m.text)
            if hit.flagged:
                self.store.record_quarantine(unit.channel_id, m.ts, hit.rules)
                run.counts.quarantined += 1

    def _delete_unit(self, row: UnitRow, run: RunState) -> None:
        removed = self.store.delete_unit(row.channel_id, row.unit_key)
        self.vectors.delete_ids(removed)
        run.counts.deleted += len(removed)
        run.done.add((row.channel_id, row.unit_key))

    def _see(self, row: UnitRow, run: RunState) -> None:
        """Count an unchanged unit's facts as seen by this run."""
        if row.fact_count:
            ids = [
                f.id for f in self.store.facts(unit_of=(row.channel_id, row.unit_key))
            ]
            run.seen_facts.extend(ids)
            if run.kind == "reconcile":
                self.store.stamp_facts(ids, run.run_id)
                self._restamp(ids, run)

    # ── indexing ─────────────────────────────────────────────────────────

    def _apply(self, changes: FactChanges, run: RunState) -> None:
        run.counts.add(changes)
        self.vectors.delete_ids(changes.removed)
        fresh = [*changes.added, *changes.updated]
        rows = self.store.facts(ids=fresh) if fresh else []
        points = [(r.id, fact_payload(r, run.run_id)) for r in rows]
        vectors = self._upsert(points)
        run.seen_facts.extend([*fresh, *changes.unchanged])
        self._restamp(changes.unchanged, run)
        for row, dense in zip(rows, vectors, strict=True):
            self._supersede(row, dense)

    def _restamp(self, ids: Sequence[str], run: RunState) -> None:
        """Stamp unchanged facts' points with this run, writing back any
        that are missing from Qdrant (a deleted or restored collection)."""
        if not ids:
            return
        present = self.vectors.existing_ids(ids)
        if present:
            self.vectors.set_payload(
                [i for i in ids if i in present], {"sync_run_id": run.run_id}
            )
        missing = [i for i in ids if i not in present]
        if missing:
            rows = self.store.facts(ids=missing)
            self._upsert([(r.id, fact_payload(r, run.run_id)) for r in rows])
            logger.warning("Wrote back %d facts missing from Qdrant", len(rows))

    def _reindex_all(self, run: RunState) -> None:
        """Delete every Slack point and re-embed every stored fact."""
        self.vectors.delete_where(source_filter(SOURCE))
        rows = self.store.facts()
        ids = [r.id for r in rows]
        self.store.stamp_facts(ids, run.run_id)
        self._upsert([(r.id, fact_payload(r, run.run_id)) for r in rows])
        run.seen_facts.extend(ids)
        logger.info(
            "Rebuild: deleted the Slack points in %s and re-embedded %d facts",
            self.vectors.collection,
            len(rows),
        )

    def _upsert(
        self, points: Sequence[tuple[str, dict[str, Any]]]
    ) -> list[list[float]]:
        dense_all: list[list[float]] = []
        for start in range(0, len(points), self.cfg.batch_size):
            batch = points[start : start + self.cfg.batch_size]
            texts = [embed_text(p) for _, p in batch]
            dense = self.embedder.embed(texts, kind="document")
            sparse = self.sparse.encode(texts)
            self.vectors.upsert(
                [
                    Point(id=pid, dense=d, sparse=s, payload=payload)
                    for (pid, payload), d, s in zip(batch, dense, sparse, strict=True)
                ]
            )
            dense_all.extend(dense)
        return dense_all

    def _supersede(self, row: FactRow, dense: list[float]) -> None:
        """Mark older, highly similar facts about the same things superseded."""
        hits = self.vectors.similar(
            dense,
            source_filter(SOURCE),
            limit=6,
        )
        mine = {e.lower() for e in row.entities}
        for hit in hits:
            if hit["id"] == row.id or hit.get("kind") == "channel":
                continue
            if hit["score"] < self.cfg.supersede_similarity:
                break
            if (
                hit.get("superseded_by")
                or str(hit.get("said_on", "")) > row.said_on.isoformat()
            ):
                continue
            if str(hit.get("doc_id", "")).startswith(
                f"{row.channel_id}:{row.unit_key}:"
            ):
                continue  # facts of one unit don't supersede each other
            theirs = {str(e).lower() for e in hit.get("entities") or []}
            same_subject = (
                bool(mine & theirs)
                if mine and theirs
                else (hit.get("kind") == row.kind)
            )
            if not same_subject:
                continue
            self.store.supersede(hit["id"], row.id)
            self.vectors.set_payload([hit["id"]], {"superseded_by": row.id})
            logger.debug("Fact %s superseded by %s", hit["id"], row.id)

    # ── reconcile sweep ──────────────────────────────────────────────────

    def _sweep(self, run: RunState) -> tuple[str, str]:
        """Delete what this full run didn't see, unless that looks like a fault."""
        seen = len(set(run.seen_facts))
        last = self.store.last_run("reconcile")
        previous = int(last["seen"]) if last else 0
        if previous and seen < previous * SAFETY_FRACTION:
            message = (
                f"Safety valve: this reconcile saw {seen} facts, less than half"
                f" of the {previous} the last one saw; nothing was deleted"
            )
            logger.error("%s", message)
            return "alert", message
        self.store.stamp_facts(sorted(set(run.seen_facts)), run.run_id)
        removed = self.store.delete_facts_before_run(run.run_id)
        self.vectors.set_payload(run.directory_ids, {"sync_run_id": run.run_id})
        self.vectors.delete_older_runs(SOURCE, run.run_id)
        run.counts.deleted += len(removed)
        return "ok", ""
