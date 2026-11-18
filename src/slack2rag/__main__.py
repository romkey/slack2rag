"""Command line: `slack2rag` or `python -m slack2rag`.

slack2rag [--check]               the worker (the default command)
slack2rag run [--check]           the same, spelled out
slack2rag query TEXT...           search the facts mined from Slack
slack2rag migrate                 create or upgrade the PostgreSQL schema
slack2rag status                  the last run, counts and quarantine size
slack2rag reconcile               run the nightly full reconcile now
slack2rag reprocess               mark every unit for reprocessing
slack2rag report                  print the wiki promotion queue
slack2rag label-template [N]      JSONL of N units to hand-label (default 200)
slack2rag eval-classifier FILE    score the relevance check on labeled units
slack2rag spot-check [N]          N random facts, with leaks flagged (default 50)
slack2rag canary STRING...        fail if a planted string is indexed anywhere
"""

import datetime
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv

from slack2rag import __version__, query
from slack2rag import main as worker
from slack2rag.config import Config, ConfigError
from slack2rag.pg_store import PgStore, migrate

logger = logging.getLogger("slack2rag")

USAGE = __doc__ or ""


def _settings() -> Config:
    """Settings from the environment, plus a .env file in the working directory."""
    load_dotenv(".env")  # a bare load_dotenv() searches from this file's directory
    return Config.from_env()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


def _count(args: list[str], default: int) -> int:
    if not args:
        return default
    if len(args) > 1 or not args[0].isdigit():
        raise ValueError("expects one whole number")
    return int(args[0])


def _out(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


def cmd_migrate(cfg: Config, args: list[str]) -> int:
    if args:
        raise ValueError("takes no arguments")
    migrate(cfg.database_url)
    return 0


def cmd_status(cfg: Config, args: list[str]) -> int:
    if args:
        raise ValueError("takes no arguments")
    store = PgStore(cfg.database_url)
    status = store.status()
    if status is None:
        _out("No run has finished yet.")
        return 0
    for key in (
        "last_kind",
        "last_status",
        "last_started_at",
        "last_finished_at",
        "added",
        "updated",
        "deleted",
        "quarantine_count",
        "errors",
        "last_error",
    ):
        _out(f"{key:18} {status[key]}")
    return 0 if status["last_status"] in ("ok", "running") else 1


def cmd_reconcile(cfg: Config, args: list[str]) -> int:
    if args:
        raise ValueError("takes no arguments")
    return 0 if worker.run_job(worker.build_pipeline(cfg), "reconcile") else 1


def cmd_reprocess(cfg: Config, args: list[str]) -> int:
    if args:
        raise ValueError("takes no arguments")
    count = PgStore(cfg.database_url).mark_all_dirty()
    _out(f"{count} units will be refetched and reprocessed by the next run.")
    return 0


def cmd_report(cfg: Config, args: list[str]) -> int:
    from slack2rag.report import promotion_candidates, render

    if args:
        raise ValueError("takes no arguments")
    pipeline = worker.build_pipeline(cfg)
    facts = promotion_candidates(pipeline.store, pipeline.vectors)
    _out(render(facts, datetime.datetime.now(tz=datetime.UTC).date()))
    return 0


def cmd_label_template(cfg: Config, args: list[str]) -> int:
    from slack2rag.evaluation import label_template

    for line in label_template(worker.build_pipeline(cfg), _count(args, 200)):
        _out(line)
    return 0


def cmd_eval_classifier(cfg: Config, args: list[str]) -> int:
    from slack2rag.evaluation import eval_classifier

    if len(args) != 1:
        raise ValueError("takes one labels file")
    report = eval_classifier(worker.build_pipeline(cfg), Path(args[0]))
    _out(report.render())
    return 0


def cmd_spot_check(cfg: Config, args: list[str]) -> int:
    from slack2rag.evaluation import spot_check
    from slack2rag.slack_client import SlackClient

    count = _count(args, 50)
    names = SlackClient(cfg.slack_bot_token, api_pause=cfg.api_pause).member_names()
    flagged = 0
    for fact, problems in spot_check(PgStore(cfg.database_url), names, count):
        flagged += bool(problems)
        mark = "  !! " + "; ".join(problems) if problems else ""
        _out(f"[{fact.kind}] {fact.question}\n    {fact.answer}{mark}")
    _out(f"\n{flagged} of {count} sampled facts flagged.")
    return 1 if flagged else 0


def cmd_canary(cfg: Config, args: list[str]) -> int:
    from qdrant_client import QdrantClient

    from slack2rag.evaluation import canaries

    if not args:
        raise ValueError("takes one or more canary strings")
    found = canaries(PgStore(cfg.database_url), QdrantClient(url=cfg.qdrant_url), args)
    for line in found:
        _out(line)
    _out("canary check: FAILED" if found else "canary check: passed")
    return 1 if found else 0


COMMANDS: dict[str, Callable[[Config, list[str]], int]] = {
    "migrate": cmd_migrate,
    "status": cmd_status,
    "reconcile": cmd_reconcile,
    "reprocess": cmd_reprocess,
    "report": cmd_report,
    "label-template": cmd_label_template,
    "eval-classifier": cmd_eval_classifier,
    "spot-check": cmd_spot_check,
    "canary": cmd_canary,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = "run"
    if args and args[0] in ("run", "query", *COMMANDS):
        command = args.pop(0)
    elif args and args[0] in ("-h", "--help"):
        sys.stdout.write(USAGE.lstrip())
        return 0
    elif args and args[0] in ("-V", "--version"):
        sys.stdout.write(f"slack2rag {__version__}\n")
        return 0

    try:
        cfg = _settings()
    except ConfigError as e:
        sys.stderr.write(f"slack2rag: {e}\n")
        return 2
    _configure_logging(cfg.log_level)

    try:
        if command == "query":
            return query.main(args, cfg=cfg)
        if command in COMMANDS:
            try:
                return COMMANDS[command](cfg, args)
            except ValueError as e:
                sys.stderr.write(f"slack2rag {command}: {e}\n")
                return 2
        return worker.main(args, cfg=cfg)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
