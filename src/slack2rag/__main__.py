"""Command line: `slack2rag` or `python -m slack2rag`.

slack2rag [--check]        index Slack into Qdrant (the default command)
slack2rag run [--check]    the same, spelled out
slack2rag query TEXT...    semantic search over the index
slack2rag migrate          create or upgrade the PostgreSQL schema
"""

import logging
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from slack2rag import __version__, query
from slack2rag import main as indexer
from slack2rag.config import Config, ConfigError
from slack2rag.pg_store import migrate

logger = logging.getLogger("slack2rag")

USAGE = __doc__ or ""


def _settings() -> Config:
    """Settings from the environment, plus a .env file in the working directory."""
    # A bare load_dotenv() searches from this file's directory, not the cwd.
    load_dotenv(".env")
    return Config.from_env()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = "run"
    if args and args[0] in ("run", "query", "migrate"):
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
        if command == "migrate":
            if args:
                sys.stderr.write("slack2rag migrate: takes no arguments\n")
                return 2
            if not cfg.database_url:
                sys.stderr.write("slack2rag migrate: DATABASE_URL is not set\n")
                return 2
            migrate(cfg.database_url)
            return 0
        return indexer.main(args, cfg=cfg)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
