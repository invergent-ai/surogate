"""CLI command for database schema migrations (Alembic wrapper)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from surogate.utils.logger import get_logger

logger = get_logger()

_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def _load_config(config_path: str | None):
    """Resolve ServerConfig from a config file.

    Falls back to ~/.surogate/config.yaml when no explicit path is given.
    """
    if not config_path:
        default = Path.home() / ".surogate" / "config.yaml"
        if default.exists():
            config_path = str(default)

    if not config_path:
        logger.error("No config file provided and ~/.surogate/config.yaml not found.")
        sys.exit(1)

    from surogate.core.config.loader import load_config
    from surogate.core.config.server_config import ServerConfig

    return load_config(ServerConfig, config_path)


def _run_surogates_migrations(surogates_db_url: str) -> None:
    from surogates.config import DatabaseSettings
    from surogates.db.engine import run_migrations

    logger.info(f"Running Surogates migrations against {surogates_db_url}")
    run_migrations(DatabaseSettings(url=surogates_db_url))


def _alembic_cfg(db_url: str):
    from alembic.config import Config

    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default=None,
        help="Path or HTTP(s) URL to config file",
    )

    sub = parser.add_subparsers(dest="action", metavar="<action>")

    # upgrade
    up = sub.add_parser("upgrade", help="Upgrade database to a revision (default: head)")
    up.add_argument("revision", nargs="?", default="head")

    # downgrade
    down = sub.add_parser("downgrade", help="Downgrade database to a revision")
    down.add_argument("revision", default="-1", nargs="?")

    # revision (autogenerate)
    rev = sub.add_parser("revision", help="Create a new migration revision")
    rev.add_argument("-m", "--message", required=True, help="Revision message")
    rev.add_argument(
        "--no-autogenerate",
        action="store_true",
        help="Create an empty revision (skip autogenerate)",
    )

    # current
    sub.add_parser("current", help="Show current database revision")

    # history
    hist = sub.add_parser("history", help="Show migration history")
    hist.add_argument("-v", "--verbose", action="store_true")

    # heads
    sub.add_parser("heads", help="Show available heads")

    # create-all  (direct metadata.create_all, bypasses Alembic)
    sub.add_parser(
        "create-all",
        help="Create all tables from model metadata (no Alembic tracking)",
    )

    return parser


def _cmd_upgrade(config, args):
    from alembic import command

    logger.info(f"Upgrading database to revision: {args.revision}")
    command.upgrade(_alembic_cfg(config.database_url), args.revision)
    logger.info("Upgrade complete.")
    _run_surogates_migrations(config.surogates_database_url)


def _cmd_downgrade(config, args):
    from alembic import command

    logger.info(f"Downgrading database to revision: {args.revision}")
    command.downgrade(_alembic_cfg(config.database_url), args.revision)
    logger.info("Downgrade complete.")


def _cmd_revision(config, args):
    from alembic import command

    autogenerate = not args.no_autogenerate
    logger.info(f"Creating revision: {args.message}  (autogenerate={autogenerate})")
    command.revision(
        _alembic_cfg(config.database_url),
        message=args.message,
        autogenerate=autogenerate,
    )


def _cmd_current(config, args):
    from alembic import command

    command.current(_alembic_cfg(config.database_url), verbose=True)


def _cmd_history(config, args):
    from alembic import command

    command.history(_alembic_cfg(config.database_url), verbose=args.verbose)


def _cmd_heads(config, args):
    from alembic import command

    command.heads(_alembic_cfg(config.database_url), verbose=True)


def _cmd_create_all(config, args):
    import asyncio
    from surogate.core.db import init_engine, create_all_tables
    from surogate.core.db.engine import get_session_factory
    from surogate.core.db.repository import auth as auth_repository
    from surogate.core.hub.lakefs import init_lakefs, seed_lakefs_user

    logger.info(f"Creating all tables from model metadata ({config.database_url})")
    init_engine(config.database_url)

    async def _seed():
        await create_all_tables()
        factory = get_session_factory()
        async with factory() as session:
            await auth_repository.seed_admin_user(session)
            await init_lakefs(config)
            await seed_lakefs_user("surogate", session, config)

    asyncio.run(_seed())
    logger.info("All tables created and seeded.")
    _run_surogates_migrations(config.surogates_database_url)


_ACTIONS = {
    "upgrade": _cmd_upgrade,
    "downgrade": _cmd_downgrade,
    "revision": _cmd_revision,
    "current": _cmd_current,
    "history": _cmd_history,
    "heads": _cmd_heads,
    "create-all": _cmd_create_all,
}


if __name__ == "__main__":
    parser = prepare_command_parser()
    args = parser.parse_args(sys.argv[1:])

    if not args.action:
        parser.print_help()
        sys.exit(1)

    config = _load_config(args.config)
    _ACTIONS[args.action](config, args)
