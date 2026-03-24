from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from alembic import command
from alembic.config import Config
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlmodel import Session, create_engine

_engine: Engine | None = None
DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 30_000


def get_data_dir() -> Path:
    raw_data_dir = os.getenv("HEV_DATA_DIR", "./data")
    return Path(raw_data_dir).expanduser().resolve()


def get_database_url() -> str:
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    db_path = os.getenv("HEV_DB_PATH")
    if db_path:
        return f"sqlite:///{Path(db_path).expanduser().resolve()}"

    return f"sqlite:///{get_data_dir() / 'ha_entity_vault.db'}"


def ensure_data_dir() -> None:
    database_url = get_database_url()
    parsed = make_url(database_url)
    if parsed.get_backend_name() != "sqlite":
        return

    database = parsed.database or ""
    if database in {"", ":memory:"}:
        return

    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def get_sqlite_busy_timeout_ms() -> int:
    raw_value = os.getenv("SQLITE_BUSY_TIMEOUT_MS")
    if raw_value is None:
        return DEFAULT_SQLITE_BUSY_TIMEOUT_MS

    try:
        timeout_ms = int(raw_value)
    except ValueError:
        return DEFAULT_SQLITE_BUSY_TIMEOUT_MS
    return max(timeout_ms, 0)


def configure_sqlite_connection(dbapi_connection: object, _: object) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.fetchone()
        cursor.execute(f"PRAGMA busy_timeout={get_sqlite_busy_timeout_ms()}")
        # WAL + NORMAL reduces write contention and fsync overhead for this single-host app.
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        database_url = get_database_url()
        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            connect_args["timeout"] = get_sqlite_busy_timeout_ms() / 1000
        _engine = create_engine(database_url, connect_args=connect_args, echo=False)
        if database_url.startswith("sqlite"):
            event.listen(_engine, "connect", configure_sqlite_connection)
    return _engine


def reset_engine_for_tests() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def get_session() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session


def run_migrations() -> None:
    ensure_data_dir()
    config_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    alembic_config = Config(str(config_path))
    alembic_config.set_main_option("sqlalchemy.url", get_database_url())
    command.upgrade(alembic_config, "head")
