from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine

_engine: Engine | None = None


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
    if get_database_url().startswith("sqlite"):
        get_data_dir().mkdir(parents=True, exist_ok=True)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        connect_args: dict[str, object] = {}
        if get_database_url().startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(get_database_url(), connect_args=connect_args, echo=False)
    return _engine


def reset_engine_for_tests() -> None:
    global _engine
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
