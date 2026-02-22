from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine.url import make_url
from sqlmodel import SQLModel

from app import models  # noqa: F401
from app.db import get_database_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

if config.get_main_option("sqlalchemy.url") == "":
    config.set_main_option("sqlalchemy.url", get_database_url())

if config.get_main_option("sqlalchemy.url").startswith("sqlite"):
    config.set_main_option("sqlalchemy.url", get_database_url())

target_metadata = SQLModel.metadata


def ensure_sqlite_parent_dir() -> None:
    url = config.get_main_option("sqlalchemy.url")
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return

    database = parsed.database or ""
    if database in {"", ":memory:"}:
        return

    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    ensure_sqlite_parent_dir()
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
