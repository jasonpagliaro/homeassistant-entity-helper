from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from app import db


def test_sqlite_engine_uses_wal_and_extended_busy_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / "data"))
    db.reset_engine_for_tests()
    db.ensure_data_dir()

    try:
        engine = db.get_engine()
        with engine.connect() as connection:
            journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar_one()
            busy_timeout = connection.execute(text("PRAGMA busy_timeout")).scalar_one()

        assert str(journal_mode).lower() == "wal"
        assert int(busy_timeout) == db.DEFAULT_SQLITE_BUSY_TIMEOUT_MS
    finally:
        db.reset_engine_for_tests()


@pytest.mark.parametrize(
    ("env_value", "expected_timeout_ms"),
    [("1234", 1234), ("not-a-number", db.DEFAULT_SQLITE_BUSY_TIMEOUT_MS)],
)
def test_sqlite_busy_timeout_env_override_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_value: str, expected_timeout_ms: int
) -> None:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / f"data-{env_value}"))
    monkeypatch.setenv("SQLITE_BUSY_TIMEOUT_MS", env_value)
    db.reset_engine_for_tests()
    db.ensure_data_dir()

    try:
        engine = db.get_engine()
        with engine.connect() as connection:
            busy_timeout = connection.execute(text("PRAGMA busy_timeout")).scalar_one()

        assert int(busy_timeout) == expected_timeout_ms
    finally:
        db.reset_engine_for_tests()
