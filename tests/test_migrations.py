from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from app import db


def test_migration_upgrade_from_0004_to_head(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / "data"))
    db.reset_engine_for_tests()

    config_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    alembic_config = Config(str(config_path))
    alembic_config.set_main_option("sqlalchemy.url", db.get_database_url())

    command.upgrade(alembic_config, "0004_profile_enablement")
    command.upgrade(alembic_config, "head")

    engine = db.get_engine()
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        table_names = {str(row[0]) for row in rows}
        entity_suggestion_columns = connection.execute(
            text("PRAGMA table_info(entity_suggestions)")
        ).fetchall()
        entity_suggestion_column_names = {str(row[1]) for row in entity_suggestion_columns}

    assert "llm_connections" in table_names
    assert "suggestion_runs" in table_names
    assert "suggestion_proposals" in table_names
    assert "suggestion_audit_events" in table_names
    assert "workflow_status" in entity_suggestion_column_names
    assert "workflow_error" in entity_suggestion_column_names
    assert "workflow_payload_json" in entity_suggestion_column_names
    assert "workflow_result_json" in entity_suggestion_column_names
    assert "workflow_updated_at" in entity_suggestion_column_names

    db.reset_engine_for_tests()
