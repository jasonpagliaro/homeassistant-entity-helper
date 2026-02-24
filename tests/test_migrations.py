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
        app_config_columns = connection.execute(text("PRAGMA table_info(app_config)")).fetchall()
        app_config_column_names = {str(row[1]) for row in app_config_columns}
        adjustment_draft_columns = connection.execute(
            text("PRAGMA table_info(automation_adjustment_drafts)")
        ).fetchall()
        adjustment_draft_column_names = {str(row[1]) for row in adjustment_draft_columns}

    assert "llm_connections" in table_names
    assert "suggestion_runs" in table_names
    assert "suggestion_proposals" in table_names
    assert "suggestion_audit_events" in table_names
    assert "automation_adjustment_drafts" in table_names
    assert "automation_adjustment_revisions" in table_names
    assert "automation_adjustment_actions" in table_names
    assert "app_config" in table_names
    assert "workflow_status" in entity_suggestion_column_names
    assert "workflow_error" in entity_suggestion_column_names
    assert "workflow_payload_json" in entity_suggestion_column_names
    assert "workflow_result_json" in entity_suggestion_column_names
    assert "workflow_updated_at" in entity_suggestion_column_names
    assert "updates_enabled" in app_config_column_names
    assert "update_repo_owner" in app_config_column_names
    assert "update_repo_name" in app_config_column_names
    assert "update_repo_branch" in app_config_column_names
    assert "update_check_interval_minutes" in app_config_column_names
    assert "last_checked_at" in app_config_column_names
    assert "last_check_state" in app_config_column_names
    assert "last_check_error" in app_config_column_names
    assert "installed_commit_sha" in app_config_column_names
    assert "latest_commit_sha" in app_config_column_names
    assert "latest_commit_url" in app_config_column_names
    assert "latest_commit_published_at" in app_config_column_names
    assert "dismissed_commit_sha" in app_config_column_names
    assert "dismissed_at" in app_config_column_names
    assert "last_update_attempt_at" in app_config_column_names
    assert "last_update_result" in app_config_column_names
    assert "source_entity_id" in adjustment_draft_column_names
    assert "source_config_key" in adjustment_draft_column_names
    assert "working_yaml_text" in adjustment_draft_column_names
    assert "working_structured_json" in adjustment_draft_column_names
    assert "queue_status" in adjustment_draft_column_names
    assert "last_test_action_id" in adjustment_draft_column_names

    db.reset_engine_for_tests()
