"""llm automation suggestions

Revision ID: 0007_llm_automation_suggestions
Revises: 0006_automation_drafts
Create Date: 2026-02-15 16:10:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0007_llm_automation_suggestions"
down_revision = "0006_automation_drafts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_connections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("provider_kind", sa.String(length=64), nullable=False),
        sa.Column("base_url", sa.String(length=512), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("api_key_env_var", sa.String(length=128), nullable=True),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("max_output_tokens", sa.Integer(), nullable=False),
        sa.Column("extra_headers_json", sa.Text(), nullable=True),
        sa.Column("is_enabled", sa.Boolean(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("profile_id", "name", name="uq_llm_connections_profile_name"),
    )
    op.create_index(op.f("ix_llm_connections_profile_id"), "llm_connections", ["profile_id"], unique=False)
    op.create_index(op.f("ix_llm_connections_name"), "llm_connections", ["name"], unique=False)
    op.create_index(
        op.f("ix_llm_connections_provider_kind"),
        "llm_connections",
        ["provider_kind"],
        unique=False,
    )
    op.create_index(
        op.f("ix_llm_connections_is_enabled"),
        "llm_connections",
        ["is_enabled"],
        unique=False,
    )
    op.create_index(
        op.f("ix_llm_connections_is_default"),
        "llm_connections",
        ["is_default"],
        unique=False,
    )

    op.create_table(
        "suggestion_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("llm_connection_id", sa.Integer(), nullable=False),
        sa.Column("config_sync_run_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("target_count", sa.Integer(), nullable=False),
        sa.Column("processed_count", sa.Integer(), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False),
        sa.Column("invalid_count", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("context_hash", sa.String(length=128), nullable=True),
        sa.Column("filters_json", sa.Text(), nullable=True),
        sa.Column("result_summary_json", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["llm_connection_id"], ["llm_connections.id"]),
        sa.ForeignKeyConstraint(["config_sync_run_id"], ["config_sync_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_suggestion_runs_profile_id"), "suggestion_runs", ["profile_id"], unique=False)
    op.create_index(
        op.f("ix_suggestion_runs_llm_connection_id"),
        "suggestion_runs",
        ["llm_connection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_runs_config_sync_run_id"),
        "suggestion_runs",
        ["config_sync_run_id"],
        unique=False,
    )
    op.create_index(op.f("ix_suggestion_runs_status"), "suggestion_runs", ["status"], unique=False)

    op.create_table(
        "suggestion_proposals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("suggestion_run_id", sa.Integer(), nullable=False),
        sa.Column("config_snapshot_id", sa.Integer(), nullable=True),
        sa.Column("target_entity_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.String(length=512), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("risk_level", sa.String(length=32), nullable=True),
        sa.Column("proposed_patch_json", sa.Text(), nullable=True),
        sa.Column("verification_steps_json", sa.Text(), nullable=True),
        sa.Column("raw_response_json", sa.Text(), nullable=True),
        sa.Column("validation_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["suggestion_run_id"], ["suggestion_runs.id"]),
        sa.ForeignKeyConstraint(["config_snapshot_id"], ["config_snapshots.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_suggestion_proposals_profile_id"),
        "suggestion_proposals",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_proposals_suggestion_run_id"),
        "suggestion_proposals",
        ["suggestion_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_proposals_config_snapshot_id"),
        "suggestion_proposals",
        ["config_snapshot_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_proposals_target_entity_id"),
        "suggestion_proposals",
        ["target_entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_proposals_status"),
        "suggestion_proposals",
        ["status"],
        unique=False,
    )

    op.create_table(
        "suggestion_audit_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("suggestion_run_id", sa.Integer(), nullable=True),
        sa.Column("proposal_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["suggestion_run_id"], ["suggestion_runs.id"]),
        sa.ForeignKeyConstraint(["proposal_id"], ["suggestion_proposals.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_suggestion_audit_events_profile_id"),
        "suggestion_audit_events",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_audit_events_suggestion_run_id"),
        "suggestion_audit_events",
        ["suggestion_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_audit_events_proposal_id"),
        "suggestion_audit_events",
        ["proposal_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_audit_events_event_type"),
        "suggestion_audit_events",
        ["event_type"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_suggestion_audit_events_event_type"), table_name="suggestion_audit_events")
    op.drop_index(op.f("ix_suggestion_audit_events_proposal_id"), table_name="suggestion_audit_events")
    op.drop_index(op.f("ix_suggestion_audit_events_suggestion_run_id"), table_name="suggestion_audit_events")
    op.drop_index(op.f("ix_suggestion_audit_events_profile_id"), table_name="suggestion_audit_events")
    op.drop_table("suggestion_audit_events")

    op.drop_index(op.f("ix_suggestion_proposals_status"), table_name="suggestion_proposals")
    op.drop_index(op.f("ix_suggestion_proposals_target_entity_id"), table_name="suggestion_proposals")
    op.drop_index(op.f("ix_suggestion_proposals_config_snapshot_id"), table_name="suggestion_proposals")
    op.drop_index(op.f("ix_suggestion_proposals_suggestion_run_id"), table_name="suggestion_proposals")
    op.drop_index(op.f("ix_suggestion_proposals_profile_id"), table_name="suggestion_proposals")
    op.drop_table("suggestion_proposals")

    op.drop_index(op.f("ix_suggestion_runs_status"), table_name="suggestion_runs")
    op.drop_index(op.f("ix_suggestion_runs_config_sync_run_id"), table_name="suggestion_runs")
    op.drop_index(op.f("ix_suggestion_runs_llm_connection_id"), table_name="suggestion_runs")
    op.drop_index(op.f("ix_suggestion_runs_profile_id"), table_name="suggestion_runs")
    op.drop_table("suggestion_runs")

    op.drop_index(op.f("ix_llm_connections_is_default"), table_name="llm_connections")
    op.drop_index(op.f("ix_llm_connections_is_enabled"), table_name="llm_connections")
    op.drop_index(op.f("ix_llm_connections_provider_kind"), table_name="llm_connections")
    op.drop_index(op.f("ix_llm_connections_name"), table_name="llm_connections")
    op.drop_index(op.f("ix_llm_connections_profile_id"), table_name="llm_connections")
    op.drop_table("llm_connections")
