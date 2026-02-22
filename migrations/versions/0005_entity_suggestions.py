"""entity suggestions

Revision ID: 0005_entity_suggestions
Revises: 0004_profile_enablement
Create Date: 2026-02-15 13:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0005_entity_suggestions"
down_revision = "0004_profile_enablement"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "entity_suggestion_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("sync_run_id", sa.Integer(), nullable=False),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entity_count", sa.Integer(), nullable=False),
        sa.Column("ready_count", sa.Integer(), nullable=False),
        sa.Column("needs_review_count", sa.Integer(), nullable=False),
        sa.Column("blocked_count", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("llm_enabled", sa.Boolean(), nullable=False),
        sa.Column("llm_model", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["sync_run_id"], ["sync_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_entity_suggestion_runs_profile_id"),
        "entity_suggestion_runs",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_suggestion_runs_sync_run_id"),
        "entity_suggestion_runs",
        ["sync_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_suggestion_runs_status"),
        "entity_suggestion_runs",
        ["status"],
        unique=False,
    )

    op.create_table(
        "entity_suggestions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("sync_run_id", sa.Integer(), nullable=False),
        sa.Column("suggestion_run_id", sa.Integer(), nullable=False),
        sa.Column("entity_snapshot_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.String(length=255), nullable=False),
        sa.Column("domain", sa.String(length=64), nullable=False),
        sa.Column("readiness_status", sa.String(length=32), nullable=False),
        sa.Column("missing_fields_json", sa.Text(), nullable=True),
        sa.Column("issues_json", sa.Text(), nullable=True),
        sa.Column("semantic_type_json", sa.Text(), nullable=True),
        sa.Column("llm_suggestions_json", sa.Text(), nullable=True),
        sa.Column("source_metadata_json", sa.Text(), nullable=True),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["sync_run_id"], ["sync_runs.id"]),
        sa.ForeignKeyConstraint(["suggestion_run_id"], ["entity_suggestion_runs.id"]),
        sa.ForeignKeyConstraint(["entity_snapshot_id"], ["entity_snapshots.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "suggestion_run_id",
            "entity_snapshot_id",
            name="uq_entity_suggestions_run_snapshot",
        ),
    )
    op.create_index(
        op.f("ix_entity_suggestions_profile_id"),
        "entity_suggestions",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_suggestions_sync_run_id"),
        "entity_suggestions",
        ["sync_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_suggestions_suggestion_run_id"),
        "entity_suggestions",
        ["suggestion_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_suggestions_entity_snapshot_id"),
        "entity_suggestions",
        ["entity_snapshot_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_suggestions_entity_id"),
        "entity_suggestions",
        ["entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_suggestions_domain"),
        "entity_suggestions",
        ["domain"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_suggestions_readiness_status"),
        "entity_suggestions",
        ["readiness_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_entity_suggestions_readiness_status"), table_name="entity_suggestions")
    op.drop_index(op.f("ix_entity_suggestions_domain"), table_name="entity_suggestions")
    op.drop_index(op.f("ix_entity_suggestions_entity_id"), table_name="entity_suggestions")
    op.drop_index(op.f("ix_entity_suggestions_entity_snapshot_id"), table_name="entity_suggestions")
    op.drop_index(op.f("ix_entity_suggestions_suggestion_run_id"), table_name="entity_suggestions")
    op.drop_index(op.f("ix_entity_suggestions_sync_run_id"), table_name="entity_suggestions")
    op.drop_index(op.f("ix_entity_suggestions_profile_id"), table_name="entity_suggestions")
    op.drop_table("entity_suggestions")

    op.drop_index(op.f("ix_entity_suggestion_runs_status"), table_name="entity_suggestion_runs")
    op.drop_index(op.f("ix_entity_suggestion_runs_sync_run_id"), table_name="entity_suggestion_runs")
    op.drop_index(op.f("ix_entity_suggestion_runs_profile_id"), table_name="entity_suggestion_runs")
    op.drop_table("entity_suggestion_runs")
