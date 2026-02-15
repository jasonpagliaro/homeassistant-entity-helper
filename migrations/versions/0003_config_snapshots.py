"""config snapshots

Revision ID: 0003_config_snapshots
Revises: 0002_entity_enrichment
Create Date: 2026-02-15 14:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_config_snapshots"
down_revision = "0002_entity_enrichment"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "config_sync_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_config_sync_runs_profile_id"),
        "config_sync_runs",
        ["profile_id"],
        unique=False,
    )

    op.create_table(
        "config_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("config_sync_run_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("entity_id", sa.String(length=255), nullable=False),
        sa.Column("config_key", sa.String(length=255), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("state", sa.String(length=255), nullable=True),
        sa.Column("fetch_status", sa.String(length=32), nullable=False),
        sa.Column("fetch_error", sa.Text(), nullable=True),
        sa.Column("summary_json", sa.Text(), nullable=True),
        sa.Column("references_json", sa.Text(), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("attributes_json", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("last_changed", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["config_sync_run_id"], ["config_sync_runs.id"]),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_config_snapshots_profile_id"),
        "config_snapshots",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_config_snapshots_config_sync_run_id"),
        "config_snapshots",
        ["config_sync_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_config_snapshots_kind"),
        "config_snapshots",
        ["kind"],
        unique=False,
    )
    op.create_index(
        op.f("ix_config_snapshots_entity_id"),
        "config_snapshots",
        ["entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_config_snapshots_config_key"),
        "config_snapshots",
        ["config_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_config_snapshots_fetch_status"),
        "config_snapshots",
        ["fetch_status"],
        unique=False,
    )
    op.create_index(
        "ix_config_snapshots_profile_run_kind",
        "config_snapshots",
        ["profile_id", "config_sync_run_id", "kind"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_config_snapshots_profile_run_kind", table_name="config_snapshots")
    op.drop_index(op.f("ix_config_snapshots_fetch_status"), table_name="config_snapshots")
    op.drop_index(op.f("ix_config_snapshots_config_key"), table_name="config_snapshots")
    op.drop_index(op.f("ix_config_snapshots_entity_id"), table_name="config_snapshots")
    op.drop_index(op.f("ix_config_snapshots_kind"), table_name="config_snapshots")
    op.drop_index(op.f("ix_config_snapshots_config_sync_run_id"), table_name="config_snapshots")
    op.drop_index(op.f("ix_config_snapshots_profile_id"), table_name="config_snapshots")
    op.drop_table("config_snapshots")

    op.drop_index(op.f("ix_config_sync_runs_profile_id"), table_name="config_sync_runs")
    op.drop_table("config_sync_runs")
