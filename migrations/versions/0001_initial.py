"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-02-15 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "profiles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("base_url", sa.String(length=512), nullable=False),
        sa.Column("token", sa.Text(), nullable=False),
        sa.Column("token_env_var", sa.String(length=128), nullable=True),
        sa.Column("verify_tls", sa.Boolean(), nullable=False),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_profiles_name"),
    )
    op.create_index(op.f("ix_profiles_name"), "profiles", ["name"], unique=False)

    op.create_table(
        "sync_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entity_count", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_sync_runs_profile_id"), "sync_runs", ["profile_id"], unique=False)

    op.create_table(
        "entity_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("sync_run_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.String(length=255), nullable=False),
        sa.Column("domain", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=255), nullable=False),
        sa.Column("attributes_json", sa.Text(), nullable=False),
        sa.Column("context_json", sa.Text(), nullable=True),
        sa.Column("last_changed", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["sync_run_id"], ["sync_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_entity_snapshots_profile_id"),
        "entity_snapshots",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_snapshots_sync_run_id"),
        "entity_snapshots",
        ["sync_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_snapshots_entity_id"),
        "entity_snapshots",
        ["entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_snapshots_domain"),
        "entity_snapshots",
        ["domain"],
        unique=False,
    )
    op.create_index(
        op.f("ix_entity_snapshots_state"),
        "entity_snapshots",
        ["state"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_entity_snapshots_state"), table_name="entity_snapshots")
    op.drop_index(op.f("ix_entity_snapshots_domain"), table_name="entity_snapshots")
    op.drop_index(op.f("ix_entity_snapshots_entity_id"), table_name="entity_snapshots")
    op.drop_index(op.f("ix_entity_snapshots_sync_run_id"), table_name="entity_snapshots")
    op.drop_index(op.f("ix_entity_snapshots_profile_id"), table_name="entity_snapshots")
    op.drop_table("entity_snapshots")

    op.drop_index(op.f("ix_sync_runs_profile_id"), table_name="sync_runs")
    op.drop_table("sync_runs")

    op.drop_index(op.f("ix_profiles_name"), table_name="profiles")
    op.drop_table("profiles")
