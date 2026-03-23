"""service catalog snapshots

Revision ID: 0013_service_catalog
Revises: 0012_automation_adjustments
Create Date: 2026-03-22 15:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0013_service_catalog"
down_revision = "0012_automation_adjustments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "service_sync_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("domain_count", sa.Integer(), nullable=False),
        sa.Column("service_count", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_service_sync_runs_profile_id"),
        "service_sync_runs",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_service_sync_runs_status"),
        "service_sync_runs",
        ["status"],
        unique=False,
    )

    op.create_table(
        "service_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("service_sync_run_id", sa.Integer(), nullable=False),
        sa.Column("domain", sa.String(length=128), nullable=False),
        sa.Column("service_name", sa.String(length=128), nullable=False),
        sa.Column("service_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("fields_json", sa.Text(), nullable=True),
        sa.Column("target_json", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["service_sync_run_id"], ["service_sync_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "service_sync_run_id",
            "service_id",
            name="uq_service_snapshots_run_service_id",
        ),
    )
    op.create_index(
        op.f("ix_service_snapshots_profile_id"),
        "service_snapshots",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_service_snapshots_service_sync_run_id"),
        "service_snapshots",
        ["service_sync_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_service_snapshots_domain"),
        "service_snapshots",
        ["domain"],
        unique=False,
    )
    op.create_index(
        op.f("ix_service_snapshots_service_name"),
        "service_snapshots",
        ["service_name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_service_snapshots_service_id"),
        "service_snapshots",
        ["service_id"],
        unique=False,
    )
    op.create_index(
        "ix_service_snapshots_profile_run_domain",
        "service_snapshots",
        ["profile_id", "service_sync_run_id", "domain"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_service_snapshots_profile_run_domain", table_name="service_snapshots")
    op.drop_index(op.f("ix_service_snapshots_service_id"), table_name="service_snapshots")
    op.drop_index(op.f("ix_service_snapshots_service_name"), table_name="service_snapshots")
    op.drop_index(op.f("ix_service_snapshots_domain"), table_name="service_snapshots")
    op.drop_index(
        op.f("ix_service_snapshots_service_sync_run_id"),
        table_name="service_snapshots",
    )
    op.drop_index(op.f("ix_service_snapshots_profile_id"), table_name="service_snapshots")
    op.drop_table("service_snapshots")

    op.drop_index(op.f("ix_service_sync_runs_status"), table_name="service_sync_runs")
    op.drop_index(op.f("ix_service_sync_runs_profile_id"), table_name="service_sync_runs")
    op.drop_table("service_sync_runs")
