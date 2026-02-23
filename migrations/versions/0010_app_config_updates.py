"""add app config update checker settings

Revision ID: 0010_app_config_updates
Revises: 0009_concept_suggestions_v2
Create Date: 2026-02-23 15:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0010_app_config_updates"
down_revision = "0009_concept_suggestions_v2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_config",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("updates_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("update_repo_owner", sa.String(length=128), nullable=False, server_default="jasonpagliaro"),
        sa.Column("update_repo_name", sa.String(length=128), nullable=False, server_default="homeassistant-entity-helper"),
        sa.Column("update_repo_branch", sa.String(length=128), nullable=False, server_default="main"),
        sa.Column("update_check_interval_minutes", sa.Integer(), nullable=False, server_default="720"),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_check_state", sa.String(length=32), nullable=False, server_default="never"),
        sa.Column("last_check_error", sa.Text(), nullable=True),
        sa.Column("installed_commit_sha", sa.String(length=64), nullable=True),
        sa.Column("latest_commit_sha", sa.String(length=64), nullable=True),
        sa.Column("latest_commit_url", sa.String(length=512), nullable=True),
        sa.Column("latest_commit_published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_commit_sha", sa.String(length=64), nullable=True),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_app_config_last_check_state"), "app_config", ["last_check_state"], unique=False)
    op.create_index(op.f("ix_app_config_latest_commit_sha"), "app_config", ["latest_commit_sha"], unique=False)
    op.create_index(
        op.f("ix_app_config_dismissed_commit_sha"),
        "app_config",
        ["dismissed_commit_sha"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_app_config_dismissed_commit_sha"), table_name="app_config")
    op.drop_index(op.f("ix_app_config_latest_commit_sha"), table_name="app_config")
    op.drop_index(op.f("ix_app_config_last_check_state"), table_name="app_config")
    op.drop_table("app_config")
