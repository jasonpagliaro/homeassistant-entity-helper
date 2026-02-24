"""add app config fields for update manager runtime status

Revision ID: 0011_update_manager_status
Revises: 0010_app_config_updates
Create Date: 2026-02-24 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0011_update_manager_status"
down_revision = "0010_app_config_updates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "app_config",
        sa.Column("last_update_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "app_config",
        sa.Column("last_update_result", sa.String(length=64), nullable=False, server_default="never"),
    )
    op.create_index(
        op.f("ix_app_config_last_update_result"),
        "app_config",
        ["last_update_result"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_app_config_last_update_result"), table_name="app_config")
    op.drop_column("app_config", "last_update_result")
    op.drop_column("app_config", "last_update_attempt_at")
