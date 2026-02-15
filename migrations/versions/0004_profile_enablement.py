"""profile enablement

Revision ID: 0004_profile_enablement
Revises: 0003_config_snapshots
Create Date: 2026-02-15 10:15:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_profile_enablement"
down_revision = "0003_config_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "profiles",
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_index(op.f("ix_profiles_is_enabled"), "profiles", ["is_enabled"], unique=False)
    op.alter_column("profiles", "is_enabled", server_default=None)


def downgrade() -> None:
    op.drop_index(op.f("ix_profiles_is_enabled"), table_name="profiles")
    op.drop_column("profiles", "is_enabled")
