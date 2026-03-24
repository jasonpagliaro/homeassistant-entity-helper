"""state snapshot enrichment fields

Revision ID: 0014_state_snapshot_enrichment
Revises: 0013_service_catalog
Create Date: 2026-03-23 20:15:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0014_state_snapshot_enrichment"
down_revision = "0013_service_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("sync_runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("registry_enrichment_available", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("registry_enrichment_error", sa.Text(), nullable=True))

    with op.batch_alter_table("entity_snapshots", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_payload_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("has_entity_registry", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("has_device_registry", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("last_reported", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("entity_snapshots", schema=None) as batch_op:
        batch_op.drop_column("last_reported")
        batch_op.drop_column("has_device_registry")
        batch_op.drop_column("has_entity_registry")
        batch_op.drop_column("source_payload_json")

    with op.batch_alter_table("sync_runs", schema=None) as batch_op:
        batch_op.drop_column("registry_enrichment_error")
        batch_op.drop_column("registry_enrichment_available")
