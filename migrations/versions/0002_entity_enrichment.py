"""entity enrichment fields

Revision ID: 0002_entity_enrichment
Revises: 0001_initial
Create Date: 2026-02-15 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_entity_enrichment"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("entity_snapshots", schema=None) as batch_op:
        batch_op.add_column(sa.Column("friendly_name", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("device_id", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("device_name", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("area_id", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("area_name", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("floor_id", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("floor_name", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("location_name", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("labels_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("metadata_json", sa.Text(), nullable=True))

    op.create_index(op.f("ix_entity_snapshots_area_name"), "entity_snapshots", ["area_name"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_entity_snapshots_area_name"), table_name="entity_snapshots")

    with op.batch_alter_table("entity_snapshots", schema=None) as batch_op:
        batch_op.drop_column("metadata_json")
        batch_op.drop_column("labels_json")
        batch_op.drop_column("location_name")
        batch_op.drop_column("floor_name")
        batch_op.drop_column("floor_id")
        batch_op.drop_column("area_name")
        batch_op.drop_column("area_id")
        batch_op.drop_column("device_name")
        batch_op.drop_column("device_id")
        batch_op.drop_column("friendly_name")
