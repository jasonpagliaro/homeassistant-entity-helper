"""entity suggestion workflow state

Revision ID: 0008_entity_suggestion_workflow
Revises: 0007_llm_automation_suggestions
Create Date: 2026-02-22 08:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0008_entity_suggestion_workflow"
down_revision = "0007_llm_automation_suggestions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("entity_suggestions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "workflow_status",
                sa.String(length=32),
                nullable=False,
                server_default="open",
            )
        )
        batch_op.add_column(sa.Column("workflow_error", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("workflow_payload_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("workflow_result_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("workflow_updated_at", sa.DateTime(timezone=True), nullable=True))

    op.create_index(
        op.f("ix_entity_suggestions_workflow_status"),
        "entity_suggestions",
        ["workflow_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_entity_suggestions_workflow_status"), table_name="entity_suggestions")

    with op.batch_alter_table("entity_suggestions", schema=None) as batch_op:
        batch_op.drop_column("workflow_updated_at")
        batch_op.drop_column("workflow_result_json")
        batch_op.drop_column("workflow_payload_json")
        batch_op.drop_column("workflow_error")
        batch_op.drop_column("workflow_status")
