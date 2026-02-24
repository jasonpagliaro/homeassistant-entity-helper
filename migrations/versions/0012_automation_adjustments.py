"""automation adjustments workflow tables

Revision ID: 0012_automation_adjustments
Revises: 0011_update_manager_status
Create Date: 2026-02-24 00:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0012_automation_adjustments"
down_revision = "0011_update_manager_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "automation_adjustment_drafts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("source_entity_id", sa.String(length=255), nullable=False),
        sa.Column("source_config_key", sa.String(length=255), nullable=True),
        sa.Column("source_alias", sa.String(length=255), nullable=True),
        sa.Column("working_yaml_text", sa.Text(), nullable=False),
        sa.Column("working_structured_json", sa.Text(), nullable=False),
        sa.Column("queue_status", sa.String(length=32), nullable=False, server_default="draft"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_test_action_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_automation_adjustment_drafts_profile_id"),
        "automation_adjustment_drafts",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_drafts_source_entity_id"),
        "automation_adjustment_drafts",
        ["source_entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_drafts_source_config_key"),
        "automation_adjustment_drafts",
        ["source_config_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_drafts_queue_status"),
        "automation_adjustment_drafts",
        ["queue_status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_drafts_last_test_action_id"),
        "automation_adjustment_drafts",
        ["last_test_action_id"],
        unique=False,
    )

    op.create_table(
        "automation_adjustment_revisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("draft_id", sa.Integer(), nullable=False),
        sa.Column("revision_index", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("section", sa.String(length=32), nullable=True),
        sa.Column("prompt_text", sa.Text(), nullable=True),
        sa.Column("yaml_text", sa.Text(), nullable=False),
        sa.Column("structured_json", sa.Text(), nullable=False),
        sa.Column("change_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["draft_id"], ["automation_adjustment_drafts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_automation_adjustment_revisions_draft_id"),
        "automation_adjustment_revisions",
        ["draft_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_revisions_revision_index"),
        "automation_adjustment_revisions",
        ["revision_index"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_revisions_source"),
        "automation_adjustment_revisions",
        ["source"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_revisions_section"),
        "automation_adjustment_revisions",
        ["section"],
        unique=False,
    )

    op.create_table(
        "automation_adjustment_actions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("draft_id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="failed"),
        sa.Column("request_payload_json", sa.Text(), nullable=True),
        sa.Column("response_payload_json", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("old_entity_id", sa.String(length=255), nullable=True),
        sa.Column("old_config_key", sa.String(length=255), nullable=True),
        sa.Column("new_entity_id", sa.String(length=255), nullable=True),
        sa.Column("new_config_key", sa.String(length=255), nullable=True),
        sa.Column("new_alias", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["draft_id"], ["automation_adjustment_drafts.id"]),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_automation_adjustment_actions_draft_id"),
        "automation_adjustment_actions",
        ["draft_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_actions_profile_id"),
        "automation_adjustment_actions",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_actions_operation"),
        "automation_adjustment_actions",
        ["operation"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_actions_status"),
        "automation_adjustment_actions",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_actions_old_entity_id"),
        "automation_adjustment_actions",
        ["old_entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_actions_old_config_key"),
        "automation_adjustment_actions",
        ["old_config_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_actions_new_entity_id"),
        "automation_adjustment_actions",
        ["new_entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_adjustment_actions_new_config_key"),
        "automation_adjustment_actions",
        ["new_config_key"],
        unique=False,
    )

    with op.batch_alter_table("automation_adjustment_drafts", schema=None) as batch_op:
        batch_op.create_foreign_key(
            "fk_automation_adjustment_drafts_last_test_action_id",
            "automation_adjustment_actions",
            ["last_test_action_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("automation_adjustment_drafts", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_automation_adjustment_drafts_last_test_action_id",
            type_="foreignkey",
        )

    op.drop_index(op.f("ix_automation_adjustment_actions_new_config_key"), table_name="automation_adjustment_actions")
    op.drop_index(op.f("ix_automation_adjustment_actions_new_entity_id"), table_name="automation_adjustment_actions")
    op.drop_index(op.f("ix_automation_adjustment_actions_old_config_key"), table_name="automation_adjustment_actions")
    op.drop_index(op.f("ix_automation_adjustment_actions_old_entity_id"), table_name="automation_adjustment_actions")
    op.drop_index(op.f("ix_automation_adjustment_actions_status"), table_name="automation_adjustment_actions")
    op.drop_index(op.f("ix_automation_adjustment_actions_operation"), table_name="automation_adjustment_actions")
    op.drop_index(op.f("ix_automation_adjustment_actions_profile_id"), table_name="automation_adjustment_actions")
    op.drop_index(op.f("ix_automation_adjustment_actions_draft_id"), table_name="automation_adjustment_actions")
    op.drop_table("automation_adjustment_actions")

    op.drop_index(op.f("ix_automation_adjustment_revisions_section"), table_name="automation_adjustment_revisions")
    op.drop_index(op.f("ix_automation_adjustment_revisions_source"), table_name="automation_adjustment_revisions")
    op.drop_index(
        op.f("ix_automation_adjustment_revisions_revision_index"),
        table_name="automation_adjustment_revisions",
    )
    op.drop_index(op.f("ix_automation_adjustment_revisions_draft_id"), table_name="automation_adjustment_revisions")
    op.drop_table("automation_adjustment_revisions")

    op.drop_index(
        op.f("ix_automation_adjustment_drafts_last_test_action_id"),
        table_name="automation_adjustment_drafts",
    )
    op.drop_index(op.f("ix_automation_adjustment_drafts_queue_status"), table_name="automation_adjustment_drafts")
    op.drop_index(
        op.f("ix_automation_adjustment_drafts_source_config_key"),
        table_name="automation_adjustment_drafts",
    )
    op.drop_index(
        op.f("ix_automation_adjustment_drafts_source_entity_id"),
        table_name="automation_adjustment_drafts",
    )
    op.drop_index(op.f("ix_automation_adjustment_drafts_profile_id"), table_name="automation_adjustment_drafts")
    op.drop_table("automation_adjustment_drafts")
