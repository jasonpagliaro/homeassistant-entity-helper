"""automation drafts

Revision ID: 0006_automation_drafts
Revises: 0005_entity_suggestions
Create Date: 2026-02-15 14:15:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_automation_drafts"
down_revision = "0005_entity_suggestions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "automation_draft_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("suggestion_run_id", sa.Integer(), nullable=False),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("generated_count", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("llm_model", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["suggestion_run_id"], ["entity_suggestion_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_automation_draft_runs_profile_id"),
        "automation_draft_runs",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_draft_runs_suggestion_run_id"),
        "automation_draft_runs",
        ["suggestion_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_draft_runs_status"),
        "automation_draft_runs",
        ["status"],
        unique=False,
    )

    op.create_table(
        "automation_drafts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("draft_run_id", sa.Integer(), nullable=False),
        sa.Column("suggestion_run_id", sa.Integer(), nullable=False),
        sa.Column("entity_suggestion_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.String(length=255), nullable=False),
        sa.Column("template_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("yaml_text", sa.Text(), nullable=True),
        sa.Column("structured_json", sa.Text(), nullable=True),
        sa.Column("rationale_json", sa.Text(), nullable=True),
        sa.Column("generation_status", sa.String(length=32), nullable=False),
        sa.Column("generation_error", sa.Text(), nullable=True),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pulled_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["draft_run_id"], ["automation_draft_runs.id"]),
        sa.ForeignKeyConstraint(["suggestion_run_id"], ["entity_suggestion_runs.id"]),
        sa.ForeignKeyConstraint(["entity_suggestion_id"], ["entity_suggestions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_automation_drafts_profile_id"),
        "automation_drafts",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_drafts_draft_run_id"),
        "automation_drafts",
        ["draft_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_drafts_suggestion_run_id"),
        "automation_drafts",
        ["suggestion_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_drafts_entity_suggestion_id"),
        "automation_drafts",
        ["entity_suggestion_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_drafts_entity_id"),
        "automation_drafts",
        ["entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_drafts_template_id"),
        "automation_drafts",
        ["template_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_drafts_generation_status"),
        "automation_drafts",
        ["generation_status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_automation_drafts_review_status"),
        "automation_drafts",
        ["review_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_automation_drafts_review_status"), table_name="automation_drafts")
    op.drop_index(op.f("ix_automation_drafts_generation_status"), table_name="automation_drafts")
    op.drop_index(op.f("ix_automation_drafts_template_id"), table_name="automation_drafts")
    op.drop_index(op.f("ix_automation_drafts_entity_id"), table_name="automation_drafts")
    op.drop_index(op.f("ix_automation_drafts_entity_suggestion_id"), table_name="automation_drafts")
    op.drop_index(op.f("ix_automation_drafts_suggestion_run_id"), table_name="automation_drafts")
    op.drop_index(op.f("ix_automation_drafts_draft_run_id"), table_name="automation_drafts")
    op.drop_index(op.f("ix_automation_drafts_profile_id"), table_name="automation_drafts")
    op.drop_table("automation_drafts")

    op.drop_index(op.f("ix_automation_draft_runs_status"), table_name="automation_draft_runs")
    op.drop_index(op.f("ix_automation_draft_runs_suggestion_run_id"), table_name="automation_draft_runs")
    op.drop_index(op.f("ix_automation_draft_runs_profile_id"), table_name="automation_draft_runs")
    op.drop_table("automation_draft_runs")
