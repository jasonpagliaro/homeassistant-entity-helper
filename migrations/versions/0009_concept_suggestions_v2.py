"""concept-first suggestions v2

Revision ID: 0009_concept_suggestions_v2
Revises: 0008_entity_suggestion_workflow
Create Date: 2026-02-23 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0009_concept_suggestions_v2"
down_revision = "0008_entity_suggestion_workflow"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("suggestion_runs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "run_kind",
                sa.String(length=32),
                nullable=False,
                server_default="concept_v2",
            )
        )
        batch_op.add_column(
            sa.Column(
                "idea_type",
                sa.String(length=64),
                nullable=False,
                server_default="general",
            )
        )
        batch_op.add_column(sa.Column("custom_intent", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "mode",
                sa.String(length=32),
                nullable=False,
                server_default="standard",
            )
        )
        batch_op.add_column(
            sa.Column(
                "top_k",
                sa.Integer(),
                nullable=False,
                server_default="10",
            )
        )
        batch_op.add_column(
            sa.Column(
                "include_existing",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        batch_op.add_column(
            sa.Column(
                "include_new",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )

    op.create_index(op.f("ix_suggestion_runs_run_kind"), "suggestion_runs", ["run_kind"], unique=False)

    with op.batch_alter_table("suggestion_proposals", schema=None) as batch_op:
        batch_op.add_column(sa.Column("concept_payload_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("concept_type", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("impact_score", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("feasibility_score", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("novelty_score", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("ranking_score", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("ranking_breakdown_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("duplicate_fingerprint", sa.String(length=128), nullable=True))
        batch_op.add_column(
            sa.Column(
                "queue_stage",
                sa.String(length=32),
                nullable=False,
                server_default="suggested",
            )
        )
        batch_op.add_column(sa.Column("queue_note", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("queue_updated_at", sa.DateTime(timezone=True), nullable=True))

    op.create_index(
        op.f("ix_suggestion_proposals_ranking_score"),
        "suggestion_proposals",
        ["ranking_score"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_proposals_duplicate_fingerprint"),
        "suggestion_proposals",
        ["duplicate_fingerprint"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_proposals_queue_stage"),
        "suggestion_proposals",
        ["queue_stage"],
        unique=False,
    )

    op.create_table(
        "suggestion_generations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("proposal_id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("llm_connection_id", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("optional_instruction", sa.Text(), nullable=True),
        sa.Column("current_step", sa.Integer(), nullable=False),
        sa.Column("pending_question_json", sa.Text(), nullable=True),
        sa.Column("planning_answers_json", sa.Text(), nullable=True),
        sa.Column("final_yaml_text", sa.Text(), nullable=True),
        sa.Column("final_structured_json", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["proposal_id"], ["suggestion_proposals.id"]),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.ForeignKeyConstraint(["llm_connection_id"], ["llm_connections.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_suggestion_generations_proposal_id"),
        "suggestion_generations",
        ["proposal_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_generations_profile_id"),
        "suggestion_generations",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_generations_llm_connection_id"),
        "suggestion_generations",
        ["llm_connection_id"],
        unique=False,
    )
    op.create_index(op.f("ix_suggestion_generations_mode"), "suggestion_generations", ["mode"], unique=False)
    op.create_index(op.f("ix_suggestion_generations_status"), "suggestion_generations", ["status"], unique=False)

    op.create_table(
        "suggestion_generation_revisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("generation_id", sa.Integer(), nullable=False),
        sa.Column("revision_index", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=True),
        sa.Column("yaml_text", sa.Text(), nullable=True),
        sa.Column("structured_json", sa.Text(), nullable=True),
        sa.Column("change_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["generation_id"], ["suggestion_generations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_suggestion_generation_revisions_generation_id"),
        "suggestion_generation_revisions",
        ["generation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_generation_revisions_revision_index"),
        "suggestion_generation_revisions",
        ["revision_index"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_generation_revisions_source"),
        "suggestion_generation_revisions",
        ["source"],
        unique=False,
    )

    op.create_table(
        "suggestion_submission_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("generation_id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("config_key", sa.String(length=255), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("previous_config_json", sa.Text(), nullable=True),
        sa.Column("request_payload_json", sa.Text(), nullable=True),
        sa.Column("response_payload_json", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["generation_id"], ["suggestion_generations.id"]),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_suggestion_submission_events_generation_id"),
        "suggestion_submission_events",
        ["generation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_submission_events_profile_id"),
        "suggestion_submission_events",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_submission_events_config_key"),
        "suggestion_submission_events",
        ["config_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_submission_events_operation"),
        "suggestion_submission_events",
        ["operation"],
        unique=False,
    )
    op.create_index(
        op.f("ix_suggestion_submission_events_status"),
        "suggestion_submission_events",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_suggestion_submission_events_status"), table_name="suggestion_submission_events")
    op.drop_index(op.f("ix_suggestion_submission_events_operation"), table_name="suggestion_submission_events")
    op.drop_index(op.f("ix_suggestion_submission_events_config_key"), table_name="suggestion_submission_events")
    op.drop_index(op.f("ix_suggestion_submission_events_profile_id"), table_name="suggestion_submission_events")
    op.drop_index(op.f("ix_suggestion_submission_events_generation_id"), table_name="suggestion_submission_events")
    op.drop_table("suggestion_submission_events")

    op.drop_index(
        op.f("ix_suggestion_generation_revisions_source"),
        table_name="suggestion_generation_revisions",
    )
    op.drop_index(
        op.f("ix_suggestion_generation_revisions_revision_index"),
        table_name="suggestion_generation_revisions",
    )
    op.drop_index(
        op.f("ix_suggestion_generation_revisions_generation_id"),
        table_name="suggestion_generation_revisions",
    )
    op.drop_table("suggestion_generation_revisions")

    op.drop_index(op.f("ix_suggestion_generations_status"), table_name="suggestion_generations")
    op.drop_index(op.f("ix_suggestion_generations_mode"), table_name="suggestion_generations")
    op.drop_index(op.f("ix_suggestion_generations_llm_connection_id"), table_name="suggestion_generations")
    op.drop_index(op.f("ix_suggestion_generations_profile_id"), table_name="suggestion_generations")
    op.drop_index(op.f("ix_suggestion_generations_proposal_id"), table_name="suggestion_generations")
    op.drop_table("suggestion_generations")

    op.drop_index(op.f("ix_suggestion_proposals_queue_stage"), table_name="suggestion_proposals")
    op.drop_index(
        op.f("ix_suggestion_proposals_duplicate_fingerprint"),
        table_name="suggestion_proposals",
    )
    op.drop_index(op.f("ix_suggestion_proposals_ranking_score"), table_name="suggestion_proposals")

    with op.batch_alter_table("suggestion_proposals", schema=None) as batch_op:
        batch_op.drop_column("queue_updated_at")
        batch_op.drop_column("queue_note")
        batch_op.drop_column("queue_stage")
        batch_op.drop_column("duplicate_fingerprint")
        batch_op.drop_column("ranking_breakdown_json")
        batch_op.drop_column("ranking_score")
        batch_op.drop_column("novelty_score")
        batch_op.drop_column("feasibility_score")
        batch_op.drop_column("impact_score")
        batch_op.drop_column("concept_type")
        batch_op.drop_column("concept_payload_json")

    op.drop_index(op.f("ix_suggestion_runs_run_kind"), table_name="suggestion_runs")

    with op.batch_alter_table("suggestion_runs", schema=None) as batch_op:
        batch_op.drop_column("include_new")
        batch_op.drop_column("include_existing")
        batch_op.drop_column("top_k")
        batch_op.drop_column("mode")
        batch_op.drop_column("custom_intent")
        batch_op.drop_column("idea_type")
        batch_op.drop_column("run_kind")
