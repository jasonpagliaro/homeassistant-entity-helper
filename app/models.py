from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Profile(SQLModel, table=True):
    __tablename__ = "profiles"
    __table_args__ = (UniqueConstraint("name", name="uq_profiles_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, max_length=128)
    base_url: str = Field(max_length=512)
    token: str = Field(default="", sa_column=Column(Text, nullable=False))
    token_env_var: Optional[str] = Field(default="HA_TOKEN", max_length=128)
    is_enabled: bool = Field(default=True, index=True)
    verify_tls: bool = Field(default=True)
    timeout_seconds: int = Field(default=10, ge=1, le=120)
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SyncRun(SQLModel, table=True):
    __tablename__ = "sync_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    pulled_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    entity_count: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    status: str = Field(default="success", max_length=32)
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))


class EntitySnapshot(SQLModel, table=True):
    __tablename__ = "entity_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    sync_run_id: int = Field(foreign_key="sync_runs.id", index=True)
    entity_id: str = Field(index=True, max_length=255)
    domain: str = Field(index=True, max_length=64)
    state: str = Field(index=True, max_length=255)
    friendly_name: Optional[str] = Field(default=None, max_length=255)
    device_id: Optional[str] = Field(default=None, max_length=255)
    device_name: Optional[str] = Field(default=None, max_length=255)
    area_id: Optional[str] = Field(default=None, max_length=255)
    area_name: Optional[str] = Field(default=None, max_length=255, index=True)
    floor_id: Optional[str] = Field(default=None, max_length=255)
    floor_name: Optional[str] = Field(default=None, max_length=255)
    location_name: Optional[str] = Field(default=None, max_length=255)
    labels_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    metadata_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    attributes_json: str = Field(sa_column=Column(Text, nullable=False))
    context_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    last_changed: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    pulled_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class ConfigSyncRun(SQLModel, table=True):
    __tablename__ = "config_sync_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    pulled_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    item_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    status: str = Field(default="success", max_length=32)
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))


class ConfigSnapshot(SQLModel, table=True):
    __tablename__ = "config_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    config_sync_run_id: int = Field(foreign_key="config_sync_runs.id", index=True)
    kind: str = Field(index=True, max_length=32)
    entity_id: str = Field(index=True, max_length=255)
    config_key: Optional[str] = Field(default=None, max_length=255)
    name: Optional[str] = Field(default=None, max_length=255)
    state: Optional[str] = Field(default=None, max_length=255)
    fetch_status: str = Field(index=True, max_length=32, default="success")
    fetch_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    summary_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    references_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    config_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    attributes_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    metadata_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    last_changed: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    pulled_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class EntitySuggestionRun(SQLModel, table=True):
    __tablename__ = "entity_suggestion_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    sync_run_id: int = Field(foreign_key="sync_runs.id", index=True)
    pulled_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    entity_count: int = Field(default=0, ge=0)
    ready_count: int = Field(default=0, ge=0)
    needs_review_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    status: str = Field(default="success", max_length=32, index=True)
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    policy_version: str = Field(default="v1", max_length=32)
    llm_enabled: bool = Field(default=False)
    llm_model: Optional[str] = Field(default=None, max_length=255)


class EntitySuggestion(SQLModel, table=True):
    __tablename__ = "entity_suggestions"
    __table_args__ = (
        UniqueConstraint(
            "suggestion_run_id",
            "entity_snapshot_id",
            name="uq_entity_suggestions_run_snapshot",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    sync_run_id: int = Field(foreign_key="sync_runs.id", index=True)
    suggestion_run_id: int = Field(foreign_key="entity_suggestion_runs.id", index=True)
    entity_snapshot_id: int = Field(foreign_key="entity_snapshots.id", index=True)
    entity_id: str = Field(index=True, max_length=255)
    domain: str = Field(index=True, max_length=64)
    readiness_status: str = Field(default="needs_review", max_length=32, index=True)
    missing_fields_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    issues_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    semantic_type_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    llm_suggestions_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    source_metadata_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    workflow_status: str = Field(default="open", max_length=32, index=True)
    workflow_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    workflow_payload_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    workflow_result_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    workflow_updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    pulled_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class AutomationDraftRun(SQLModel, table=True):
    __tablename__ = "automation_draft_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    suggestion_run_id: int = Field(foreign_key="entity_suggestion_runs.id", index=True)
    pulled_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    candidate_count: int = Field(default=0, ge=0)
    generated_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    status: str = Field(default="success", max_length=32, index=True)
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    llm_model: Optional[str] = Field(default=None, max_length=255)


class AutomationDraft(SQLModel, table=True):
    __tablename__ = "automation_drafts"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    draft_run_id: int = Field(foreign_key="automation_draft_runs.id", index=True)
    suggestion_run_id: int = Field(foreign_key="entity_suggestion_runs.id", index=True)
    entity_suggestion_id: int = Field(foreign_key="entity_suggestions.id", index=True)
    entity_id: str = Field(index=True, max_length=255)
    template_id: str = Field(index=True, max_length=64)
    title: str = Field(max_length=255)
    yaml_text: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    structured_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    rationale_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    generation_status: str = Field(default="success", max_length=32, index=True)
    generation_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    review_status: str = Field(default="pending", max_length=32, index=True)
    review_note: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    reviewed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    pulled_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class LLMConnection(SQLModel, table=True):
    __tablename__ = "llm_connections"
    __table_args__ = (
        UniqueConstraint("profile_id", "name", name="uq_llm_connections_profile_name"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    name: str = Field(index=True, max_length=128)
    provider_kind: str = Field(default="openai_compatible", index=True, max_length=64)
    base_url: str = Field(max_length=512)
    model: str = Field(max_length=255)
    api_key_env_var: Optional[str] = Field(default=None, max_length=128)
    timeout_seconds: int = Field(default=20, ge=1, le=300)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=900, ge=1, le=8192)
    extra_headers_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    is_enabled: bool = Field(default=True, index=True)
    is_default: bool = Field(default=False, index=True)
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SuggestionRun(SQLModel, table=True):
    __tablename__ = "suggestion_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    llm_connection_id: int = Field(foreign_key="llm_connections.id", index=True)
    config_sync_run_id: Optional[int] = Field(default=None, foreign_key="config_sync_runs.id", index=True)
    run_kind: str = Field(default="concept_v2", max_length=32, index=True)
    idea_type: str = Field(default="general", max_length=64)
    custom_intent: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    mode: str = Field(default="standard", max_length=32)
    top_k: int = Field(default=10, ge=1, le=25)
    include_existing: bool = Field(default=True)
    include_new: bool = Field(default=True)
    status: str = Field(default="queued", max_length=32, index=True)
    target_count: int = Field(default=0, ge=0)
    processed_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    invalid_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    context_hash: Optional[str] = Field(default=None, max_length=128)
    filters_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    result_summary_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    started_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    finished_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SuggestionProposal(SQLModel, table=True):
    __tablename__ = "suggestion_proposals"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    suggestion_run_id: int = Field(foreign_key="suggestion_runs.id", index=True)
    config_snapshot_id: Optional[int] = Field(default=None, foreign_key="config_snapshots.id", index=True)
    target_entity_id: str = Field(index=True, max_length=255)
    status: str = Field(default="proposed", index=True, max_length=32)
    schema_version: str = Field(default="haev.automation.suggestion.v1", max_length=64)
    summary: Optional[str] = Field(default=None, max_length=512)
    confidence: Optional[float] = Field(default=None)
    risk_level: Optional[str] = Field(default=None, max_length=32)
    concept_payload_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    concept_type: Optional[str] = Field(default=None, max_length=64)
    impact_score: Optional[float] = Field(default=None)
    feasibility_score: Optional[float] = Field(default=None)
    novelty_score: Optional[float] = Field(default=None)
    ranking_score: Optional[float] = Field(default=None, index=True)
    ranking_breakdown_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    duplicate_fingerprint: Optional[str] = Field(default=None, max_length=128, index=True)
    queue_stage: str = Field(default="suggested", max_length=32, index=True)
    queue_note: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    queue_updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    proposed_patch_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    verification_steps_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    raw_response_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    validation_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SuggestionAuditEvent(SQLModel, table=True):
    __tablename__ = "suggestion_audit_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    suggestion_run_id: Optional[int] = Field(default=None, foreign_key="suggestion_runs.id", index=True)
    proposal_id: Optional[int] = Field(default=None, foreign_key="suggestion_proposals.id", index=True)
    event_type: str = Field(index=True, max_length=64)
    actor: str = Field(default="system", max_length=64)
    payload_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SuggestionGeneration(SQLModel, table=True):
    __tablename__ = "suggestion_generations"

    id: Optional[int] = Field(default=None, primary_key=True)
    proposal_id: int = Field(foreign_key="suggestion_proposals.id", index=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    llm_connection_id: int = Field(foreign_key="llm_connections.id", index=True)
    mode: str = Field(default="auto", max_length=32, index=True)
    status: str = Field(default="running", max_length=32, index=True)
    optional_instruction: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    current_step: int = Field(default=0, ge=0)
    pending_question_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    planning_answers_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    final_yaml_text: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    final_structured_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    finished_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


class SuggestionGenerationRevision(SQLModel, table=True):
    __tablename__ = "suggestion_generation_revisions"

    id: Optional[int] = Field(default=None, primary_key=True)
    generation_id: int = Field(foreign_key="suggestion_generations.id", index=True)
    revision_index: int = Field(default=0, ge=0, index=True)
    source: str = Field(default="initial", max_length=32, index=True)
    prompt_text: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    yaml_text: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    structured_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    change_summary: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SuggestionSubmissionEvent(SQLModel, table=True):
    __tablename__ = "suggestion_submission_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    generation_id: int = Field(foreign_key="suggestion_generations.id", index=True)
    profile_id: int = Field(foreign_key="profiles.id", index=True)
    config_key: str = Field(max_length=255, index=True)
    operation: str = Field(max_length=32, index=True)
    previous_config_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    request_payload_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    response_payload_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    status: str = Field(default="failed", max_length=32, index=True)
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
