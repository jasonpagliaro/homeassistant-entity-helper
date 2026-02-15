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
