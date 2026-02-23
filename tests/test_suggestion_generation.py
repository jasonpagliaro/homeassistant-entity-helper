from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app import db
from app.main import create_app
from app.models import (
    LLMConnection,
    Profile,
    SuggestionGeneration,
    SuggestionGenerationRevision,
    SuggestionProposal,
    SuggestionRun,
    utcnow,
)


def extract_csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SESSION_SECRET", "generation-test-secret")
    db.reset_engine_for_tests()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    db.reset_engine_for_tests()


def seed_generation_data() -> tuple[int, int]:
    with Session(db.get_engine()) as session:
        profile = Profile(
            name="generation-home",
            base_url="http://ha.local:8123",
            token="token",
            token_env_var="HA_TOKEN",
            verify_tls=True,
            timeout_seconds=10,
            is_enabled=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(profile)
        session.commit()
        session.refresh(profile)
        assert profile.id is not None

        connection = LLMConnection(
            profile_id=profile.id,
            name="Generation LLM",
            provider_kind="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3.1",
            api_key_env_var="GENERATION_LLM_KEY",
            timeout_seconds=20,
            temperature=0.2,
            max_output_tokens=900,
            extra_headers_json=None,
            is_enabled=True,
            is_default=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(connection)
        session.commit()
        session.refresh(connection)
        assert connection.id is not None

        run = SuggestionRun(
            profile_id=profile.id,
            llm_connection_id=connection.id,
            config_sync_run_id=None,
            run_kind="concept_v2",
            idea_type="general",
            custom_intent=None,
            mode="standard",
            top_k=10,
            include_existing=True,
            include_new=True,
            status="succeeded",
            target_count=1,
            processed_count=1,
            success_count=1,
            invalid_count=0,
            error_count=0,
            error=None,
            context_hash="hash",
            filters_json=json.dumps({}),
            result_summary_json=json.dumps({"success_count": 1}),
            started_at=utcnow(),
            finished_at=utcnow(),
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        assert run.id is not None

        proposal = SuggestionProposal(
            profile_id=profile.id,
            suggestion_run_id=run.id,
            config_snapshot_id=None,
            target_entity_id="automation.generated_test",
            status="proposed",
            schema_version="haev.automation.concept.v2",
            summary="Create a resilient reminder automation.",
            confidence=0.8,
            risk_level="low",
            concept_payload_json=json.dumps(
                {
                    "title": "Resilient reminder",
                    "summary": "Create a resilient reminder automation.",
                    "concept_type": "general",
                    "target_kind": "new_automation",
                    "involved_entities": ["sensor.test"],
                }
            ),
            concept_type="general",
            impact_score=0.7,
            feasibility_score=0.8,
            novelty_score=0.4,
            ranking_score=0.72,
            ranking_breakdown_json=json.dumps({}),
            duplicate_fingerprint="fp",
            queue_stage="queued",
            queue_note=None,
            queue_updated_at=utcnow(),
            proposed_patch_json=None,
            verification_steps_json=json.dumps([]),
            raw_response_json=json.dumps({}),
            validation_error=None,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(proposal)
        session.commit()
        session.refresh(proposal)
        assert proposal.id is not None

        return profile.id, proposal.id


def test_generate_auto_from_concept_creates_revision(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_id, proposal_id = seed_generation_data()
    settings = client.get(f"/settings?profile_id={profile_id}")
    assert settings.status_code == 200
    csrf_token = extract_csrf(settings.text)

    async def fake_generate_yaml_from_concept(**_: Any) -> tuple[str, dict[str, Any]]:
        payload = {
            "alias": "Resilient reminder",
            "description": "Generated from concept",
            "trigger": [{"platform": "state", "entity_id": "sensor.test"}],
            "condition": [],
            "action": [{"service": "notify.notify", "data": {"message": "hi"}}],
            "mode": "single",
        }
        return (
            (
                "alias: Resilient reminder\n"
                "description: Generated from concept\n"
                "trigger:\n"
                "- platform: state\n"
                "  entity_id: sensor.test\n"
                "condition: []\n"
                "action:\n"
                "- service: notify.notify\n"
                "  data:\n"
                "    message: hi\n"
                "mode: single\n"
            ),
            payload,
        )

    monkeypatch.setattr("app.main.generate_yaml_from_concept", fake_generate_yaml_from_concept)

    response = client.post(
        f"/suggestions/proposals/{proposal_id}/generate",
        data={
            "csrf_token": csrf_token,
            "generation_mode": "auto",
            "optional_instruction": "Keep it concise",
            "next_url": f"/suggestions/queue?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers.get("location", "")
    match = re.search(r"/suggestions/generations/(\d+)\?", location)
    assert match is not None
    generation_id = int(match.group(1))

    with Session(db.get_engine()) as session:
        generation = session.get(SuggestionGeneration, generation_id)
        assert generation is not None
        assert generation.status == "completed"
        assert generation.final_yaml_text is not None

        revisions = list(
            session.exec(
                select(SuggestionGenerationRevision).where(
                    SuggestionGenerationRevision.generation_id == generation_id
                )
            ).all()
        )
        assert len(revisions) == 1

        proposal = session.get(SuggestionProposal, proposal_id)
        assert proposal is not None
        assert proposal.queue_stage == "yaml_generated"
