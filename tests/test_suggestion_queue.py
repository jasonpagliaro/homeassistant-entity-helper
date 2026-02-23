from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app import db
from app.main import create_app
from app.models import LLMConnection, Profile, SuggestionProposal, SuggestionRun, utcnow


def extract_csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SESSION_SECRET", "queue-test-secret")
    db.reset_engine_for_tests()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    db.reset_engine_for_tests()


def seed_queue_data() -> tuple[int, int, int]:
    with Session(db.get_engine()) as session:
        profile = Profile(
            name="queue-home",
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
            name="Queue LLM",
            provider_kind="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3.1",
            api_key_env_var="QUEUE_LLM_KEY",
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
            target_count=2,
            processed_count=2,
            success_count=2,
            invalid_count=0,
            error_count=0,
            error=None,
            context_hash="hash",
            filters_json=json.dumps({}),
            result_summary_json=json.dumps({"success_count": 2}),
            started_at=utcnow(),
            finished_at=utcnow(),
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        assert run.id is not None

        concept_payload = {
            "title": "Shared concept",
            "summary": "Shared summary",
            "concept_type": "general",
            "target_kind": "new_automation",
            "involved_entities": ["sensor.temp"],
            "impact_score": 0.7,
            "feasibility_score": 0.7,
            "novelty_score": 0.4,
            "confidence": 0.8,
            "risk_level": "low",
        }
        fingerprint = "dupe-fingerprint"
        one = SuggestionProposal(
            profile_id=profile.id,
            suggestion_run_id=run.id,
            config_snapshot_id=None,
            target_entity_id="automation.generated_a",
            status="proposed",
            schema_version="haev.automation.concept.v2",
            summary="Shared summary",
            confidence=0.8,
            risk_level="low",
            concept_payload_json=json.dumps(concept_payload),
            concept_type="general",
            impact_score=0.7,
            feasibility_score=0.7,
            novelty_score=0.4,
            ranking_score=0.7,
            ranking_breakdown_json=json.dumps({}),
            duplicate_fingerprint=fingerprint,
            queue_stage="suggested",
            queue_note=None,
            queue_updated_at=utcnow(),
            proposed_patch_json=None,
            verification_steps_json=json.dumps([]),
            raw_response_json=json.dumps({}),
            validation_error=None,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        two = SuggestionProposal(
            profile_id=profile.id,
            suggestion_run_id=run.id,
            config_snapshot_id=None,
            target_entity_id="automation.generated_b",
            status="proposed",
            schema_version="haev.automation.concept.v2",
            summary="Shared summary",
            confidence=0.8,
            risk_level="low",
            concept_payload_json=json.dumps(concept_payload),
            concept_type="general",
            impact_score=0.7,
            feasibility_score=0.7,
            novelty_score=0.4,
            ranking_score=0.7,
            ranking_breakdown_json=json.dumps({}),
            duplicate_fingerprint=fingerprint,
            queue_stage="suggested",
            queue_note=None,
            queue_updated_at=utcnow(),
            proposed_patch_json=None,
            verification_steps_json=json.dumps([]),
            raw_response_json=json.dumps({}),
            validation_error=None,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(one)
        session.add(two)
        session.commit()
        session.refresh(one)
        session.refresh(two)
        assert one.id is not None
        assert two.id is not None

        return profile.id, one.id, two.id


def test_queue_duplicate_warning_then_force(client: TestClient) -> None:
    profile_id, first_id, second_id = seed_queue_data()
    settings = client.get(f"/settings?profile_id={profile_id}")
    assert settings.status_code == 200
    csrf_token = extract_csrf(settings.text)

    queue_one = client.post(
        f"/suggestions/proposals/{first_id}/queue",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/suggestions/queue?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert queue_one.status_code == 303

    with Session(db.get_engine()) as session:
        first = session.get(SuggestionProposal, first_id)
        assert first is not None
        assert first.queue_stage == "queued"

    queue_two_without_force = client.post(
        f"/suggestions/proposals/{second_id}/queue",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/suggestions/queue?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert queue_two_without_force.status_code == 303

    with Session(db.get_engine()) as session:
        second = session.get(SuggestionProposal, second_id)
        assert second is not None
        assert second.queue_stage == "suggested"

    queue_two_force = client.post(
        f"/suggestions/proposals/{second_id}/queue",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/suggestions/queue?profile_id={profile_id}",
            "force_duplicate": "on",
        },
        follow_redirects=False,
    )
    assert queue_two_force.status_code == 303

    with Session(db.get_engine()) as session:
        second = session.get(SuggestionProposal, second_id)
        assert second is not None
        assert second.queue_stage == "queued"
