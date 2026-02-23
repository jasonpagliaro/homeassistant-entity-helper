from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session

from app import db
from app.main import create_app
from app.models import (
    ConfigSnapshot,
    ConfigSyncRun,
    EntitySnapshot,
    LLMConnection,
    Profile,
    SuggestionGeneration,
    SuggestionProposal,
    SuggestionRun,
    SuggestionSubmissionEvent,
    SyncRun,
    utcnow,
)
from app.suggestion_context import (
    build_automation_context,
    build_concept_suggestion_context,
    build_known_entity_ids,
)


def test_build_automation_context_includes_linked_entities_and_related_config(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("APP_NAME", "Context Test")
    db.reset_engine_for_tests()


def test_build_concept_suggestion_context_includes_recent_concepts_and_submissions(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("APP_NAME", "Context Test")
    db.reset_engine_for_tests()

    app = create_app()
    with TestClient(app):
        pass

    with Session(db.get_engine()) as session:
        profile = Profile(
            name="ctx-concept",
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

        sync_run = SyncRun(
            profile_id=profile.id,
            pulled_at=utcnow(),
            entity_count=1,
            duration_ms=1,
            status="success",
            error=None,
        )
        session.add(sync_run)
        session.commit()
        session.refresh(sync_run)
        assert sync_run.id is not None

        session.add(
            EntitySnapshot(
                profile_id=profile.id,
                sync_run_id=sync_run.id,
                entity_id="sensor.office_temp",
                domain="sensor",
                state="72",
                friendly_name="Office Temp",
                attributes_json=json.dumps({}),
                pulled_at=utcnow(),
            )
        )
        session.commit()

        config_run = ConfigSyncRun(
            profile_id=profile.id,
            pulled_at=utcnow(),
            item_count=1,
            success_count=1,
            error_count=0,
            duration_ms=1,
            status="success",
            error=None,
        )
        session.add(config_run)
        session.commit()
        session.refresh(config_run)
        assert config_run.id is not None

        session.add(
            ConfigSnapshot(
                profile_id=profile.id,
                config_sync_run_id=config_run.id,
                kind="automation",
                entity_id="automation.office_temp_alert",
                config_key="office_temp_alert",
                name="Office Temp Alert",
                state="on",
                fetch_status="success",
                summary_json=json.dumps({"name": "Office Temp Alert"}),
                references_json=json.dumps({"entity_id": ["sensor.office_temp"]}),
                config_json=json.dumps({"alias": "Office Temp Alert"}),
                attributes_json=json.dumps({}),
                metadata_json=json.dumps({}),
                pulled_at=utcnow(),
            )
        )
        session.commit()

        connection = LLMConnection(
            profile_id=profile.id,
            name="Context LLM",
            provider_kind="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3.1",
            api_key_env_var="CTX_LLM_KEY",
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
            config_sync_run_id=config_run.id,
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
            result_summary_json=json.dumps({}),
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
            target_entity_id="automation.generated_context",
            status="proposed",
            schema_version="haev.automation.concept.v2",
            summary="Context concept",
            confidence=0.8,
            risk_level="low",
            concept_payload_json=json.dumps(
                {
                    "title": "Context concept",
                    "summary": "Context concept",
                    "concept_type": "general",
                    "target_kind": "new_automation",
                    "involved_entities": ["sensor.office_temp"],
                }
            ),
            concept_type="general",
            impact_score=0.7,
            feasibility_score=0.8,
            novelty_score=0.4,
            ranking_score=0.72,
            ranking_breakdown_json=json.dumps({}),
            duplicate_fingerprint="context-fp",
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

        generation = SuggestionGeneration(
            proposal_id=proposal.id,
            profile_id=profile.id,
            llm_connection_id=connection.id,
            mode="auto",
            status="submitted",
            optional_instruction=None,
            current_step=0,
            pending_question_json=None,
            planning_answers_json=None,
            final_yaml_text="alias: Context concept\\ntrigger: []\\ncondition: []\\naction: []\\nmode: single\\n",
            final_structured_json=json.dumps({}),
            error=None,
            created_at=utcnow(),
            updated_at=utcnow(),
            finished_at=utcnow(),
        )
        session.add(generation)
        session.commit()
        session.refresh(generation)
        assert generation.id is not None

        session.add(
            SuggestionSubmissionEvent(
                generation_id=generation.id,
                profile_id=profile.id,
                config_key="context_concept",
                operation="create",
                previous_config_json=None,
                request_payload_json=json.dumps({}),
                response_payload_json=json.dumps({"result": "ok"}),
                status="success",
                error=None,
                created_at=utcnow(),
            )
        )
        session.commit()

        context = build_concept_suggestion_context(
            session,
            profile,
            idea_type="general",
            mode="standard",
            custom_intent="",
            include_existing=True,
            include_new=True,
            max_context_bytes=20_000,
        )
        assert context["schema_version"] == "haev.automation.concept_context.v2"
        assert context["existing_automations"]
        assert context["recent_concepts"]
        assert context["recent_submissions"]

    db.reset_engine_for_tests()

    app = create_app()
    with TestClient(app):
        pass

    with Session(db.get_engine()) as session:
        profile = Profile(
            name="ctx",
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

        sync_run = SyncRun(
            profile_id=profile.id,
            pulled_at=utcnow(),
            entity_count=2,
            duration_ms=1,
            status="success",
            error=None,
        )
        session.add(sync_run)
        session.commit()
        session.refresh(sync_run)
        assert sync_run.id is not None

        session.add(
            EntitySnapshot(
                profile_id=profile.id,
                sync_run_id=sync_run.id,
                entity_id="person.jason",
                domain="person",
                state="home",
                friendly_name="Jason",
                attributes_json=json.dumps({}),
                pulled_at=utcnow(),
            )
        )
        session.add(
            EntitySnapshot(
                profile_id=profile.id,
                sync_run_id=sync_run.id,
                entity_id="light.kitchen",
                domain="light",
                state="on",
                friendly_name="Kitchen Light",
                attributes_json=json.dumps({}),
                pulled_at=utcnow(),
            )
        )
        session.commit()

        config_run = ConfigSyncRun(
            profile_id=profile.id,
            pulled_at=utcnow(),
            item_count=2,
            success_count=2,
            error_count=0,
            duration_ms=1,
            status="success",
            error=None,
        )
        session.add(config_run)
        session.commit()
        session.refresh(config_run)
        assert config_run.id is not None

        target = ConfigSnapshot(
            profile_id=profile.id,
            config_sync_run_id=config_run.id,
            kind="automation",
            entity_id="automation.evening_mode",
            config_key="evening_mode",
            name="Evening Mode",
            state="on",
            fetch_status="success",
            summary_json=json.dumps({"name": "Evening Mode"}),
            references_json=json.dumps({"entity_id": ["person.jason", "light.kitchen"]}),
            config_json=json.dumps({"alias": "Evening Mode"}),
            attributes_json=json.dumps({}),
            metadata_json=json.dumps({}),
            pulled_at=utcnow(),
        )
        related = ConfigSnapshot(
            profile_id=profile.id,
            config_sync_run_id=config_run.id,
            kind="script",
            entity_id="script.goodnight",
            config_key="goodnight",
            name="Goodnight",
            state="off",
            fetch_status="success",
            summary_json=json.dumps({"name": "Goodnight"}),
            references_json=json.dumps({"entity_id": ["light.kitchen"]}),
            config_json=json.dumps({"alias": "Goodnight"}),
            attributes_json=json.dumps({}),
            metadata_json=json.dumps({}),
            pulled_at=utcnow(),
        )
        session.add(target)
        session.add(related)
        session.commit()
        session.refresh(target)
        assert target.id is not None

        context = build_automation_context(session, profile, target, max_context_bytes=12_000)
        assert context["target"]["entity_id"] == "automation.evening_mode"
        assert any(item["entity_id"] == "person.jason" for item in context["linked_entities"])
        assert any(item["entity_id"] == "script.goodnight" for item in context["related_config"])

        known_ids = build_known_entity_ids(session, profile.id, config_run.id)
        assert "person.jason" in known_ids
        assert "automation.evening_mode" in known_ids

    db.reset_engine_for_tests()
