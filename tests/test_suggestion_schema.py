from __future__ import annotations

from app.suggestion_schema import validate_suggestion_payload


def test_validate_suggestion_payload_accepts_valid_item() -> None:
    payload = {
        "suggestions": [
            {
                "target_entity_id": "automation.evening_mode",
                "summary": "Add an explicit guard before actions.",
                "confidence": 0.8,
                "risk_level": "low",
                "proposed_patch": [
                    {
                        "op": "replace",
                        "path": "/description",
                        "value": "Updated automation description",
                    }
                ],
                "verification_steps": ["Run automation manually and inspect traces."],
            }
        ]
    }
    valid, errors = validate_suggestion_payload(
        payload,
        known_entity_ids={"automation.evening_mode", "person.jason", "light.kitchen"},
        max_patch_ops=6,
    )
    assert errors == []
    assert len(valid) == 1
    assert valid[0]["target_entity_id"] == "automation.evening_mode"


def test_validate_suggestion_payload_rejects_unknown_entities() -> None:
    payload = {
        "suggestions": [
            {
                "target_entity_id": "automation.evening_mode",
                "summary": "Reference a missing entity.",
                "confidence": 0.5,
                "risk_level": "medium",
                "proposed_patch": [
                    {
                        "op": "add",
                        "path": "/action/0/target/entity_id",
                        "value": "light.unknown_room",
                    }
                ],
                "verification_steps": ["Run automation."],
            }
        ]
    }
    valid, errors = validate_suggestion_payload(
        payload,
        known_entity_ids={"automation.evening_mode"},
        max_patch_ops=6,
    )
    assert valid == []
    assert any("unknown_entities" in error for error in errors)
