from __future__ import annotations

import json

from app.models import EntitySnapshot
from app.suggestions import (
    WORKFLOW_FIXABLE_ISSUE_CODES,
    evaluate_entity_snapshot,
    is_supported_entity,
    split_issues_by_fixability,
)


def build_snapshot(
    *,
    entity_id: str,
    domain: str,
    state: str,
    friendly_name: str | None,
    area_id: str | None,
    area_name: str | None,
    device_id: str | None,
    device_name: str | None,
    attributes: dict[str, object],
    metadata: dict[str, object],
    labels: dict[str, object] | None = None,
) -> EntitySnapshot:
    return EntitySnapshot(
        profile_id=1,
        sync_run_id=1,
        entity_id=entity_id,
        domain=domain,
        state=state,
        friendly_name=friendly_name,
        device_id=device_id,
        device_name=device_name,
        area_id=area_id,
        area_name=area_name,
        attributes_json=json.dumps(attributes),
        metadata_json=json.dumps(metadata),
        labels_json=json.dumps(labels or {}),
    )


def test_missing_area_is_blocked() -> None:
    snapshot = build_snapshot(
        entity_id="sensor.kitchen_temp",
        domain="sensor",
        state="on",
        friendly_name="Kitchen Temperature",
        area_id=None,
        area_name=None,
        device_id="dev1",
        device_name="Kitchen Sensor",
        attributes={"device_class": "temperature", "state_class": "measurement"},
        metadata={"attribute_device_class": "temperature"},
        labels={"names": ["Climate"]},
    )
    result = evaluate_entity_snapshot(snapshot)
    assert result.readiness_status == "blocked"
    assert "area" in result.missing_fields
    assert any(issue["code"] == "missing_area" for issue in result.issues)


def test_sensor_missing_semantic_type_is_blocked() -> None:
    snapshot = build_snapshot(
        entity_id="sensor.outdoor_temp",
        domain="sensor",
        state="72",
        friendly_name="Outdoor Temperature",
        area_id="outside",
        area_name="Outside",
        device_id="dev1",
        device_name="Outdoor Station",
        attributes={},
        metadata={},
        labels={"names": ["Weather"]},
    )
    result = evaluate_entity_snapshot(snapshot)
    assert result.readiness_status == "blocked"
    assert any(issue["code"] == "missing_sensor_semantic_type" for issue in result.issues)


def test_unavailable_with_no_blockers_is_needs_review() -> None:
    snapshot = build_snapshot(
        entity_id="sensor.basement_humidity",
        domain="sensor",
        state="unavailable",
        friendly_name="Basement Humidity",
        area_id="basement",
        area_name="Basement",
        device_id="dev1",
        device_name="Basement Sensor",
        attributes={"device_class": "humidity"},
        metadata={"attribute_device_class": "humidity"},
        labels={"names": ["Indoor"]},
    )
    result = evaluate_entity_snapshot(snapshot)
    assert result.readiness_status == "needs_review"
    assert any(issue["code"] == "unhealthy_state" for issue in result.issues)


def test_fully_populated_entity_is_ready() -> None:
    snapshot = build_snapshot(
        entity_id="sensor.living_temp",
        domain="sensor",
        state="70",
        friendly_name="Living Room Temperature",
        area_id="living_room",
        area_name="Living Room",
        device_id="dev1",
        device_name="Living Room Sensor",
        attributes={"device_class": "temperature", "state_class": "measurement"},
        metadata={
            "attribute_device_class": "temperature",
            "attribute_state_class": "measurement",
        },
        labels={"names": ["Climate"]},
    )
    result = evaluate_entity_snapshot(snapshot)
    assert result.readiness_status == "ready"
    assert result.missing_fields == []


def test_missing_area_is_warning_when_area_check_is_degraded() -> None:
    snapshot = build_snapshot(
        entity_id="sensor.guest_room_temp",
        domain="sensor",
        state="70",
        friendly_name="Guest Room Temperature",
        area_id=None,
        area_name=None,
        device_id="dev1",
        device_name="Guest Room Sensor",
        attributes={"device_class": "temperature", "state_class": "measurement"},
        metadata={"attribute_device_class": "temperature"},
        labels={"names": ["Climate"]},
    )
    result = evaluate_entity_snapshot(snapshot, area_required=False)
    assert result.readiness_status == "needs_review"
    assert "area" in result.missing_fields
    assert any(issue["code"] == "missing_area_enrichment_unavailable" for issue in result.issues)
    assert not any(issue["severity"] == "blocker" for issue in result.issues)


def test_supported_entity_domain_gate() -> None:
    assert is_supported_entity(
        build_snapshot(
            entity_id="sensor.temp",
            domain="sensor",
            state="1",
            friendly_name="Temp",
            area_id="a",
            area_name="A",
            device_id="d",
            device_name="D",
            attributes={"device_class": "temperature"},
            metadata={"attribute_device_class": "temperature"},
        )
    )
    assert is_supported_entity(
        build_snapshot(
            entity_id="binary_sensor.motion",
            domain="binary_sensor",
            state="off",
            friendly_name="Motion",
            area_id="a",
            area_name="A",
            device_id="d",
            device_name="D",
            attributes={"device_class": "motion"},
            metadata={"attribute_device_class": "motion"},
        )
    )
    assert is_supported_entity(
        build_snapshot(
            entity_id="lock.front_door",
            domain="lock",
            state="locked",
            friendly_name="Front Door Lock",
            area_id="a",
            area_name="A",
            device_id="d",
            device_name="D",
            attributes={},
            metadata={},
        )
    )
    assert not is_supported_entity(
        build_snapshot(
            entity_id="event.scene_button",
            domain="event",
            state="x",
            friendly_name="Scene Button",
            area_id="a",
            area_name="A",
            device_id="d",
            device_name="D",
            attributes={},
            metadata={},
        )
    )
    assert not is_supported_entity(
        build_snapshot(
            entity_id="light.kitchen",
            domain="light",
            state="on",
            friendly_name="Kitchen Light",
            area_id="a",
            area_name="A",
            device_id="d",
            device_name="D",
            attributes={},
            metadata={},
        )
    )
    assert not is_supported_entity(
        build_snapshot(
            entity_id="sensor_bad_shape",
            domain="sensor",
            state="on",
            friendly_name="Bad Shape",
            area_id="a",
            area_name="A",
            device_id="d",
            device_name="D",
            attributes={},
            metadata={},
        )
    )


def test_split_issues_by_fixability() -> None:
    issues = [
        {"code": "missing_area", "severity": "blocker"},
        {"code": "missing_labels", "severity": "warning"},
        {"code": "missing_device_linkage", "severity": "warning"},
        {"code": "invalid_entity_id_shape", "severity": "blocker"},
    ]
    fixable, manual_only = split_issues_by_fixability(issues)
    fixable_codes = {item["code"] for item in fixable}
    manual_codes = {item["code"] for item in manual_only}

    assert "missing_area" in WORKFLOW_FIXABLE_ISSUE_CODES
    assert "missing_labels" in WORKFLOW_FIXABLE_ISSUE_CODES
    assert fixable_codes == {"missing_area", "missing_labels"}
    assert manual_codes == {"missing_device_linkage", "invalid_entity_id_shape"}
