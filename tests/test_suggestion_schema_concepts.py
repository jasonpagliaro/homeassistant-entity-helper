from __future__ import annotations

from app.suggestion_schema import rank_concept_suggestions, validate_concept_suggestion_payload


def test_validate_concept_suggestion_payload_accepts_valid_items() -> None:
    payload = {
        "suggestions": [
            {
                "title": "Hallway motion nudge",
                "summary": "Notify if hallway motion is detected during work focus blocks.",
                "concept_type": "comfort",
                "target_kind": "new_automation",
                "involved_entities": ["binary_sensor.hallway_motion", "person.jason"],
                "impact_score": 0.7,
                "feasibility_score": 0.8,
                "novelty_score": 0.4,
                "confidence": 0.9,
                "risk_level": "low",
                "prerequisites": ["Hallway motion sensor online"],
                "verification_outline": ["Trigger motion and confirm notification"],
            },
            {
                "title": "Improve evening mode guard",
                "summary": "Add occupancy checks before turning on lights.",
                "concept_type": "safety",
                "target_kind": "existing_automation",
                "target_entity_id": "automation.evening_mode",
                "impact_score": 0.85,
                "feasibility_score": 0.88,
                "novelty_score": 0.2,
                "confidence": 0.8,
                "risk_level": "low",
            },
        ]
    }

    valid, errors = validate_concept_suggestion_payload(
        payload,
        known_entity_ids={"automation.evening_mode", "binary_sensor.hallway_motion", "person.jason"},
    )

    assert errors == []
    assert len(valid) == 2
    assert valid[0]["target_kind"] == "new_automation"
    assert valid[1]["target_kind"] == "existing_automation"


def test_rank_concept_suggestions_modes() -> None:
    concepts = [
        {
            "title": "High impact baseline",
            "summary": "...",
            "concept_type": "general",
            "impact_score": 0.95,
            "feasibility_score": 0.8,
            "novelty_score": 0.1,
            "confidence": 0.9,
        },
        {
            "title": "Novel idea",
            "summary": "...",
            "concept_type": "obscure",
            "impact_score": 0.45,
            "feasibility_score": 0.35,
            "novelty_score": 0.95,
            "confidence": 0.7,
        },
        {
            "title": "Another novel idea",
            "summary": "...",
            "concept_type": "obscure",
            "impact_score": 0.55,
            "feasibility_score": 0.45,
            "novelty_score": 0.85,
            "confidence": 0.7,
        },
    ]

    ranked_standard = rank_concept_suggestions(concepts, mode="standard", top_k=2)
    assert len(ranked_standard) == 2
    assert ranked_standard[0]["title"] == "High impact baseline"

    ranked_obscure = rank_concept_suggestions(concepts, mode="obscure", top_k=3)
    assert len(ranked_obscure) == 2
    assert ranked_obscure[0]["title"] in {"Novel idea", "Another novel idea"}

    ranked_surprise = rank_concept_suggestions(concepts, mode="surprise", top_k=3)
    assert len(ranked_surprise) == 3
    assert all("ranking_breakdown" in item for item in ranked_surprise)


def test_rank_concept_suggestions_applies_score_sanity_adjustment() -> None:
    concepts = [
        {
            "title": "Saturated perfect score",
            "summary": "...",
            "concept_type": "general",
            "impact_score": 1.0,
            "feasibility_score": 1.0,
            "novelty_score": 1.0,
            "confidence": 0.6,
            "risk_level": "medium",
            "involved_entities": ["sensor.a", "sensor.b", "sensor.c", "sensor.d", "sensor.e"],
            "prerequisites": ["One", "Two", "Three"],
            "verification_outline": ["Check once"],
        }
    ]

    ranked = rank_concept_suggestions(concepts, mode="standard", top_k=1)
    assert len(ranked) == 1
    item = ranked[0]
    assert item["impact_score"] < 1.0
    assert item["feasibility_score"] < 1.0
    assert item["novelty_score"] < 1.0
    breakdown = item["ranking_breakdown"]
    assert breakdown["raw_scores"]["impact"] == 1.0
    assert breakdown["adjusted_scores"]["impact"] < 1.0
    assert "calibration_multiplier" in breakdown["calibration"]
