from __future__ import annotations

import re
from pathlib import Path

TEMPLATE_TABLE_LABELS: dict[str, tuple[str, ...]] = {
    "entities.html": (
        "Entity ID",
        "Name",
        "Domain",
        "State",
        "Area",
        "Location",
        "Last Updated",
        "Pulled At",
        "Actions",
    ),
    "config_items.html": (
        "ID",
        "Kind",
        "Entity ID",
        "Name",
        "Config Key",
        "Status",
        "Pulled At",
        "Actions",
    ),
    "suggestions.html": (
        "Run",
        "Status",
        "Type",
        "Mode",
        "Progress",
        "Results",
        "Created",
        "Actions",
    ),
    "entity_suggestions.html": (
        "ID",
        "Entity ID",
        "Domain",
        "Readiness",
        "Workflow",
        "Missing Fields",
        "Pulled At",
        "Actions",
    ),
    "entity_suggestion_workflow_queue.html": (
        "ID",
        "Entity ID",
        "Domain",
        "Readiness",
        "Workflow",
        "Fixable Issues",
        "Manual Issues",
        "Actions",
    ),
    "automation_drafts.html": (
        "ID",
        "Entity ID",
        "Template",
        "Title",
        "Generation",
        "Review",
        "Pulled At",
        "Actions",
    ),
}

HIGH_IMPACT_BUTTON_TOOLTIP_LABELS: dict[str, dict[str, int]] = {
    "config.html": {
        "Check Now": 1,
        "Reset Banner Dismissal": 1,
    },
    "settings.html": {
        "Run Suggestions Check": 1,
        "Disable Profile": 1,
        "Enable Profile": 1,
        "Delete Profile": 2,
        "Delete LLM": 1,
    },
    "entities.html": {
        "Sync Now": 1,
        "Sync Config Items": 1,
        "Run Suggestions Check": 1,
    },
    "config_items.html": {
        "Sync Config Items": 1,
        "Suggest Automation Updates": 1,
        "Suggest": 1,
    },
    "config_item_detail.html": {
        "Suggest Improvement Concepts": 1,
    },
    "suggestions.html": {
        "Generate Concept Suggestions": 1,
    },
    "suggestion_run_detail.html": {
        "Add All to Queue": 1,
        "Generate YAML": 1,
    },
    "entity_suggestions.html": {
        "Run Suggestions Check": 1,
        "Generate Drafts (Ready)": 1,
    },
    "entity_suggestion_workflow_queue.html": {
        "Recheck Now": 1,
    },
    "entity_suggestion_workflow_detail.html": {
        "Apply to Home Assistant": 1,
        "Skip for Now": 1,
    },
    "automation_draft_detail.html": {
        "Accept Draft": 1,
        "Reject Draft": 1,
    },
}


def _read_template(template_name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "app" / "templates" / template_name).read_text(encoding="utf-8")


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _button_counts(template: str, label: str) -> tuple[int, int]:
    total = 0
    with_tooltip = 0

    for match in re.finditer(r"<button\b([^>]*)>(.*?)</button>", template, flags=re.DOTALL):
        attrs = match.group(1)
        text = re.sub(r"<[^>]+>", " ", match.group(2))
        if _normalize_text(text) != label:
            continue

        total += 1
        if re.search(r'\bdata-tooltip\s*=\s*(["\']).+?\1', attrs):
            with_tooltip += 1

    return total, with_tooltip


def test_responsive_table_markup_contract() -> None:
    for template_name, labels in TEMPLATE_TABLE_LABELS.items():
        template = _read_template(template_name)
        assert 'class="data-table"' in template

        missing_data_label = re.findall(r"<td(?![^>]*\bdata-label=)[^>]*>", template)
        assert missing_data_label == []

        for label in labels:
            assert f'data-label="{label}"' in template


def test_high_impact_buttons_have_tooltips() -> None:
    for template_name, labels in HIGH_IMPACT_BUTTON_TOOLTIP_LABELS.items():
        template = _read_template(template_name)

        for label, expected_count in labels.items():
            total_count, tooltip_count = _button_counts(template, label)

            assert total_count == expected_count, (
                f"{template_name}: expected {expected_count} '{label}' button(s), "
                f"found {total_count}"
            )
            assert tooltip_count == expected_count, (
                f"{template_name}: expected all '{label}' button(s) to include data-tooltip, "
                f"found {tooltip_count}/{expected_count}"
            )


def test_settings_llm_preset_form_contract() -> None:
    template = _read_template("settings.html")
    assert 'data-llm-form="true"' in template
    assert 'name="preset_slug"' in template
    assert "data-llm-model-select" in template
    assert "data-llm-feature-select" in template
    assert "data-llm-required-fields" in template
    assert "data-llm-test-draft" in template
    assert "data-llm-temperature-help" in template
    assert "data-llm-temperature-guidance" not in template
    assert 'id="llm-temperature-suggestions"' in template
    assert "/api/llm/models" in template
    assert "/api/llm/test-draft" in template


def test_config_page_form_contract() -> None:
    template = _read_template("config.html")
    assert 'action="/config/update-settings"' in template
    assert 'action="/config/check-updates"' in template
    assert 'action="/config/update-banner/reset"' in template
    assert 'name="updates_enabled"' in template
    assert 'name="update_repo_owner"' in template
    assert 'name="update_repo_name"' in template
    assert 'name="update_repo_branch"' in template
    assert 'name="update_check_interval_minutes"' in template
    assert 'data-local-datetime' in template
    assert "Intl.DateTimeFormat" in template
