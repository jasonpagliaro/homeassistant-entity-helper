from __future__ import annotations

import re
from pathlib import Path

TEMPLATE_TABLE_LABELS: dict[str, tuple[str, ...]] = {
    "entities.html": (
        "Entity ID",
        "Name",
        "Domain",
        "Device",
        "Area",
        "Location",
        "Pulled At",
        "Actions",
    ),
    "states.html": (
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
    "states.html": {
        "Sync Now": 1,
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


def _action_wrapper_contains_label(template: str, label: str) -> bool:
    for match in re.finditer(
        r'<div class="grid-form-actions">([\s\S]*?)</div>',
        template,
        flags=re.DOTALL,
    ):
        text = re.sub(r"<[^>]+>", " ", match.group(1))
        if _normalize_text(label) in _normalize_text(text):
            return True
    return False


def _form_has_class(template: str, action: str, class_name: str) -> bool:
    for match in re.finditer(r"<form\b([^>]*)>", template, flags=re.DOTALL):
        attrs = match.group(1)
        if f'action="{action}"' not in attrs:
            continue
        if re.search(rf'\bclass\s*=\s*"[^"]*\b{re.escape(class_name)}\b[^"]*"', attrs):
            return True
    return False


def _label_for_input_has_class(template: str, input_name: str, class_name: str) -> bool:
    for match in re.finditer(r"<label\b([^>]*)>([\s\S]*?)</label>", template, flags=re.DOTALL):
        attrs = match.group(1)
        body = match.group(2)
        if f'name="{input_name}"' not in body:
            continue
        if re.search(rf'\bclass\s*=\s*"[^"]*\b{re.escape(class_name)}\b[^"]*"', attrs):
            return True
    return False


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


def test_grid_form_action_wrappers_present_on_key_pages() -> None:
    automation_adjustment_template = _read_template("automation_adjustment_detail.html")
    assert _action_wrapper_contains_label(automation_adjustment_template, "Save Manual Edit")
    assert _action_wrapper_contains_label(automation_adjustment_template, "Apply AI Whole Edit")

    suggestion_queue_template = _read_template("suggestion_queue.html")
    assert _action_wrapper_contains_label(suggestion_queue_template, "Apply")
    assert _action_wrapper_contains_label(suggestion_queue_template, "Reset")

    config_template = _read_template("config.html")
    assert _action_wrapper_contains_label(config_template, "Save Update Settings")


def test_action_form_layout_contracts_on_adjustment_pages() -> None:
    automation_adjustment_template = _read_template("automation_adjustment_detail.html")
    assert 'class="action-form-row"' in automation_adjustment_template
    assert _form_has_class(
        automation_adjustment_template,
        "/automation-adjustments/drafts/{{ draft.id }}/submit",
        "action-form",
    )
    assert _form_has_class(
        automation_adjustment_template,
        "/automation-adjustments/drafts/{{ draft.id }}/test",
        "action-form",
    )
    assert _form_has_class(
        automation_adjustment_template,
        "/automation-adjustments/drafts/{{ draft.id }}/test",
        "action-form--wide",
    )
    assert _label_for_input_has_class(
        automation_adjustment_template,
        "test_alias_suffix",
        "action-form-field",
    )
    assert _label_for_input_has_class(automation_adjustment_template, "confirm_submit", "checkbox-row")
    assert _label_for_input_has_class(automation_adjustment_template, "confirm_test", "checkbox-row")
    assert _label_for_input_has_class(automation_adjustment_template, "confirm_revert", "checkbox-row")

    automation_adjustments_template = _read_template("automation_adjustments.html")
    assert _label_for_input_has_class(automation_adjustments_template, "confirm_test", "checkbox-row")


def test_action_form_layout_contracts_on_draft_and_suggestion_pages() -> None:
    automation_draft_template = _read_template("automation_draft_detail.html")
    assert 'class="action-form-row"' in automation_draft_template
    assert _form_has_class(
        automation_draft_template,
        "/automation-drafts/{{ draft.id }}/accept",
        "action-form",
    )
    assert _form_has_class(
        automation_draft_template,
        "/automation-drafts/{{ draft.id }}/accept",
        "action-form--wide",
    )
    assert _form_has_class(
        automation_draft_template,
        "/automation-drafts/{{ draft.id }}/reject",
        "action-form",
    )
    assert _form_has_class(
        automation_draft_template,
        "/automation-drafts/{{ draft.id }}/reject",
        "action-form--wide",
    )
    assert _label_for_input_has_class(automation_draft_template, "review_note", "action-form-field")

    suggestions_template = _read_template("suggestions.html")
    assert _label_for_input_has_class(suggestions_template, "include_existing", "checkbox-row")
    assert _label_for_input_has_class(suggestions_template, "include_new", "checkbox-row")

    suggestion_generation_template = _read_template("suggestion_generation_detail.html")
    assert _label_for_input_has_class(
        suggestion_generation_template,
        "confirm_submit",
        "checkbox-row",
    )


def test_flow_editor_mount_points_and_hidden_fields_are_present() -> None:
    config_item_template = _read_template("config_item_detail.html")
    assert 'data-flow-editor-root="true"' in config_item_template
    assert 'data-flow-toggle-target="flow-panel-config-' in config_item_template
    assert "Edit in Adjustment Flow" in config_item_template

    automation_adjustment_template = _read_template("automation_adjustment_detail.html")
    assert 'data-flow-editor-root="true"' in automation_adjustment_template
    assert 'name="automation_json"' in automation_adjustment_template
    assert 'name="edit_origin"' in automation_adjustment_template
    assert 'id="flow-editor-adjustment-form-' in automation_adjustment_template

    suggestion_generation_template = _read_template("suggestion_generation_detail.html")
    assert 'data-flow-editor-root="true"' in suggestion_generation_template
    assert 'name="automation_json"' in suggestion_generation_template
    assert 'name="edit_origin"' in suggestion_generation_template
    assert 'id="flow-editor-generation-form-' in suggestion_generation_template


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
    assert 'id="copy-manual-update-commands"' in template
    assert 'id="manual-update-commands"' in template
    assert 'class="manual-update-window"' in template
    assert 'class="manual-update-window__body"' in template
    assert 'class="button-secondary manual-update-copy-button"' in template
    assert 'aria-label="Manual update commands"' in template
    assert 'wrap="off"' not in template
    assert "<textarea" not in template
    assert "Copy Text" not in template
    assert re.search(r'id="copy-manual-update-commands"[\s\S]*?>\s*Copy\s*</button>', template)
    assert "# one-time cleanup for older installs that pinned HEV_BUILD_COMMIT_SHA in .env" not in template
    assert "sed -i.bak '/^HEV_BUILD_COMMIT_SHA=/d' .env && rm -f .env.bak" not in template
    assert "curl -fsS http://localhost:23010/healthz" not in template
    assert "HEV_BUILD_COMMIT_SHA=$(git rev-parse HEAD) docker compose up -d --build --force-recreate" in template
    assert "navigator.clipboard.writeText" in template
    assert 'document.execCommand("copy")' in template
    assert 'fallbackEl.style.left = "-9999px"' in template
    assert "Local Docker Update Commands" not in template
    assert template.index("<h2>Update Status</h2>") < template.index('aria-label="Manual update commands"')
    assert template.index('aria-label="Manual update commands"') < template.index("<h2>Update Settings</h2>")
    assert 'data-local-datetime' in template
    assert "Intl.DateTimeFormat" in template
