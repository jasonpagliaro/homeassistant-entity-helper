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
        "Progress",
        "Results",
        "Created",
        "Finished",
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


def _read_template(template_name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "app" / "templates" / template_name).read_text(encoding="utf-8")


def test_responsive_table_markup_contract() -> None:
    for template_name, labels in TEMPLATE_TABLE_LABELS.items():
        template = _read_template(template_name)
        assert 'class="data-table"' in template

        missing_data_label = re.findall(r"<td(?![^>]*\bdata-label=)[^>]*>", template)
        assert missing_data_label == []

        for label in labels:
            assert f'data-label="{label}"' in template
