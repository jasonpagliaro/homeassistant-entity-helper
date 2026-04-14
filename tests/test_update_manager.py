from __future__ import annotations

import shlex
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPDATE_MANAGER = REPO_ROOT / "scripts" / "update-manager.sh"
UPDATE_MANAGER_BASH = UPDATE_MANAGER.relative_to(REPO_ROOT).as_posix()


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["bash", "-s"],
        cwd=REPO_ROOT,
        capture_output=True,
        input=script.replace("\r\n", "\n").encode(),
        check=False,
    )
    return subprocess.CompletedProcess(
        result.args,
        result.returncode,
        result.stdout.decode(),
        result.stderr.decode(),
    )


def test_ensure_sqlite_volume_resolved_prefers_running_compose_volume() -> None:
    script = textwrap.dedent(
        f"""
        set -euo pipefail
        source {shlex.quote(UPDATE_MANAGER_BASH)}
        SQLITE_VOLUME="hev_data"
        SQLITE_DB_PATH="/data/ha_entity_vault.db"

        compose_container_id() {{
          printf 'cid-123\\n'
        }}

        docker() {{
          local last_arg="${{!#}}"

          if [[ "$1" == "volume" && "$2" == "inspect" && "$3" == "hev_data" ]]; then
            return 0
          fi
          if [[ "$1" == "volume" && "$2" == "inspect" && "$3" == "homeassistant-entity-helper_hev_data" ]]; then
            return 0
          fi
          if [[ "$1" == "run" && "$2" == "--rm" && "$4" == "hev_data:/data:ro" ]]; then
            return 1
          fi
          if [[ "$1" == "run" && "$2" == "--rm" && "$4" == "homeassistant-entity-helper_hev_data:/data:ro" ]]; then
            return 0
          fi
          if [[ "$1" == "inspect" && "$2" == "--format" && "$last_arg" == "cid-123" ]]; then
            printf 'volume|/data|homeassistant-entity-helper_hev_data\\n'
            return 0
          fi

          printf 'unexpected docker call: %s\\n' "$*" >&2
          return 99
        }}

        ensure_sqlite_volume_resolved
        printf '%s\\n' "$SQLITE_VOLUME"
        """
    )

    result = run_bash(script)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "homeassistant-entity-helper_hev_data"


def test_ensure_sqlite_volume_resolved_keeps_valid_explicit_volume() -> None:
    script = textwrap.dedent(
        f"""
        set -euo pipefail
        source {shlex.quote(UPDATE_MANAGER_BASH)}
        SQLITE_VOLUME="custom-sqlite-volume"
        SQLITE_DB_PATH="/data/ha_entity_vault.db"

        compose_container_id() {{
          printf 'cid-should-not-be-used\\n'
        }}

        docker() {{
          local last_arg="${{!#}}"

          if [[ "$1" == "volume" && "$2" == "inspect" && "$3" == "custom-sqlite-volume" ]]; then
            return 0
          fi
          if [[ "$1" == "run" && "$2" == "--rm" && "$4" == "custom-sqlite-volume:/data:ro" ]]; then
            return 0
          fi
          if [[ "$1" == "inspect" && "$2" == "--format" && "$last_arg" == "cid-should-not-be-used" ]]; then
            printf 'container lookup should not have been used\\n' >&2
            return 99
          fi

          printf 'unexpected docker call: %s\\n' "$*" >&2
          return 99
        }}

        ensure_sqlite_volume_resolved
        printf '%s\\n' "$SQLITE_VOLUME"
        """
    )

    result = run_bash(script)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "custom-sqlite-volume"
