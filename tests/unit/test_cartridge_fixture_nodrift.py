from __future__ import annotations

from pathlib import Path

import yaml

from tests.fixtures.builtin_profiles import BUILTIN_PROFILES


def test_builtin_fixture_matches_base_yaml() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "app" / "profiles" / "base.yaml"
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    assert BUILTIN_PROFILES["base"] == payload


def test_builtin_fixture_matches_iso_auditor_yaml() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "app" / "profiles" / "iso_auditor.yaml"
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    assert BUILTIN_PROFILES["iso_auditor"] == payload
