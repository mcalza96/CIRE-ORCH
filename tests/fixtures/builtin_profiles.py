from __future__ import annotations

from pathlib import Path

import yaml


def _load_profile(profile_id: str) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "app" / "cartridges" / f"{profile_id}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid fixture payload for {profile_id}")
    return payload


BUILTIN_PROFILES: dict[str, dict] = {
    "base": _load_profile("base"),
    "iso_auditor": _load_profile("iso_auditor"),
}
