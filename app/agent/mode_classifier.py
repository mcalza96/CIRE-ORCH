from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.cartridges.models import AgentProfile
from app.agent.models import QueryMode
from app.agent.policies import extract_requested_scopes


@dataclass(frozen=True)
class ModeClassification:
    mode: QueryMode
    confidence: float
    reasons: tuple[str, ...] = ()
    features: dict[str, Any] | None = None
    blocked_modes: tuple[QueryMode, ...] = ()


def classify_mode_v2(query: str, profile: AgentProfile | None = None) -> ModeClassification:
    requested = extract_requested_scopes(query, profile=profile)
    features = {
        "requested_scopes_count": len(requested),
        "requested_scopes": list(requested),
    }

    if profile is not None and profile.query_modes.modes:
        default_mode = str(profile.query_modes.default_mode or "").strip()
        if default_mode and default_mode in profile.query_modes.modes:
            return ModeClassification(
                mode=default_mode,
                confidence=0.5,
                reasons=("profile_default_mode",),
                features=features,
            )
        first_mode = next(iter(profile.query_modes.modes.keys()), "default")
        return ModeClassification(
            mode=first_mode,
            confidence=0.45,
            reasons=("profile_first_mode",),
            features=features,
        )

    return ModeClassification(
        mode="default",
        confidence=0.35,
        reasons=("generic_default_mode",),
        features=features,
    )
