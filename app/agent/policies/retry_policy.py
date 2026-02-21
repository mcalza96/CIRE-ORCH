from app.agent.types.models import RetrievalPlan, QueryIntent, QueryMode
from app.profiles.models import AgentProfile


class RetryPolicy:
    def _resolve_relaxed_mode(
        self,
        *,
        current_mode: str,
        profile: AgentProfile | None,
        multi_scope: bool,
    ) -> str:
        if profile is not None and profile.query_modes.modes:
            candidates: list[str] = []
            for mode_name, mode_cfg in profile.query_modes.modes.items():
                if mode_name == current_mode:
                    continue
                if bool(mode_cfg.require_literal_evidence):
                    continue
                if multi_scope and "logical_comparison" in mode_cfg.tool_hints:
                    return mode_name
                candidates.append(mode_name)
            if candidates:
                return candidates[0]
            default_mode = str(profile.query_modes.default_mode or "").strip()
            if default_mode and default_mode != current_mode:
                return default_mode

        del multi_scope
        return "default"

    def determine_next_intent(
        self, current_plan: RetrievalPlan, reason: str, profile: AgentProfile | None = None
    ) -> QueryIntent | None:
        """
        Determines the intent for the next retry attempt based on failure reason and current plan.
        Returns None if no specific strategy applies (caller should decide or keep current intent).
        """
        # Conservative retry policy for retrieval failures.
        if reason in {
            "scope_mismatch",
            "clause_missing",
            "low_score",
            "empty_retrieval",
            "graph_fallback_no_multihop",
        }:
            # Strategy: Relax constraints if literal-evidence plan failed
            if bool(current_plan.require_literal_evidence):
                fallback_mode: QueryMode = self._resolve_relaxed_mode(
                    current_mode=str(current_plan.mode or ""),
                    profile=profile,
                    multi_scope=len(current_plan.requested_standards) >= 2,
                )

                # Only return new intent if it's different
                if fallback_mode != current_plan.mode:
                    return QueryIntent(
                        mode=fallback_mode,
                        rationale=f"retry_relax_from_{current_plan.mode}_reason_{reason}",
                    )

        return None

    def should_force_literal_retry(self, query: str, plan: RetrievalPlan) -> bool:
        """
        Check if we should force keeping literal mode despite failure/retry.
        """
        # This logic relies on query analysis.
        # A richer signal can be passed from planner/classifier in future iterations.
        # Note: This method is a placeholder for more complex logic.
        return False
