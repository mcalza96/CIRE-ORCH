
from app.agent.models import RetrievalPlan, QueryIntent, QueryMode
from app.cartridges.models import AgentProfile

class RetryPolicy:
    def determine_next_intent(
        self, 
        current_plan: RetrievalPlan, 
        reason: str, 
        profile: AgentProfile | None = None
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
            # Strategy: Relax constraints if literal mode failed
            if current_plan.mode in {"literal_normativa", "literal_lista"}:
                fallback_mode: QueryMode = (
                    "comparativa" 
                    if len(current_plan.requested_standards) >= 2 
                    else "explicativa"
                )
                
                # Only return new intent if it's different
                if fallback_mode != current_plan.mode:
                    return QueryIntent(
                        mode=fallback_mode,
                        rationale=f"retry_relax_from_{current_plan.mode}_reason_{reason}"
                    )
                    
        return None

    def should_force_literal_retry(
        self, 
        query: str, 
        plan: RetrievalPlan
    ) -> bool:
        """
        Check if we should force keeping literal mode despite failure/retry.
        """
        # This logic relies on query analysis.
        # A richer signal can be passed from planner/classifier in future iterations.
        # Note: This method is a placeholder for more complex logic.
        return False
