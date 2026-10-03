"""Deterministic scoring engine for model selection.

Computes a final score for each candidate model given a task category,
using capability match, size preference, and benchmark history.
No LLM involvement — purely rule-based.
"""

import time
from typing import Dict, List, Optional, Tuple

from mileage.core.logger import logger
from mileage.router.schemas import (
    Capability,
    ModelProfile,
    RouterConfig,
    RoutingDecision,
    TaskCategory,
    TaskRule,
)
from mileage.router.storage import RouterStorage


class ModelScorer:
    """Deterministic scoring engine for model routing.

    Scoring formula:
        final_score = (capability_score × (1 - benchmark_weight))
                    + (benchmark_score × benchmark_weight)
                    + size_bonus

    Where:
        capability_score = weighted sum of required + preferred capability matches
        benchmark_score  = success_rate × speed_factor (from SQLite history)
        size_bonus       = small bonus for smaller models when prefer_smallest is set
    """

    def __init__(
        self,
        config: RouterConfig,
        storage: Optional[RouterStorage] = None,
    ):
        self.config = config
        self.storage = storage

    def score_model(
        self,
        profile: ModelProfile,
        task_category: TaskCategory,
    ) -> Tuple[float, str]:
        """Score a single model for a task category.

        Returns:
            (score, reason_string)
        """
        rule = self.config.get_rule(task_category)

        # ── 1. Capability Match Score ────────────────────────────────────
        cap_score, cap_details = self._score_capabilities(profile, rule)

        # ── 2. Benchmark Score ───────────────────────────────────────────
        bench_score = 0.0
        bench_detail = ""
        if self.config.enable_benchmarks and self.storage:
            bench_score, bench_detail = self._score_benchmarks(profile)

        # ── 3. Size / Speed Bonus ────────────────────────────────────────
        size_bonus, size_detail = self._score_size(profile, rule)

        # ── 4. Composite Score ───────────────────────────────────────────
        bw = self.config.benchmark_weight if self.config.enable_benchmarks else 0.0
        final = (cap_score * (1.0 - bw)) + (bench_score * bw) + size_bonus

        # Clamp
        final = max(0.0, min(1.0, final))

        # Build reason
        parts = [cap_details]
        if bench_detail:
            parts.append(bench_detail)
        if size_detail:
            parts.append(size_detail)
        reason = " | ".join(parts)

        return round(final, 4), reason

    def score_all(
        self,
        profiles: List[ModelProfile],
        task_category: TaskCategory,
    ) -> List[Tuple[ModelProfile, float, str]]:
        """Score all candidate models, sorted by score descending.

        Returns:
            List of (profile, score, reason) tuples.
        """
        scored: List[Tuple[ModelProfile, float, str]] = []
        for p in profiles:
            score, reason = self.score_model(p, task_category)
            scored.append((p, score, reason))

        scored.sort(key=lambda x: -x[1])
        return scored

    def select_best(
        self,
        profiles: List[ModelProfile],
        task_category: TaskCategory,
    ) -> RoutingDecision:
        """Select the best model using deterministic scoring.

        Returns a RoutingDecision with full transparency.
        """
        start = time.perf_counter()
        rule = self.config.get_rule(task_category)
        scored = self.score_all(profiles, task_category)

        if not scored:
            return RoutingDecision(
                selected_model=self.config.fallback_model or "unknown",
                reason="No models available for scoring.",
                task_category=task_category,
                confidence=0.0,
            )

        # All scores for transparency
        all_scores = {p.name: s for p, s, _ in scored}

        # Select best above confidence threshold
        best_profile, best_score, best_reason = scored[0]
        confidence = best_score

        # Find fallback (second-best, or global fallback)
        fallback = None
        if len(scored) > 1:
            fallback = scored[1][0].name
        elif self.config.fallback_model:
            fallback = self.config.fallback_model

        # Check confidence threshold
        if confidence < self.config.confidence_threshold:
            # Use fallback instead
            selected = self.config.fallback_model or best_profile.name
            reason = (
                f"Low confidence ({confidence:.2f} < {self.config.confidence_threshold}). "
                f"Using fallback: {selected}. Original best: {best_profile.name} — {best_reason}"
            )
        else:
            selected = best_profile.name
            reason = best_reason

        decision_ms = (time.perf_counter() - start) * 1000

        decision = RoutingDecision(
            selected_model=selected,
            fallback_model=fallback,
            reason=reason,
            task_category=task_category,
            required_capabilities=rule.required_capabilities,
            expected_latency_ms=best_profile.avg_latency_ms,
            expected_tokens_per_sec=best_profile.avg_tokens_per_sec,
            capability_match_score=best_score,
            confidence=round(confidence, 4),
            scores=all_scores,
            decision_time_ms=round(decision_ms, 2),
        )

        # Persist to storage
        if self.storage:
            try:
                self.storage.save_decision(decision)
            except Exception as e:
                logger.debug("Failed to save routing decision: %s", e)

        return decision

    # ── Internal Scoring Components ──────────────────────────────────────

    def _score_capabilities(
        self, profile: ModelProfile, rule: TaskRule
    ) -> Tuple[float, str]:
        """Score based on required + preferred capability match."""
        if not rule.required_capabilities:
            return 0.5, f"{profile.name}: no requirements (0.50)"

        # Required capabilities — must all be present
        required_scores: List[float] = []
        missing_required: List[str] = []
        for cap in rule.required_capabilities:
            cap_score = profile.get_capability_score(cap)
            weight = self.config.get_capability_weight(cap)
            required_scores.append(cap_score * weight)
            if cap_score < 0.2:
                missing_required.append(cap.value)

        # If any required capability is missing, severe penalty
        if missing_required:
            penalty = 0.15 * len(missing_required)
            avg_req = sum(required_scores) / len(required_scores) if required_scores else 0
            final = max(0.0, avg_req - penalty)
            return round(final, 4), (
                f"{profile.name}: missing {', '.join(missing_required)} "
                f"(cap={final:.2f})"
            )

        # Weighted average of required capabilities
        total_weight = sum(
            self.config.get_capability_weight(c) for c in rule.required_capabilities
        )
        weighted_sum = sum(required_scores)
        avg_required = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Preferred capabilities — bonus on top
        preferred_bonus = 0.0
        if rule.preferred_capabilities:
            for cap in rule.preferred_capabilities:
                preferred_bonus += profile.get_capability_score(cap) * 0.1
            preferred_bonus = min(0.15, preferred_bonus)  # Cap the bonus

        final = min(1.0, avg_required + preferred_bonus)

        return round(final, 4), (
            f"{profile.name}: cap={final:.2f} "
            f"(req={avg_required:.2f}, pref=+{preferred_bonus:.2f})"
        )

    def _score_benchmarks(
        self, profile: ModelProfile,
    ) -> Tuple[float, str]:
        """Score based on historical benchmark performance from SQLite."""
        if not self.storage:
            return 0.0, ""

        stats = self.storage.get_model_stats(profile.name)
        if stats["total"] == 0:
            return 0.0, "no benchmarks"

        success_rate = stats["success_rate"]
        avg_latency = stats["avg_latency_ms"]

        # Speed factor: faster is better (normalize against 5-second baseline)
        speed_factor = min(1.0, 5000.0 / max(avg_latency, 100.0))

        bench_score = (success_rate * 0.7) + (speed_factor * 0.3)

        # Update profile with latest stats
        profile.avg_latency_ms = avg_latency
        profile.avg_tokens_per_sec = stats["avg_tps"]
        profile.benchmark_success_rate = success_rate
        profile.benchmark_run_count = int(stats["total"])

        return round(bench_score, 4), (
            f"bench={bench_score:.2f} "
            f"({stats['total']} runs, {success_rate:.0%} pass, {avg_latency:.0f}ms)"
        )

    def _score_size(
        self, profile: ModelProfile, rule: TaskRule
    ) -> Tuple[float, str]:
        """Score bonus/penalty based on model size preferences."""
        if not self.config.prefer_smallest and not rule.prefer_fastest:
            return 0.0, ""

        param_b = profile.parameter_count_billions

        # Check max_parameter_size constraint
        if rule.max_parameter_size_b and param_b:
            if param_b > rule.max_parameter_size_b:
                return -0.1, f"exceeds {rule.max_parameter_size_b}B limit"

        # Size bonus: smaller is better (when prefer_smallest or prefer_fastest)
        if param_b is not None and (self.config.prefer_smallest or rule.prefer_fastest):
            if param_b <= 2.0:
                bonus = 0.08
            elif param_b <= 4.0:
                bonus = 0.05
            elif param_b <= 8.0:
                bonus = 0.02
            elif param_b <= 14.0:
                bonus = 0.0
            else:
                bonus = -0.03  # Slight penalty for very large models

            detail = f"size={param_b}B → {'+' if bonus >= 0 else ''}{bonus:.2f}"
            return bonus, detail

        return 0.0, ""
