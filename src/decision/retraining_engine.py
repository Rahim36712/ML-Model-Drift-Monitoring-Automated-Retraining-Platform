"""Rule-based retraining decision engine for the MLOps platform.

Evaluates drift-detection summaries against configurable thresholds and
emits a ``RetrainingDecision`` that tells the orchestrator whether to
retrain, flag for human review, or take no action.

The rule-set is evaluated in priority order; *all* matching rules are
collected, and the highest urgency among them determines the final
decision.

Usage::

    from src.config.settings import get_settings
    from src.data.database import get_database
    from src.decision import RetrainingDecisionEngine, RetrainingDecision

    engine = RetrainingDecisionEngine(
        thresholds=get_settings().drift_thresholds,
        db=get_database(),
    )
    decision = engine.evaluate(drift_summary)
    if decision.should_retrain:
        trigger_retraining_pipeline(decision.reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import desc, func

from src.data.database import DriftResult, RetrainingEvent

if TYPE_CHECKING:
    from src.config.settings import DriftThresholds
    from src.data.database import DatabaseManager

logger = logging.getLogger(__name__)

# Urgency ranking (lower index = higher urgency)
_URGENCY_RANK: dict[str, int] = {
    "HIGH": 0,
    "MEDIUM": 1,
    "LOW": 2,
    "NONE": 3,
}


# ===================================================================
# Data container
# ===================================================================

@dataclass
class RetrainingDecision:
    """Immutable outcome of the retraining-decision evaluation.

    Attributes:
        should_retrain: ``True`` when an automated retrain is
            recommended.
        action: One of ``RETRAIN``, ``FLAG_FOR_REVIEW``, ``NO_ACTION``.
        urgency: ``HIGH``, ``MEDIUM``, ``LOW``, or ``NONE``.
        reason: Human-readable explanation of why this decision was
            made.
        triggered_rules: Names of every rule that fired during
            evaluation.
        timestamp: UTC time the decision was produced.
    """

    should_retrain: bool
    action: str  # RETRAIN | FLAG_FOR_REVIEW | NO_ACTION
    urgency: str  # HIGH | MEDIUM | LOW | NONE
    reason: str
    triggered_rules: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ===================================================================
# Decision engine
# ===================================================================

class RetrainingDecisionEngine:
    """Evaluate drift summaries and decide whether to retrain.

    Args:
        thresholds: ``DriftThresholds`` from the application settings.
        db: Optional ``DatabaseManager`` used for sustained-warning
            look-back and decision persistence.
    """

    def __init__(
        self,
        thresholds: DriftThresholds,
        db: DatabaseManager | None = None,
    ) -> None:
        self._thresholds = thresholds
        self._db = db
        logger.info("RetrainingDecisionEngine initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, drift_summary: Any) -> RetrainingDecision:
        """Run the rule-set against a drift summary and return a decision.

        The ``drift_summary`` is expected to expose:
        - ``data_drift`` (with ``.overall_psi: float``)
        - ``prediction_drift`` (with ``.hellinger_distance: float``)
        - ``concept_drift`` (with ``.metric_deltas: dict[str, float]``)
        - ``overall_status: str``

        All attributes may be ``None`` when the corresponding detector
        was not executed.

        Args:
            drift_summary: Result object from the drift-detection
                pipeline.

        Returns:
            A populated ``RetrainingDecision``.
        """
        triggered_rules: list[str] = []
        highest_urgency: str = "NONE"
        action: str = "NO_ACTION"
        reasons: list[str] = []

        # ---- Rule 1: Critical data drift (PSI) -----------------------
        try:
            data_drift = getattr(drift_summary, "data_drift", None)
            if data_drift is not None:
                overall_psi: float = getattr(data_drift, "overall_psi", 0.0)
                psi_critical: float = self._thresholds.data_drift.psi.critical
                if overall_psi > psi_critical:
                    triggered_rules.append("critical_data_drift")
                    reasons.append(
                        f"Critical data drift: PSI {overall_psi:.4f} > "
                        f"{psi_critical:.4f}"
                    )
                    highest_urgency = self._higher_urgency(highest_urgency, "HIGH")
                    action = "RETRAIN"
        except Exception:  # noqa: BLE001
            logger.exception("Error evaluating rule 'critical_data_drift'.")

        # ---- Rule 2: F1 degradation -----------------------------------
        try:
            concept_drift = getattr(drift_summary, "concept_drift", None)
            if concept_drift is not None:
                metric_deltas: dict[str, float] = getattr(
                    concept_drift, "metric_deltas", {},
                )
                f1_delta = abs(metric_deltas.get("f1", 0.0))
                f1_critical: float = self._thresholds.concept_drift.f1_drop.critical
                if f1_delta > f1_critical:
                    triggered_rules.append("f1_degradation")
                    reasons.append(
                        f"F1 degradation: |Δf1| {f1_delta:.4f} > "
                        f"{f1_critical:.4f}"
                    )
                    highest_urgency = self._higher_urgency(highest_urgency, "HIGH")
                    action = "RETRAIN"
        except Exception:  # noqa: BLE001
            logger.exception("Error evaluating rule 'f1_degradation'.")

        # ---- Rule 3: Moderate drift + performance drop ----------------
        try:
            if data_drift is not None and concept_drift is not None:
                overall_psi_val: float = getattr(data_drift, "overall_psi", 0.0)
                psi_warning: float = self._thresholds.data_drift.psi.warning
                deltas = getattr(concept_drift, "metric_deltas", {})
                f1_delta_val = abs(deltas.get("f1", 0.0))

                if overall_psi_val > psi_warning and f1_delta_val > 0.02:
                    triggered_rules.append("moderate_drift_with_perf_drop")
                    reasons.append(
                        f"Moderate drift with perf drop: PSI {overall_psi_val:.4f} > "
                        f"{psi_warning:.4f} AND |Δf1| {f1_delta_val:.4f} > 0.02"
                    )
                    highest_urgency = self._higher_urgency(highest_urgency, "MEDIUM")
                    if action != "RETRAIN":
                        action = "RETRAIN"
        except Exception:  # noqa: BLE001
            logger.exception("Error evaluating rule 'moderate_drift_with_perf_drop'.")

        # ---- Rule 4: Prediction distribution shift --------------------
        try:
            prediction_drift = getattr(drift_summary, "prediction_drift", None)
            if prediction_drift is not None:
                hellinger: float = getattr(
                    prediction_drift, "hellinger_distance", 0.0,
                )
                hellinger_critical: float = (
                    self._thresholds.prediction_drift.hellinger.critical
                )
                if hellinger > hellinger_critical:
                    triggered_rules.append("prediction_shift")
                    reasons.append(
                        f"Prediction shift: Hellinger {hellinger:.4f} > "
                        f"{hellinger_critical:.4f}"
                    )
                    highest_urgency = self._higher_urgency(highest_urgency, "MEDIUM")
                    if action not in ("RETRAIN",):
                        action = "FLAG_FOR_REVIEW"
        except Exception:  # noqa: BLE001
            logger.exception("Error evaluating rule 'prediction_shift'.")

        # ---- Rule 5: Sustained warning streak -------------------------
        try:
            consecutive = self._get_consecutive_warnings()
            if consecutive >= 3:
                triggered_rules.append("sustained_warning")
                reasons.append(
                    f"Sustained warnings: {consecutive} consecutive "
                    f"breaches (threshold: 3)"
                )
                highest_urgency = self._higher_urgency(highest_urgency, "MEDIUM")
                if action not in ("RETRAIN",):
                    action = "RETRAIN"
        except Exception:  # noqa: BLE001
            logger.exception("Error evaluating rule 'sustained_warning'.")

        # ---- Assemble decision ----------------------------------------
        should_retrain = action == "RETRAIN"
        reason = " | ".join(reasons) if reasons else "No drift rules triggered."

        decision = RetrainingDecision(
            should_retrain=should_retrain,
            action=action,
            urgency=highest_urgency,
            reason=reason,
            triggered_rules=triggered_rules,
        )

        logger.info(
            "Retraining decision: action=%s urgency=%s rules=%s",
            decision.action,
            decision.urgency,
            decision.triggered_rules,
        )
        return decision

    def get_decision_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve past retraining decisions from the database.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of dictionaries with retraining-event fields, ordered
            newest-first.  Returns an empty list if the database is
            unavailable.
        """
        if self._db is None:
            logger.warning("No DB available – cannot fetch decision history.")
            return []

        try:
            with self._db.get_session() as session:
                rows = (
                    session.query(RetrainingEvent)
                    .order_by(desc(RetrainingEvent.timestamp))
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "id": r.id,
                        "timestamp": r.timestamp.isoformat()
                        if r.timestamp
                        else None,
                        "trigger_reason": r.trigger_reason,
                        "old_version": r.old_version,
                        "new_version": r.new_version,
                        "old_f1": r.old_f1,
                        "new_f1": r.new_f1,
                        "status": r.status,
                    }
                    for r in rows
                ]
        except Exception:  # noqa: BLE001
            logger.exception("Error fetching decision history.")
            return []

    def record_decision(self, decision: RetrainingDecision) -> None:
        """Persist a retraining decision to the ``retraining_events`` table.

        Only ``RETRAIN`` and ``FLAG_FOR_REVIEW`` actions are recorded;
        ``NO_ACTION`` decisions are silently skipped to avoid table
        bloat.

        Args:
            decision: The ``RetrainingDecision`` to store.
        """
        if self._db is None:
            logger.warning("No DB available – decision not recorded.")
            return

        if decision.action == "NO_ACTION":
            logger.debug("NO_ACTION decision skipped (not recorded).")
            return

        try:
            with self._db.get_session() as session:
                event = RetrainingEvent(
                    trigger_reason=decision.reason,
                    old_version=0,  # filled by the retraining pipeline
                    old_f1=0.0,     # filled by the retraining pipeline
                    status="STARTED" if decision.should_retrain else "REJECTED",
                )
                session.add(event)
            logger.info(
                "Retraining decision recorded (action=%s, urgency=%s).",
                decision.action,
                decision.urgency,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to record retraining decision.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_consecutive_warnings(self) -> int:
        """Count the most recent unbroken streak of breached drift results.

        Scans ``drift_results`` in reverse chronological order and
        counts rows where ``is_breached = True`` until the first
        non-breached row is encountered.

        Returns:
            Number of consecutive breached results.  Returns ``0`` when
            the database is unavailable or empty.
        """
        if self._db is None:
            return 0

        try:
            with self._db.get_session() as session:
                results = (
                    session.query(DriftResult.is_breached)
                    .order_by(desc(DriftResult.timestamp))
                    .limit(50)
                    .all()
                )

            count = 0
            for (is_breached,) in results:
                if is_breached:
                    count += 1
                else:
                    break
            return count
        except Exception:  # noqa: BLE001
            logger.exception("Error counting consecutive warnings.")
            return 0

    @staticmethod
    def _higher_urgency(current: str, candidate: str) -> str:
        """Return whichever urgency level is more severe.

        Args:
            current: Current highest urgency.
            candidate: New urgency to consider.

        Returns:
            The more severe of the two urgency strings.
        """
        cur_rank = _URGENCY_RANK.get(current, 3)
        cand_rank = _URGENCY_RANK.get(candidate, 3)
        return current if cur_rank <= cand_rank else candidate
