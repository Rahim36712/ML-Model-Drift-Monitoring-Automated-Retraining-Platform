"""Binary fraud metrics and explicit champion/challenger safety gates."""

from dataclasses import asdict, dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    f1: float
    precision: float
    recall: float
    auc_roc: float | None
    confusion_matrix: list
    classification_report_str: str

    def to_dict(self):
        return asdict(self)


@dataclass
class ComparisonResult:
    is_improved: bool
    metric_deltas: dict
    recommendation: str
    reason: str


class ModelEvaluator:
    def evaluate(self, model, X, y) -> EvaluationResult:
        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]
        return EvaluationResult(
            float(accuracy_score(y, pred)),
            float(f1_score(y, pred, zero_division=0)),
            float(precision_score(y, pred, zero_division=0)),
            float(recall_score(y, pred, zero_division=0)),
            float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
            confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            classification_report(y, pred, zero_division=0),
        )

    def compare_models(self, champion, challenger, min_improvement=0.005):
        delta = {
            k: getattr(challenger, k) - getattr(champion, k)
            for k in ["f1", "accuracy", "precision", "recall"]
        }
        improved = delta["f1"] >= min_improvement and delta["recall"] >= -0.02
        return ComparisonResult(
            improved,
            delta,
            "DEPLOY" if improved else "REJECT",
            "F1 improvement and recall gate passed."
            if improved
            else "Candidate did not pass F1 improvement / recall gates.",
        )


def promotion_gates(historical, recent=None, settings=None):
    """Pairs are (champion metrics, challenger metrics) on identical raw rows."""
    improvement = getattr(settings, "min_f1_improvement", 0.005)
    f1_limit = getattr(settings, "max_historical_f1_drop", 0.01)
    recall_limit = getattr(settings, "max_recall_drop", 0.02)
    selected = recent or historical
    checks = {
        "selected_f1_improvement": selected[1].f1 - selected[0].f1 >= improvement - 1e-12,
        "historical_f1_safety": historical[1].f1 - historical[0].f1 >= -f1_limit - 1e-12,
        "historical_recall_safety": historical[1].recall - historical[0].recall
        >= -recall_limit - 1e-12,
    }
    if recent:
        checks["recent_recall_safety"] = (
            recent[1].recall - recent[0].recall >= -recall_limit - 1e-12
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selected_set": "recent" if recent else "historical",
        "historical": [m.to_dict() for m in historical],
        "recent": [m.to_dict() for m in recent] if recent else None,
        "limits": {
            "min_f1_improvement": improvement,
            "max_historical_f1_drop": f1_limit,
            "max_recall_drop": recall_limit,
        },
    }
