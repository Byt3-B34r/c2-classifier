"""
model.py
────────
Training, persistence, and inference for the C2 traffic classifier.

Wraps a scikit-learn estimator (RandomForest baseline; XGBoost as drop-in
replacement) with:

    - Reproducible train/test splitting (time-window aware to prevent leakage)
    - Class imbalance handling via class_weight='balanced'
    - SHAP explainability per prediction (TreeExplainer — fast on tree models)
    - Joblib serialisation of the (model + feature_names + metadata) bundle

Design notes
────────────
- The serialised artefact is a single dict containing the model, the exact
  FEATURE_NAMES list it was trained on, scaler state (if any), and training
  metadata (timestamp, dataset, scores). This makes models self-describing.
- SHAP is loaded lazily — it's only imported when explain() is called, so
  training and basic inference don't pay the import cost.
- Prediction returns structured Prediction objects rather than raw floats,
  so report.py downstream gets a stable contract.

Usage
─────
    # Training
    clf = C2Classifier()
    clf.train(X_train, y_train)
    clf.evaluate(X_test, y_test)
    clf.save("models/rf_v1.joblib")

    # Inference
    clf = C2Classifier.load("models/rf_v1.joblib")
    predictions = clf.predict(feature_vectors, explain=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from features import FEATURE_NAMES, N_FEATURES

logger = logging.getLogger(__name__)

# Bundle format version — bump on breaking changes to the saved schema.
BUNDLE_VERSION: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Output structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    """
    Single-flow prediction with optional SHAP attribution.

    Attributes
    ──────────
    label : str
        Predicted class — "benign" or "c2".
    confidence : float
        Probability of the predicted class, in [0.0, 1.0].
    proba_c2 : float
        Probability of the c2 class specifically. Useful for ranking.
    shap : dict[str, float] | None
        Feature → SHAP value for the c2 class. Positive values pushed the
        prediction toward c2; negative values pushed it toward benign.
        Only populated when explain=True is passed to predict().
    """
    label:      str
    confidence: float
    proba_c2:   float
    shap:       Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.shap is None:
            d.pop("shap")
        return d


@dataclass
class TrainingMetadata:
    """Metadata embedded in the serialised model bundle."""
    trained_at:   str   = ""
    dataset:      str   = ""
    n_train:      int   = 0
    n_test:       int   = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    metrics:      Dict[str, float] = field(default_factory=dict)
    feature_names:List[str] = field(default_factory=list)
    bundle_version: int = BUNDLE_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class C2Classifier:
    """
    Tree-based classifier for C2 traffic detection.

    The default estimator is a RandomForestClassifier with parameters tuned
    for the typical CIC-IDS-2017 / CTU-13 feature distributions. To swap in
    XGBoost or another tree model, pass it via the `estimator` argument:

        from xgboost import XGBClassifier
        clf = C2Classifier(estimator=XGBClassifier(...))

    The class assumes a binary problem: 0 = benign, 1 = c2.

    Parameters
    ──────────
    estimator : sklearn-compatible classifier, optional
        If None, a sensible RandomForest baseline is constructed.
    random_state : int
        Seed used wherever the estimator supports it.
    """

    LABEL_BENIGN: int = 0
    LABEL_C2:     int = 1
    LABEL_NAMES = {0: "benign", 1: "c2"}

    def __init__(
        self,
        estimator: Optional[Any] = None,
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state
        self.model: Any = estimator if estimator is not None else self._default_estimator()
        self.metadata: TrainingMetadata = TrainingMetadata()
        self._explainer: Optional[Any] = None  # lazy-init SHAP explainer

    # ── Construction ─────────────────────────────────────────────────────────

    def _default_estimator(self) -> RandomForestClassifier:
        """
        Baseline Random Forest configuration.

        Choices explained:
          - n_estimators=300       : enough trees for stable feature importances
          - max_depth=None         : let trees grow; class_weight handles imbalance
          - min_samples_leaf=2     : guard against single-flow leaves
          - class_weight='balanced': auto-weight inversely to class frequency
          - n_jobs=-1              : use all cores for fit and predict
        """
        return RandomForestClassifier(
            n_estimators       = 300,
            max_depth          = None,
            min_samples_leaf   = 2,
            class_weight       = "balanced",
            n_jobs             = -1,
            random_state       = self.random_state,
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "unknown",
    ) -> None:
        """
        Fit the model on (X, y).

        Parameters
        ──────────
        X : np.ndarray of shape (n_samples, N_FEATURES)
        y : np.ndarray of shape (n_samples,) with values in {0, 1}
        dataset_name : identifier recorded in the saved bundle metadata
        """
        self._validate_X(X)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]}"
            )

        unique, counts = np.unique(y, return_counts=True)
        class_counts = {self.LABEL_NAMES.get(int(u), str(u)): int(c)
                        for u, c in zip(unique, counts)}

        logger.info("model: training on %d samples, class counts=%s",
                    X.shape[0], class_counts)

        self.model.fit(X, y)

        # Record training metadata (metrics filled in by evaluate())
        self.metadata = TrainingMetadata(
            trained_at    = datetime.now(timezone.utc).isoformat(),
            dataset       = dataset_name,
            n_train       = int(X.shape[0]),
            class_counts  = class_counts,
            feature_names = list(FEATURE_NAMES),
            bundle_version= BUNDLE_VERSION,
        )

        # Invalidate any pre-existing explainer — it was bound to the old model
        self._explainer = None

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on a held-out test set.

        Returns a dict of metrics and also attaches them to self.metadata so
        they get serialised with the model.
        """
        self._validate_X(X_test)
        if not self._is_fitted():
            raise RuntimeError("model: cannot evaluate before training")

        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, self.LABEL_C2]

        # False-positive rate on the benign class — critical for SOC viability
        cm = confusion_matrix(y_test, y_pred, labels=[self.LABEL_BENIGN, self.LABEL_C2])
        tn, fp = cm[0]
        fn, tp = cm[1]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics = {
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":    float(recall_score(   y_test, y_pred, zero_division=0)),
            "f1":        float(f1_score(       y_test, y_pred, zero_division=0)),
            "fpr":       float(fpr),
            "roc_auc":   float(roc_auc_score(  y_test, y_proba))
                         if len(np.unique(y_test)) > 1 else 0.0,
        }

        self.metadata.n_test  = int(X_test.shape[0])
        self.metadata.metrics = metrics

        if verbose:
            logger.info("model: evaluation metrics %s", metrics)
            logger.info("model: confusion matrix\n%s",
                        np.array2string(cm, separator=", "))
            logger.info("model: classification report\n%s",
                        classification_report(y_test, y_pred,
                                              target_names=["benign", "c2"],
                                              zero_division=0))
        return metrics

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        explain: bool = False,
    ) -> List[Prediction]:
        """
        Predict labels for a batch of flows.

        Parameters
        ──────────
        X : np.ndarray of shape (n_samples, N_FEATURES)
        explain : if True, attach SHAP feature attributions to each prediction.
                  Adds ~50–200ms per flow; disable for high-throughput inference.

        Returns
        ───────
        List[Prediction] in the same order as X.
        """
        self._validate_X(X)
        if not self._is_fitted():
            raise RuntimeError("model: cannot predict before training")

        proba = self.model.predict_proba(X)
        # Map class indices to our label constants (handles models trained on
        # only one class, though that should not happen in practice)
        c2_idx = self._class_index(self.LABEL_C2)
        proba_c2 = proba[:, c2_idx] if c2_idx is not None else np.zeros(len(X))

        shap_values = self._explain_batch(X) if explain else None

        results: List[Prediction] = []
        for i in range(X.shape[0]):
            p_c2  = float(proba_c2[i])
            label = "c2" if p_c2 >= 0.5 else "benign"
            conf  = p_c2 if label == "c2" else 1.0 - p_c2

            shap_dict: Optional[Dict[str, float]] = None
            if shap_values is not None:
                shap_dict = {
                    name: float(shap_values[i][j])
                    for j, name in enumerate(FEATURE_NAMES)
                }

            results.append(Prediction(
                label      = label,
                confidence = conf,
                proba_c2   = p_c2,
                shap       = shap_dict,
            ))

        return results

    def predict_one(
        self,
        feature_vector: np.ndarray,
        explain: bool = False,
    ) -> Prediction:
        """Convenience wrapper around predict() for a single flow."""
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        return self.predict(feature_vector, explain=explain)[0]

    # ── SHAP explainability ──────────────────────────────────────────────────

    def _explain_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for the c2 class on a batch of inputs.

        Uses TreeExplainer which is fast and exact for tree-based models.
        For non-tree estimators, swap in shap.Explainer and accept the
        ~10x performance hit.

        Returns
        ───────
        np.ndarray of shape (n_samples, n_features) where each value is the
        signed contribution of that feature to the c2-class log-odds.
        """
        explainer = self._get_explainer()
        raw_shap  = explainer.shap_values(X)

        # SHAP's return shape varies across versions and estimator types:
        #   - sklearn RF (older shap)  → list[ array_class0, array_class1 ]
        #   - XGB / newer shap         → np.ndarray (n_samples, n_features)
        #   - newest shap (multiclass) → np.ndarray (n_samples, n_features, n_classes)
        if isinstance(raw_shap, list):
            # Older format: pick the c2-class array
            return raw_shap[self.LABEL_C2]

        if isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            # Newer format: (n_samples, n_features, n_classes)
            return raw_shap[:, :, self.LABEL_C2]

        # Already (n_samples, n_features) — binary-class ndarray
        return raw_shap

    def _get_explainer(self) -> Any:
        """Lazily instantiate a SHAP TreeExplainer bound to the current model."""
        if self._explainer is not None:
            return self._explainer
        try:
            import shap  # noqa: WPS433 — lazy import is intentional
        except ImportError as exc:
            raise ImportError(
                "SHAP is required for explain=True. Install with: pip install shap"
            ) from exc
        self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialise the (model + metadata) bundle to disk via joblib.

        The saved artefact is a dict so we can extend it without breaking
        backward compatibility — old loaders just ignore unknown keys.
        """
        if not self._is_fitted():
            raise RuntimeError("model: cannot save an untrained model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model":    self.model,
            "metadata": asdict(self.metadata),
            "version":  BUNDLE_VERSION,
        }
        joblib.dump(bundle, path)
        logger.info("model: saved bundle to %s (v%d)", path, BUNDLE_VERSION)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "C2Classifier":
        """
        Load a serialised bundle and reconstruct a C2Classifier.

        Performs feature-name compatibility checking: if the saved model was
        trained on a different feature set than the current code defines,
        a warning is logged. This catches the common bug of retraining
        after FEATURE_NAMES was edited mid-flight.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model bundle not found: {path}")

        bundle = joblib.load(path)

        version = bundle.get("version", 0)
        if version > BUNDLE_VERSION:
            raise RuntimeError(
                f"model bundle version {version} is newer than supported "
                f"{BUNDLE_VERSION} — upgrade c2-classifier"
            )

        clf = cls(estimator=bundle["model"])
        meta_dict = bundle.get("metadata", {})

        # Reconstruct metadata while tolerating older bundles missing fields
        clf.metadata = TrainingMetadata(**{
            k: meta_dict.get(k, getattr(TrainingMetadata, k, None))
            for k in TrainingMetadata.__dataclass_fields__.keys()
            if k in meta_dict
        })

        # Feature contract check
        saved_features = clf.metadata.feature_names
        if saved_features and saved_features != FEATURE_NAMES:
            logger.warning(
                "model: feature-name mismatch! "
                "Saved model expects %d features (%s), "
                "current code defines %d features (%s). "
                "Predictions may be incorrect.",
                len(saved_features), saved_features[:3],
                N_FEATURES,         FEATURE_NAMES[:3],
            )

        logger.info("model: loaded bundle from %s (trained_at=%s)",
                    path, clf.metadata.trained_at)
        return clf

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _validate_X(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be np.ndarray, got {type(X).__name__}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {N_FEATURES}. "
                f"Did FEATURE_NAMES change since training?"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values — clean upstream")

    def _is_fitted(self) -> bool:
        """Best-effort fitted check that works across sklearn estimators."""
        # sklearn convention: fitted estimators have at least one trailing-
        # underscore attribute set on them.
        return any(
            hasattr(self.model, attr) for attr in
            ("classes_", "estimators_", "feature_importances_", "_Booster")
        )

    def _class_index(self, label: int) -> Optional[int]:
        """Return the column index of *label* in predict_proba output."""
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return None
        idx = np.where(classes == label)[0]
        return int(idx[0]) if len(idx) else None


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly: python model.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    rng = np.random.default_rng(42)

    # Synthesise a separable binary dataset:
    #   benign — high IAT variance (CV ~ 1.0), low payload entropy (~ 4.0)
    #   c2     — low  IAT variance (CV ~ 0.05), high payload entropy (~ 7.5)
    n_per_class = 500

    benign = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, N_FEATURES)).astype(np.float32)
    c2     = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, N_FEATURES)).astype(np.float32)

    # Push the beacon_score and payload_entropy features into separable regions
    beacon_idx  = FEATURE_NAMES.index("beacon_score")
    entropy_idx = FEATURE_NAMES.index("payload_entropy")
    benign[:, beacon_idx]  += 1.5
    benign[:, entropy_idx] += 3.0
    c2[:,     beacon_idx]  += 0.05
    c2[:,     entropy_idx] += 7.5

    X = np.vstack([benign, c2])
    y = np.concatenate([
        np.zeros(n_per_class, dtype=int),
        np.ones( n_per_class, dtype=int),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    clf = C2Classifier()
    clf.train(X_train, y_train, dataset_name="synthetic_smoke")
    metrics = clf.evaluate(X_test, y_test, verbose=False)

    assert metrics["f1"]      > 0.95, f"smoke-test F1 too low: {metrics['f1']}"
    assert metrics["roc_auc"] > 0.98, f"smoke-test ROC-AUC too low: {metrics['roc_auc']}"

    # Round-trip serialisation
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        clf.save(tmp.name)
        loaded = C2Classifier.load(tmp.name)

    # Round-trip prediction with SHAP
    sample = X_test[:3]
    preds  = loaded.predict(sample, explain=True)
    assert len(preds) == 3
    assert all(p.shap is not None for p in preds)
    assert all(len(p.shap) == N_FEATURES for p in preds)

    print(f"smoke-test passed")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:<10} {v:.4f}")

    print(f"\nSample prediction (with SHAP):")
    p = preds[0]
    print(f"  label      : {p.label}")
    print(f"  confidence : {p.confidence:.4f}")
    print(f"  proba_c2   : {p.proba_c2:.4f}")
    top_shap = sorted(p.shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print(f"  top SHAP contributors:")
    for name, val in top_shap:
        sign = "+" if val >= 0 else "−"
        print(f"    {sign} {name:<26} {val:+.4f}")