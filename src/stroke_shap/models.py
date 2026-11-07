"""Model pipelines, decision threshold and test-set evaluation."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from stroke_shap import SEED

N_FOLDS = 5


class BMIImputer(BaseEstimator, TransformerMixin):
    """Fill missing BMI with the median of the recorded BMIs seen in ``fit``.

    Missing rows are found through the ``bmi_missing`` flag, because
    :func:`stroke_shap.data.load_split` has already filled them with the full
    training median. As a pipeline step, the median is refitted on each
    cross-validation fold's training part, so the held-out fold never informs it.
    On the full training set it reproduces the median used by ``load_split``.
    Without a ``bmi_missing`` column (the baseline) it does nothing.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "BMIImputer":
        """Learn the median of the recorded BMIs (rows with ``bmi_missing == 0``)."""
        self.active_ = "bmi_missing" in X.columns
        if self.active_:
            self.median_ = float(X.loc[X["bmi_missing"] == 0, "bmi"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with flagged rows' BMI set to the learned median."""
        if not self.active_:
            return X
        X = X.copy()
        X.loc[X["bmi_missing"] == 1, "bmi"] = self.median_
        return X


def make_random_forest() -> Pipeline:
    """Return BMI imputation, minority oversampling, then a 100-tree random forest.

    Imputation and oversampling sit inside the pipeline, so they are fitted on
    training data only, including within each cross-validation fold. The
    oversampler and forest are the provided model.
    """
    return make_pipeline(
        BMIImputer(), RandomOverSampler(random_state=SEED), RandomForestClassifier(random_state=SEED)
    )


def make_mlp(**params: object) -> Pipeline:
    """Return BMI imputation, standardisation, minority oversampling, then a PyTorch MLP.

    The scaler comes before the oversampler, so it is fitted on the real (not
    oversampled) training patients.

    Args:
        **params: Passed to :class:`stroke_shap.mlp.TorchMLPClassifier`.
    """
    from stroke_shap.mlp import TorchMLPClassifier  # torch is only needed for this model

    return make_pipeline(
        BMIImputer(), StandardScaler(), RandomOverSampler(random_state=SEED), TorchMLPClassifier(**params)
    )


def out_of_fold_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Score every training patient with a copy of ``model`` that did not see them.

    Stratified 5-fold cross-validation: each fold refits the whole pipeline on the
    other four folds and predicts the held-out fold.

    Returns:
        Out-of-fold stroke probabilities, one per row of ``X``.
    """
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    return cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]


def youden_threshold(y: pd.Series, scores: np.ndarray) -> float:
    """Return the threshold that maximises Youden's J = recall + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate(y: pd.Series, proba: pd.Series, threshold: float) -> dict[str, float]:
    """Return confusion counts, threshold metrics and ranking metrics for one model.

    Args:
        y: True labels (1 = recorded stroke).
        proba: Predicted stroke probabilities.
        threshold: Probability at or above which a patient is flagged.
    """
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "Strokes Caught (TP)": tp,
        "Strokes Missed (FN)": fn,
        "False Alarms (FP)": fp,
        "Correctly Cleared (TN)": tn,
        "Recall": recall_score(y, pred),
        "Precision": precision_score(y, pred),
        "F1": f1_score(y, pred),
        "Balanced Accuracy": balanced_accuracy_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
        "ROC AUC": roc_auc_score(y, proba),
        "Average Precision": average_precision_score(y, proba),
    }


def select_main_case(y: pd.Series, proba: pd.Series, threshold: float, bmi_missing: pd.Series) -> int:
    """Return the patient ``id`` of the main case for the local explanation.

    Rule, fixed before looking at any explanation: among true positives with a
    recorded BMI, take the one with the median predicted probability (lower
    median), i.e. a typical flagged stroke rather than the most confident one.
    """
    candidates = proba[(y == 1) & (proba >= threshold) & (bmi_missing == 0)].sort_values(kind="stable")
    return int(candidates.index[(len(candidates) - 1) // 2])
