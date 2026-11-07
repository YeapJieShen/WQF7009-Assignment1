"""SHAP explainers for the two models, additivity check and grouped importance."""

import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline

from stroke_shap.data import variable_of


def tree_shap(
    model: Pipeline, X: pd.DataFrame, background: pd.DataFrame | None
) -> tuple[pd.DataFrame, float]:
    """Explain the random forest's stroke probability with ``shap.TreeExplainer``.

    Args:
        model: Fitted pipeline ending in the random forest.
        X: Patients to explain.
        background: Reference patients for the *interventional* algorithm. ``None``
            uses the *tree path-dependent* algorithm, whose reference is the
            (oversampled) data the trees were trained on.

    Returns:
        SHAP values for the stroke class, shaped and indexed like ``X``, and the base value.
    """
    forest = model[-1]
    if background is None:
        explainer = shap.TreeExplainer(forest)
    else:
        explainer = shap.TreeExplainer(forest, data=background, feature_perturbation="interventional")
    # Additivity is checked separately (additivity_residual); shap raises on small residuals otherwise
    values = explainer.shap_values(X, check_additivity=False)[..., 1]
    base_value = float(explainer.expected_value[1])
    return pd.DataFrame(values, index=X.index, columns=X.columns), base_value


def deep_shap(model: Pipeline, X: pd.DataFrame, background: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Explain the MLP's stroke probability with ``shap.DeepExplainer``.

    Inputs are standardised with the pipeline's fitted scaler, as the network sees them.

    Args:
        model: Fitted pipeline: BMI imputer, scaler, oversampler,
            :class:`~stroke_shap.mlp.TorchMLPClassifier`.
        X: Patients to explain.
        background: Reference patients (unscaled).

    Returns:
        SHAP values, shaped and indexed like ``X``, and the base value.
    """
    import torch

    scaler, network = model.named_steps["standardscaler"], model[-1].module_
    reference = torch.as_tensor(scaler.transform(background), dtype=torch.float32)
    inputs = torch.as_tensor(scaler.transform(X), dtype=torch.float32)
    explainer = shap.DeepExplainer(network, reference)
    values = explainer.shap_values(inputs, check_additivity=False)[..., 0]
    base_value = float(np.ravel(explainer.expected_value)[0])
    return pd.DataFrame(values, index=X.index, columns=X.columns), base_value


def additivity_residual(base_value: float, values: pd.DataFrame, proba: pd.Series) -> pd.Series:
    """Return |base value + sum of SHAP values - model output| per patient (0 if exact)."""
    return (base_value + values.sum(axis=1) - proba).abs()


def by_variable(values: pd.DataFrame) -> pd.DataFrame:
    """Sum each patient's SHAP values over the one-hot columns of each original variable.

    Returns:
        Signed SHAP values, one column per variable, named by display name.
    """
    return values.T.groupby(variable_of).sum().T


def importance_by_variable(values: pd.DataFrame) -> pd.Series:
    """Return the mean |SHAP value| of each original variable, largest first."""
    return by_variable(values).abs().mean().sort_values(ascending=False)
