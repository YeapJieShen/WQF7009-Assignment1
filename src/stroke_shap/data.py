"""Load, encode, split and impute the Kaggle stroke dataset.

The encoding and split reproduce ``scripts/utils.py`` (deepfindr, 2021) exactly,
so the baseline pipeline matches the provided code. The patient ``id`` is kept
as the index instead of being dropped, so that explained patients can be traced
back to the CSV.
"""

from pathlib import Path
from typing import Literal, NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from stroke_shap import SEED

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "healthcare-dataset-stroke-data.csv"
TEST_SIZE = 0.2

CATEGORICAL_NAMES = {
    "gender": "Gender",
    "ever_married": "Ever Married",
    "work_type": "Work Type",
    "Residence_type": "Residence",
    "smoking_status": "Smoking",
}
"""Categorical columns (one-hot encoded) and their display names."""

NUMERIC_NAMES = {
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "avg_glucose_level": "Average Glucose",
    "bmi": "BMI",
    "bmi_missing": "BMI Missing",
}
"""Columns used as they are, and their display names."""

UNITS = {"age": "years", "avg_glucose_level": "mg/dL", "bmi": "kg/m²"}
"""Measurement units of the continuous columns, for axis labels."""


class Split(NamedTuple):
    """Train/test predictors and targets, indexed by patient ``id``."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_split(pipeline: Literal["baseline", "improved"]) -> Split:
    """Load the dataset, one-hot encode it and split it 80/20.

    Args:
        pipeline: ``"baseline"`` fills missing BMI with 0, as in the provided code.
            ``"improved"`` adds a ``bmi_missing`` flag and fills missing BMI with
            the training median (see :func:`impute_bmi`).

    Returns:
        The split, with identical rows in both pipelines.
    """
    data = pd.read_csv(DATA_PATH, index_col="id")
    if pipeline == "improved":
        data.insert(data.columns.get_loc("bmi") + 1, "bmi_missing", data["bmi"].isna().astype(int))

    categorical = list(CATEGORICAL_NAMES)
    encoded = pd.get_dummies(data[categorical], prefix=categorical, dtype=int)
    X = pd.concat([encoded, data.drop(columns=[*categorical, "stroke"])], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data["stroke"], test_size=TEST_SIZE, random_state=SEED
    )

    if pipeline == "baseline":
        X_train, X_test = X_train.fillna({"bmi": 0}), X_test.fillna({"bmi": 0})
    else:
        X_train, X_test = impute_bmi(X_train, X_test)
    return Split(X_train, X_test, y_train, y_test)


def impute_bmi(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing BMI with the median of the recorded training BMIs.

    The median is computed on the training set only, so no test information
    leaks into training. Inside cross-validation, the pipeline step
    :class:`stroke_shap.models.BMIImputer` refits it on each fold.

    Args:
        X_train: Training predictors with missing ``bmi`` values.
        X_test: Test predictors with missing ``bmi`` values.

    Returns:
        Copies of both sets with ``bmi`` filled.
    """
    median = X_train["bmi"].median()
    return X_train.fillna({"bmi": median}), X_test.fillna({"bmi": median})


def variable_of(column: str) -> str:
    """Return the display name of the original variable an encoded column belongs to.

    Examples:
        >>> variable_of("smoking_status_formerly smoked")
        'Smoking'
        >>> variable_of("avg_glucose_level")
        'Average Glucose'
    """
    for prefix, name in CATEGORICAL_NAMES.items():
        if column.startswith(f"{prefix}_"):
            return name
    return NUMERIC_NAMES.get(column, column)


def display_label(column: str) -> str:
    """Return a plain-English label for an encoded column.

    Examples:
        >>> display_label("smoking_status_formerly smoked")
        'Smoking: Formerly Smoked'
        >>> display_label("bmi")
        'BMI'
        >>> display_label("smoking_status")
        'Smoking'
    """
    for prefix, name in CATEGORICAL_NAMES.items():
        if column.startswith(f"{prefix}_"):
            level = column.removeprefix(f"{prefix}_").replace("_", " ")
            return f"{name}: {level.title()}"
    return {**CATEGORICAL_NAMES, **NUMERIC_NAMES}.get(column, column)


def axis_label(column: str) -> str:
    """Return :func:`display_label` with the column's unit, if it has one.

    Examples:
        >>> axis_label("avg_glucose_level")
        'Average Glucose (mg/dL)'
        >>> axis_label("hypertension")
        'Hypertension'
    """
    label = display_label(column)
    return f"{label} ({UNITS[column]})" if column in UNITS else label
