"""Classifier factories for traditional and transformer models."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC


TRANSFORMER_MODELS = {
    "minilm": "nreimers/MiniLM-L6-H384-uncased",
    "nreimers/minilm-l6-h384-uncased": "nreimers/MiniLM-L6-H384-uncased",
    "albert": "albert-base-v2",
    "albert-base-v2": "albert-base-v2",
    "deberta": "microsoft/deberta-v3-small",
    "deberta-v3-small": "microsoft/deberta-v3-small",
    "microsoft/deberta-v3-small": "microsoft/deberta-v3-small",
}


@dataclass(frozen=True)
class ClassifierFactory:
    """Build multi-label classifiers from concise classifier names."""

    classifier_type: str
    c_param: float = 1.0
    random_state: int = 42

    def build_traditional(self) -> MultiOutputClassifier:
        name = normalize_classifier_name(self.classifier_type)
        if name == "logistic_regression":
            estimator = LogisticRegression(
                C=self.c_param,
                solver="liblinear",
                max_iter=1_000,
                class_weight="balanced",
                random_state=self.random_state,
            )
        elif name == "svm":
            estimator = LinearSVC(
                C=self.c_param,
                class_weight="balanced",
                random_state=self.random_state,
            )
        elif name == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"{self.classifier_type} is not a traditional classifier.")

        return MultiOutputClassifier(estimator, n_jobs=1)


def normalize_classifier_name(classifier_type: str) -> str:
    cleaned = classifier_type.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "lr": "logistic_regression",
        "logreg": "logistic_regression",
        "logistic": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "svm": "svm",
        "linear_svc": "svm",
        "linearsvc": "svm",
        "rf": "random_forest",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
    }
    if cleaned in aliases:
        return aliases[cleaned]
    if cleaned in TRANSFORMER_MODELS:
        return "transformer"
    raise ValueError(f"Unsupported classifier_type: {classifier_type}")


def resolve_transformer_model_name(classifier_type: str) -> str:
    cleaned = classifier_type.lower().strip()
    if cleaned not in TRANSFORMER_MODELS:
        raise ValueError(
            "Transformer classifier_type must be one of: minilm, albert-base-v2, "
            "microsoft/deberta-v3-small."
        )
    return TRANSFORMER_MODELS[cleaned]
