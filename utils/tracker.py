"""Weights & Biases tracking isolation for emotion experiments."""

from __future__ import annotations

from datetime import datetime
from typing import Any


class EmotionTracker:
    """Encapsulate W&B initialization and all metric logging."""

    def __init__(
        self,
        classifier_type: str,
        representation_type: str,
        processing_variant: str,
        c_param: float,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self.classifier_type = classifier_type
        self.representation_type = representation_type
        self.processing_variant = processing_variant
        self.c_param = c_param
        self.enabled = enabled
        self.run_name = self._build_run_name()
        self._run = None

        if self.enabled:
            try:
                import wandb

                self._run = wandb.init(
                    entity="matteostaehlin-zhaw",
                    project="NLP_p3",
                    name=self.run_name,
                    config=config or {},
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Weights & Biases is required for tracking. Install it with "
                    "`pip install wandb`, or disable tracking in the pipeline."
                ) from exc

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        """Log scalar metrics to the active W&B dashboard."""
        if not self.enabled or self._run is None:
            return

        payload = {
            f"{prefix}/{key}" if prefix else key: value
            for key, value in metrics.items()
            if _is_scalar(value)
        }
        self._run.log(payload, step=step)

    def log_evaluation(self, metrics: dict[str, Any], split: str) -> None:
        """Parse and log aggregate plus per-class multi-label metrics."""
        self.log_metrics(
            {
                "macro_f1": metrics.get("macro_f1"),
                "accuracy": metrics.get("accuracy"),
                "macro_roc_auc": metrics.get("macro_roc_auc"),
            },
            prefix=split,
        )

        for class_name, f1_value in metrics.get("f1_per_class", {}).items():
            self.log_metrics({f"f1_{class_name}": f1_value}, prefix=split)

    def finish(self) -> None:
        """Close the W&B run."""
        if self.enabled and self._run is not None:
            self._run.finish()

    def _build_run_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            f"{self.classifier_type}_{self.representation_type}_"
            f"{self.processing_variant}_C{self.c_param}_{timestamp}"
        )


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None
