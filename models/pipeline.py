"""Training, validation, evaluation, and metrics logging orchestration."""

from __future__ import annotations

import csv
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data.data_loader import EmotionDataLoader
from data.data_processing import EMOTION_COLUMNS, DatasetSplits, normalize_variant_name
from models.classifiers import (
    ClassifierFactory,
    normalize_classifier_name,
    resolve_transformer_model_name,
)
from models.representations import TextRepresentation, normalize_representation_name
from utils.tracker import EmotionTracker


@dataclass(frozen=True)
class PipelineResult:
    """Summary returned from a completed training run."""

    run_name: str
    test_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    metrics_registry_path: Path


class EmotionTrainingPipeline:
    """Coordinate data loading, model fitting, evaluation, and persistence."""

    def __init__(
        self,
        classifier_type: str,
        representation_type: str,
        processing_variant: str,
        c_param: float,
        data_dir: str | Path = "data",
        metrics_registry_path: str | Path = "metrics_registry.csv",
        tracking_enabled: bool = True,
    ) -> None:
        self.classifier_type = classifier_type
        self.representation_type = representation_type
        self.processing_variant = processing_variant
        self.c_param = float(c_param)
        self.data_loader = EmotionDataLoader(data_dir=data_dir)
        self.metrics_registry_path = Path(metrics_registry_path)
        self.tracking_enabled = tracking_enabled
        self.label_columns = EMOTION_COLUMNS

    def run(self) -> PipelineResult:
        normalized_classifier = normalize_classifier_name(self.classifier_type)
        if normalized_classifier == "transformer":
            return self._run_transformer()
        return self._run_traditional()

    def _run_traditional(self) -> PipelineResult:
        variant = normalize_variant_name(self.processing_variant)
        representation_name = normalize_representation_name(self.representation_type)
        splits = self.data_loader.load(variant)
        tracker = self._build_tracker(variant, representation_name)

        try:
            train_texts, y_train = self._texts_and_labels(splits.train)
            test_texts, y_test = self._texts_and_labels(splits.test)
            validation_texts, y_validation = self._texts_and_labels(splits.validation)

            representation = TextRepresentation(representation_name)
            x_train = representation.fit_transform(train_texts)
            x_test = representation.transform(test_texts)
            x_validation = representation.transform(validation_texts)

            # Build the model using the factory
            classifier = ClassifierFactory(
                self.classifier_type, c_param=self.c_param
            ).build_traditional()
            
            # Standard, straightforward fitting
            classifier.fit(x_train, y_train)

            test_metrics = self._evaluate_model(classifier, x_test, y_test)
            tracker.log_evaluation(test_metrics, split="test")

            validation_metrics = self._evaluate_model(classifier, x_validation, y_validation)
            tracker.log_evaluation(validation_metrics, split="validation")
            self._append_registry_row(tracker.run_name, validation_metrics)

            return PipelineResult(
                run_name=tracker.run_name,
                test_metrics=test_metrics,
                validation_metrics=validation_metrics,
                metrics_registry_path=self.metrics_registry_path,
            )
        finally:
            tracker.finish()

    def _run_transformer(self) -> PipelineResult:
        variant = "raw"
        splits = self.data_loader.load_raw()
        model_name = resolve_transformer_model_name(self.classifier_type)
        tracker = self._build_tracker(variant, "transformer")

        try:
            test_metrics, validation_metrics = self._train_transformer_model(
                model_name=model_name,
                splits=splits,
                run_name=tracker.run_name,
            )
            tracker.log_evaluation(test_metrics, split="test")
            tracker.log_evaluation(validation_metrics, split="validation")
            self._append_registry_row(tracker.run_name, validation_metrics)

            return PipelineResult(
                run_name=tracker.run_name,
                test_metrics=test_metrics,
                validation_metrics=validation_metrics,
                metrics_registry_path=self.metrics_registry_path,
            )
        finally:
            tracker.finish()

    def _train_transformer_model(
        self, model_name: str, splits: DatasetSplits, run_name: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            import torch
            from datasets import Dataset
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                EarlyStoppingCallback,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Transformer training requires torch, datasets, and transformers."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        def to_dataset(frame):
            subset = frame.select(["text", *self.label_columns]).to_pandas()
            dataset = Dataset.from_pandas(subset, preserve_index=False)

            def tokenize(batch):
                encoded = tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=128,
                    padding="max_length",
                )
                labels = np.array(
                    [[batch[label][i] for label in self.label_columns] for i in range(len(batch["text"]))],
                    dtype=np.float32,
                )
                encoded["labels"] = labels.tolist()
                return encoded

            columns_to_remove = ["text", *self.label_columns]
            return dataset.map(tokenize, batched=True, remove_columns=columns_to_remove)

        train_dataset = to_dataset(splits.train)
        test_dataset = to_dataset(splits.test)
        validation_dataset = to_dataset(splits.validation)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_columns),
            id2label={index: label for index, label in enumerate(self.label_columns)},
            label2id={label: index for index, label in enumerate(self.label_columns)},
            problem_type="multi_label_classification",
        )

        args = self._training_arguments(
            TrainingArguments,
            run_name,
            fp16=torch.cuda.is_available(),
        )

        def compute_metrics(eval_prediction):
            logits, labels = eval_prediction
            scores = torch.sigmoid(torch.tensor(logits)).numpy()
            predictions = (scores >= 0.5).astype(int)
            return self._calculate_metrics(labels, predictions, scores)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()

        test_output = trainer.predict(test_dataset)
        validation_output = trainer.predict(validation_dataset)

        test_scores = torch.sigmoid(torch.tensor(test_output.predictions)).numpy()
        validation_scores = torch.sigmoid(torch.tensor(validation_output.predictions)).numpy()

        test_metrics = self._calculate_metrics(
            test_output.label_ids, (test_scores >= 0.5).astype(int), test_scores
        )
        validation_metrics = self._calculate_metrics(
            validation_output.label_ids,
            (validation_scores >= 0.5).astype(int),
            validation_scores,
        )
        return test_metrics, validation_metrics

    def _training_arguments(self, training_arguments_cls, run_name: str, fp16: bool):
        output_dir = Path("checkpoints") / run_name
        kwargs = {
            "output_dir": str(output_dir),
            "num_train_epochs": 10,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_f1",
            "greater_is_better": True,
            "save_strategy": "epoch",
            "logging_strategy": "epoch",
            "report_to": ["wandb"] if self.tracking_enabled else [],
            "fp16": fp16,
        }
        signature = inspect.signature(training_arguments_cls.__init__)
        if "eval_strategy" in signature.parameters:
            kwargs["eval_strategy"] = "epoch"
        else:
            kwargs["evaluation_strategy"] = "epoch"
        return training_arguments_cls(**kwargs)

    def _evaluate_model(self, model, features, y_true: np.ndarray) -> dict[str, Any]:
        predictions = np.asarray(model.predict(features), dtype=int)
        scores = self._score_outputs(model, features, predictions)
        return self._calculate_metrics(y_true, predictions, scores)

    def _score_outputs(self, model, features, fallback_predictions: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            if isinstance(probabilities, list):
                return np.column_stack([prob[:, 1] for prob in probabilities])
            probabilities_array = np.asarray(probabilities)
            if probabilities_array.ndim == 3:
                return probabilities_array[:, :, 1].T
            return probabilities_array

        if hasattr(model, "decision_function"):
            decision_values = model.decision_function(features)
            if isinstance(decision_values, list):
                decision_values = np.column_stack(decision_values)
            decision_values = np.asarray(decision_values, dtype=float)
            return 1.0 / (1.0 + np.exp(-decision_values))

        return fallback_predictions.astype(float)

    def _calculate_metrics(
        self, y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray
    ) -> dict[str, Any]:
        y_true = np.asarray(y_true, dtype=int)
        predictions = np.asarray(predictions, dtype=int)
        scores = np.asarray(scores, dtype=float)

        f1_values = f1_score(
            y_true,
            predictions,
            average=None,
            zero_division=0,
        )
        per_class_auc = []
        for index in range(y_true.shape[1]):
            try:
                per_class_auc.append(roc_auc_score(y_true[:, index], scores[:, index]))
            except ValueError:
                per_class_auc.append(np.nan)

        macro_roc_auc = float(np.nanmean(per_class_auc)) if per_class_auc else float("nan")
        return {
            "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "macro_roc_auc": macro_roc_auc,
            "f1_per_class": {
                label: float(score) for label, score in zip(self.label_columns, f1_values)
            },
            "roc_auc_per_class": {
                label: float(score) if not np.isnan(score) else None
                for label, score in zip(self.label_columns, per_class_auc)
            },
        }

    def _texts_and_labels(self, frame) -> tuple[list[str], np.ndarray]:
        texts = [str(text) for text in frame.get_column("text").to_list()]
        labels = frame.select(self.label_columns).to_numpy().astype(int)
        return texts, labels

    def _build_tracker(self, variant: str, representation_name: str) -> EmotionTracker:
        return EmotionTracker(
            classifier_type=self.classifier_type,
            representation_type=representation_name,
            processing_variant=variant,
            c_param=self.c_param,
            enabled=self.tracking_enabled,
            config={
                "classifier_type": self.classifier_type,
                "representation_type": representation_name,
                "processing_variant": variant,
                "c_param": self.c_param,
                "label_count": len(self.label_columns),
            },
        )

    def _append_registry_row(self, run_name: str, validation_metrics: dict[str, Any]) -> None:
        fieldnames = ["id", "accuracy", "macro_roc_auc"] + [
            f"f1_{label}" for label in self.label_columns
        ]
        row = {
            "id": run_name,
            "accuracy": validation_metrics["accuracy"],
            "macro_roc_auc": validation_metrics["macro_roc_auc"],
        }
        row.update(
            {
                f"f1_{label}": validation_metrics["f1_per_class"][label]
                for label in self.label_columns
            }
        )

        file_exists = self.metrics_registry_path.exists()
        with self.metrics_registry_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
