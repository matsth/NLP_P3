"""Runnable entrypoint for multi-label GoEmotions experiments with systematic sweeping."""

from __future__ import annotations

import itertools
from pathlib import Path
from models.pipeline import EmotionTrainingPipeline, PipelineResult


def training_emotions(
    classifier_type: str,
    representation_type: str,
    processing_variant: str,
    c_param: float,
    metrics_registry_path: str | Path = "metrics_registry.csv"
) -> PipelineResult:
    """Train and validate a GoEmotions classifier and ensure save boundaries.

    Examples:
        training_emotions("logistic_regression", "word_tfidf", "variant_b", 1.0)
        training_emotions("svm", "char_ngrams", "variant_c", 0.5)
        training_emotions("minilm", "transformer", "raw", 1.0)
    """
    pipeline = EmotionTrainingPipeline(
        classifier_type=classifier_type,
        representation_type=representation_type,
        processing_variant=processing_variant,
        c_param=c_param,
        metrics_registry_path=metrics_registry_path,
        tracking_enabled=True,
    )
    return pipeline.run()


def run_all_experiments(registry_filename: str = "metrics_registry.csv") -> None:
    """Systematically runs every combination of preprocessing, representation, and model.
    
    Ensures language models automatically run on raw text and skip 
    traditional bag-of-words vectorization. Safeguards against local OOM and script crashes.
    """
    registry_path = Path(registry_filename).resolve()
    
    # 1. Define evaluation sweeping matrices
    variants = ["variant_a", "variant_b", "variant_c"]
    traditional_representations = ["word_tfidf", "char_ngrams"]
    
    traditional_classifiers = ["logistic_regression", "svm", "random_forest"]
    transformer_classifiers = ["minilm", "albert"]
    
    c_params = [1.0]  # Add other regularization strengths if desired (e.g. [0.1, 1.0, 10.0])

    print("=" * 70)
    print("STARTING MULTI-LABEL EMOTION EXPERIMENTAL SWEEP")
    print(f"Destination Registry: {registry_path}")
    print("=" * 70)

    # --- Phase 1: Sweep Traditional Classifiers ---
    # Setup standard combinations (Total 18 variations for 1 C parameter)
    traditional_combinations = list(itertools.product(
        traditional_classifiers, traditional_representations, variants, c_params
    ))

    print(f"\nPhase 1: Scheduling {len(traditional_combinations)} Traditional Model runs...")
    for idx, (clf, rep, variant, c) in enumerate(traditional_combinations, 1):
        
        
        if idx < 13:
            print(f"[{idx}/{len(traditional_combinations)}] Already completed. Skipping...")
            continue
        print(f"\n[{idx}/{len(traditional_combinations)}] Running: Config -> {clf} | {rep} | {variant} | C={c}")
        try:
            result = training_emotions(
                classifier_type=clf,
                representation_type=rep,
                processing_variant=variant,
                c_param=c,
                metrics_registry_path=registry_path
            )
            print(f"Metric logged successfully for run: {result.run_name}")
        except Exception as e:
            print(f"Execution Failure on Traditional Config [{clf} | {rep} | {variant}]: {e}")
            print("Skipping to next hyperparameter target to ensure execution continuity...")

    # --- Phase 2: Sweep Language Models (Transformers) ---
    print(f"\nPhase 2: Scheduling {len(transformer_classifiers)} Transformer Model sweeps...")
    for idx, transformer in enumerate(transformer_classifiers, 1):
        # Language models bypass all text preprocessing and use raw strings.
        print(f"\n[{idx}/{len(transformer_classifiers)}] Running: Transformer -> {transformer} | Text=raw")
        try:
            result = training_emotions(
                classifier_type=transformer,
                representation_type="transformer",
                processing_variant="raw",
                c_param=1.0,
                metrics_registry_path=registry_path
            )
            print(f"Metric logged successfully for run: {result.run_name}")
        except Exception as e:
            print(f"Execution Failure on Transformer [{transformer}]: {e}")
            print("Continuing grid iterations...")

    print("\n" + "=" * 70)
    print("ALL BENCHMARK SWEEPS FINISHED EXECUTING!")
    print(f"Confirm aggregate rows via local registry path: {registry_path}")
    print("=" * 70)


if __name__ == "__main__":
    import wandb
    wandb.login(key="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")
    # Execute the crash-resilient bulk runner loop
    run_all_experiments()