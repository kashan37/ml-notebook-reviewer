from typing import TypedDict, Optional, List
import uuid
from datetime import datetime


# ==============================
# W2D1— VISSUALIZATION INTELLIGENCE
# ==============================

class EpochMetrics(TypedDict):
    """Single epoch's metrics — building block for loss curves."""
    epoch: int
    loss: Optional[float]
    val_loss: Optional[float]
    accuracy: Optional[float]
    val_accuracy: Optional[float]


class LossCurveData(TypedDict):
    """
    Full training history extracted from output cells.
    Week 1 only grabbed the LAST epoch. Week 2 grabs ALL of them.
    This is what powers the loss curve chart.
    """
    epochs: List[EpochMetrics]          # one entry per epoch
    total_epochs_found: int             # how many epochs we actually extracted
    has_validation_curves: bool         # did we find val_loss / val_accuracy


class OverfittingScore(TypedDict):
    """
    Heuristic-based overfitting assessment.
    Score is 0-100. Higher = more likely overfitting.
    Never claim certainty — these are signals not verdicts.
    """
    score: Optional[int]                # 0-100, None if not enough data
    risk_level: Optional[str]           # "low" | "moderate" | "high" | None
    widening_gap_detected: bool         # val_loss diverging from train_loss
    unstable_validation_detected: bool  # val_loss oscillating
    memorization_detected: bool         # train acc >> val acc consistently
    evidence: List[str]                 # human readable list of what triggered it


class TrainingRiskFlag(TypedDict):
    """Single training configuration risk finding."""
    severity: str                       # "warning" | "info"
    message: str                        # human readable, phrased as potential risk
    category: str                       # "epochs" | "batch_size" | "callbacks" | etc



# ==============================
# NOTEBOOK SNAPSHOT STRUCTURE
# ==============================

class ExtractedMetrics(TypedDict):
    accuracy: Optional[float]
    val_accuracy: Optional[float]
    loss: Optional[float]
    val_loss: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    epochs_trained: Optional[int]
    batch_size: Optional[int]
    learning_rate: Optional[float]


class ReproducibilitySnapshot(TypedDict):
    random_seeds: bool
    train_test_split: bool
    callbacks: bool
    logging: bool


class StructuralFeatures(TypedDict):
    has_preprocessing: bool
    has_augmentation: bool
    has_validation_split: bool
    architecture_keywords: List[str]
    optimizer: Optional[str]
    loss_function: Optional[str]
    has_early_stopping: bool
    has_model_checkpoint: bool


class NotebookSnapshot(TypedDict):
    filename: str
    focus: str
    char_count: int
    file_size_kb: float
    stats: dict
    reproducibility: ReproducibilitySnapshot
    extracted_metrics: ExtractedMetrics
    structural_features: StructuralFeatures
    # --- Week 2 additions ---
    loss_curve: Optional[LossCurveData]
    overfitting_score: Optional[OverfittingScore]
    training_risks: List[TrainingRiskFlag]
    training_summary: Optional[str]


# ==============================
# COMPARISON RESULT STRUCTURE
# ==============================

class MetricDeltas(TypedDict):
    accuracy_delta: Optional[float]   #notebook_b - notebook_a (positive = B is better)
    val_accuracy_delta: Optional[float]
    loss_delta: Optional[float]           # negative = B is better for loss
    val_loss_delta: Optional[float]
    f1_delta: Optional[float]


class StructuralDiff(TypedDict):
    same_focus: bool
    architecture_overlap: List[str]       # keywords present in both
    config_differences: List[str]         # fields where they differ
    reproducibility_gaps: List[str]       # things A has that B doesn't, or vice versa


class ComparisonResult(TypedDict):
    winner: Optional[str]                 # "notebook_a" | "notebook_b" | "inconclusive" | None
    confidence: Optional[str]            # "high" | "medium" | "low" | None
    metric_deltas: MetricDeltas
    structural_diff: StructuralDiff
    risk_flags: List[str]                 # e.g. ["suspicious_val_accuracy_jump", "no_seeds_in_b"]
    llm_review: Optional[str]           


# ==============================
# TOP LEVEL COMPARISON OBJECT
# ==============================

class ComparisonSchema(TypedDict):
    comparison_id: str
    comparison_type: str                  # "NotebookVsNotebook"|"run_vs_run"
    timestamp: str
    notebook_a: NotebookSnapshot
    notebook_b: NotebookSnapshot
    comparison_result: ComparisonResult


# ==============================
# FACTORY FUNCTIONS--to create fresh empty objects
# ==============================

def empty_metrics() -> ExtractedMetrics:
    return {
        "accuracy": None,
        "val_accuracy": None,
        "loss": None,
        "val_loss": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "epochs_trained": None,
        "batch_size": None,
        "learning_rate": None,
    }


def empty_reproducibility() -> ReproducibilitySnapshot:
    return {
        "random_seeds": False,
        "train_test_split": False,
        "callbacks": False,
        "logging": False,
    }


def empty_structural_features() -> StructuralFeatures:
    return {
        "has_preprocessing": False,
        "has_augmentation": False,
        "has_validation_split": False,
        "architecture_keywords": [],
        "optimizer": None,
        "loss_function": None,
        "has_early_stopping": False,
        "has_model_checkpoint": False,
    }


def empty_snapshot(filename: str = "") -> NotebookSnapshot:
    return {
        "filename": filename,
        "focus": "Unknown",
        "char_count": 0,
        "file_size_kb": 0.0,
        "stats": {"total_cells": 0, "code_cells": 0, "markdown_cells": 0},
        "reproducibility": empty_reproducibility(),
        "extracted_metrics": empty_metrics(),
        "structural_features": empty_structural_features(),
        #  W2 
        "loss_curve": None,
        "overfitting_score": None,
        "training_risks": [],
        "training_summary": None,
    }


def empty_comparison_result() -> ComparisonResult:
    return {
        "winner": None,
        "confidence": None,
        "metric_deltas": {
            "accuracy_delta": None,
            "val_accuracy_delta": None,
            "loss_delta": None,
            "val_loss_delta": None,
            "f1_delta": None,
        },
        "structural_diff": {
            "same_focus": False,
            "architecture_overlap": [],
            "config_differences": [],
            "reproducibility_gaps": [],
        },
        "risk_flags": [],
        "llm_review": None,
    }


def create_comparison(comparison_type: str = "notebook_vs_notebook") -> ComparisonSchema:
    """
    Entry point. Call this whenever a new comparison session starts.
    comparison_type: "notebook_vs_notebook" or "run_vs_run"
    """
    return {
        "comparison_id": str(uuid.uuid4()),
        "comparison_type": comparison_type,
        "timestamp": datetime.utcnow().isoformat(),
        "notebook_a": empty_snapshot(),
        "notebook_b": empty_snapshot(),
        "comparison_result": empty_comparison_result(),
    }