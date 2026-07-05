"""
training_risk_analyzer.py

SEVERITY LEVELS:
- "warning" : something that commonly causes real problems
- "info"    : something worth knowing but not necessarily problematic

CATEGORIES:
- epochs        : training duration risks
- batch_size    : batch configuration risks  
- callbacks     : missing safety nets
- validation    : validation strategy risks
- precision     : mixed precision / optimization
- architecture  : model capacity risks
- reproducibility: experiment reliability risks
- learning_rate : learning rate configuration risks
"""

from typing import List
from comparison_schema import TrainingRiskFlag, NotebookSnapshot


def _flag(severity: str, category: str, message: str) -> TrainingRiskFlag:
    """
    Factory function for creating a TrainingRiskFlag.
    Using this instead of building dicts manually everywhere.
    """
    return {
        "severity": severity,
        "category": category,
        "message":  message,
    }


def analyze_training_risks(snapshot: NotebookSnapshot) -> List[TrainingRiskFlag]:
    """
    Runs all risk checks and returns a list of findings.
    Empty list = no risks detected.
    
    Pulls data from three places in the snapshot:
    - extracted_metrics:   epochs_trained, batch_size, learning_rate
    - structural_features: has_early_stopping, has_validation_split,
                           optimizer, loss_function, architecture_keywords
    - reproducibility:     random_seeds, logging
    """
    flags  = []
    m      = snapshot["extracted_metrics"]
    struct = snapshot["structural_features"]
    repro  = snapshot["reproducibility"]

    # Pull relevant values upfront for readability
    epochs        = m.get("epochs_trained")
    batch_size    = m.get("batch_size")
    learning_rate = m.get("learning_rate")
    optimizer     = struct.get("optimizer")
    arch_keywords = struct.get("architecture_keywords", [])

    # ==============================
    # EPOCH RISKS
    # ==============================

    if epochs is not None:
        if epochs > 200:
            flags.append(_flag(
                "warning", "epochs",
                f"Training uses {epochs} epochs without visible EarlyStopping — "
                f"high risk of overfitting past the optimal checkpoint."
            ))
        elif epochs > 100 and not struct["has_early_stopping"]:
            flags.append(_flag(
                "warning", "epochs",
                f"Training uses {epochs} epochs with no EarlyStopping detected — "
                f"consider adding it to prevent unnecessary overfitting."
            ))
        elif epochs > 50 and not struct["has_early_stopping"]:
            flags.append(_flag(
                "info", "epochs",
                f"Training uses {epochs} epochs without visible EarlyStopping. "
                f"May be intentional, but worth monitoring val_loss manually."
            ))

    # ==============================
    # BATCH SIZE RISKS
    # ==============================

    if batch_size is not None:
        if batch_size <= 4:
            flags.append(_flag(
                "warning", "batch_size",
                f"Batch size of {batch_size} is very small — "
                f"may result in noisy gradient updates and unstable training."
            ))
        elif batch_size <= 8:
            flags.append(_flag(
                "info", "batch_size",
                f"Batch size of {batch_size} is on the small side — "
                f"this may slow training depending on hardware constraints."
            ))
        elif batch_size > 512:
            flags.append(_flag(
                "info", "batch_size",
                f"Batch size of {batch_size} is very large — "
                f"large batches can hurt generalization if learning rate isn't scaled accordingly."
            ))

    # ==============================
    # CALLBACK RISKS
    # ==============================

    if not struct["has_early_stopping"]:
        flags.append(_flag(
            "warning", "callbacks",
            "No EarlyStopping callback detected — "
            "training will run for the full epoch count regardless of whether "
            "the model has stopped improving."
        ))

    if not struct["has_model_checkpoint"]:
        flags.append(_flag(
            "info", "callbacks",
            "No ModelCheckpoint detected — "
            "best model weights may not be saved if training degrades in later epochs."
        ))

    # ==============================
    # VALIDATION RISKS
    # ==============================

    if not struct["has_validation_split"]:
        flags.append(_flag(
            "warning", "validation",
            "No visible validation split detected — "
            "unable to assess generalization performance from notebook evidence."
        ))

    # ==============================
    # MIXED PRECISION RISKS
    # ==============================

    # Only flag absence of mixed precision for larger architectures
    # Flagging it on a logistic regression would be absurd
    large_arch_keywords = [
        "resnet", "efficientnet", "mobilenet", "vgg", "inception",
        "bert", "gpt", "t5", "llama", "mistral", "transformer",
        "unet", "u-net", "yolo"
    ]
    using_large_arch = any(kw in arch_keywords for kw in large_arch_keywords)

    notebook_text_lower = snapshot.get("filename", "").lower()
    has_mixed_precision = any(kw in str(snapshot).lower() for kw in [
        "mixed_precision", "float16", "fp16", "tf.keras.mixed_precision",
        "torch.cuda.amp", "autocast", "gradscaler"
    ])

    if using_large_arch and not has_mixed_precision:
        flags.append(_flag(
            "info", "precision",
            f"Large architecture detected ({', '.join(kw for kw in arch_keywords if kw in large_arch_keywords)}) "
            f"with no mixed precision usage found — "
            f"mixed precision (fp16) could reduce memory usage and speed up training."
        ))

    # ==============================
    # LEARNING RATE RISKS
    # ==============================

    if learning_rate is not None:
        if learning_rate > 0.1:
            flags.append(_flag(
                "warning", "learning_rate",
                f"Learning rate of {learning_rate} appears high for most optimizers — "
                f"may cause unstable training or divergence."
            ))
        elif learning_rate < 1e-6:
            flags.append(_flag(
                "info", "learning_rate",
                f"Learning rate of {learning_rate} is very small — "
                f"training may converge extremely slowly or stall entirely."
            ))

    # ==============================
    # REPRODUCIBILITY RISKS
    # ==============================

    if not repro["random_seeds"]:
        flags.append(_flag(
            "warning", "reproducibility",
            "No random seed detected — "
            "results may not be reproducible across runs."
        ))

    if not repro["logging"]:
        flags.append(_flag(
            "info", "reproducibility",
            "No experiment logging detected (WandB, MLflow, TensorBoard) — "
            "tracking experiment history manually is error prone."
        ))

    # ==============================
    # ARCHITECTURE SANITY CHECKS
    # ==============================

    # CNN keywords on what looks like tabular data
    has_cnn    = any(kw in arch_keywords for kw in ["conv2d", "conv1d", "conv3d"])
    has_tabular = any(kw in str(snapshot.get("structural_features", {})).lower() for kw in [
        "dataframe", "pd.read", "csv", "tabular", "feature"
    ])

    if has_cnn and has_tabular:
        flags.append(_flag(
            "info", "architecture",
            "CNN architecture detected alongside tabular data patterns — "
            "ensure Conv layers are appropriate for this data type."
        ))

    # Dense-only architecture with very deep stack
    has_only_dense = (
        "dense" in arch_keywords and
        not any(kw in arch_keywords for kw in [
            "conv2d", "conv1d", "lstm", "gru", "transformer",
            "attention", "resnet", "efficientnet"
        ])
    )
    if has_only_dense and len(arch_keywords) == 1:
        flags.append(_flag(
            "info", "architecture",
            "Only dense/linear layers detected — "
            "ensure this architecture is appropriate for your data modality."
        ))

    return flags