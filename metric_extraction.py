"""
metric_extraction.py
Week 1, Day 2 — Extracts training metrics from notebook output and code cells.

ARCHITECTURE DECISION:
notebook_text already has [CODE], [OUTPUT], [MARKDOWN] tags from load_notebook().
We split on those tags first — never run metric regex on code cells or vice versa.
This prevents false matches like batch_size appearing in a printed dict output,
or a loss: label inside a comment being mistaken for a real training metric.

EXTRACTION ORDER (each metric):
1. Keras/TF epoch logs     → most structured, take LAST epoch (final training state)
2. Pytorch
3. Labeled float fallback  → catches print("Accuracy:", acc) style outputs  
4. sklearn report parser   → targets "weighted avg" row specifically
5. Code cell scanner       → ONLY for batch_size and learning_rate (config, not results)

Why "last match" everywhere:
People print metrics multiple times. The last value is the final/best one.
First match gives you epoch 1 accuracy. Last match gives you epoch 50 accuracy.
"""

import re
from typing import Optional, List
from comparison_schema import ExtractedMetrics, empty_metrics


# ==============================
# SECTION SPLITTERS
# First thing we do — always separate output text from code text
# ==============================

def _extract_output_sections(notebook_text: str) -> str:
    """
    Pulls everything between [OUTPUT] tags.
    Result is lowercased — metric names in outputs are case-insensitive noise.
    'Accuracy: 0.87' and 'accuracy: 0.87' are the same thing.
    """
    output_blocks = re.findall(
        r'\[OUTPUT\](.*?)(?=\[CODE\]|\[MARKDOWN\]|\Z)',
        notebook_text,
        re.DOTALL
    )
    return "\n".join(output_blocks).lower()


def _extract_code_sections(notebook_text: str) -> str:
    """
    Pulls everything between [CODE] tags.
    NOT lowercased — we preserve case because batch_size, BATCH_SIZE,
    and learning_rate all need to match via re.IGNORECASE, not lowercasing.
    Subtle difference: lowercasing kills scientific notation like 1E-4.
    """
    code_blocks = re.findall(
        r'\[CODE\](.*?)(?=\[OUTPUT\]|\[MARKDOWN\]|\Z)',
        notebook_text,
        re.DOTALL
    )
    return "\n".join(code_blocks)


# ==============================
# PASS 1 — KERAS / TF EPOCH LOGS
# Most structured format. Always try this first.
# ==============================

def _extract_keras_epoch_metrics(output_text: str) -> dict:
    """
    Parses Keras/TF training logs and returns the LAST epoch's metric values.
    
    Handles both common TF formats:
    
    Format A (two lines — TF2 default):
        Epoch 10/10
        313/313 [====] - 2s 7ms/step - loss: 0.2044 - accuracy: 0.9144 - val_loss: 0.2273 - val_accuracy: 0.9089
    
    Format B (one line — older TF / custom):
        Epoch 10/10 - 45s - loss: 0.1234 - accuracy: 0.9234 - val_loss: 0.2345 - val_accuracy: 0.8934
    
    Both formats always have a line containing "loss:" — that's our anchor.
    We find ALL such lines, take the last one, then extract each metric from it.
    """
    result = {
        "accuracy": None,
        "val_accuracy": None,
        "loss": None,
        "val_loss": None,
        "epochs_trained": None,
    }

    # Extract how many epochs ran — "epoch 10/10" → epochs_trained = 10
    epoch_header_matches = re.findall(r'epoch\s+(\d+)/(\d+)', output_text)
    if epoch_header_matches:
        last_epoch_num, _ = epoch_header_matches[-1]
        result["epochs_trained"] = int(last_epoch_num)

    # Find all lines that contain metric key-value pairs
    # Anchor: any line with "loss:" in it (Keras always logs loss)
    # re.MULTILINE so ^ and $ match line boundaries
    epoch_metric_blocks = re.findall(
        r'epoch\s+\d+/\d+\r?\n[^\r\n]*loss\s*:\s*[\d.]+[^\r\n]*',
        output_text,
        re.MULTILINE
    )

    if epoch_metric_blocks:
        metric_lines = [block.split('\n')[-1].strip('\r') for block in epoch_metric_blocks]
    else:
        # no epoch headers found — fallback to old behavior
        metric_lines = re.findall(
            r'^[^\n]*loss\s*:\s*[\d.]+[^\n]*$',
            output_text,
            re.MULTILINE
        )

    if not metric_lines:
        return result

    last_line = metric_lines[-1]

    # Individual metric patterns — order matters for val_ vs non-val_ disambiguation
    # We check val_loss BEFORE loss, val_accuracy BEFORE accuracy
    # Otherwise "val_loss: 0.23" could partially match "loss: 0.23"
    metric_patterns = {
        "val_loss":     r'\bval_loss\s*:\s*([\d.]+(?:e[+-]?\d+)?)',
        "val_accuracy": r'\bval_(?:accuracy|acc)\s*:\s*([\d.]+(?:e[+-]?\d+)?)',
        "loss":         r'(?<![a-z_])loss\s*:\s*([\d.]+(?:e[+-]?\d+)?)',
        "accuracy":     r'(?<![a-z_])(?:accuracy|acc)\s*:\s*([\d.]+(?:e[+-]?\d+)?)',
    }

    for metric_name, pattern in metric_patterns.items():
        match = re.search(pattern, last_line, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Normalize percentage accuracy to 0.0–1.0 range
            # Some setups log accuracy as 91.44 instead of 0.9144
            if metric_name in ("accuracy", "val_accuracy") and value > 1.0:
                value = value / 100.0
            result[metric_name] = round(value, 4)

    return result

# ==============================
# PASS 1B — PYTORCH EPOCH LOGS
# Handles manual PyTorch training loop outputs
# ==============================

def _extract_pytorch_epoch_metrics(output_text: str) -> dict:
    """
    Handles the three most common PyTorch training output patterns:

    Pattern A — manual loop (most common):
        Epoch [10/10], Loss: 0.2044, Acc: 91.44%
        Epoch [10/10], Train Loss: 0.2044, Val Loss: 0.2273, Val Acc: 90.89%

    Pattern B — PyTorch Lightning:
        Epoch 9: 100%|██| loss=0.2044, val_loss=0.2273, val_acc=0.9089

    Pattern C — tqdm with inline metrics:
        100%|████| 313/313 loss=0.2044 acc=0.9144
    """
    result = {
        "accuracy": None,
        "val_accuracy": None,
        "loss": None,
        "val_loss": None,
        "epochs_trained": None,
    }

    # epochs_trained: handles both "Epoch [10/10]" and "Epoch 10/10"
    # The \[? means "optional opening bracket"
    epoch_matches = re.findall(r'epoch\s+\[?(\d+)/(\d+)\]?', output_text)
    if epoch_matches:
        last_epoch, _ = epoch_matches[-1]
        result["epochs_trained"] = int(last_epoch)

    # Find all lines that look like PyTorch epoch output
    candidate_lines = re.findall(
        r'^[^\n]*(?:epoch|loss\s*[=:]|acc\s*[=:])[^\n]*$',
        output_text,
        re.MULTILINE
    )

    if not candidate_lines:
        return result

    last_line = candidate_lines[-1]

    # Metric patterns — both = and : separators covered
    # val_ variants checked BEFORE plain variants (same reason as Keras parser)
    metric_patterns = {
        "val_loss":     r'\bval(?:idation)?[_\s]?loss\s*[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)',
        "val_accuracy": r'\bval(?:idation)?[_\s]?(?:accuracy|acc)\s*[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)',
        "loss":         r'(?<![a-z_])(?:train[_\s]?)?loss\s*[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)',
        "accuracy":     r'(?<![a-z_])(?:train[_\s]?)?(?:accuracy|acc)\s*[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)',
    }

    for metric_name, pattern in metric_patterns.items():
        match = re.search(pattern, last_line, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Normalize percentage to 0.0–1.0
            if metric_name in ("accuracy", "val_accuracy") and value > 1.0:
                value = value / 100.0
            result[metric_name] = round(value, 4)

    return result


# ==============================
# PASS 2 — LABELED FLOAT FALLBACK
# Catches print() statements in all their chaotic glory
# ==============================

def _extract_labeled_float(text: str, labels: List[str]) -> Optional[float]:
    """
    Generic pattern matcher for outputs like:
        "Test Accuracy: 0.8734"
        "val_accuracy = 0.91"  
        "Final Accuracy: 87.34%"
        "accuracy: 0.87"
    
    Takes the LAST match from the output (final reported value, not intermediate).
    Normalizes percentages to 0.0–1.0 range automatically.
    
    The negative lookbehind (?<![a-zA-Z_]) ensures:
    - Searching for "loss" won't match "val_loss" (preceded by _)
    - Searching for "accuracy" won't match "val_accuracy" (preceded by _)
    This is why val_ variants are queried separately with their full name.
    """
    for label in labels:
        escaped = re.escape(label)
        pattern = rf'(?<![a-zA-Z_]){escaped}\s*(?:[:=])\s*([\d.]+)\s*(%?)'
        matches = re.findall(pattern, text, re.IGNORECASE)

        if matches:
            value_str, pct_sign = matches[-1]  # last = final/best reported value
            value = float(value_str)
            if pct_sign == "%" or value > 1.0:
                value = value / 100.0
            return round(value, 4)

    return None


# ==============================
# PASS 3 — SKLEARN CLASSIFICATION REPORT
# Very specific structure — parse it specifically
# ==============================

def _extract_sklearn_report_metrics(output_text: str) -> dict:
    """
    Parses sklearn's classification_report output format:
    
                  precision    recall  f1-score   support
         class 0       0.88      0.92      0.90       100
         class 1       0.86      0.79      0.82        80
        accuracy                           0.87       180
       macro avg       0.87      0.86      0.86       180
    weighted avg       0.87      0.86      0.86       180
    
    We target "weighted avg" first — it accounts for class imbalance.
    Falls back to "macro avg" if weighted isn't present.
    
    Why not per-class rows? Because the number of classes is unknown
    and "weighted avg" is the single most representative summary row.
    """
    result = {"precision": None, "recall": None, "f1": None}

    for row_label in ["weighted avg", "macro avg"]:
        escaped = re.escape(row_label)
        # Matches: "weighted avg    0.87    0.86    0.86    180"
        pattern = rf'{escaped}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, output_text, re.IGNORECASE)

        if match:
            result["precision"] = round(float(match.group(1)), 4)
            result["recall"]    = round(float(match.group(2)), 4)
            result["f1"]        = round(float(match.group(3)), 4)
            return result  # stop at first successful row (weighted > macro)

    return result


# ==============================
# PASS 4 HELPERS — CODE CELL CONFIG
# batch_size and learning_rate live in CODE, not outputs
# ==============================

def _extract_batch_size(code_text: str) -> Optional[int]:
    """
    Scans code cells for batch_size assignment.
    Takes the LAST definition — most likely the one passed to model.fit().
    
    Matches: batch_size=32, batch_size = 64, BATCH_SIZE = 128
    Does NOT match: x.reshape(-1, batch_size) — that's usage, not assignment.
    The \s*=\s* with no leading operator ensures we catch assignments only.
    """
    pattern = r'\bbatch_size\s*=\s*(\d+)'
    matches = re.findall(pattern, code_text, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None


def _extract_learning_rate(code_text: str) -> Optional[float]:
    """
    Scans code cells for learning rate assignment.
    Scientific notation (1e-4, 1E-4) is very common in ML code — handle it.
    
    Matches: lr=0.001, learning_rate=1e-4, LR = 0.0001, LEARNING_RATE = 1E-3
    Priority: checks 'learning_rate' before 'lr' since 'lr' is a substring
    of 'learning_rate' and we want the more specific match first.
    """
    # patterns = [
    #     r'\blearning_rate\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)',
    #     r'(?<![a-zA-Z_])lr\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)',
    # ]

    patterns = [
        r'\blearning_rate\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)(?!\s*[*/%+])',
        r'(?<![a-zA-Z_])lr\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)(?!\s*[*/%+])',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, code_text, re.IGNORECASE)
        if matches:
            return float(matches[-1])

    return None


# ==============================
# MAIN ENTRY POINT
# This is the only function comparison_engine.py calls
# ==============================

def extract_metrics_from_notebook(notebook_text: str) -> ExtractedMetrics:
    """
    Full metric extraction pipeline. Called by build_notebook_snapshot() on Day 3.
    
    Returns ExtractedMetrics dict. Every field defaults to None — never crashes.
    None means "not found in this notebook", which is valid and useful information
    (e.g. no val_accuracy = this notebook has no validation split).
    """
    metrics = empty_metrics()

    output_text = _extract_output_sections(notebook_text)
    code_text   = _extract_code_sections(notebook_text)

    # === PASS 1: Keras epoch logs ===
    # Best case — structured, unambiguous, complete
    keras_metrics = _extract_keras_epoch_metrics(output_text)
    for key, value in keras_metrics.items():
        if value is not None:
            metrics[key] = value

    # === PASS 1B: PyTorch epoch logs ===
    # Only fills fields still None after Keras pass
    # No risk of overwriting a good Keras value with a worse PyTorch match
    pytorch_metrics = _extract_pytorch_epoch_metrics(output_text)
    for key, value in pytorch_metrics.items():
        if value is not None and metrics[key] is None:
            metrics[key] = value

    # === PASS 2: Labeled float fallback ===
    # Only runs for fields still None after Keras extraction
    if metrics["accuracy"] is None:
        metrics["accuracy"] = _extract_labeled_float(output_text, [
            "accuracy", "acc", "test accuracy", "test_accuracy",
            "train accuracy", "final accuracy"
        ])

    if metrics["val_accuracy"] is None:
        metrics["val_accuracy"] = _extract_labeled_float(output_text, [
            "val_accuracy", "val accuracy", "validation accuracy", "val_acc"
        ])

    if metrics["loss"] is None:
        metrics["loss"] = _extract_labeled_float(output_text, [
            "loss", "train loss", "training loss", "final loss"
        ])

    if metrics["val_loss"] is None:
        metrics["val_loss"] = _extract_labeled_float(output_text, [
            "val_loss", "val loss", "validation loss"
        ])

    # === PASS 3: sklearn classification report ===
    sklearn_metrics = _extract_sklearn_report_metrics(output_text)
    for key in ["precision", "recall", "f1"]:
        if metrics[key] is None:
            metrics[key] = sklearn_metrics.get(key)

    # Fallback print-based extraction for precision/recall/f1 if report wasn't found
    if metrics["precision"] is None:
        metrics["precision"] = _extract_labeled_float(output_text, ["precision"])
    if metrics["recall"] is None:
        metrics["recall"] = _extract_labeled_float(output_text, ["recall"])
    if metrics["f1"] is None:
        metrics["f1"] = _extract_labeled_float(output_text, [
            "f1", "f1-score", "f1 score", "f1_score"
        ])

    # === PASS 4: Code cell config ===
    # epochs_trained fallback if Keras logs weren't present
    if metrics["epochs_trained"] is None:
        metrics["epochs_trained"] = _extract_epochs_trained(output_text)

    if metrics["batch_size"] is None:
        metrics["batch_size"] = _extract_batch_size(code_text)

    if metrics["learning_rate"] is None:
        metrics["learning_rate"] = _extract_learning_rate(code_text)

    return metrics


def _extract_epochs_trained(output_text: str) -> Optional[int]:
    """
    Handles both:
    - Keras:   epoch 10/10
    - PyTorch: Epoch [10/10]
    """
    matches = re.findall(r'epoch\s+\[?(\d+)/\d+\]?', output_text)
    if matches:
        return int(matches[-1])
    return None