"""
loss_curve_extractor.py
W2D2 — Extracts per-epoch metrics from training output cells.

Unlike metric_extraction.py which grabs the LAST epoch only, this module grabs EVERY epoch to build the full training history.
That history powers loss curve charts and overfitting detection.
DESIGN NOTE:
reusing the same [OUTPUT] section splitting logic from metric_extraction.py but process EVERY epoch block instead of just the last one.
The result is a chronologically ordered list of EpochMetrics dicts.
"""

import re
from typing import List, Optional
from comparison_schema import LossCurveData, EpochMetrics


# ==============================
# OUTPUT SECTION EXTRACTOR. Same logic as metric_extraction.py — only looks at output cells
# ==============================

def _extract_output_text(notebook_text: str) -> str:
    """
    Pulls all [OUTPUT] sections and normalizes line endings.
    NO stripping of any kind — let the epoch parser handle what it finds.
    Stripping was causing more problems than it solved.
    """
    output_blocks = re.findall(
        r'\[OUTPUT\](.*?)(?=\[CODE\]|\[MARKDOWN\]|\Z)',
        notebook_text,
        re.DOTALL
    )
    combined = "\n".join(output_blocks).lower()

    # Normalize line endings only — nothing else
    combined = combined.replace('\r\n', '\n').replace('\r', '\n')

    # Strip ANSI escape codes only — these are invisible characters
    # that break regex matching, not actual content
    combined = re.sub(r'\x1b\[[0-9;]*m', '', combined)

    return combined


# ==============================
# EPOCH BLOCK PARSER. Core of Day 2 — extract metrics from EVERY epoch
# ==============================

def _parse_all_epoch_blocks(output_text: str) -> List[EpochMetrics]:
    results = []

    # --- Keras: epoch header + next line has metrics ---
    keras_blocks = re.findall(
        r'epoch\s+\[?(\d+)/\d+\]?\r?\n([^\r\n]*loss[^\r\n]*)',
        output_text,
        re.MULTILINE
    )

    for epoch_num_str, metric_line in keras_blocks:
        results.append(_parse_metric_line(metric_line, int(epoch_num_str)))

    # --- PyTorch: everything on one line, only if Keras found nothing ---
    if not results:
        pytorch_blocks = re.findall(
            r'epoch\s+\[(\d+)/\d+\]([^\r\n]*loss[^\r\n]*)',
            output_text,
            re.MULTILINE
        )
        for epoch_num_str, metric_line in pytorch_blocks:
            results.append(_parse_metric_line(metric_line, int(epoch_num_str)))

    # Detect training run resets — when epoch counter goes back to 1
    # Keep only the LAST complete training run
    last_reset_idx = 0
    for i in range(1, len(results)):
        if results[i]["epoch"] <= results[i-1]["epoch"]:
            # Counter reset or repeated — new training run started here
            last_reset_idx = i
    
    results = results[last_reset_idx:]



    return results


def _parse_metric_line(line: str, epoch_num: int) -> EpochMetrics:
    """
    Extracts loss, val_loss, accuracy, val_accuracy from a single metric line.
    Handles both : and = separators (Keras uses :, PyTorch f-strings use =).
    Normalizes percentage accuracy (91.44 → 0.9144).

    val_ variants are checked BEFORE plain variants — same reason as Week 1:
    searching for 'loss' would match inside 'val_loss' without this ordering.
    """
    metric = {
        "epoch":        epoch_num,
        "loss":         None,
        "val_loss":     None,
        "accuracy":     None,
        "val_accuracy": None,
    }

    patterns = {
    "val_loss":     r'\bval(?:idation)?[_\s]?loss\s*[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)',
    "val_accuracy": r'\bval(?:idation)?[_\s]?(?:accuracy|acc)\s*[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)',
    "loss":         r'(?<![a-z_])(?:train[_\s]?)?loss\s*[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)',
    "accuracy":     r'(?<![a-z_])(?:train[_\s]?)?(?:accuracy|acc)\s*[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)',
}

    for key, pattern in patterns.items():
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Normalize percentage to 0.0–1.0
            if key in ("accuracy", "val_accuracy") and value > 1.0:
                value = value / 100.0
            metric[key] = round(value, 4)

    return metric


# ==============================
# MAIN ENTRY POINT. Called by build_notebook_snapshot() in comparison_engine.py
# ==============================

def extract_loss_curves(notebook_text: str) -> LossCurveData:
    """
    Full loss curve extraction pipeline.
    Returns LossCurveData with all epochs found.
    Returns empty LossCurveData (not None) if nothing found —
    downstream code can check total_epochs_found == 0 instead of None checks.
    """
    output_text = _extract_output_text(notebook_text)

    epochs      = _parse_all_epoch_blocks(output_text)

    # Check if any validation curves exist
    has_val = any(
        e["val_loss"] is not None or e["val_accuracy"] is not None
        for e in epochs
    )

    return {
        "epochs":               epochs,
        "total_epochs_found":   len(epochs),
        "has_validation_curves": has_val,
    }