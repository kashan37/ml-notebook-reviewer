"""
overfitting_detector.py
W2,D3 — Heuristic overfitting detection from loss curve data.
SCORING SYSTEM:
Each heuristic contributes points to a 0-100 score.
- Widening gap:          up to 40 points
- Unstable validation:   up to 30 points  
- Memorization:          up to 30 points
Risk levels:
- 0-29:  low
- 30-59: moderate
- 60+:   high
"""

import re
from typing import List, Tuple, Optional
from comparison_schema import LossCurveData, OverfittingScore


# ==============================
# MAIN ENTRY POINT
# ==============================

def detect_overfitting(loss_curve: LossCurveData) -> OverfittingScore:
    """
    Runs all three overfitting heuristics and returns a scored assessment.
    Returns None-filled OverfittingScore if not enough data to assess.
    
    Minimum requirement: at least 3 epochs with loss data.
    Below that we simply don't have enough signal to say anything meaningful.
    """
    empty_score = {
        "score": None,
        "risk_level": None,
        "widening_gap_detected": False,
        "unstable_validation_detected": False,
        "memorization_detected": False,
        "evidence": [],
    }

    epochs = loss_curve.get("epochs", [])

    # Need at least 3 epochs to detect any meaningful pattern
    if len(epochs) < 3:
        empty_score["evidence"].append(
            "Not enough epochs to assess overfitting (minimum 3 required)."
        )
        return empty_score

    # Need loss data at minimum
    has_loss = any(e["loss"] is not None for e in epochs)
    if not has_loss:
        empty_score["evidence"].append(
            "No loss values found in training history."
        )
        return empty_score

    total_score = 0
    evidence    = []

    #  Heuristic 1: Widening Gap 
    gap_detected, gap_points, gap_evidence = _detect_widening_gap(epochs)
    total_score += gap_points
    evidence.extend(gap_evidence)

    #Heuristic 2: Unstable Validation 
    unstable_detected, unstable_points, unstable_evidence = _detect_unstable_validation(epochs)
    total_score += unstable_points
    evidence.extend(unstable_evidence)

    #   Heuristic 3: Memorization
    memo_detected, memo_points, memo_evidence = _detect_memorization(epochs)
    total_score += memo_points
    evidence.extend(memo_evidence)

    # Cap at 100
    total_score = min(total_score, 100)

    # Determine risk level
    if total_score >= 60:
        risk_level = "high"
    elif total_score >= 30:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "score":                      total_score,
        "risk_level":                 risk_level,
        "widening_gap_detected":      gap_detected,
        "unstable_validation_detected": unstable_detected,
        "memorization_detected":      memo_detected,
        "evidence":                   evidence,
    }


# ==============================
# HEURISTIC 1 — WIDENING GAP
# val_loss diverging from train_loss over time
# ==============================

def _detect_widening_gap(epochs: list) -> Tuple[bool, int, List[str]]:
    """
    Checks if the gap between val_loss and train_loss is consistently
    growing over the second half of training.
    
    Why second half only?
    Early training almost always shows val_loss > train_loss — that's normal.
    The DANGEROUS pattern is when this gap keeps GROWING in later epochs
    after the model should have started converging.
    
    Returns: (detected, score_contribution, evidence_strings)
    """
    evidence = []

    # Need both loss and val_loss
    valid_epochs = [
        e for e in epochs
        if e["loss"] is not None and e["val_loss"] is not None
    ]

    if len(valid_epochs) < 3:
        return False, 0, []

    # Compute gap at each epoch
    gaps = [e["val_loss"] - e["loss"] for e in valid_epochs]

    # Look at second half only
    midpoint   = len(gaps) // 2
    first_half = gaps[:midpoint]
    second_half = gaps[midpoint:]

    avg_gap_first  = sum(first_half) / len(first_half)
    avg_gap_second = sum(second_half) / len(second_half)
    gap_growth     = avg_gap_second - avg_gap_first

    # Count how many consecutive epochs in second half show increasing gap
    consecutive_increases = 0
    max_consecutive = 0
    for i in range(1, len(second_half)):
        if second_half[i] > second_half[i-1]:
            consecutive_increases += 1
            max_consecutive = max(max_consecutive, consecutive_increases)
        else:
            consecutive_increases = 0

    # Scoring:
    # Gap grew by more than 0.05 between halves → 20 points
    # 3+ consecutive increases in second half → additional 20 points
    score = 0
    detected = False

    if gap_growth > 0.05:
        score += 20
        detected = True
        evidence.append(
            f"Val loss gap widened by {gap_growth:.3f} between first and second half of training."
        )

    if max_consecutive >= 3:
        score += 20
        detected = True
        evidence.append(
            f"Val loss increased for {max_consecutive} consecutive epochs in second half of training."
        )

    if not detected:
        evidence.append(
            f"No significant val/train loss divergence detected "
            f"(gap change: {gap_growth:+.3f})."
        )

    return detected, score, evidence


# ==============================
# HEURISTIC 2 — UNSTABLE VALIDATION
# val_loss oscillating instead of converging
# ==============================

def _detect_unstable_validation(epochs: list) -> Tuple[bool, int, List[str]]:
    """
    Checks if val_loss is oscillating instead of converging smoothly.
    
    Method: count direction reversals in val_loss sequence.
    A reversal is when val_loss goes down then up (or up then down).
    Stable training: 0-1 reversals (monotonically decreasing mostly).
    Unstable training: 3+ reversals (bouncing around like a rubber ball).
    
    We also check the coefficient of variation in the second half —
    high variance relative to mean = unstable.
    """
    evidence = []

    valid_epochs = [e for e in epochs if e["val_loss"] is not None]

    if len(valid_epochs) < 4:
        return False, 0, []

    val_losses = [e["val_loss"] for e in valid_epochs]

    # Count direction reversals
    reversals = 0
    for i in range(1, len(val_losses) - 1):
        prev_dir = val_losses[i] - val_losses[i-1]
        next_dir = val_losses[i+1] - val_losses[i]
        if (prev_dir > 0 and next_dir < 0) or (prev_dir < 0 and next_dir > 0):
            reversals += 1

    # Coefficient of variation in second half
    second_half = val_losses[len(val_losses)//2:]
    mean_val    = sum(second_half) / len(second_half)
    variance    = sum((x - mean_val)**2 for x in second_half) / len(second_half)
    std_dev     = variance ** 0.5
    cv          = (std_dev / mean_val) if mean_val > 0 else 0

    score    = 0
    detected = False

    if reversals >= 3:
        points = min(20, reversals * 5)
        score += points
        detected = True
        evidence.append(
            f"Val loss changed direction {reversals} times — "
            f"suggests unstable training or learning rate too high."
        )

    if cv > 0.10:
        score += 10
        detected = True
        evidence.append(
            f"Val loss shows high variance in second half of training "
            f"(CV: {cv:.3f}) — training may not have converged."
        )

    if not detected:
        evidence.append(
            f"Validation loss appears stable ({reversals} direction changes)."
        )

    return detected, score, evidence


# ==============================
# HEURISTIC 3 — MEMORIZATION
# train_accuracy consistently >> val_accuracy
# ==============================

def _detect_memorization(epochs: list) -> Tuple[bool, int, List[str]]:
    """
    Checks if train_accuracy is consistently and significantly higher
    than val_accuracy across the second half of training.
    
    Why second half?
    Early epochs always show some gap — model is still learning.
    Persistent gap in LATER epochs means it memorized training data
    instead of learning generalizable features.
    
    Threshold: average gap > 5% in second half = memorization signal.
    Gap > 10% = strong memorization signal.
    """
    evidence = []

    valid_epochs = [
        e for e in epochs
        if e["accuracy"] is not None and e["val_accuracy"] is not None
    ]

    if len(valid_epochs) < 3:
        return False, 0, []

    # Second half only
    second_half = valid_epochs[len(valid_epochs)//2:]
    gaps = [e["accuracy"] - e["val_accuracy"] for e in second_half]
    avg_gap = sum(gaps) / len(gaps)

    score    = 0
    detected = False

    if avg_gap > 0.10:
        score += 30
        detected = True
        evidence.append(
            f"Train accuracy exceeds val accuracy by {avg_gap:.2%} on average "
            f"in second half of training — strong memorization signal."
        )
    elif avg_gap > 0.05:
        score += 15
        detected = True
        evidence.append(
            f"Train accuracy exceeds val accuracy by {avg_gap:.2%} on average "
            f"in second half of training — moderate memorization signal."
        )
    else:
        evidence.append(
            f"Train/val accuracy gap is acceptable ({avg_gap:.2%} average)."
        )

    return detected, score, evidence