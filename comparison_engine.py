"""
comparison_engine.py
Orchestrates the full notebook comparison pipeline.

Day 2 fills: extract_metrics_from_notebook()
Day 3 fills: build_structural_diff()
Day 4 fills: score_comparison(), detect_risk_flags()
Day 5 fills: generate_llm_comparison_review()
"""

import logging
log = logging.getLogger("notebook_lens")

from prompts import build_comparison_prompt
from metric_extraction import extract_metrics_from_notebook

from comparison_schema import (
    ComparisonSchema,
    NotebookSnapshot,
    create_comparison,
    empty_snapshot,
    empty_structural_features,
)

from comparison_schema import (
    ComparisonSchema,
    NotebookSnapshot,
    empty_snapshot,
    empty_structural_features,
    create_comparison,
)

from notebook_utils import get_notebook_stats, get_file_size
from detection import detect_notebook_focus, detect_reproducibility_dict

# ==============================
#  STRUCTURAL DIFF
# ==============================

def build_notebook_snapshot(notebook, uploaded_file, notebook_text: str) -> NotebookSnapshot:
    """
    Single entry point that produces a complete NotebookSnapshot.
    Reuses all existing V1 functions.
    """
    snapshot = empty_snapshot(filename=uploaded_file.name)

    # Existing V1 functions  
    snapshot["stats"]   = get_notebook_stats(notebook)
    snapshot["focus"]   = detect_notebook_focus(notebook_text)
    snapshot["file_size_kb"] = get_file_size(uploaded_file)
    snapshot["char_count"]   = len(notebook_text)

    # Reproducibility as booleans
    raw_repro = detect_reproducibility_dict(notebook_text)
    snapshot["reproducibility"] = {
        "random_seeds":    raw_repro["random_seeds"],
        "train_test_split": raw_repro["train_test_split"],
        "callbacks":       raw_repro["callbacks"],
        "logging":         raw_repro["logging"],
    }

    # metric extractor
    snapshot["extracted_metrics"] = extract_metrics_from_notebook(notebook_text)
    #structural extractor
    snapshot["structural_features"] = extract_structural_features(notebook_text)

    return snapshot



def extract_structural_features(notebook_text: str) -> dict:
    """
    Scans CODE cells only — structural features live in code, not outputs.
    """
    features = empty_structural_features()

    # Pull code sections only — same helper logic as metric_extraction.py
    import re
    code_blocks = re.findall(
        r'\[CODE\](.*?)(?=\[OUTPUT\]|\[MARKDOWN\]|\Z)',
        notebook_text,
        re.DOTALL
    )
    code_text = "\n".join(code_blocks).lower()

    # --- Preprocessing ---
    features["has_preprocessing"] = any(kw in code_text for kw in [
        "standardscaler", "minmaxscaler", "robustscaler", "normalize",
        "preprocessing", "fillna", "dropna", "imputer", "pipeline"
    ])

    # --- Augmentation ---
    features["has_augmentation"] = any(kw in code_text for kw in [
        "imagedatagenerator", "albumentations", "augment", "randomflip",
        "randomrotation", "randomzoom", "randomcrop", "transforms.compose",
        "a.compose", "v2.compose"
    ])

    # --- Validation split ---
    features["has_validation_split"] = any(kw in code_text for kw in [
        "validation_split", "val_split", "train_test_split",
        "kfold", "stratifiedkfold", "validationdataset"
    ])

    # --- Early stopping + checkpointing ---
    features["has_early_stopping"] = "earlystopping" in code_text
    features["has_model_checkpoint"] = "modelcheckpoint" in code_text

    # --- Architecture keywords (list of found ones) ---
    architecture_candidates = [
        "conv2d", "conv1d", "conv3d",
        "lstm", "gru", "rnn", "bilstm",
        "transformer", "multiheadattention", "attention",
        "bert", "gpt", "t5", "llama", "mistral", "gemma",
        "resnet", "efficientnet", "mobilenet", "vgg", "inception",
        "dense", "linear",
        "unet", "u-net",
        "yolo",
        "autoencoder",
        "gan", "generator", "discriminator",
    ]
    features["architecture_keywords"] = [
        kw for kw in architecture_candidates if kw in code_text
    ]

    # --- Optimizer ---
    optimizer_candidates = [
        "adamw", "adam", "sgd", "rmsprop",
        "adagrad", "adadelta", "nadam", "lion"
    ]
    for opt in optimizer_candidates:
        if opt in code_text:
            features["optimizer"] = opt
            break  # take the first found (most prominent, usually the main one)

    # --- Loss function ---
    loss_candidates = [
        "sparse_categorical_crossentropy",
        "categorical_crossentropy",
        "binary_crossentropy",
        "mean_squared_error",
        "mean_absolute_error",
        "huber",
        "focal_loss",
        "ctc_loss",
        "kl_divergence",
        "cosine_similarity",
    ]
    for loss in loss_candidates:
        if loss in code_text:
            features["loss_function"] = loss
            break

    return features


def build_structural_diff(snapshot_a: NotebookSnapshot, snapshot_b: NotebookSnapshot) -> dict:
    """
    Compares two snapshots and returns a StructuralDiff dict.

    config_differences: only reports fields where BOTH snapshots have a value
    but those values differ. 

    reproducibility_gaps: reports which signals are present in one notebook
    but absent in the other. 
    """
    diff = {
        "same_focus": False,
        "architecture_overlap": [],
        "config_differences": [],
        "reproducibility_gaps": [],
    }

    # Focus match 
    diff["same_focus"] = snapshot_a["focus"] == snapshot_b["focus"]

    # Architecture overlap 
    keywords_a = set(snapshot_a["structural_features"]["architecture_keywords"])
    keywords_b = set(snapshot_b["structural_features"]["architecture_keywords"])
    diff["architecture_overlap"] = sorted(keywords_a & keywords_b)

    # Config differences.  Compare fields that are meaningful training configuration choices
    
    config_fields = {
        "optimizer":       ("structural_features", "optimizer"),
        "loss_function":   ("structural_features", "loss_function"),
        "batch_size":      ("extracted_metrics",   "batch_size"),
        "learning_rate":   ("extracted_metrics",   "learning_rate"),
        "epochs_trained":  ("extracted_metrics",   "epochs_trained"),
    }

    for field_label, (section, key) in config_fields.items():
        val_a = snapshot_a[section][key]
        val_b = snapshot_b[section][key]

        # Only flag as a difference if BOTH have values AND they differ
        if val_a is not None and val_b is not None and val_a != val_b:
            diff["config_differences"].append(
                f"{field_label}: A={val_a} vs B={val_b}"
            )

    #Reproducibility gaps
    repro_labels = {
        "random_seeds":     "Random seeds",
        "train_test_split": "Train/test split",
        "callbacks":        "Callbacks",
        "logging":          "Logging",
    }

    for key, label in repro_labels.items():
        has_a = snapshot_a["reproducibility"][key]
        has_b = snapshot_b["reproducibility"][key]

        if has_a and not has_b:
            diff["reproducibility_gaps"].append(f"{label}: A has it, B doesn't")
        elif has_b and not has_a:
            diff["reproducibility_gaps"].append(f"{label}: B has it, A doesn't")

    return diff


# ==============================
# SCORING + RISK FLAGS
# ==============================

def score_comparison(schema: ComparisonSchema) -> ComparisonSchema:
    """
    Compares two notebooks using metric deltas (B - A) and decides a winner.

    Rules:
    - For accuracy-like metrics: higher is better
    - For loss: lower is better
    - Small differences (< 0.005) are ignored

    Winner is decided by voting across metrics:
    each metric votes A, B, or skips if difference is too small.

    Confidence:
    - High: clear win across multiple metrics
    - Medium: slight or limited agreement
    - Low: very little signal
    - None: no usable metrics
    """
    result = schema["comparison_result"]
    metrics_a = schema["notebook_a"]["extracted_metrics"]
    metrics_b = schema["notebook_b"]["extracted_metrics"]

    NOISE_THRESHOLD = 0.005  # deltas smaller than this are noise, not signal

    # Compute raw deltas.None if either side is missing the metric 
   
    def delta(key):
        a = metrics_a.get(key)
        b = metrics_b.get(key)
        if a is not None and b is not None:
            return round(b - a, 4)
        return None

    result["metric_deltas"]["accuracy_delta"]     = delta("accuracy")
    result["metric_deltas"]["val_accuracy_delta"] = delta("val_accuracy")
    result["metric_deltas"]["loss_delta"]         = delta("loss")
    result["metric_deltas"]["val_loss_delta"]     = delta("val_loss")
    result["metric_deltas"]["f1_delta"]           = delta("f1")

    #Vote tally
    votes_a = 0
    votes_b = 0
    total_votes_cast = 0

    higher_is_better = ["accuracy_delta", "val_accuracy_delta", "f1_delta"]
    lower_is_better  = ["loss_delta", "val_loss_delta"]

    for key in higher_is_better:
        d = result["metric_deltas"][key]
        if d is None:
            continue
        
        # val metrics count double — generalization matters more than train performance
        weight = 2 if key in ("val_accuracy_delta", "f1_delta") else 1
        
        if d > NOISE_THRESHOLD:
            votes_b += weight
            total_votes_cast += weight
        elif d < -NOISE_THRESHOLD:
            votes_a += weight
            total_votes_cast += weight
        else:
            total_votes_cast += weight

    for key in lower_is_better:
        d = result["metric_deltas"][key]
        if d is None:
            continue

        weight = 2 if key == "val_loss_delta" else 1

        if d < -NOISE_THRESHOLD:
            votes_b += weight
            total_votes_cast += weight
        elif d > NOISE_THRESHOLD:
            votes_a += weight
            total_votes_cast += weight
        else:
            total_votes_cast += weight

    #Determine winner
    if total_votes_cast == 0:
        result["winner"] = None
        result["confidence"] = None
        return schema

    if votes_a == votes_b:
        result["winner"] = "inconclusive"
    elif votes_a > votes_b:
        result["winner"] = "notebook_a"
    else:
        result["winner"] = "notebook_b"

    #Determine confidence
    margin = abs(votes_a - votes_b)

    if total_votes_cast >= 3 and margin >= 2:
        result["confidence"] = "high"
    elif total_votes_cast == 1 or margin == 0:
        result["confidence"] = "low"
    else:
        result["confidence"] = "medium"

    schema["comparison_result"] = result
    schema["comparison_result"]["risk_flags"] = detect_risk_flags(schema)
    return schema


def detect_risk_flags(schema: ComparisonSchema) -> list:
    """
    Scans both snapshots for suspicious patterns.
    Returns a list of human-readable flag strings.
    These feed directly into the LLM prompt on Day 5.

    FLAGS:
    - regression_in_b:              B is meaningfully worse than A (>3% drop)
    - suspicious_accuracy_gain:     accuracy jumped >15% with no architecture change
    - overfitting_risk_a/b:         val_loss much higher than train_loss (gap > 0.15)
    - val_train_divergence_a/b:     val_accuracy trails train_accuracy by >15%
    - suspicious_gain_no_val_lift:  accuracy improved but val_accuracy didn't follow
    - seed_inconsistency:           one notebook has seeds, other doesn't
    - high_epochs_no_earlystopping: >50 epochs with no EarlyStopping callback
    - no_validation_strategy:       neither notebook has a validation split

    RULE: only flags when there's actual evidence. Never flag based on absence
    of something unless that absence itself IS the problem (e.g. no seeds).
    """
    flags = []

    metrics_a = schema["notebook_a"]["extracted_metrics"]
    metrics_b = schema["notebook_b"]["extracted_metrics"]
    struct_a  = schema["notebook_a"]["structural_features"]
    struct_b  = schema["notebook_b"]["structural_features"]
    repro_a   = schema["notebook_a"]["reproducibility"]
    repro_b   = schema["notebook_b"]["reproducibility"]
    deltas    = schema["comparison_result"]["metric_deltas"]

    # Regression in B 
    # B is clearly worse than A on val_accuracy or accuracy by more than 3%
    acc_delta = deltas.get("val_accuracy_delta") or deltas.get("accuracy_delta")
    if acc_delta is not None and acc_delta < -0.03:
        flags.append(
            f"regression_in_b: B performs worse than A "
            f"(accuracy delta: {acc_delta:+.2%})"
        )

    # Suspicious accuracy gain
    # Val accuracy jumped more than 15% with no meaningful architecture change.This pattern often means: different dataset split, data leakage,
 
    val_acc_delta = deltas.get("val_accuracy_delta")
    arch_overlap  = schema["comparison_result"]["structural_diff"]["architecture_overlap"]

    if val_acc_delta is not None and val_acc_delta > 0.15:
        # Only suspicious if architectures are similar (same overlap keywords)If architecture changed significantly, a big jump is explainable
        
        keywords_a = set(struct_a["architecture_keywords"])
        keywords_b = set(struct_b["architecture_keywords"])
        arch_changed = len(keywords_b - keywords_a) > 2  # B added 2+ new components

        if not arch_changed:
            flags.append(
                f"suspicious_accuracy_gain: val_accuracy improved by "
                f"{val_acc_delta:+.2%} with no significant architecture change"
            )

    # Overfitting risk
    # Val loss significantly higher than train loss within the same notebook.
    for label, metrics in [("a", metrics_a), ("b", metrics_b)]:
        loss     = metrics.get("loss")
        val_loss = metrics.get("val_loss")
        if loss is not None and val_loss is not None:
            gap = val_loss - loss
            if gap > 0.30:
                flags.append(
                    f"overfitting_risk_{label}: val_loss exceeds train_loss "
                    f"by {gap:.3f} (serious gap)"
                )
            elif gap > 0.15:
                flags.append(
                    f"overfitting_risk_{label}: val_loss exceeds train_loss "
                    f"by {gap:.3f} (moderate gap)"
                )

    # Val/train accuracy divergence 
    # Val accuracy trails train accuracy by more than 15% — overfitting signal
    for label, metrics in [("a", metrics_a), ("b", metrics_b)]:
        acc     = metrics.get("accuracy")
        val_acc = metrics.get("val_accuracy")
        if acc is not None and val_acc is not None:
            gap = acc - val_acc
            if gap > 0.15:
                flags.append(
                    f"val_train_divergence_{label}: train_accuracy exceeds "
                    f"val_accuracy by {gap:.2%} — likely overfitting"
                )

    # Suspicious gain with no val lift
    # Train accuracy improved significantly but val_accuracy didn't follow.
    acc_d     = deltas.get("accuracy_delta")
    val_acc_d = deltas.get("val_accuracy_delta")
    if acc_d is not None and val_acc_d is not None:
        if acc_d > 0.05 and val_acc_d < 0.01:
            flags.append(
                f"suspicious_gain_no_val_lift: train accuracy improved "
                f"{acc_d:+.2%} but val_accuracy only moved {val_acc_d:+.2%} — "
                f"possible overfitting in B"
            )

    # Seed inconsistency 
    # One notebook sets random seeds, the other doesn't.
    if repro_a["random_seeds"] != repro_b["random_seeds"]:
        has_seeds = "A" if repro_a["random_seeds"] else "B"
        flags.append(
            f"seed_inconsistency: only notebook {has_seeds} sets random seeds — "
            f"results may not be directly comparable"
        )

    # High epochs without EarlyStopping 
    # Training for 50+ epochs without EarlyStopping risks wasted compute    
    for label, metrics, struct in [("a", metrics_a, struct_a), ("b", metrics_b, struct_b)]:
        epochs = metrics.get("epochs_trained")
        if epochs is not None and epochs > 50 and not struct["has_early_stopping"]:
            flags.append(
                f"high_epochs_no_earlystopping_{label}: {epochs} epochs "
                f"with no EarlyStopping detected"
            )

    #No validation strategy in either notebook
    # If neither notebook has a validation split, the entire comparison is based on training metrics only 
    a_has_val_metrics = (
        schema["notebook_a"]["extracted_metrics"]["val_accuracy"] is not None or
        schema["notebook_a"]["extracted_metrics"]["val_loss"] is not None
    )
    b_has_val_metrics = (
        schema["notebook_b"]["extracted_metrics"]["val_accuracy"] is not None or
        schema["notebook_b"]["extracted_metrics"]["val_loss"] is not None
    )

    if (not struct_a["has_validation_split"] and
        not struct_b["has_validation_split"] and
        not a_has_val_metrics and
        not b_has_val_metrics):
        flags.append(
            "no_validation_strategy: neither notebook has a visible "
            "validation split — comparison reliability is low"
        )

    return flags

# ==============================
# LLM REVIEW
# ==============================

def generate_llm_comparison_review(schema: ComparisonSchema, gemini_call_fn) -> str:
    """
    Builds comparison prompt from schema and calls Gemini.
    
    Why gemini_call_fn is passed in rather than imported directly:
    Keeps comparison_engine.py decoupled from gemini_service.py.
    Makes testing easier — pass a mock function, get a mock response.
    On Day 6 when app.py calls this, it passes call_gemini directly.
    """
    prompt = build_comparison_prompt(schema)

    try:
        review = gemini_call_fn(prompt)
        schema["comparison_result"]["llm_review"] = review
        return review
    except Exception as e:
        log.error(f"LLM comparison review failed | {type(e).__name__}: {e}")  #TEMPP-----------------------
        fallback = (
            "Comparative review could not be generated. "
            "Structured analysis above is still valid."
        )
        schema["comparison_result"]["llm_review"] = fallback
        return fallback
    


# ==============================
# MAIN ORCHESTRATOR
# ==============================

def run_comparison(
    notebook_a,
    file_a,
    text_a: str,
    notebook_b,
    file_b,
    text_b: str,
    comparison_type: str = "notebook_vs_notebook",
    gemini_call_fn=None
) -> ComparisonSchema:
    """
    Full pipeline. This is the only function app.py calls on Day 6.
    
    Order matters:
    1. Build snapshots independently (no dependency between them)
    2. Build structural diff (needs both snapshots)
    3. Score comparison (needs structural diff for suspicious_gain flag)
    4. LLM review (needs everything above — always last)
    """
    schema = create_comparison(comparison_type)

    # Step 1 — Build snapshots
    schema["notebook_a"] = build_notebook_snapshot(notebook_a, file_a, text_a)
    schema["notebook_b"] = build_notebook_snapshot(notebook_b, file_b, text_b)

    # Step 2 — Structural diff
    schema["comparison_result"]["structural_diff"] = build_structural_diff(
        schema["notebook_a"],
        schema["notebook_b"]
    )

    # Step 3 — Score + risk flags (detect_risk_flags called inside score_comparison)
    schema = score_comparison(schema)

    # Step 4 — LLM review (optional — skip if no gemini function passed)
    if gemini_call_fn is not None:
        generate_llm_comparison_review(schema, gemini_call_fn)

    return schema