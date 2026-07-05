"""
training_summary.py
Auto-generates concise experiment summary using Gemini.

OUTPUT FORMAT:
- Experiment Overview  (what this notebook did)
- Strengths           (what's working)
- Weaknesses          (what needs attention)
"""

from comparison_schema import NotebookSnapshot


def build_summary_prompt(snapshot: NotebookSnapshot) -> str:
    """
    Assembles snapshot data into a structured Gemini prompt.
    Only includes sections where data actually exists —
    no "Loss: not found" noise cluttering the prompt.
    """
    m      = snapshot["extracted_metrics"]
    struct = snapshot["structural_features"]
    repro  = snapshot["reproducibility"]
    risks  = snapshot.get("training_risks", [])
    overfit = snapshot.get("overfitting_score")
    loss_curve = snapshot.get("loss_curve")

    # --- Metrics block ---
    metrics_lines = []
    if m["accuracy"]     is not None: metrics_lines.append(f"  Train Accuracy:    {m['accuracy']:.2%}")
    if m["val_accuracy"] is not None: metrics_lines.append(f"  Val Accuracy:      {m['val_accuracy']:.2%}")
    if m["loss"]         is not None: metrics_lines.append(f"  Train Loss:        {m['loss']:.4f}")
    if m["val_loss"]     is not None: metrics_lines.append(f"  Val Loss:          {m['val_loss']:.4f}")
    if m["f1"]           is not None: metrics_lines.append(f"  F1 Score:          {m['f1']:.4f}")
    if m["epochs_trained"] is not None: metrics_lines.append(f"  Epochs Trained:    {m['epochs_trained']}")
    if m["batch_size"]   is not None: metrics_lines.append(f"  Batch Size:        {m['batch_size']}")
    if m["learning_rate"] is not None: metrics_lines.append(f"  Learning Rate:     {m['learning_rate']}")

    metrics_block = "TRAINING METRICS:\n" + (
        "\n".join(metrics_lines) if metrics_lines else "  No metrics found in outputs."
    )

    # --- Config block ---
    config_lines = [
        f"  Focus:             {snapshot['focus']}",
        f"  Optimizer:         {struct['optimizer'] or 'not detected'}",
        f"  Loss Function:     {struct['loss_function'] or 'not detected'}",
        f"  Architecture:      {', '.join(struct['architecture_keywords']) or 'not detected'}",
        f"  Has Preprocessing: {'Yes' if struct['has_preprocessing'] else 'No'}",
        f"  Has Augmentation:  {'Yes' if struct['has_augmentation'] else 'No'}",
        f"  Validation Split:  {'Yes' if struct['has_validation_split'] else 'No'}",
        f"  Early Stopping:    {'Yes' if struct['has_early_stopping'] else 'No'}",
        f"  Model Checkpoint:  {'Yes' if struct['has_model_checkpoint'] else 'No'}",
        f"  Random Seeds:      {'Yes' if repro['random_seeds'] else 'No'}",
        f"  Experiment Logging:{'Yes' if repro['logging'] else 'No'}",
    ]
    config_block = "CONFIGURATION:\n" + "\n".join(config_lines)

    # --- Loss curve summary ---
    if loss_curve and loss_curve["total_epochs_found"] > 0:
        first = loss_curve["epochs"][0]
        last  = loss_curve["epochs"][-1]
        curve_block = (
            f"TRAINING CURVE SUMMARY:\n"
            f"  Epochs extracted:  {loss_curve['total_epochs_found']}\n"
            f"  Has val curves:    {'Yes' if loss_curve['has_validation_curves'] else 'No'}\n"
            f"  Epoch 1 loss:      {first['loss']}\n"
            f"  Final loss:        {last['loss']}\n"
            f"  Epoch 1 val_loss:  {first['val_loss']}\n"
            f"  Final val_loss:    {last['val_loss']}\n"
        )
    else:
        curve_block = "TRAINING CURVE SUMMARY:\n  No epoch-level data found."

    # --- Overfitting block ---
    if overfit and overfit["score"] is not None:
        overfit_block = (
            f"OVERFITTING ASSESSMENT:\n"
            f"  Score:             {overfit['score']}/100\n"
            f"  Risk Level:        {overfit['risk_level']}\n"
            f"  Widening Gap:      {'Yes' if overfit['widening_gap_detected'] else 'No'}\n"
            f"  Unstable Val:      {'Yes' if overfit['unstable_validation_detected'] else 'No'}\n"
            f"  Memorization:      {'Yes' if overfit['memorization_detected'] else 'No'}\n"
            f"  Evidence:\n" +
            "\n".join(f"    - {e}" for e in overfit["evidence"])
        )
    else:
        overfit_block = "OVERFITTING ASSESSMENT:\n  Not enough data to assess."

    # --- Risk flags block ---
    if risks:
        warnings = [r for r in risks if r["severity"] == "warning"]
        infos    = [r for r in risks if r["severity"] == "info"]
        risk_lines = []
        for r in warnings:
            risk_lines.append(f"  ⚠ [{r['category']}] {r['message']}")
        for r in infos:
            risk_lines.append(f"  i [{r['category']}] {r['message']}")
        risk_block = "TRAINING RISKS DETECTED:\n" + "\n".join(risk_lines)
    else:
        risk_block = "TRAINING RISKS DETECTED:\n  None."

    # --- Assemble full prompt ---
    prompt = f"""You are a senior ML engineer reviewing a single training experiment.
You have been given pre-computed structured analysis from a deterministic engine.
Your job is to write a concise, honest experiment summary — like a code review comment, not a sales pitch.

STRICT RULES:
1. Only reference data explicitly provided below. Never invent metrics or findings.
2. If a metric says "not found", do not mention it.
3. Be direct and specific. No filler phrases.
4. Maximum 250 words total.
5. Maintain an honest tone — don't oversell weak results,be friendly and corny in your response.

---
NOTEBOOK: {snapshot['filename']}
{metrics_block}

{config_block}

{curve_block}

{overfit_block}

{risk_block}
---

Write your summary using EXACTLY this structure:

## Experiment Overview
[2-3 sentences. What did this notebook train, what approach was used, 
what were the headline results. Be specific — reference actual values.]

## Strengths
[2-3 bullet points. What this notebook does well based on the evidence above.
Only mention things actually supported by the data.]

## Weaknesses & Recommendations
[2-3 bullet points. What needs attention, referencing specific risk flags
and overfitting signals. Phrase as actionable recommendations.]"""

    return prompt


def generate_training_summary(snapshot: NotebookSnapshot, gemini_call_fn) -> str:
    """
    Builds prompt from snapshot and calls Gemini.
    Returns summary string. Falls back gracefully if Gemini fails.
    gemini_call_fn is call_gemini from gemini_service.py — passed in, not imported.
    """
    try:
        prompt  = build_summary_prompt(snapshot)
        summary = gemini_call_fn(prompt)
        return summary
    except Exception as e:
        return (
            "Training summary could not be generated. "
            "Structured analysis above is still valid."
        )