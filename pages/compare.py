import streamlit as st
import nbformat
from google.genai import errors
import logging
import plotly.graph_objects as go

from styles import inject_styles
from gemini_service import call_gemini
from comparison_engine import run_comparison
from notebook_utils import load_notebook, validate_notebook_content

log = logging.getLogger("notebook_lens")


def render_loss_curve(loss_curve: dict, title: str):
    """
    Renders an interactive Plotly loss curve chart.
    Dark themed to match the app. Shows train and val curves
    on the same chart if both exist.
    Returns early with a caption if no curve data exists.
    """
    if loss_curve is None or loss_curve["total_epochs_found"] == 0:
        st.caption("No epoch-level training data found for this notebook.")
        return

    epochs    = [e["epoch"] for e in loss_curve["epochs"]]
    train_loss = [e["loss"] for e in loss_curve["epochs"]]
    val_loss   = [e["val_loss"] for e in loss_curve["epochs"]]
    train_acc  = [e["accuracy"] for e in loss_curve["epochs"]]
    val_acc    = [e["val_accuracy"] for e in loss_curve["epochs"]]

    # Only plot series that actually have data
    has_train_loss = any(v is not None for v in train_loss)
    has_val_loss   = any(v is not None for v in val_loss)
    has_train_acc  = any(v is not None for v in train_acc)
    has_val_acc    = any(v is not None for v in val_acc)

    if not has_train_loss and not has_val_loss:
        st.caption("No loss values found in training history.")
        return

    fig = go.Figure()

    # --- Loss curves ---
    if has_train_loss:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_loss,
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#4facfe", width=2),
            marker=dict(size=4),
            connectgaps=True,
        ))

    if has_val_loss:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_loss,
            mode="lines+markers",
            name="Val Loss",
            line=dict(color="#e05c5c", width=2, dash="dash"),
            marker=dict(size=4),
            connectgaps=True,
        ))

    # --- Accuracy curves on secondary y-axis ---
    if has_train_acc:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_acc,
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(color="#76b900", width=2),
            marker=dict(size=4),
            yaxis="y2",
            connectgaps=True,
        ))

    if has_val_acc:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_acc,
            mode="lines+markers",
            name="Val Accuracy",
            line=dict(color="#f0a500", width=2, dash="dash"),
            marker=dict(size=4),
            yaxis="y2",
            connectgaps=True,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#f5f5f7", size=14)),
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
        font=dict(color="#a0a0b0", family="Inter"),
        xaxis=dict(
            title="Epoch",
            gridcolor="#2a2d3a",
            zerolinecolor="#2a2d3a",
        ),
        yaxis=dict(
            title="Loss",
            gridcolor="#2a2d3a",
            zerolinecolor="#2a2d3a",
            side="left",
        ),
        yaxis2=dict(
            title="Accuracy",
            overlaying="y",
            side="right",
            gridcolor="#2a2d3a",
            zerolinecolor="#2a2d3a",
            range=[0, 1],
        ),
        legend=dict(
            bgcolor="#111318",
            bordercolor="#2a2d3a",
            borderwidth=1,
            font=dict(color="#a0a0b0"),
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=320,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_overfitting_score(overfit: dict):
    """
    Renders overfitting score card with risk level and evidence.
    """
    if overfit is None or overfit["score"] is None:
        st.caption("Not enough epoch data to assess overfitting.")
        return

    score     = overfit["score"]
    risk      = overfit["risk_level"]
    evidence  = overfit["evidence"]

    risk_color = {
        "low":      "#76b900",
        "moderate": "#f0a500",
        "high":     "#e05c5c",
    }.get(risk, "#666")

    st.markdown(f"""
    <div style="
        background: #111318;
        border: 1px solid #2a2d3a;
        border-left: 4px solid {risk_color};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    ">
        <div style="font-size:13px; color:#a0a0b0; 
                    text-transform:uppercase; letter-spacing:0.06em;
                    font-weight:600; margin-bottom:6px;">
            Overfitting Risk
        </div>
        <div style="font-size:24px; font-weight:700; color:{risk_color};">
            {score}/100 — {risk.title()}
        </div>
        <div style="margin-top:10px;">
            {''.join(f'<div style="font-size:12px; color:#a0a0b0; margin-top:4px;">· {e}</div>' 
                     for e in evidence)}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_training_risks(risks: list):
    """
    Renders training risk flags with severity icons.
    Warnings first, then info.
    """
    if not risks:
        st.caption("No training configuration risks detected.")
        return

    warnings = [r for r in risks if r["severity"] == "warning"]
    infos    = [r for r in risks if r["severity"] == "info"]

    for r in warnings:
        st.markdown(f"""
        <div style="
            background: #1a1008;
            border-left: 4px solid #e05c5c;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            margin-bottom: 6px;
            font-size: 13px;
            color: #f5f5f7;
        ">⚠️ <b>[{r['category']}]</b> {r['message']}</div>
        """, unsafe_allow_html=True)

    for r in infos:
        st.markdown(f"""
        <div style="
            background: #0f1420;
            border-left: 4px solid #4facfe;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            margin-bottom: 6px;
            font-size: 13px;
            color: #f5f5f7;
        ">ℹ️ <b>[{r['category']}]</b> {r['message']}</div>
        """, unsafe_allow_html=True)

st.set_page_config(
    page_title="Compare Notebooks — Notebook Lens",
    page_icon="🛰️",
    layout="wide"
)

inject_styles()

# =========================
# HEADER
# =========================
st.markdown("""
<div class="brand-header">
    <div class="brand-logo">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="7" stroke="#4facfe" stroke-width="1.5" fill="none"/>
            <circle cx="16" cy="16" r="2.5" fill="#4facfe"/>
            <line x1="16" y1="2" x2="16" y2="8" stroke="#4facfe" stroke-width="1.5" stroke-linecap="round"/>
            <line x1="16" y1="24" x2="16" y2="30" stroke="#4facfe" stroke-width="1.5" stroke-linecap="round"/>
            <line x1="2" y1="16" x2="8" y2="16" stroke="#4facfe" stroke-width="1.5" stroke-linecap="round"/>
            <line x1="24" y1="16" x2="30" y2="16" stroke="#4facfe" stroke-width="1.5" stroke-linecap="round"/>
            <circle cx="7" cy="7" r="1.5" fill="#76b900"/>
            <circle cx="25" cy="7" r="1.5" fill="#76b900"/>
            <circle cx="7" cy="25" r="1.5" fill="#76b900"/>
            <circle cx="25" cy="25" r="1.5" fill="#76b900"/>
        </svg>
    </div>
    <div>
        <div class="app-title">Notebook Lens</div>
        <div class="app-subtitle">Compare Notebooks · Find What Actually Improved</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin: 0 0 2rem 0;">
    <p style="color: #a0a0b0; font-size: 15px; line-height: 1.7; max-width: 640px;">
        Upload two notebooks to compare approaches, training runs, or experiments.
        The engine extracts metrics, diffs structure, detects risk patterns, 
        and generates an ML engineering verdict.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# COMPARISON TYPE SELECTOR
# =========================
st.markdown("#### What are you comparing?")
comparison_type = st.radio(
    label="comparison_type",
    options=["notebook_vs_notebook", "run_vs_run"],
    format_func=lambda x: (
        "Two different notebooks — different approaches, architectures, or authors"
        if x == "notebook_vs_notebook"
        else "Two training runs — same notebook, different hyperparameters or configs"
    ),
    label_visibility="collapsed"
)

st.markdown("---")

# =========================
# FILE UPLOADERS — SIDE BY SIDE
# =========================
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Notebook A")
    file_a = st.file_uploader(
        "Upload Notebook A",
        type=["ipynb"],
        key="file_a",
        label_visibility="collapsed"
    )

with col_b:
    st.markdown("#### Notebook B")
    file_b = st.file_uploader(
        "Upload Notebook B",
        type=["ipynb"],
        key="file_b",
        label_visibility="collapsed"
    )

# =========================
# RESET STATE ON NEW UPLOAD
# =========================
current_files = (
    file_a.name if file_a else None,
    file_b.name if file_b else None
)

if "last_compared_files" not in st.session_state:
    st.session_state["last_compared_files"] = (None, None)

if current_files != st.session_state["last_compared_files"]:
    st.session_state["comparison_schema"] = None
    st.session_state["comparison_ready"]  = False
    st.session_state["last_compared_files"] = current_files

# =========================
# PARSE + VALIDATE BOTH
# Only runs when both files are uploaded
# =========================
if file_a and file_b:

    # --- Parse ---
    try:
        notebook_a = nbformat.read(file_a, as_version=4)
    except Exception:
        st.error("Notebook A could not be read. File may be corrupted.")
        st.stop()

    try:
        notebook_b = nbformat.read(file_b, as_version=4)
    except Exception:
        st.error("Notebook B could not be read. File may be corrupted.")
        st.stop()

    # --- Validate ---
    valid_a, reason_a = validate_notebook_content(notebook_a)
    if not valid_a:
        st.error(f"Notebook A: {reason_a}")
        st.stop()

    valid_b, reason_b = validate_notebook_content(notebook_b)
    if not valid_b:
        st.error(f"Notebook B: {reason_b}")
        st.stop()

    # --- Load text ---
    text_a = load_notebook(notebook_a)
    text_b = load_notebook(notebook_b)

    # --- Show upload confirmation ---
    col_a2, col_b2 = st.columns(2)
    with col_a2:
        st.success(f"A: {file_a.name}")
    with col_b2:
        st.success(f"B: {file_b.name}")

    # =========================
    # COMPARE BUTTON
    # =========================
    if st.button("Compare Notebooks", type="primary"):
        progress_bar = st.progress(0)
        status       = st.empty()

        try:
            status.text("Building notebook snapshots...")
            progress_bar.progress(20)

            status.text("Extracting metrics and structural features...")
            progress_bar.progress(45)

            status.text("Running comparison scoring engine...")
            progress_bar.progress(65)

            status.text("Generating ML engineering review...")
            progress_bar.progress(80)

            schema = run_comparison(
                notebook_a, file_a, text_a,
                notebook_b, file_b, text_b,
                comparison_type=comparison_type,
                gemini_call_fn= call_gemini  #### for no gemini call
            )

            progress_bar.progress(100)
            status.text("Comparison complete.")

            #st.write("DEBUG — Notebook B metrics:", schema["notebook_b"]["extracted_metrics"])#TODO TEMPORARY to see the extracted values 

            st.session_state["comparison_schema"] = schema
            st.session_state["comparison_ready"]  = True

        except errors.ClientError as e:
            error_text = str(e)
            progress_bar.empty()
            status.empty()
            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                st.error("Review limit reached. Please wait and try again.")
            else:
                st.error("The AI review service could not complete the request.")

        except errors.ServerError:
            progress_bar.empty()
            status.empty()
            st.error("The AI review service is temporarily busy. Please try again.")

        except Exception as e:
            log.error(f"Comparison error | {type(e).__name__}: {e}")
            progress_bar.empty()
            status.empty()
            st.error("Something went wrong during comparison. Please try again.")

        finally:
            progress_bar.empty()
            status.empty()

# =========================
# RENDER RESULTS
# =========================
if st.session_state.get("comparison_ready") and st.session_state.get("comparison_schema"):
    schema  = st.session_state["comparison_schema"]
    result  = schema["comparison_result"]
    nb_a    = schema["notebook_a"]
    nb_b    = schema["notebook_b"]
    deltas  = result["metric_deltas"]
    flags   = result["risk_flags"]
    diff    = result["structural_diff"]

    st.markdown("---")

    # =========================
    # VERDICT BANNER
    # =========================
    winner     = result["winner"]
    confidence = result["confidence"]

    winner_label = {
        "notebook_a":   f"Notebook A wins  ({file_a.name})",
        "notebook_b":   f"Notebook B wins  ({file_b.name})",
        "inconclusive": "Inconclusive — too close to call",
        None:           "No metrics found — structural comparison only"
    }.get(winner, "Unknown")

    confidence_color = {
        "high":   "#76b900",
        "medium": "#f0a500",
        "low":    "#e05c5c",
        None:     "#666"
    }.get(confidence, "#666")

    st.markdown(f"""
    <div style="
        background: #111318;
        border: 1px solid #2a2d3a;
        border-left: 4px solid {confidence_color};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <div style="font-size: 18px; font-weight: 600; color: #f5f5f7;">
            {winner_label}
        </div>
        <div style="font-size: 13px; color: #a0a0b0; margin-top: 4px;">
            Confidence: {confidence or 'undetermined'} 
            &nbsp;·&nbsp; 
            Type: {schema['comparison_type'].replace('_', ' ').title()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # METRIC COMPARISON PANELS
    # =========================
    st.markdown("#### Metrics")

    def metric_row(label, val_a, val_b, delta, higher_is_better=True):
        """Renders one metric row with color coded delta."""
        if val_a is None and val_b is None:
            return

        def fmt(v):
            return f"{v:.4f}" if v is not None else "—"

        def fmt_delta(d):
            if d is None:
                return "—", "#666"
            if abs(d) <= 0.005:
                return f"{d:+.4f}", "#a0a0b0"
            better = (d > 0) == higher_is_better
            color  = "#76b900" if better else "#e05c5c"
            return f"{d:+.4f}", color

        delta_str, delta_color = fmt_delta(delta)

        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 1rem;
            background: #111318;
            border-radius: 8px;
            margin-bottom: 6px;
            font-size: 14px;
        ">
            <span style="color:#a0a0b0; width: 160px;">{label}</span>
            <span style="color:#f5f5f7; width: 100px; text-align:center;">{fmt(val_a)}</span>
            <span style="color:#f5f5f7; width: 100px; text-align:center;">{fmt(val_b)}</span>
            <span style="color:{delta_color}; width: 100px; text-align:right;">{delta_str}</span>
        </div>
        """, unsafe_allow_html=True)

    # Header row
    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 1rem;
        font-size: 12px;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">
        <span style="width:160px;">Metric</span>
        <span style="width:100px; text-align:center;">A</span>
        <span style="width:100px; text-align:center;">B</span>
        <span style="width:100px; text-align:right;">Delta (B-A)</span>
    </div>
    """, unsafe_allow_html=True)

    ma = nb_a["extracted_metrics"]
    mb = nb_b["extracted_metrics"]

    metric_row("Accuracy",     ma["accuracy"],     mb["accuracy"],     deltas["accuracy_delta"])
    metric_row("Val Accuracy", ma["val_accuracy"], mb["val_accuracy"], deltas["val_accuracy_delta"])
    metric_row("Loss",         ma["loss"],         mb["loss"],         deltas["loss_delta"],     higher_is_better=False)
    metric_row("Val Loss",     ma["val_loss"],     mb["val_loss"],     deltas["val_loss_delta"], higher_is_better=False)
    metric_row("F1 Score",     ma["f1"],           mb["f1"],           deltas["f1_delta"])

    # =========================
    # STRUCTURAL DIFF + CONFIG
    # =========================
    st.markdown("#### Structural Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Config Differences**")
        if diff["config_differences"]:
            for item in diff["config_differences"]:
                st.markdown(f"""
                <div style="
                    background:#111318; border-radius:6px;
                    padding: 0.5rem 0.8rem; margin-bottom:5px;
                    font-size:13px; color:#f5f5f7;
                    border-left: 3px solid #4facfe;
                ">
                    {item}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No config differences detected.")

    with col2:
        st.markdown("**Reproducibility Gaps**")
        if diff["reproducibility_gaps"]:
            for item in diff["reproducibility_gaps"]:
                st.markdown(f"""
                <div style="
                    background:#111318; border-radius:6px;
                    padding: 0.5rem 0.8rem; margin-bottom:5px;
                    font-size:13px; color:#f5f5f7;
                    border-left: 3px solid #f0a500;
                ">
                    {item}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No reproducibility gaps detected.")

    st.markdown(f"""
    <div style="font-size:13px; color:#a0a0b0; margin-top:0.5rem;">
        Same Focus: {'Yes' if diff['same_focus'] else 'No'} &nbsp;·&nbsp;
        Architecture Overlap: {', '.join(diff['architecture_overlap']) or 'none'}
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # RISK FLAGS
    # =========================
    if flags:
        st.markdown("#### Risk Flags")
        for flag in flags:
            st.markdown(f"""
            <div style="
                background: #1a1008;
                border-left: 4px solid #e05c5c;
                border-radius: 8px;
                padding: 0.7rem 1rem;
                margin-bottom: 8px;
                font-size: 13px;
                color: #f5f5f7;
            ">
                ⚠️ {flag}
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # LLM REVIEW
    # =========================
    st.markdown("#### ML Engineering Review")
    llm_review = result.get("llm_review")

    if llm_review:
        with st.expander("View full comparative review", expanded=True):
            st.markdown(llm_review)
    else:
        st.caption("LLM review not available.")

    # =========================
    # PER NOTEBOOK ANALYSIS. Loss curves, overfitting, risks, summaries
    # =========================
    st.markdown("---")
    st.markdown("#### Per-Notebook Analysis")
    st.markdown(
        '<div class="dashboard-caption">Loss curves, overfitting assessment, '
        'and training configuration risks for each notebook.</div>',
        unsafe_allow_html=True
    )

    tab_a, tab_b = st.tabs([
        f"Notebook A — {nb_a['filename']}",
        f"Notebook B — {nb_b['filename']}",
    ])

    with tab_a:
        st.markdown("**Loss Curves**")
        render_loss_curve(nb_a.get("loss_curve"), "Notebook A — Training History")

        st.markdown("**Overfitting Assessment**")
        render_overfitting_score(nb_a.get("overfitting_score"))

        st.markdown("**Training Configuration Risks**")
        render_training_risks(nb_a.get("training_risks", []))

        st.markdown("**Training Summary**")
        summary_a = nb_a.get("training_summary")
        if summary_a:
            with st.expander("View training summary", expanded=False):
                st.markdown(summary_a)
        else:
            st.caption("Training summary not available.")

    with tab_b:
        st.markdown("**Loss Curves**")
        render_loss_curve(nb_b.get("loss_curve"), "Notebook B — Training History")

        st.markdown("**Overfitting Assessment**")
        render_overfitting_score(nb_b.get("overfitting_score"))

        st.markdown("**Training Configuration Risks**")
        render_training_risks(nb_b.get("training_risks", []))

        st.markdown("**Training Summary**")
        summary_b = nb_b.get("training_summary")
        if summary_b:
            with st.expander("View training summary", expanded=False):
                st.markdown(summary_b)
        else:
            st.caption("Training summary not available.")


    # =========================
    # FOOTER
    # =========================
    st.markdown("""
    <div class="footer">
        Notebook Lens · Review. Improve. Iterate faster
        <a href="https://github.com/kashan37/notebook-lens" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)