import streamlit as st
import nbformat
from google.genai import errors
import html
import logging
from styles import inject_styles
from gemini_service import call_gemini
from sample_outputs import TEST_REVIEW_OUTPUT

from config import (
    DEBUG_MODE,
    TEST_MODE,
    CHAT_TEST_MODE,
    MAX_CHARS,
)
from notebook_utils import (
    get_notebook_stats,
    validate_notebook_content,
    get_file_size,
    load_notebook,
)
from detection import (
    detect_notebook_focus,
    detect_reproducibility_signals,
    get_chat_suggestions,
)
from review_parsing import (
    extract_section,
    extract_section_body,
    extract_top_priorities,
    clean_preview_text,
)
from exports import (
    build_markdown_export,
    build_pdf_export,
)
from prompts import (
    build_chat_prompt,
    build_prompt,
)
from ui_components import (
    update_loading_state,
    render_sections_as_expanders,
    render_top_priorities,
    render_review_dashboard,
    render_header,
    render_landing_intro,
    render_file_information,
    render_detected_focus,
    render_notebook_preview,
    render_notebook_statistics,
    render_download_buttons,
)
from focus_instructions import (
    focus_instructions,
    DEFAULT_FOCUS_INSTRUCTION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("notebook_lens")

st.set_page_config(
    page_title="Notebook Lens",
    page_icon="🛰️",
    layout="wide"
)

# =========================
# UI STYLES
# =========================
inject_styles()


render_header()
render_landing_intro()


uploaded_file = st.file_uploader(
    "",
    type=["ipynb"],
    label_visibility="collapsed"
)

# =========================
# SESSION STATE INIT
# =========================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

# ============================
# STREAMLIT UI + APP LOGIC
# ============================
if uploaded_file is not None:

    try:
        notebook = nbformat.read(uploaded_file, as_version=4)
    except Exception as e:
        st.error("Could not read this notebook. The file may be corrupted or not a valid .ipynb.")
        st.caption("Try re-exporting it fresh from Jupyter or Colab.")
        if DEBUG_MODE:
            st.write("Parse error:", type(e).__name__)
            st.exception(e)
        st.stop()
    stats = get_notebook_stats(notebook)
    size = get_file_size(uploaded_file)
    notebook_text = load_notebook(notebook)
    notebook_focus = detect_notebook_focus(notebook_text)
    reproducibility_context = detect_reproducibility_signals(notebook_text)
    
    is_valid, reason = validate_notebook_content(notebook)
    if not is_valid:
        st.error(f"No usable code cells found in notebook. {reason}")
        st.caption("Please upload a notebook that contains actual code cells.")
        st.stop()

    log.info(f"Parse successful | Cells: {stats['total_cells']} | Code: {stats['code_cells']} | Markdown: {stats['markdown_cells']} | Focus: {notebook_focus}")
    log.info(f"Upload received | File: {uploaded_file.name} | Size: {size} KB")
    st.success("Upload successful. Ready for analysis.")

    if len(notebook_text) > MAX_CHARS:
            st.warning(
                f"This notebook is large ({len(notebook_text):,} characters). "
                f"Only the first {MAX_CHARS:,} characters were sent for review. "
                "Later cells may not be covered."
                )


    render_file_information(uploaded_file, size)
    render_detected_focus(notebook_focus)
    render_notebook_preview(notebook_text)
    render_notebook_statistics(stats, size)


    dynamic_instruction = focus_instructions.get(
        notebook_focus,
        DEFAULT_FOCUS_INSTRUCTION
        )

    if st.button("Analyze Notebook", type="primary"):
        safe_notebook_text = notebook_text[:MAX_CHARS]

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            update_loading_state(
                progress_bar,
                status_text,
                10,
                "Reading notebook structure..."
            )

            update_loading_state(
                progress_bar,
                status_text,
                30,
                "Preparing review criteria..."
            )

            # =========================
            # GEMINI API CALL
            # =========================
            # build prompt
            prompt = build_prompt(
                dynamic_instruction,
                reproducibility_context,
                safe_notebook_text
            )

            update_loading_state(
                progress_bar,
                status_text,
                65,
                "Evaluating notebook quality..."
            )

            # call gemini

            if TEST_MODE:
                output = TEST_REVIEW_OUTPUT
                st.session_state["review_output"] = output
            else:
                with st.spinner("Review engine is thinking..."):
                    output = call_gemini(prompt)
                    st.session_state["review_output"] = output

            update_loading_state(
                progress_bar,
                status_text,
                85,
                "Organizing review dashboard...",
                delay = 0.6
            )

            update_loading_state(
                progress_bar,
                status_text,
                100,
                "Analysis complete.",
                delay = 0.4

            )

            st.success("Analysis complete.")
            progress_bar.empty()
            status_text.empty()
            st.session_state["review_ready"] = True

        except errors.ClientError as e:
            log.warning(f"ClientError from Gemini | {e}")
            progress_bar.empty()
            status_text.empty()

            error_text = str(e)

            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                st.error("Review limit reached.")
                st.caption("The review service is temporarily unavailable. Please wait a moment and try again.")
            else:
                st.error("The AI review service could not complete the request.")
                st.caption("Please try again in a moment.")

                if DEBUG_MODE:
                    st.write("Error type:", type(e).__name__)
                    st.exception(e)

        except errors.ServerError as e:
            log.error(f"ServerError from Gemini | {e}")
            progress_bar.empty()
            status_text.empty()

            st.error("The AI review service is temporarily busy.")
            st.caption("The model is experiencing high demand. Please wait a moment and try again.")

            if DEBUG_MODE:
                st.write("Error type:", type(e).__name__)
                st.exception(e)

        except Exception as e:
            log.error(f"Unexpected error | {type(e).__name__}: {e}")
            progress_bar.empty()
            status_text.empty()

            st.error("Something went wrong while preparing the review.")
            st.caption("Please try again. If the issue continues, try a smaller notebook or clear very large output cells.")

            if DEBUG_MODE:
                st.write("Error type:", type(e).__name__)
                st.exception(e)

    if st.session_state.get("review_ready") and st.session_state.get("review_output"):
        output = st.session_state["review_output"]
        st.markdown("---")

        markdown_export = build_markdown_export(
            output,
            uploaded_file.name,
            notebook_focus,
            stats,
            size
        )
        pdf_export = build_pdf_export(
            output,
            uploaded_file.name,
            notebook_focus,
            stats,
            size
        )

        render_download_buttons(markdown_export, pdf_export, uploaded_file)

        render_top_priorities(output)
        render_review_dashboard(output, notebook_focus, stats, size)

        summary_tab, mistakes_tab, improvements_tab, questions_tab = st.tabs([
            "Summary",
            "Technical Audit",
            "Improvements",
            "Review Questions"
        ])

        with summary_tab:
            render_sections_as_expanders(output, [
                "Project Summary",
                "Evidence Found",
                "What Looks Good",
                "Notebook Scores",
                "Final Verdict"
            ])

        with mistakes_tab:
            render_sections_as_expanders(output, [
                "Mistakes & Bad Practices",
                "Data & Preprocessing Review",
                "Model & Training Review",
                "Reproducibility Review",
                "Overfitting / Underfitting Analysis"
            ])

        with improvements_tab:
            render_sections_as_expanders(output, [
                "Improvements"
            ])

        with questions_tab:
            render_sections_as_expanders(output, [
                "Technical Questions",
                "Technical Review Questions"
            ])

        # =========================
        # CHAT WITH NOTEBOOK
        # =========================
        chat_col1, chat_col2 = st.columns([6, 1])
        with chat_col1:
            st.markdown("### Chat with your Notebook")
            st.markdown(
                '<div class="dashboard-caption">Ask follow-up questions about your notebook and review.</div>',
                unsafe_allow_html=True
            )
        with chat_col2:
            if st.button("Clear Chat", type = "secondary"):
                st.session_state["chat_history"] = []
                st.rerun()

        # Show suggestions only when chat is empty
        if not st.session_state["chat_history"]:
            st.markdown(
                '<div class="dashboard-caption" style="margin-top: 12px;">Suggested questions — click to ask:</div>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="suggestion-buttons">', unsafe_allow_html=True)

            suggestions = get_chat_suggestions(notebook_focus)
            sug_cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                with sug_cols[i]:
                    if st.button(suggestion, key=f"sug_{i}", use_container_width= True):
                        st.session_state["chat_history"].append({
                            "role": "user",
                            "content": suggestion
                        })
                        st.session_state["pending_question"] = suggestion
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Display existing chat history
        for message in st.session_state["chat_history"]:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(f'<div style="color:#f5f5f7; font-size:15px; line-height:1.6;">{html.escape(message["content"])}</div>', unsafe_allow_html=True)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

        # Chat input box
        user_question = st.chat_input("Ask about your code, results, or improvements...")

        # Handle both typed questions and suggestion button clicks
        active_question = user_question or st.session_state.pop("pending_question", None)

        if active_question:
            if user_question:
                # Typed question — add to history and show immediately
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": active_question
                })

                with st.chat_message("user"):
                    st.markdown(
                        f'<div style="color:#f5f5f7; font-size:15px; line-height:1.6;">{html.escape(active_question)}</div>',
                        unsafe_allow_html=True
                    )

            # Build prompt and call Gemini
            with st.chat_message("assistant"):
                if CHAT_TEST_MODE:
                    chat_response = chat_response = f"Test mode is on, so Gemini is currently pretending to be deep in thought. Your question was: '{active_question}'"
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": chat_response
                    })
                    st.markdown(chat_response)
                else:
                    with st.spinner("Thinking..."):
                        try:
                            chat_prompt = build_chat_prompt(
                                active_question,
                                notebook_text,
                                output,
                                st.session_state["chat_history"]
                            )
                            chat_response = call_gemini(chat_prompt)
                            st.session_state["chat_history"].append({
                                "role": "assistant",
                                "content": chat_response
                            })
                            st.markdown(chat_response)

                        except errors.ClientError as e:
                            error_text = str(e)
                            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                                st.error("Review limit reached. Please wait and try again.")
                            else:
                                st.error("Could not get a response. Please try again.")

                        except errors.ServerError as e:
                            log.error(f"Chat server error | {e}")
                            st.error("The AI chat service is temporarily busy. Please try again in a moment.")

                        except Exception as e:
                            log.error(f"Chat error | {type(e).__name__}: {e}")
                            st.error("Something went wrong. Please try again.")

st.markdown("""
<div class="footer">
    Notebook Lens · Review. Improve. Iterate faster 
    <a href="https://github.com/kashan37/notebook-lens" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)