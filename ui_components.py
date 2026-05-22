import time
import html
import re
import streamlit as st

from review_parsing import (
    extract_section_body,
    extract_top_priorities,
    clean_preview_text,
)

# =========================
# LOADING UI HELPERS
# =========================
def update_loading_state(progress_bar, status_text, progress, message, delay=0.15):
    progress_bar.progress(progress)
    status_text.markdown(f"**{message}**")
    time.sleep(delay)



def render_sections_as_expanders(review_text, headings, first_expanded=True):
    matched_sections = []

    for heading in headings:
        pattern = rf"### {re.escape(heading)}(.*?)(?=\n### |\Z)"
        match = re.search(pattern, review_text, re.DOTALL)

        if match:
            matched_sections.append({
                "heading": heading,
                "content": match.group(1).strip()
            })

    if not matched_sections:
        st.info("No content found for this tab.")
        return

    for index, section in enumerate(matched_sections):
        with st.expander(
            section["heading"],
            expanded=(first_expanded and index == 0)
        ):
            st.markdown(section["content"])


def render_top_priorities(review_text):
    priorities = extract_top_priorities(review_text)

    if not priorities:
        return

    st.markdown("###  Top 3 Priorities")
    st.markdown(
        '<div class="dashboard-caption">Fix these first. Everything else can wait.</div>',
        unsafe_allow_html=True
    )

    for i, priority in enumerate(priorities, 1):
        st.markdown(f"""
        <div class="info-card" style="border-left: 3px solid #4facfe; margin-bottom: 8px;">
            <span style="color:#4facfe; font-weight:700; font-size:18px;">#{i}</span>
            &nbsp;&nbsp;{html.escape(priority)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)



def render_review_card(label, value, detail):
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_detail = html.escape(str(detail))

    st.markdown(f"""
    <div class="review-card">
        <div class="review-card-label">{safe_label}</div>
        <div class="review-card-value">{safe_value}</div>
        <div class="review-card-detail">{safe_detail}</div>
    </div>
    """, unsafe_allow_html=True)

def render_review_dashboard(output, notebook_focus, stats, size):
    mistakes_text = extract_section_body(output, "Mistakes & Bad Practices")
    improvements_text = extract_section_body(output, "Improvements")
    verdict_text = extract_section_body(output, "Final Verdict")

    issue_preview = clean_preview_text(mistakes_text)
    improvement_preview = clean_preview_text(improvements_text)
    verdict_preview = clean_preview_text(verdict_text)

    st.markdown("### Review Dashboard")
    st.markdown(
        '<div class="dashboard-caption">A quick executive snapshot before the detailed review.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_review_card(
            "Detected Focus",
            notebook_focus,
            "Primary ML/data task detected from notebook content."
        )

    with col2:
        render_review_card(
            "Notebook Shape",
            f"{stats['total_cells']} cells",
            f"{stats['code_cells']} code cells · {stats['markdown_cells']} markdown cells"
        )

    with col3:
        render_review_card(
            "Key Issue",
            "Needs Review",
            issue_preview
        )

    with col4:
        render_review_card(
            "Top Improvement",
            "Next Step",
            improvement_preview
        )

    st.markdown("<br>", unsafe_allow_html=True)

    render_review_card(
        "Final Verdict Preview",
        "Overall Take",
        verdict_preview
    )

def render_header():
    st.markdown("""
<div class="brand-header">
    <div class="brand-logo">
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="16" cy="16" r="15" fill="none" stroke="#4facfe" stroke-width="1.8"/>
    <circle cx="16" cy="16" r="12" fill="#4facfe" fill-opacity="0.05"/>
    <line x1="9" y1="9" x2="16" y2="13" stroke="#4facfe" stroke-opacity="0.5" stroke-width="1"/>
    <line x1="9" y1="16" x2="16" y2="13" stroke="#4facfe" stroke-opacity="0.5" stroke-width="1"/>
    <line x1="9" y1="23" x2="16" y2="19" stroke="#4facfe" stroke-opacity="0.5" stroke-width="1"/>
    <line x1="9" y1="9" x2="16" y2="19" stroke="#4facfe" stroke-opacity="0.5" stroke-width="1"/>
    <line x1="9" y1="23" x2="16" y2="13" stroke="#4facfe" stroke-opacity="0.5" stroke-width="1"/>
    <line x1="16" y1="13" x2="23" y2="9" stroke="#76b900" stroke-opacity="0.6" stroke-width="1"/>
    <line x1="16" y1="13" x2="23" y2="16" stroke="#76b900" stroke-opacity="0.6" stroke-width="1"/>
    <line x1="16" y1="19" x2="23" y2="16" stroke="#76b900" stroke-opacity="0.6" stroke-width="1"/>
    <line x1="16" y1="19" x2="23" y2="23" stroke="#76b900" stroke-opacity="0.6" stroke-width="1"/>
    <circle cx="9" cy="9" r="2.2" fill="#4facfe"/>
    <circle cx="9" cy="16" r="2.2" fill="#4facfe"/>
    <circle cx="9" cy="23" r="2.2" fill="#4facfe"/>
    <circle cx="16" cy="13" r="2.2" fill="#4facfe" opacity="0.85"/>
    <circle cx="16" cy="19" r="2.2" fill="#4facfe" opacity="0.85"/>
    <circle cx="23" cy="9" r="2.2" fill="#76b900"/>
    <circle cx="23" cy="16" r="2.2" fill="#76b900"/>
    <circle cx="23" cy="23" r="2.2" fill="#76b900"/>
    <path d="M25 25 L31 31" stroke="#4facfe" stroke-width="2.8" stroke-linecap="round"/>
    <path d="M21 21 Q23 20 25 22" fill="none" stroke="#ffffff" stroke-opacity="0.25" stroke-width="1.2" stroke-linecap="round"/>
</svg>
    </div>
    <div>
        <div class="app-title">Notebook Lens</div>
        <div class="app-subtitle">Your notebook deserves an honest code review</div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_landing_intro():
    st.markdown("""
<div class="landing-grid">
    <div class="landing-feature">
        <div class="landing-feature-icon">&lt; / &gt;</div>
        <div class="landing-feature-title">Automatic Notebook Review</div>
        <div class="landing-feature-desc">Upload any Jupyter or Colab notebook and get a structured ML review in seconds.
Notebook Lens analyzes your code, training setup, evaluation logic, and workflow automatically..</div>
    </div>
    <div class="landing-feature">
        <div class="landing-feature-icon">⚡</div>
        <div class="landing-feature-title">Actionable Feedback</div>
        <div class="landing-feature-desc">No vague AI advice. No giant wall of text. Get the highest priority issues and improvements based on actual notebook evidence..</div>
    </div>
    <div class="landing-feature">
        <div class="landing-feature-icon">💬</div>
        <div class="landing-feature-title">Chat With Your Notebook</div>
        <div class="landing-feature-desc">Ask follow up questions about your model, preprocessing, training decisions, or results. Responses stay grounded in your notebook and review context..</div>
    </div>
</div>
<div class="landing-who">
    <b>Built for:</b> Developers reviewing ML workflows before shipping • Students refining portfolio projects • Kaggle practitioners improving experiments • Bootcamp graduates strengthening notebook quality.
</div>
""", unsafe_allow_html=True)
    
def render_file_information(uploaded_file, size):
    st.markdown("### File Information")

    st.markdown(f"""
    <div class="info-card">
        <b>Notebook Name:</b> {uploaded_file.name}<br>
        <b>File Type:</b> Jupyter / Colab Notebook (.ipynb)<br>
        <b>File Size:</b> {size} KB
    </div>
""", unsafe_allow_html=True)


def render_detected_focus(notebook_focus):
    st.markdown(f"""
    ### Detected Notebook Focus
    <span style="
        background: linear-gradient(135deg, rgba(79,172,254,0.15), rgba(79,172,254,0.05));
        border: 1px solid rgba(79,172,254,0.35);
        color: #4facfe;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        margin-top: 6px;
    ">
    {notebook_focus}
    </span>
    """, unsafe_allow_html=True)


def render_notebook_preview(notebook_text):
    st.markdown("### Notebook Preview")

    st.text_area(
        "Notebook Content Preview",
        notebook_text[:100000],
        height=400
    )


def render_notebook_statistics(stats, size):
    st.markdown("### Notebook Statistics")

    st.markdown(f"""
    <div class="info-card">
        <b>Total Cells:</b> {stats['total_cells']}<br>
        <b>Code Cells:</b> {stats['code_cells']}<br>
        <b>Markdown Cells:</b> {stats['markdown_cells']}<br>
        <b>File Size:</b> {size} KB
    </div>
    """, unsafe_allow_html=True)

def render_download_buttons(markdown_export, pdf_export, uploaded_file):
    st.markdown('<div style="display:flex; flex-direction:column; gap:6px; margin-bottom:28px;">', unsafe_allow_html=True)

    st.download_button(
        label="⬇️ Download Review (.md)",
        data=markdown_export,
        file_name=f"notebook_lens_{uploaded_file.name.replace('.ipynb', '')}.md",
        mime="text/markdown"
    )

    st.download_button(
        label="⬇️ Download Report (.pdf)",
        data=pdf_export,
        file_name=f"notebook_lens_{uploaded_file.name.replace('.ipynb', '')}.pdf",
        mime="application/pdf"
    )

    st.markdown('</div>', unsafe_allow_html=True)