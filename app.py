import streamlit as st
import nbformat
from google import genai

client = genai.Client()

def get_notebook_stats(notebook):
    total_cells = len(notebook.cells)

    code_cells = sum(1 for c in notebook.cells if c.cell_type == "code")
    markdown_cells = sum(1 for c in notebook.cells if c.cell_type == "markdown")

    return {
        "total_cells": total_cells,
        "code_cells": code_cells,
        "markdown_cells": markdown_cells
    }

def get_file_size(uploaded_file):

    size_kb = uploaded_file.size / 1024
    return round(size_kb, 2)




def extract_markdown(cell):
    if cell.cell_type == "markdown":
        return f"[MARKDOWN]\n{cell.source}\n"

    return ""


def extract_code(cell):
    if cell.cell_type == "code":
        return f"[CODE]\n{cell.source}\n"

    return ""


def extract_outputs(cell):
    output_sections = []

    if cell.cell_type == "code" and hasattr(cell, "outputs") and cell.outputs:
        output_sections.append("[OUTPUT]")

        for output in cell.outputs:
            if output.get("output_type") == "stream":
                output_sections.append(output.get("text", ""))

            elif output.get("output_type") in ["execute_result", "display_data"]:
                data = output.get("data", {})

                if "text/plain" in data:
                    output_sections.append(data["text/plain"])

            elif output.get("output_type") == "error":
                output_sections.append("ERROR:")
                output_sections.append(output.get("ename", ""))
                output_sections.append(output.get("evalue", ""))

    return "\n".join(output_sections)


def load_notebook(notebook):
    sections = []

    for cell in notebook.cells:
        markdown = extract_markdown(cell)
        code = extract_code(cell)
        outputs = extract_outputs(cell)

        if markdown:
            sections.append(markdown)

        if code:
            sections.append(code)

        if outputs:
            sections.append(outputs)

    return "\n".join(sections)




# st.title("ML Notebook Reviewer")
st.markdown("""
<h1 style="
color: #e6e6e6;
font-size: 46px;
font-weight: 600;
border-left: 5px solid #4facfe;
padding-left: 12px;
">
ML Notebook Reviewer
</h1>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload notebook for analysis (Jupyter / Colab .ipynb),", 
     type=["ipynb"]
    )


if uploaded_file is not None:

    notebook = nbformat.read(uploaded_file, as_version=4)
    stats = get_notebook_stats(notebook)
    size = get_file_size(uploaded_file)
    notebook_text = load_notebook(notebook)

    st.success("Upload successful. Ready for analysis.")

    # FILE INFO CARD
    st.markdown("### File Information")

    st.markdown(f"""
    <div style="
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #333;
    color: #e6e6e6;
    font-size: 15px;
    line-height: 1.6;
    ">

    <b>Notebook Name:</b> {uploaded_file.name}<br>
    <b>File Type:</b> Jupyter / Colab Notebook (.ipynb)<br>
    <b>File Size:</b> {size} KB

    </div>
    """, unsafe_allow_html=True)

    # NOTEBOOK CONTENT
    st.markdown("### Notebook Preview")

    st.text_area(
        "Content",
        notebook_text[:100000],
        height=400
    )

    # STATS SECTION
    st.markdown("### Notebook Statistics")

    st.markdown(f"""
    <div style="
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #333;
    color: #e6e6e6;
    font-size: 15px;
    line-height: 1.6;
    ">

    <b>Total Cells:</b> {stats['total_cells']}<br>
    <b>Code Cells:</b> {stats['code_cells']}<br>
    <b>Markdown Cells:</b> {stats['markdown_cells']}<br>
    <b>File Size:</b> {size} KB

    </div>
    """, unsafe_allow_html=True)

    if st.button("Analyze Notebook"):
        MAX_CHARS = 80000
        safe_notebook_text = notebook_text[:MAX_CHARS]
        prompt = f"""
        You are a senior ML engineer reviewing a Jupyter notebook for a junior data scientist.

Your goal is to give a helpful, friedly, practical review that is easy to read.
Be honest about problems, Always Roast a little bit and dont be boring.
Avoid generic advice. Tie every point to something visible in the notebook when possible.

Your job is to:
- Be precise and technical
- Avoid vague advice
- Only comment based on evidence in the notebook
- If something is unclear or missing, explicitly say: "Not enough information"

Do NOT hallucinate missing components.
Only rewrite or improve code inside the "Mistakes & Bad Practices" and "Improvements" sections if applicable.
Do NOT generate corrected code in any other section.

Return your response in this STRICT format:

### Project Summary
Briefly explain what the notebook is trying to do, what ML task it appears to solve, and what the final output/model seems to be.

### What Looks Good
Mention 2-4 things the notebook does well, even if the project has issues.

### Mistakes & Bad Practices
List the main problems in the notebook.
For each issue, explain:
- what the problem is
- why it matters
- how to fix it

### Data & Preprocessing Review
Comment on missing values, encoding, scaling, feature selection, data leakage, train/test split, and whether preprocessing is done correctly.

### Model & Training Review
Review model choice, training approach, evaluation metrics, validation strategy, and whether the chosen metric fits the problem.

### Overfitting / Underfitting Analysis
Explain any signs or risks of overfitting or underfitting.
Suggest practical ways to reduce those risks.

### Improvements
Give clear, prioritized improvements.
Label them as:
- Quick wins
- Medium improvements
- Advanced improvements

### Interview Questions
Generate 5-7 interview questions based on this notebook.
Make them specific to the project, not generic ML questions.

### Final Verdict
Give a short friendly verdict:
- overall quality
- biggest strength
- biggest thing to fix next
- readiness level: Beginner / Improving / Solid / Portfolio-ready

        Notebook: {safe_notebook_text} """


        with st.spinner("Analyzing notebook... this may take a few seconds"):

            response = client.models.generate_content( model="gemini-2.5-flash",
                                                      contents=prompt)
            output = response.text
            st.success("Analysis complete")
            st.markdown("---")
            st.markdown(output)