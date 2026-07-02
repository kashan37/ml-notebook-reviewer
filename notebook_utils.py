import streamlit as st

# ===============================
# CORE NOTEBOOK PARSING FUNCTIONS
# ===============================
@st.cache_data
def get_notebook_stats(notebook):
    total_cells = len(notebook.cells)

    code_cells = sum(1 for c in notebook.cells if c.cell_type == "code")
    markdown_cells = sum(1 for c in notebook.cells if c.cell_type == "markdown")

    return {
        "total_cells": total_cells,
        "code_cells": code_cells,
        "markdown_cells": markdown_cells
    }

def validate_notebook_content(notebook):
    """
    Returns (is_valid, reason).
    is_valid = False blocks analysis with a clean error.
    """
    if not notebook.cells:
        return False, "This notebook has no cells at all."

    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    if not code_cells:
        return False, "No code cells found. This appears to be a markdown-only notebook."

    non_empty_code = [c for c in code_cells if c.source.strip()]
    if not non_empty_code:
        return False, "All code cells are empty. There is no code to review."

    total_code_chars = sum(len(c.source.strip()) for c in non_empty_code)
    if total_code_chars < 80:
        return False, "The notebook contains very little code (under 80 characters). Not enough to review."
    
    return True, None

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
                stream_text = output.get("text", "")

                if isinstance(stream_text, list):
                    stream_text = "\n".join(stream_text)

                output_sections.append(str(stream_text)[:8000])

            elif output.get("output_type") in ["execute_result", "display_data"]:
                data = output.get("data", {})

                if "text/plain" in data:
                    text_output = data["text/plain"]

                    if isinstance(text_output, list):
                        text_output = "\n".join(text_output)

                    output_sections.append(str(text_output)[:1000])

            elif output.get("output_type") == "error":
                output_sections.append("ERROR:")
                output_sections.append(str(output.get("ename", "")))
                output_sections.append(str(output.get("evalue", "")))

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