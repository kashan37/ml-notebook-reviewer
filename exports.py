import re
from fpdf import FPDF
from review_parsing import extract_section_body

# =========================
# DOWNLOAD HELPERS
# =========================
def build_markdown_export(output, notebook_name, notebook_focus, stats, size):
    lines = []
    lines.append(f"# Notebook Lens Review")
    lines.append(f"\n**Notebook:** {notebook_name}")
    lines.append(f"**Focus:** {notebook_focus}")
    lines.append(f"**Cells:** {stats['total_cells']} total · {stats['code_cells']} code · {stats['markdown_cells']} markdown")
    lines.append(f"**File Size:** {size} KB")
    lines.append(f"\n---\n")
    lines.append(output)
    return "\n".join(lines)

def build_pdf_export(output, notebook_name, notebook_focus, stats, size):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(79, 172, 254)
    pdf.cell(0, 12, "Notebook Lens Review", new_x="LMARGIN", new_y="NEXT")

    # Metadata
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(160, 160, 170)
    pdf.cell(0, 6, f"Notebook: {notebook_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Focus: {notebook_focus}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Cells: {stats['total_cells']} total  |  {stats['code_cells']} code  |  {stats['markdown_cells']} markdown", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"File Size: {size} KB", new_x="LMARGIN", new_y="NEXT")

    # Divider
    pdf.ln(4)
    pdf.set_draw_color(79, 172, 254)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # Sections
    section_headings = [
        "Top 3 Priorities",
        "Project Summary",
        "Evidence Found",
        "What Looks Good",
        "Mistakes & Bad Practices",
        "Data & Preprocessing Review",
        "Model & Training Review",
        "Reproducibility Review",
        "Overfitting / Underfitting Analysis",
        "Improvements",
        "Notebook Scores",
        "Technical Questions",
        "Final Verdict"
    ]
    for heading in section_headings:
        body = extract_section_body(output, heading)
        if not body:
            continue

        # Section heading
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(79, 172, 254)
        pdf.cell(0, 10, heading, new_x="LMARGIN", new_y="NEXT")

        # Section body — clean markdown symbols
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 60)

        clean_body = re.sub(r"#{1,6}\s*", "", body)
        clean_body = re.sub(r"\*\*(.*?)\*\*", r"\1", clean_body)
        clean_body = re.sub(r"\*(.*?)\*", r"\1", clean_body)
        clean_body = re.sub(r"(.*?)", r"\1", clean_body)
        clean_body = re.sub(r'```.*?```', '[code block]', clean_body, flags=re.DOTALL)

        safe_body = clean_body.encode("latin-1", errors="replace").decode("latin-1")

        pdf.multi_cell(0, 6, safe_body)
        pdf.ln(4)

    return bytes(pdf.output())