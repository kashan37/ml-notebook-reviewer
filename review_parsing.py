import streamlit as st
import re

# =========================
# REVIEW OUTPUT PARSING
# =========================
def extract_section(review_text, heading):
    pattern = rf"### {re.escape(heading)}(.*?)(?=\n### |\Z)"
    match = re.search(pattern, review_text, re.DOTALL)

    if not match:
        return ""

    return f"### {heading}\n{match.group(1).strip()}"

def extract_section_body(review_text, heading):
    pattern = rf"### {re.escape(heading)}(.*?)(?=\n### |\Z)"
    match = re.search(pattern, review_text, re.DOTALL)

    if not match:
        return ""

    return match.group(1).strip()

# =========================
# TOP 3 PRIORITIES
# =========================
def extract_top_priorities(review_text):
    body = extract_section_body(review_text, "Top 3 Priorities")
    if not body:
        return []

    priorities = []
    for line in body.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove leading "1. " "2. " etc
            clean = re.sub(r"^\d+\.\s*", "", line).strip()
            if clean:
                priorities.append(clean)

    return priorities[:3]

# =========================
# DASHBOARD CARD HELPERS
# =========================
def clean_preview_text(text, max_chars=260):
    if not text:
        return "No clear signal found yet."

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"[*_>]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    intro_patterns = [
        r"^here are some prioritized improvements.*?:\s*",
        r"^here are the prioritized improvements.*?:\s*",
        r"^the following improvements.*?:\s*",
        r"^below are some.*?:\s*",
    ]

    for pattern in intro_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    return text
