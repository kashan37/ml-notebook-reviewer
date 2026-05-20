import streamlit as st
import nbformat
from google import genai
from google.genai import errors
import re
import html
import time
import logging
from fpdf import FPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("notebook_lens")
# =========================
# IMPORTS & SETUP
# =========================

client = genai.Client()
DEBUG_MODE = False
TEST_MODE = False  #set False for real model calls
CHAT_TEST_MODE = False

st.set_page_config(
    page_title="Notebook Lens",
    page_icon="🛰️",
    layout="wide"
)

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

                output_sections.append(str(stream_text)[:2000])

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


# =========================
# UI STYLES
# =========================
st.markdown("""
<style>
            
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

span[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-icons {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
            
:root {
    --nl-bg: #0b0c0f;
    --nl-surface: #111318;
    --nl-surface-soft: #151821;
    --nl-border: rgba(255, 255, 255, 0.08);
    --nl-border-strong: rgba(255, 255, 255, 0.14);
    --nl-text: #f5f5f7;
    --nl-muted: #a1a1aa;
    --nl-subtle: #73737d;
    --nl-accent: #4facfe;
    --nl-accent-2: #76b900;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(79, 172, 254, 0.08), transparent 28rem),
        var(--nl-bg);
}

.block-container {
    max-width: 1180px;
    padding-top: 4.8rem;
    padding-bottom: 4rem;
}
            
.brand-header {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 30px;
}

.brand-logo {
    width: 64px;
    height: 64px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        linear-gradient(145deg, rgba(79, 172, 254, 0.28), rgba(118, 185, 0, 0.10)),
        linear-gradient(145deg, #151821, #0f1117);
    border: 1px solid var(--nl-border-strong);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
}

.app-title {
    color: var(--nl-text);
    font-size: 48px;
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.05;
    margin-bottom: 5px;
}

.app-subtitle {
    color: var(--nl-muted);
    font-size: 18px;
    line-height: 1.45;
    margin-bottom: 0;
}

.app-subtitle {
    color: #a1a1aa;
    font-size: 18px;
    line-height: 1.55;
    margin: 0 0 28px 0;
    padding-left: 2px; /* tiny optical alignment */
}

h1, h2, h3 {
    color: #f5f5f7;
    letter-spacing: 0;
}

h3 {
    font-size: 1.22rem;
    font-weight: 650;
    margin-top: 1.8rem;
    margin-bottom: 0.75rem;
}

p, li {
    line-height: 1.65;
}

div[data-testid="stMarkdownContainer"] {
    line-height: 1.65;
}

div[data-testid="stTextArea"] textarea {
    background-color: #0f1117;
    color: #d7d7dd;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    font-size: 13px;
    line-height: 1.55;
}

pre {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-x: auto !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

code {
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.075);
    border-radius: 14px;
    overflow: hidden;
}

div[data-testid="stExpander"] summary {
    font-weight: 650;
    color: #f5f5f7;
}



div[data-testid="stTabs"] button {
    color: #a1a1aa;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f5f5f7;
}

div[data-testid="stFileUploader"] {
    background: linear-gradient(145deg, rgba(21, 24, 33, 0.72), rgba(15, 17, 23, 0.72));
    border: 1px solid var(--nl-border);
    border-radius: 16px;
    padding: 16px;
}

div[data-testid="stFileUploader"]::before {
    content: "Upload notebook for analysis (Jupyter / Colab .ipynb)";
    display: block;
    color: #a1a1aa;
    font-size: 14px;
    margin-bottom: 12px;
}

div[data-testid="stFileUploader"] section {
    border: 1px dashed rgba(79, 172, 254, 0.28);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.018);
}
            
div[data-testid="stFileUploader"] label span:last-of-type {
    display: none !important;
}

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--nl-accent), #2f7df4);
    color: white;
    border: 0;
    padding: 0.65rem 1.1rem;
    box-shadow: 0 10px 28px rgba(79, 172, 254, 0.18);
}

.stButton > button:hover {
    border: 0;
    color: white;
    filter: brightness(1.06);
}

div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--nl-accent), var(--nl-accent-2));
}


.review-card {
    background: linear-gradient(145deg, var(--nl-surface-soft), #0f1117);
    border: 1px solid var(--nl-border);
    border-radius: 16px;
    padding: 20px;
    min-height: 165px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}
            
.review-card:hover {
    border-color: rgba(79, 172, 254, 0.22);
    transform: translateY(-1px);
    transition: border-color 180ms ease, transform 180ms ease;
}

.review-card-label {
    color: #7a7a7a;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.review-card-value {
    color: #f5f5f7;
    font-size: 22px;
    font-weight: 650;
    line-height: 1.2;
    margin-bottom: 10px;
}

.review-card-detail {
    color: #b8b8b8;
    font-size: 14px;
    line-height: 1.5;
    overflow-wrap: anywhere;
    word-break: break-word;
}

.review-card-accent {
    color: #4facfe;
}

.info-card {
    background: linear-gradient(145deg, var(--nl-surface-soft), #0f1117);
    border: 1px solid var(--nl-border);
    border-radius: 16px;
    padding: 18px 20px;
    color: #f5f5f7;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 8px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}

.info-card b {
    color: #a1a1aa;
    font-weight: 600;
}
            

.dashboard-caption {
    color: #9b9b9b;
    font-size: 14px;
    margin-top: -8px;
    margin-bottom: 18px;
}
            
div[data-testid="stDownloadButton"] > button {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: linear-gradient(145deg, #151821, #0f1117) !important;
    color: #f5f5f7 !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18) !important;
    border-radius: 16px !important;
    font-weight: 600 !important;
    width: fit-content !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    border-color: rgba(79, 172, 254, 0.22) !important;
    filter: brightness(1.08) !important;
}
            
[data-testid="stBottom"] {
    background: var(--nl-bg) !important;
    border-top: none !important;
    padding: 0px 0 !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
}

div[data-testid="stChatMessageContainer"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f5f5f7 !important;
    border: none !important;
    box-shadow: none !important;
    min-height: 20px !important;
    padding: 6px 0 !important;
}
            
div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 0 !important;
}

div[data-testid="stChatMessage"] p {
    color: #f5f5f7 !important;
    line-height: 1.6 !important;
    font-size: 15px !important;
}

div[data-testid="stChatMessage"] code {
    background: rgba(79, 172, 254, 0.1) !important;
    color: #4facfe !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 13px !important;
}

div[data-testid="stChatMessage"] pre {
    background: #0f1117 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

.suggestion-buttons .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(79, 172, 254, 0.3) !important;
    color: #a1a1aa !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.8rem !important;
    box-shadow: none !important;
    border-radius: 12px !important;
    white-space: normal !important;
    min-height: 72px !important;
    line-height: 1.4 !important;
    width: 100% !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
            
            
.landing-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 0 0 16px 0;
}

.landing-feature {
    background: linear-gradient(145deg, #151821, #0f1117);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 20px;
}

.landing-feature-icon {
    font-size: 22px;
    margin-bottom: 10px;
}

.landing-feature-title {
    color: #f5f5f7;
    font-size: 15px;
    font-weight: 650;
    margin-bottom: 6px;
}

.landing-feature-desc {
    color: #a1a1aa;
    font-size: 13px;
    line-height: 1.55;
}

.landing-who {
    background: linear-gradient(145deg, rgba(79,172,254,0.06), rgba(79,172,254,0.02));
    border: 1px solid rgba(79,172,254,0.15);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 0 0 32px 0;
    color: #a1a1aa;
    font-size: 14px;
    line-height: 1.7;
}

.landing-who b {
    color: #f5f5f7;
    font-weight: 600;
}
            

.suggestion-buttons .stButton > button:hover {
    border-color: rgba(79, 172, 254, 0.7) !important;
    color: #f5f5f7 !important;
    background: rgba(79, 172, 254, 0.08) !important;
}
            
/* Secondary buttons: chat suggestions + clear chat */
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(79, 172, 254, 0.3) !important;
    color: #a1a1aa !important;
    box-shadow: none !important;
}

button[kind="secondary"]:hover {
    background: rgba(79, 172, 254, 0.08) !important;
    border-color: rgba(79, 172, 254, 0.7) !important;
    color: #f5f5f7 !important;
}

            
.footer {
    color: #73737d;
    font-size: 12px;
    text-align: center;
    padding: 32px 0 8px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 48px;
}

.footer a {
    color: #4facfe;
    text-decoration: none;
}
            
</style>
""", unsafe_allow_html=True)


# st.title("ML Notebook Reviewer")
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


uploaded_file = st.file_uploader(
    "",
    type=["ipynb"],
    label_visibility="collapsed"
)



def count_keywords(text, keywords):
    return sum(1 for keyword in keywords if keyword in text)


# ==============================
# NOTEBOOK TYPE + TASK DETECTION
# ==============================
@st.cache_data
def detect_notebook_focus(notebook_text):
    text = notebook_text.lower()

    categories = {
        "Diffusion Model": [
            "stable diffusion",
            "diffusionpipeline",
            "stablediffusionpipeline",
            "diffusers",
            "noise_scheduler",
            "denoising",
            "ddpm",
            "ddim",
            "unet2dconditionmodel",
            "latent diffusion"
        ],
        "Transformer / LLM": [
            "transformers",
            "autotokenizer",
            "automodelforcausallm",
            "automodelforsequenceclassification",
            "bert",
            "gpt",
            "llama",
            "mistral",
            "gemma",
            "t5",
            "attention",
            "self-attention",
            "llm",
            "lora",
            "qlora",
            "peft"
        ],
        "Autoencoder": [
            "autoencoder",
            "variational autoencoder",
            "reconstruction_loss",
            "reconstruction loss",
            "latent_dim",
            "bottleneck"
        ],
        "GAN": [
            "gan",
            "generator",
            "discriminator",
            "adversarial",
            "generator loss",
            "discriminator loss"
        ],
        "Computer Vision": [
            "conv2d",
            "convolution",
            "cnn",
            "opencv",
            "cv2",
            "imagedatagenerator",
            "image_dataset_from_directory",
            "flow_from_directory",
            "resnet",
            "efficientnet",
            "mobilenet"
        ],
        "NLP": [
            "tokenizer",
            "tfidfvectorizer",
            "countvectorizer",
            "nltk",
            "spacy",
            "word_tokenize",
            "stemming",
            "lemmatization",
            "embedding"
        ],
        "Feature Engineering": [
            "feature engineering",
            "onehotencoder",
            "labelencoder",
            "standardscaler",
            "minmaxscaler",
            "robustscaler",
            "get_dummies",
            "fillna",
            "dropna",
            "isnull",
            "missing values",
            "feature selection",
            "selectkbest",
            "mutual_info",
            "pca",
            "encoding",
            "scaling",
            "normalization"
        ],
        "Exploratory Data Analysis": [
            "describe()",
            "value_counts",
            "corr()",
            "sns.",
            "plt.",
            "hist",
            "boxplot",
            "heatmap",
            "pairplot",
            "eda"
        ],
        "Time Series": [
            "datetime",
            "timestamp",
            "resample",
            "rolling",
            "shift",
            "arima",
            "sarima",
            "forecast",
            "seasonality",
            "trend"
        ],
        "Classification": [
            "classification_report",
            "accuracy_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "confusion_matrix",
            "logisticregression",
            "randomforestclassifier",
            "categorical_crossentropy",
            "binary_crossentropy",
            "naive bayes",
            "naivebayes",
            "gaussiannb",
            "multinomialnb",
            "bernoullinb",
            "prior probability",
            "posterior probability",
            "likelihood",
            "bayes theorem",
            "class probability"
        ],
        "Regression": [
            "mean_squared_error",
            "mean_absolute_error",
            "r2_score",
            "linearregression",
            "randomforestregressor",
            "mae",
            "mse",
            "rmse"
        ],
        "Clustering": [
            "kmeans",
            "dbscan",
            "agglomerativeclustering",
            "silhouette_score",
            "clustering"
        ],
        "Reinforcement Learning": [
            "reinforcement learning",
            "rl",
            "q-learning",
            "qlearning",
            "dqn",
            "policy gradient",
            "reward",
            "environment",
            "gym",
            "gymnasium",
            "agent",
            "episode",
            "epsilon",
            "bellman"
        ],
        "Object Detection": [
            "yolo",
            "yolov",
            "faster rcnn",
            "fasterrcnn",
            "ssd",
            "object detection",
            "bounding box",
            "anchor box",
            "iou",
            "map",
            "nms",
            "non maximum suppression",
            "detectron"
        ],
        "Image Segmentation": [
            "segmentation",
            "unet",
            "u-net",
            "semantic segmentation",
            "instance segmentation",
            "mask rcnn",
            "maskrcnn",
            "pixel",
            "deeplabv"
        ],
        "Audio / Speech": [
            "librosa",
            "torchaudio",
            "mel spectrogram",
            "mfcc",
            "waveform",
            "speech recognition",
            "whisper",
            "audio",
            "sound",
            "wav2vec"
        ],
        "Recommendation System": [
            "collaborative filtering",
            "content based",
            "matrix factorization",
            "svd",
            "cosine similarity",
            "user item",
            "rating",
            "recommendation",
            "recommender"
        ],
        "Anomaly Detection": [
            "anomaly detection",
            "outlier",
            "isolation forest",
            "one class svm",
            "novelty detection",
            "fraud detection",
            "anomaly score"
        ],
        "Data Visualization": [
            "plotly",
            "dash",
            "bokeh",
            "altair",
            "matplotlib",
            "seaborn",
            "visualization",
            "interactive plot",
            "dashboard"
        ],
        "MLOps / Deployment": [
            "docker",
            "kubernetes",
            "fastapi",
            "flask",
            "streamlit",
            "gradio",
            "onnx",
            "torchscript",
            "triton",
            "bentoml",
            "mlflow",
            "model serving",
            "inference",
            "pipeline"
        ]
    }

    minimum_scores = {
        "Diffusion Model": 1,
        "Transformer / LLM": 2,
        "Autoencoder": 1,
        "GAN": 2,
        "Computer Vision": 2,
        "NLP": 2,
        "Feature Engineering": 3,
        "Exploratory Data Analysis": 5,
        "Time Series": 2,
        "Classification": 1,
        "Regression": 2,
        "Clustering": 2,
        "Reinforcement Learning": 2,
        "Object Detection": 2,
        "Image Segmentation": 1,
        "Audio / Speech": 2,
        "Recommendation System": 2,
        "Anomaly Detection": 1,
        "Data Visualization": 3,
        "MLOps / Deployment": 2
    }

    scores = {
        category: count_keywords(text, keywords)
        for category, keywords in categories.items()
    }

    best_category = max(scores, key=scores.get)

    if scores[best_category] >= minimum_scores[best_category]:
        return best_category

    if any(keyword in text for keyword in [
        "sklearn",
        "train_test_split",
        ".fit(",
        ".predict(",
        "model.fit",
        "model.predict"
    ]):
        return "General ML / Data Science"

    return "General Notebook"

# ==============================
# REPRODUCIBILITY ANALYSIS
# =============================
@st.cache_data
def detect_reproducibility_signals(notebook_text):
    text = notebook_text.lower()
    signals = {
        "Random seeds": any(keyword in text for keyword in [
            "random_state",
            "np.random.seed",
            "random.seed",
            "torch.manual_seed",
            "tf.random.set_seed",
            "seed="
        ]),
        "Train/test split": any(keyword in text for keyword in [
            "train_test_split",
            "validation_split",
            "stratifiedkfold",
            "kfold",
            "cross_val_score",
            "train_df",
            "test_df",
            "x_train",
            "x_test",
            "y_train",
            "y_test"
        ]),
        "Callbacks": any(keyword in text for keyword in [
            "earlystopping",
            "modelcheckpoint",
            "reducelronplateau",
            "tensorboard",
            "callbacks"
        ]),
        "Logging / experiment tracking": any(keyword in text for keyword in [
            "mlflow",
            "wandb",
            "tensorboard",
            "comet",
            "neptune",
            "history.history",
            "training log",
            "logs"
        ])
    }

    result = ["Reproducibility signals detected in the notebook:"]

    for name, found in signals.items():
        status = "Found" if found else "Not found"
        result.append(f"- {name}: {status}")

    return "\n".join(result)

# =========================
# LOADING UI HELPERS
# =========================
def update_loading_state(progress_bar, status_text, progress, message, delay=0.15):
    progress_bar.progress(progress)
    status_text.markdown(f"**{message}**")
    time.sleep(delay)


# =========================
# CHAT SUGGESTIONS
# =========================
def get_chat_suggestions(notebook_focus):
    base_suggestions = [
        "What should I fix first in my notebook?",
        "How can I make my results more reproducible?",
        "What interview questions should I prepare for this project?",
    ]

    focus_suggestions = {
        "Computer Vision": [
            "Why might my model be overfitting on the training images?",
            "What augmentation strategy would improve my model?",
            "How can I improve my validation accuracy?",
        ],
        "Classification": [
            "How can I handle class imbalance in my dataset?",
            "Which evaluation metric is most appropriate for my problem?",
            "How would I improve my confusion matrix results?",
        ],
        "Regression": [
            "What residual issues should I check for?",
            "How can I reduce my MSE score?",
            "Are my features properly scaled for this model?",
        ],
        "NLP": [
            "Is my tokenization strategy appropriate for this task?",
            "How can I improve my text preprocessing pipeline?",
            "What embeddings would work better for this problem?",
        ],
        "GAN": [
            "What signs of mode collapse should I watch for?",
            "How can I stabilize my GAN training?",
            "How do I evaluate the quality of generated samples?",
        ],
        "Transformer / LLM": [
            "Is my fine-tuning setup appropriate for this task?",
            "How can I reduce hallucination risks in my outputs?",
            "What evaluation strategy fits this LLM task?",
        ],
        "Time Series": [
            "How can I check for data leakage in my time series?",
            "Is my validation strategy appropriate for time series data?",
            "How should I handle seasonality in my forecasting?",
        ],
        "Clustering": [
            "How do I know if my cluster count is appropriate?",
            "Should I scale my features before clustering?",
            "How can I evaluate the quality of my clusters?",
        ],
        "Reinforcement Learning": [
            "How can I tell if my agent is learning effectively?",
            "What reward shaping issues should I check for?",
            "How do I prevent my agent from overfitting to the environment?",
        ],
        "Object Detection": [
            "How can I improve my mAP score?",
            "Is my anchor box configuration appropriate?",
            "How should I handle class imbalance in object detection?",
        ],
        "Exploratory Data Analysis": [
            "What key insights am I missing from my EDA?",
            "How can I improve my visualization choices?",
            "What statistical tests should I run on this data?",
        ],
        "Feature Engineering": [
            "Am I at risk of data leakage in my preprocessing pipeline?",
            "Which feature selection method fits my dataset best?",
            "How should I handle my missing values differently?",
        ],
    }

    specific = focus_suggestions.get(notebook_focus, base_suggestions)
    return specific[:3]

# =========================
# CHAT PROMPT BUILDER
# =========================
def build_chat_prompt(user_question, notebook_text, review_output, chat_history):

    history_text = ""
    for message in chat_history[:-1]:
        role = "User" if message["role"] == "user" else "Assistant"
        history_text += f"{role}: {message['content']}\n"

    prompt = f"""
You are a strict ML code review assistant. You have access to a Jupyter notebook and its review.
Your job is to answer follow-up questions from the author about their notebook.

STRICT RULES — follow every single one without exception:
- Answer ONLY from evidence explicitly visible in the notebook text or review provided below.
- NEVER invent, assume, or extrapolate architecture details, results, metrics, or functions not shown.
- NEVER say "typically", "usually", "in most cases" — only talk about THIS specific notebook.
- If the notebook does not contain enough evidence to answer confidently, say exactly: "The notebook does not provide enough information to answer this confidently."
- Keep answers to 3-5 sentences maximum. No long explanations.
- Answer directly. No filler phrases like "Great question!", "Certainly!", or "Of course!".
- Reference specific variable names, function names, or outputs from the notebook when possible.
- Do not repeat or summarize the full review. Answer only the specific question asked.
- If the user asks something unrelated to the notebook or ML, say: "I can only answer questions about your notebook and its review."
- Do not make up improvement suggestions unless they are directly supported by notebook evidence.

NOTEBOOK TEXT:
{notebook_text[:40000]}
FULL REVIEW:
{review_output[:8000]}
CONVERSATION SO FAR:
{history_text}
USER QUESTION:
{user_question}

Answer in 3-5 sentences maximum. Be specific. Be honest about uncertainty.
"""
    return prompt


# =============================
        # PROMPT BUILDER (GEMINI INPUT)
        # =============================

def build_prompt(dynamic_instruction, reproducibility_context, safe_notebook_text):
    prompt = f"""
You are a senior Machine Learning engineer and technical reviewer evaluating a Jupyter notebook.

Your goal is to give a helpful, friendly, practical review that is easy to read.
Be honest about problems, always roast a little bit and don't be boring.
Avoid generic advice. Tie every single point to something visible in the notebook.

Your job is to:
- Be precise and technical
- Avoid vague advice
- Only comment based on evidence in the notebook
- If something is unclear or missing, explicitly say: "Not enough information"
- Reference specific notebook evidence whenever possible
- Mention specific functions, models, preprocessing steps, metrics, libraries, or outputs seen in the notebook
- Quote short relevant snippets or behaviors from the notebook when useful
- Do not make generic ML comments unless supported by notebook evidence

CRITICAL RULES — follow these without exception:
- Every single point in every section must reference specific code, functions, variable names, or outputs from the notebook. If you cannot tie a point to specific evidence, do not include it.
- NEVER mention a library, function, metric, or output that is not explicitly visible in the notebook text provided. If unsure whether something exists in the notebook, do not mention it.
- Do NOT hallucinate missing components.
- If evidence for a claim is weak or missing, clearly state that the notebook does not provide enough evidence.
- Only suggest code that preserves data integrity assumptions.
- If dataset structure is unclear, first recommend validation or inspection steps before transformations.
- Do not assume ordering, pairing, or schema correctness unless explicitly shown in the notebook evidence.
- Be conservative with scores. A notebook with no validation split, no seeds, and no callbacks cannot score above 5 in ML Rigor regardless of other qualities.
- Do not give high scores unless strong notebook evidence supports them.
- Avoid inflated scoring.

{dynamic_instruction}
{reproducibility_context}
Use the reproducibility signals above to guide your review, but do not overstate them.
If a signal is marked "Not found", mention it only where relevant and say the notebook does not provide enough evidence.
Only include rewritten or improved code when it directly helps explain a problem or improvement.
Place code examples only inside the "Mistakes & Bad Practices" or "Improvements" sections.
Do NOT generate corrected code in any other section.
Do NOT rewrite large parts of the notebook unless the notebook evidence clearly supports it.

Return your response in this STRICT format:
Use the exact section headings shown below. Do not rename headings, because the app uses them to organize the review into tabs:

### Top 3 Priorities
List exactly 3 most critical things the author should fix or improve first.
Each priority must name a specific thing from the notebook — a specific function, variable, pattern, or behavior.
Each priority must be one clear, specific, actionable sentence.
Number them 1, 2, 3.
No explanations, no sub-bullets. Just 3 lines.
BAD example: "Improve your validation strategy."
GOOD example: "Add a validation_split parameter to model.fit() since no validation data is currently passed."
Base every priority on actual evidence found in the notebook.

### Project Summary
Briefly explain what the notebook appears to be doing, what ML/data task it seems to address, and what the final output, model, or analysis appears to be.
If the goal is unclear, say "Not enough information."

### Evidence Found
List the most important concrete evidence found in the notebook.
Mention relevant libraries, functions, models, preprocessing steps, metrics, outputs, or notebook patterns.
Do not invent evidence.

### What Looks Good
Mention 2-4 things the notebook does well.
Tie each point to specific evidence from the notebook.

### Mistakes & Bad Practices
List the main problems in the notebook.
For each issue, include:
- Problem
- Evidence from the notebook
- Why it matters
- How to fix it

Only include issues that are supported by notebook evidence.
If something is only a risk, label it as a risk, not a confirmed mistake.

### Data & Preprocessing Review
Review missing values, encoding, scaling, feature selection, data leakage, train/test split, and preprocessing quality.
Reference actual preprocessing steps, functions, or code patterns found in the notebook.
If any area is not shown in the notebook, say "Not enough information."

### Model & Training Review
Review model choice, training approach, evaluation metrics, validation strategy, and whether the chosen metric fits the problem.
Reference actual models, metrics, callbacks, losses, logs, or evaluation outputs detected in the notebook.
If no model or training process is visible, say "Not enough information."

### Reproducibility Review
Review whether the notebook is reproducible and easy to verify.
Comment on:
- random seeds
- train/test split or validation setup
- callbacks such as EarlyStopping, ModelCheckpoint, or learning-rate scheduling
- logging or experiment tracking
- whether results can be rerun reliably

Use only notebook evidence.
If any item is missing or unclear, say "Not enough information."

### Overfitting / Underfitting Analysis
Explain any signs or risks of overfitting or underfitting.
Suggest practical ways to reduce those risks.
Use notebook evidence such as training logs, validation metrics, learning curves, or output behavior when making conclusions.
If no training metrics are visible, say "Not enough training metrics found to confidently evaluate overfitting."

### Improvements
Give clear, prioritized improvements.

Label them as:
- Quick wins
- Medium improvements
- Advanced improvements

For each improvement, explain:
- what to change specifically — name the function, variable, or section
- why it improves the notebook
- where it applies based on notebook evidence

### Notebook Scores
Give scores from 1-10 for the following areas.

For each score:
- give the numeric score
- briefly justify the score using notebook evidence

Categories:
- Code Quality
- ML Rigor
- Experimentation
- Readability

Scoring Guidelines:
- 1-3 = weak
- 4-6 = developing
- 7-8 = strong
- 9-10 = exceptional

Scoring Rules:
- No validation split = ML Rigor cannot exceed 5
- No random seeds = ML Rigor cannot exceed 6
- No evaluation metrics = ML Rigor cannot exceed 4
- Generic or missing comments = Readability cannot exceed 5

### Technical Questions
Generate 5-7 questions that would come up in a professional ML code review or portfolio review.
Questions should test the author's reasoning about data preprocessing, modeling choices, metrics, validation, limitations, and deployment readiness.
Each question must reference something specific and visible in the notebook.
Avoid generic ML questions that could apply to any notebook.

### Final Verdict
Give a short friendly verdict:
- overall quality
- biggest strength
- biggest thing to fix next
- reliability of the current results
- readiness level: Beginner / Improving / Solid / Portfolio-ready
- Briefly summarize how the scores reflect the overall notebook quality and engineering maturity.

Notebook: {safe_notebook_text}
"""     
    return prompt


# =========================
# GEMINI API CALL
# =========================
def call_gemini(prompt):
    log.info(f"Gemini request sent | Prompt length: {len(prompt)} chars")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        log.info(f"Gemini response received | Response length: {len(response.text)} chars")
        return response.text
    except Exception as e:
        log.error(f"Gemini call failed | {type(e).__name__}: {e}")
        raise

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

def render_top_priorities(review_text):
    priorities = extract_top_priorities(review_text)

    if not priorities:
        return

    st.markdown("### 🎯 Top 3 Priorities")
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
    MAX_CHARS = 60000

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

    st.markdown("### File Information")

    st.markdown(f"""
    <div class="info-card">
        <b>Notebook Name:</b> {uploaded_file.name}<br>
        <b>File Type:</b> Jupyter / Colab Notebook (.ipynb)<br>
        <b>File Size:</b> {size} KB
    </div>
""", unsafe_allow_html=True)


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
    <div class="info-card">
        <b>Total Cells:</b> {stats['total_cells']}<br>
        <b>Code Cells:</b> {stats['code_cells']}<br>
        <b>Markdown Cells:</b> {stats['markdown_cells']}<br>
        <b>File Size:</b> {size} KB
    </div>
    """, unsafe_allow_html=True)


    focus_instructions = {
    "Diffusion Model": """
Focus heavily on:
- diffusion pipeline usage
- prompt quality
- scheduler choice
- denoising steps
- generated output quality
- fine-tuning risks if present
""",
    "Transformer / LLM": """
Focus heavily on:
- tokenizer usage
- model loading
- prompt design
- fine-tuning setup if present
- evaluation quality
- hallucination or output validation risks
""",
    "Autoencoder": """
Focus heavily on:
- encoder/decoder architecture
- latent space design
- reconstruction loss
- bottleneck size
- anomaly detection or compression goal
""",
    "GAN": """
Focus heavily on:
- generator and discriminator balance
- training stability
- mode collapse
- generated sample quality
""",
    "Computer Vision": """
Focus heavily on:
- image preprocessing
- augmentation
- CNN architecture
- overfitting risks
- validation accuracy
""",
    "NLP": """
Focus heavily on:
- text preprocessing
- tokenization
- embeddings
- sequence handling
- evaluation metrics
""",
    "Feature Engineering": """
Focus heavily on:
- missing value handling
- encoding choices
- feature scaling
- feature selection
- data leakage risks
- whether transformations happen before or after train/test split
""",
    "Exploratory Data Analysis": """
Focus heavily on:
- visualization quality
- data cleaning
- statistical insights
- feature understanding
- missing value analysis
""",
    "Time Series": """
Focus heavily on:
- date/time handling
- trend and seasonality
- leakage from future data
- rolling features
- forecasting validation
""",
    "Classification": """
Focus heavily on:
- class imbalance
- evaluation metrics
- confusion matrix
- precision and recall
- classification performance
""",
    "Regression": """
Focus heavily on:
- regression metrics
- feature scaling
- residual issues
- overfitting
- regression assumptions
""",
    "Clustering": """
Focus heavily on:
- clustering method choice
- feature scaling
- cluster evaluation
- silhouette score or similar metrics
- interpretability of clusters
"""
}
    dynamic_instruction = focus_instructions.get(notebook_focus, """
    Focus heavily on:
     - notebook objective and whether the goal is clearly defined
     - data quality, missing values, and preprocessing logic
     - feature handling and possible data leakage
     - modeling choices and whether they match the task
     - evaluation reliability, metrics, validation strategy, and reproducibility
     - clarity of conclusions and limitations
    """)

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
                output = """### Top 3 Priorities
1. Add a validation_split parameter to model.fit() since no validation data is currently passed.
2. Set a random seed using np.random.seed() to make results reproducible.
3. Add EarlyStopping callback to prevent overfitting during training.

### Project Summary
This notebook builds a CNN image classifier using TensorFlow. It loads data using image_dataset_from_directory and trains a sequential model with Conv2D layers.

### Evidence Found
- TensorFlow and Keras imported
- image_dataset_from_directory used for data loading
- Conv2D, MaxPooling2D, Dense layers detected
- model.fit() called without validation_split
- No random seed detected

### What Looks Good
- Clean data loading pipeline using image_dataset_from_directory
- Proper use of Conv2D and MaxPooling2D layers
- Model compiled with appropriate loss function

### Mistakes & Bad Practices
- Problem: No validation split
- Evidence: model.fit() called without validation_split or validation_data
- Why it matters: Cannot detect overfitting during training
- How to fix it: Add validation_split=0.2 to model.fit()

### Data & Preprocessing Review
Preprocessing appears minimal. image_dataset_from_directory handles basic loading but no augmentation detected. Not enough information about normalization.

### Model & Training Review
Sequential CNN model detected. No callbacks found. Training runs for fixed epochs with no early stopping.

### Reproducibility Review
- Random seeds: Not found
- Train/test split: Not enough information
- Callbacks: Not found
- Logging: Not found

### Overfitting / Underfitting Analysis
Not enough training metrics found to confidently evaluate overfitting. No validation loss curve visible in the notebook.

### Improvements
Quick wins:
- Add validation_split=0.2 to model.fit()
- Set np.random.seed(42) at the top

Medium improvements:
- Add EarlyStopping and ModelCheckpoint callbacks
- Add image augmentation using RandomFlip and RandomRotation

Advanced improvements:
- Use transfer learning with EfficientNetB0 as base model

### Notebook Scores
- Code Quality: 5 — Basic structure present but missing reproducibility setup
- ML Rigor: 4 — No validation split or callbacks detected
- Experimentation: 3 — No hyperparameter tuning or experiment tracking
- Readability: 6 — Code is readable but lacks markdown explanations

### Technical Questions
1. Why did you choose not to include a validation split in model.fit()?
2. How would you detect overfitting without a validation curve?
3. What augmentation strategy would you apply to this dataset?
4. Why is a random seed important for reproducibility in this notebook?
5. How would you improve this notebook for production deployment?

### Final Verdict
This notebook shows a solid understanding of CNN basics but needs reproducibility improvements. Biggest strength is the clean data pipeline. Biggest fix is adding validation split. Currently Improving level."""
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

            st.success("Analysis complete")
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
            if st.button("Clear Chat"):
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
                    chat_response = f"This is a test response for: '{active_question}'. Gemini is not being called right now."
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