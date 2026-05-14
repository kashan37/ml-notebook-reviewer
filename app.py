import streamlit as st
import nbformat
from google import genai
from google.genai import errors
import re
import html
import time
# =========================
# IMPORTS & SETUP
# =========================

client = genai.Client()
DEBUG_MODE = False


# ===============================
# CORE NOTEBOOK PARSING FUNCTIONS
# ===============================
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
                stream_text = output.get("text", "")

                if isinstance(stream_text, list):
                    stream_text = "\n".join(stream_text)

                output_sections.append(str(stream_text))

            elif output.get("output_type") in ["execute_result", "display_data"]:
                data = output.get("data", {})

                if "text/plain" in data:
                    text_output = data["text/plain"]

                    if isinstance(text_output, list):
                        text_output = "\n".join(text_output)

                    output_sections.append(str(text_output))

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
.review-card {
    background: linear-gradient(145deg, #151821, #0f1117);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    min-height: 145px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);

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
    font-size: 24px;
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
    background: linear-gradient(145deg, #151821, #0f1117);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 18px 20px;
    color: #f5f5f7;
    font-size: 15px;
    line-height: 1.7;
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
</style>
""", unsafe_allow_html=True)


# st.title("ML Notebook Reviewer")
st.markdown("""
<h1 style="
color: #e6e6e6;
font-size: 46px;
font-weight: 600;
border-left: 5px solid #4facfe;
padding-left: 12px;
">
Notebook Lens
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
color: #b8b8b8;
font-size: 18px;
margin-top: -10px;
margin-left: 17px;
">
AI-powered reviews for Jupyter and Colab ML notebooks.
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload notebook for analysis (Jupyter / Colab .ipynb),", 
     type=["ipynb"]
    )

def count_keywords(text, keywords):
    return sum(1 for keyword in keywords if keyword in text)


# ==============================
# NOTEBOOK TYPE + TASK DETECTION
# ==============================
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
            "binary_crossentropy"
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
        "Exploratory Data Analysis": 3,
        "Time Series": 2,
        "Classification": 2,
        "Regression": 2,
        "Clustering": 2
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

def score_notebook(notebook_text):
    # Placeholder for V2 scoring engine
    scores = {
        "Code Quality": None,
        "ML Rigor": None,
        "Experimentation": None,
        "Readability": None
    }
    return scores


# ==============================
# REPRODUCIBILITY ANALYSIS
# =============================
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
# GEMINI API CALL
# =========================
def call_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

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


def combine_sections(review_text, headings): #NOT IN USE CURRENTLY
    sections = []
    for heading in headings:
        section = extract_section(review_text, heading)
        if section:
            sections.append(section)
    if not sections:
        return "_No content found for this tab._"
    return "\n\n".join(sections)

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
# DASHBOARD CARD HELPERS
# =========================
def clean_preview_text(text, max_chars=260):
    if not text:
        return "No clear signal found yet."

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"[*_`>]", "", text)
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


# ============================
# STREAMLIT UI + APP LOGIC
# ============================
if uploaded_file is not None:

    notebook = nbformat.read(uploaded_file, as_version=4)
    stats = get_notebook_stats(notebook)
    size = get_file_size(uploaded_file)

    notebook_text = load_notebook(notebook)
    notebook_focus = detect_notebook_focus(notebook_text)
    reproducibility_context = detect_reproducibility_signals(notebook_text)

    st.success("Upload successful. Ready for analysis.")

    # FILE INFO CARD
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

**{notebook_focus}**
""")

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

    if st.button("Analyze Notebook"):
        MAX_CHARS = 60000
        safe_notebook_text = notebook_text[:MAX_CHARS]

        # =============================
        # PROMPT BUILDER (GEMINI INPUT)
        # =============================

        def build_prompt(dynamic_instruction, reproducibility_context, safe_notebook_text):
            prompt = f"""
You are a senior Machine learning engineer and technical reviewer evaluating a Jupyter notebook..

Your goal is to give a helpful, friendly, practical review that is easy to read.
Be honest about problems, always roast a little bit and don't be boring.
Avoid generic advice. Tie every point to something visible in the notebook when possible.

Your job is to:
- Be precise and technical
- Avoid vague advice
- Only comment based on evidence in the notebook
- If something is unclear or missing, explicitly say: "Not enough information"
- Reference specific notebook evidence whenever possible
- Mention specific functions, models, preprocessing steps, metrics, libraries, or outputs seen in the notebook
- Quote short relevant snippets or behaviors from the notebook when useful
- Do not make generic ML comments unless supported by notebook evidence


Do NOT hallucinate missing components.
If evidence for a claim is weak or missing, clearly state that the notebook does not provide enough evidence.
Only suggest code that preserves data integrity assumptions.
If dataset structure is unclear, first recommend validation or inspection steps before transformations.
Do not assume ordering, pairing, or schema correctness unless explicitly shown in the notebook evidence.
Do not give high scores unless strong notebook evidence supports them.
Avoid inflated scoring.
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

### Improvements
Give clear, prioritized improvements.

Label them as:
- Quick wins
- Medium improvements
- Advanced improvements

For each improvement, explain:
- what to change
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

### Technical Questions
Generate 5-7 questions that would come up in a professional ML code review or portfolio review.
Questions should test the author’s reasoning about data preprocessing, modeling choices, metrics, validation, limitations, and deployment readiness.
Each question must be tied to something visible in the notebook.
Avoid generic ML questions.

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
            output = call_gemini(prompt)

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
            st.markdown("---")
            
            render_review_dashboard(output, notebook_focus, stats, size)

            summary_tab, mistakes_tab, improvements_tab, questions_tab = st.tabs([
                "Summary",
                "Mistakes",
                "Improvements",
                "Technical Questions"
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

        except errors.ClientError as e:
            progress_bar.empty()
            status_text.empty()

            error_text = str(e)

            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                st.error("Gemini quota limit reached.")
                st.caption("You hit the current request limit. Please wait and try again later.")
            else:
                st.error("The AI review service could not complete the request.")
                st.caption("Please try again in a moment.")

                if DEBUG_MODE:
                    st.write("Error type:", type(e).__name__)
                    st.exception(e)

        except errors.ServerError as e:
            progress_bar.empty()
            status_text.empty()

            st.error("The AI review service is temporarily busy.")
            st.caption("The model is experiencing high demand. Please wait a moment and try again.")

            if DEBUG_MODE:
                st.write("Error type:", type(e).__name__)
                st.exception(e)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()

            st.error("Something went wrong while preparing the review.")
            st.caption("Please try again. If the issue continues, try a smaller notebook or clear very large output cells.")

            if DEBUG_MODE:
                st.write("Error type:", type(e).__name__)
                st.exception(e)
