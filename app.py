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


def detect_notebook_type(notebook_text):
    text = notebook_text.lower()

    categories = {
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
        "Natural Language Processing": [
            "tokenizer",
            "bert",
            "transformer",
            "word2vec",
            "tfidfvectorizer",
            "countvectorizer",
            "nltk",
            "spacy",
            "word_tokenize",
            "stemming",
            "lemmatization"
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
            "grayscale",
            "rgb",
            "resize"
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
        ]
    }

    scores = {
        category: count_keywords(text, keywords)
        for category, keywords in categories.items()
    }

    if scores["Natural Language Processing"] >= 2:
        return "Natural Language Processing"

    if scores["Computer Vision"] >= 2:
        return "Computer Vision"

    if scores["Time Series"] >= 2:
        return "Time Series"

    if scores["Feature Engineering"] >= 3:
        return "Feature Engineering"

    if scores["Exploratory Data Analysis"] >= 3:
        return "Exploratory Data Analysis"

    if any(keyword in text for keyword in [
        "sklearn",
        "train_test_split",
        ".fit(",
        ".predict(",
        "randomforest",
        "xgboost",
        "lightgbm"
    ]):
        return "Tabular ML / General ML"

    return "General Notebook"





if uploaded_file is not None:

    def detect_ml_task(notebook_text):
        text = notebook_text.lower()

        classification_keywords = [
            "classification_report",
            "accuracy_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "confusion_matrix",
            "logisticregression",
            "randomforestclassifier",
            "svc",
            "categorical_crossentropy",
            "binary_crossentropy"
        ]

        regression_keywords = [
            "mean_squared_error",
            "mean_absolute_error",
            "r2_score",
            "linearregression",
            "randomforestregressor",
            "mae",
            "mse",
            "rmse"
        ]

        clustering_keywords = [
            "kmeans",
            "dbscan",
            "agglomerativeclustering",
            "silhouette_score",
            "clustering"
        ]

        forecasting_keywords = [
            "forecast",
            "arima",
            "sarima",
            "prophet",
            "seasonality",
            "time series prediction"
        ]

        gan_keywords = [
            "discriminator",
            "adversarial",
            "gan",
            "generator loss",
            "discriminator loss"
        ]

        scores = {
            "Classification": count_keywords(text, classification_keywords),
            "Regression": count_keywords(text, regression_keywords),
            "Clustering": count_keywords(text, clustering_keywords),
            "Forecasting": count_keywords(text, forecasting_keywords),
            "GAN": count_keywords(text, gan_keywords)
        }

        best_task = max(scores, key=scores.get)

        if scores[best_task] >= 2:
            return best_task

        return "No clear ML task detected"




    notebook = nbformat.read(uploaded_file, as_version=4)
    stats = get_notebook_stats(notebook)
    size = get_file_size(uploaded_file)
    notebook_text = load_notebook(notebook)
    notebook_type = detect_notebook_type(notebook_text)
    ml_task = detect_ml_task(notebook_text)

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

    st.markdown(f"""
### Detected Notebook Focus

**{notebook_type}**
""")
    
    st.markdown(f"""
### ML Task Detected

**{ml_task}**
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


    dynamic_instruction = ""
    
    if notebook_type == "Computer Vision":
        dynamic_instruction = """
    Focus heavily on:
    - image preprocessing
    - augmentation
    - CNN architecture
    - overfitting risks
    - validation accuracy
    """

    elif notebook_type == "Natural Language Processing":
            dynamic_instruction = """
        Focus heavily on:
        - tokenization
        - embeddings
        - sequence handling
        - NLP preprocessing
        - transformer usage
        """

    elif notebook_type == "Regression":
        dynamic_instruction = """
    Focus heavily on:
    - regression metrics
    - feature scaling
    - residual issues
    - overfitting
    - regression assumptions
    """

    elif notebook_type == "Classification":
        dynamic_instruction = """
    Focus heavily on:
    - class imbalance
    - evaluation metrics
    - confusion matrix
    - precision/recall
    - classification performance
    """

    elif notebook_type == "Exploratory Data Analysis":
        dynamic_instruction = """
    Focus heavily on:
    - visualization quality
    - data cleaning
    - statistical insights
    - feature understanding
    - missing value analysis
    """
        
    elif notebook_type == "Feature Engineering":
        dynamic_instruction = """
    Focus heavily on:
    - missing value handling
    - encoding choices
    - feature scaling
    - feature selection
    - data leakage risks
    - whether transformations are applied before or after train/test split
    """
        
    elif notebook_type == "Tabular ML / General ML":
        dynamic_instruction = """
    Focus heavily on:
    - data preprocessing
    - train/test split
    - feature handling
    - model evaluation
    - data leakage risks
"""

    elif ml_task == "No clear ML task detected":
        task_instruction = """
    No clear final ML task was detected.
    Focus on whether the notebook is mainly exploratory, preprocessing-focused, or incomplete.
    Do not pretend there is a classification or regression task unless the notebook clearly shows it.
    """
    
    
    task_instruction = ""

    if ml_task == "Classification":
        task_instruction = """
Additional ML task focus:
- precision / recall
- confusion matrix
- class imbalance
- F1-score
"""

    elif ml_task == "Regression":
        task_instruction = """
Additional ML task focus:
- MAE / RMSE / R2
- residual analysis
- prediction error distribution
"""

    elif ml_task == "GAN":
        task_instruction = """
Additional ML task focus:
- generator vs discriminator balance
- mode collapse
- training stability
"""

    elif ml_task == "NLP":
        task_instruction = """
Additional ML task focus:
- tokenization quality
- embeddings
- sequence handling
"""

    elif ml_task == "Computer Vision":
        task_instruction = """
Additional ML task focus:
- augmentation
- CNN structure
- overfitting in images
"""


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
{dynamic_instruction}
{task_instruction}
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