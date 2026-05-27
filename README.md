# Notebook Lens

AI powered reviews for Jupyter and Colab machine learning notebooks.

Upload any `.ipynb` notebook and get a structured technical review in seconds,focused on code quality, ML practices, training setup, reproducibility, overfitting risks, and engineering decisions.

Notebook Lens was built to help ML students, Kaggle practitioners, bootcamp graduates, and developers get honest feedback on their notebooks before sharing projects publicly.

## Features

### Automatic Notebook Review

Upload a notebook and receive a structured ML-focused review without writing prompts.

The system analyzes:

* preprocessing pipelines
* training setup
* reproducibility practices
* model configuration
* overfitting risks
* engineering quality
* notebook structure

---

### Actionable Priorities

Instead of overwhelming users with giant reports, Notebook Lens highlights the most important improvements first.

Focused feedback > 4,000 words of generic critique.

---

### Chat With Your Notebook

Ask follow-up questions after the review.

Examples:

* "Why does this model look overfit?"
* "What should I improve first?"
* "Is my validation strategy reliable?"
* "What interview questions could come from this project?"

Responses are grounded only in notebook evidence and generated review context.

## Current Status

### V1 Completed

Version 1 focuses on:

* structured notebook reviews
* grounded conversational analysis
* hallucination reduction
* dynamic review generation
* notebook type detection
* export support
* interactive review dashboard

## V2 (In Progress)

Version 2 expands Notebook Lens into a deeper AI engineering system with:

* notebook comparison engine
* experiment analysis
* visualization intelligence
* contextual notebook understanding
* evaluation + reliability framework
* embeddings and retrieval systems (RAG)
* grounded multi-turn notebook reasoning

The goal is to move beyond "AI summaries" toward a genuinely useful ML engineering assistant.

---

## Tech Stack

* Python
* Streamlit
* Gemini API
* nbformat
* Markdown/PDF export pipeline

---

## Why I Built This

A lot of notebook reviews online are either:

* extremely shallow
* generic AI summaries
* or completely disconnected from actual ML engineering problems

I wanted to build something that behaves more like a technically opinionated reviewer instead of a chatbot that says:

> "Great work! Consider tuning hyperparameters."

## Roadmap

Planned future directions include:

* experiment tracking awareness
* notebook memory/retrieval
* smarter grounded reasoning
* adaptive review strictness
* better hallucination evaluation
* deeper training diagnostics

---

## Running Locally

```bash
git clone https://github.com/kashan37/notebook-lens.git
cd notebook-lens
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Status

Actively under development.

---

## License

MIT License
