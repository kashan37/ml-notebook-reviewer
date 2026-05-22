
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

DEFAULT_FOCUS_INSTRUCTION = """
    Focus heavily on:
     - notebook objective and whether the goal is clearly defined
     - data quality, missing values, and preprocessing logic
     - feature handling and possible data leakage
     - modeling choices and whether they match the task
     - evaluation reliability, metrics, validation strategy, and reproducibility
     - clarity of conclusions and limitations
    """