import streamlit as st

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



def detect_reproducibility_dict(notebook_text):
    """
    Returns raw boolean dict of reproducibility signals.
    Used by comparison engine — needs booleans, not formatted strings.
    """
    text = notebook_text.lower()
    return {
        "random_seeds": any(keyword in text for keyword in [
            "random_state", "np.random.seed", "random.seed",
            "torch.manual_seed", "tf.random.set_seed", "seed="
        ]),
        "train_test_split": any(keyword in text for keyword in [
            "train_test_split", "validation_split", "stratifiedkfold",
            "kfold", "cross_val_score", "train_df", "test_df",
            "x_train", "x_test", "y_train", "y_test"
        ]),
        "callbacks": any(keyword in text for keyword in [
            "earlystopping", "modelcheckpoint", "reducelronplateau",
            "tensorboard", "callbacks"
        ]),
        "logging": any(keyword in text for keyword in [
            "mlflow", "wandb", "tensorboard", "comet", "neptune",
            "history.history", "training log", "logs"
        ])
    }

# ==============================
# REPRODUCIBILITY ANALYSIS
# =============================
@st.cache_data
def detect_reproducibility_signals(notebook_text):
    signals_dict = detect_reproducibility_dict(notebook_text)

    label_map = {
        "random_seeds": "Random seeds",
        "train_test_split": "Train/test split",
        "callbacks": "Callbacks",
        "logging": "Logging / experiment tracking",
    }

    result = ["Reproducibility signals detected in the notebook:"]
    for key, label in label_map.items():
        status = "Found" if signals_dict[key] else "Not found"
        result.append(f"- {label}: {status}")

    return "\n".join(result)


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

