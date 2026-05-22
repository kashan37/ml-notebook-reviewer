import streamlit as st

def inject_styles():
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
    content: "Upload your notebook (.ipynb)";
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
    margin-top: 0;
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
    padding: 0 0 80px 0 !important;
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
     