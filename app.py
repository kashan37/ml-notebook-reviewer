import streamlit as st
import nbformat
from google import genai

client = genai.Client()


def load_notebook(uploaded_file):
    notebook = nbformat.read(uploaded_file, as_version=4)
    text = ""

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            text += "[MARKDOWN CELL]\n"
            text += cell.source + "\n\n"
            
        elif cell.cell_type == "code":
            text += "[CODE CELL]\n"
            text += cell.source + "\n\n"

    return text



# st.title("ML Notebook Reviewer")
st.markdown("""
<h1 style="
color: #e6e6e6;
font-size: 46px;
font-weight: 600;
border-left: 5px solid #4facfe;
padding-left: 12px;
">
ML Notebook Reviewer
</h1>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload notebook for analysis (Jupyter / Colab .ipynb),", 
     type=["ipynb"]
    )


if uploaded_file is not None:

    st.success("Upload successful. Time to judge your ML decisions.")
    st.write(uploaded_file.name)

    st.markdown(f"""
### 📁 File Info

📄 **Notebook:** {uploaded_file.name}  
📦 **Type:** Jupyter / Colab Notebook (.ipynb)  
📏 **Size:** {round(uploaded_file.size / 1024, 2)} KB  
""")
    

    notebook_text = load_notebook(uploaded_file)
    st.text_area(
        "Notebook Content",
        notebook_text[:100000],
        height=400)

    if st.button("Analyze Notebook"):
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

        Notebook: {notebook_text} """


        with st.spinner("Analyzing notebook... this may take a few seconds 🤖"):

            response = client.models.generate_content( model="gemini-2.5-flash",
                                                      contents=prompt)
            output = response.text
            st.success("Analysis complete")
            st.markdown("---")
            st.markdown(output)