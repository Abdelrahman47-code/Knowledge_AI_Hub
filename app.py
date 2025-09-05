import streamlit as st
from core.config import OPENROUTER_KEY
from features import idea_generator, expense_parser, resume_parser, job_matcher, rag_qa

# Basic page config + styling
st.set_page_config(page_title="Smart AI Hub", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1100px;}
    .stMetric .stMetricDelta {direction:ltr}
    </style>
    """,
    unsafe_allow_html=True,
)

if not OPENROUTER_KEY:
    st.error("❌ OPENROUTER_KEY not found in .env file! Please add it and restart.")
    st.stop()

st.title("📚 Smart AI Knowledge & Productivity Hub")

# Sidebar model picker
st.sidebar.markdown("### 🧠 Model")
model_choice = st.sidebar.selectbox(
    "Model selection",
    ["Auto (detect language)", "mistralai/mistral-7b-instruct", "openai/gpt-4o-mini", "qwen/Qwen2-72B-Instruct", "anthropic/claude-3-sonnet"],
    index=0
)

mode = st.sidebar.radio(
    "🔍 Select a Tool",
    ["💡 Idea Generator", "🧾 Expense Extraction", "📄 Resume Extraction", "📊 Resume-Job Match", "📚 PDF Q&A (RAG)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    👨‍💻 **Author:** Abdelrahman Eldaba  
    🔗 [LinkedIn](https://www.linkedin.com/in/abdelrahmaneldaba/)  
    AI Engineer - Data Scientist
    """
)

# Routes
if mode == "💡 Idea Generator":
    idea_generator.run(model_choice)

elif mode == "🧾 Expense Extraction":
    expense_parser.run(model_choice)

elif mode == "📄 Resume Extraction":
    resume_parser.run(model_choice)

elif mode == "📊 Resume-Job Match":
    job_matcher.run(model_choice)

elif mode == "📚 PDF Q&A (RAG)":
    rag_qa.run(model_choice)
