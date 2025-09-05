# app.py
import streamlit as st
import requests
import json
import re
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.document_loaders import PyPDFLoader

# ------------------------------
# Load API Key
# ------------------------------
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

if not OPENROUTER_KEY:
    st.error("❌ OPENROUTER_KEY not found in .env file!")
    st.stop()

headers = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Smart AI Hub"
}

# ------------------------------
# Utils
# ------------------------------
def or_generate(prompt, model="mistralai/mistral-7b-instruct", max_tokens=300, temperature=0.8):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"]
    return "⚠️ No response from model"

def extract_json_block(text):
    # First try to capture fenced JSON
    fenced = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if fenced:
        return fenced[-1]

    # If no fenced block, try to extract the first {...} JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)

    return "{}"


def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def load_resume(file):
    if file.type == "application/pdf":
        # Save temporarily because PyPDFLoader needs a path
        with open("temp_resume.pdf", "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader("temp_resume.pdf")
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    else:
        st.warning("⚠️ Please upload a PDF file only.")
        return ""
    
# ------------------------------
# Prompts
# ------------------------------
idea_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are a professional startup idea generator. 
    Create a UNIQUE and INNOVATIVE app idea related to {topic}.
    Be original, avoid generic phrases. 
    Return the idea in 2-3 sentences.
    """
)

feature_prompt = PromptTemplate(
    input_variables=["app_idea"],
    template="""
    Based on this app idea: {app_idea},
    list exactly 3 UNIQUE, practical features in bullet points.
    Do NOT repeat the idea, only focus on features.
    """
)

tagline_prompt = PromptTemplate(
    input_variables=["app_idea", "features"],
    template="""
    Write a catchy tagline (max 12 words) for this app:
    Idea: {app_idea}
    Features: {features}
    """
)

resume_prompt = """
You are a professional HR assistant. 
Extract the following details from this resume text if available:
- Full Name
- Email
- Phone
- LinkedIn
- GitHub/Portfolio
- Education
- Work Experience
- Skills

Return in a structured JSON format.
Resume Text:
{resume_text}
"""

rag_prompt = """
You are a helpful assistant. Answer the user question based ONLY on the following document:

Document:
{context}

Question:
{question}
"""

# ------------------------------
# Idea Generator
# ------------------------------
def idea_generator(topic, n=3, model="mistralai/mistral-7b-instruct"):
    ideas = []
    for _ in range(n):
        app_idea = or_generate(idea_prompt.format(topic=topic), model=model, max_tokens=120)
        features = or_generate(feature_prompt.format(app_idea=app_idea), model=model, max_tokens=120)
        tagline = or_generate(tagline_prompt.format(app_idea=app_idea, features=features), model=model, max_tokens=40)
        ideas.append({
            "idea": app_idea.strip(),
            "features": features.strip(),
            "tagline": tagline.strip()
        })
    return ideas

# ------------------------------
# Structured Extraction
# ------------------------------
def structured_extraction(user_input, model="mistralai/mistral-7b-instruct"):
    purchase_schema = ResponseSchema(
        name="purchases",
        description="A list of purchased items with their names and amounts."
    )
    total_schema = ResponseSchema(
        name="total",
        description="The total amount of money spent on the purchases."
    )
    response_schemas = [purchase_schema, total_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """
    You are a smart assistant that extracts purchases from a user's description of their day.
    Ignore returned/refunded items when calculating totals.

    Extract all purchased items and the amount spent on each, then calculate the total amount spent.

    Respond ONLY in JSON format as follows:
    {format_instructions}

    Now extract from the following input:
    "{user_input}"
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["user_input", "format_instructions"]
    ).format(user_input=user_input, format_instructions=format_instructions)

    response = or_generate(prompt, model=model, max_tokens=400)
    json_text = extract_json_block(response)
    return output_parser.parse(json_text)

# ------------------------------
# Resume Extraction Schema
# ------------------------------
full_name_schema = ResponseSchema(name="full_name", description="The candidate's full name.")
email_schema = ResponseSchema(name="email", description="The candidate's email address.")
linkedin_schema = ResponseSchema(name="linkedin", description="The candidate's LinkedIn URL if available.")
phone_schema = ResponseSchema(name="phone", description="The candidate's phone number if available.")
education_schema = ResponseSchema(
    name="education",
    description="A list of education entries, each with degree, institution, and year."
)
skills_schema = ResponseSchema(name="skills", description="A list of the candidate's skills.")
experience_schema = ResponseSchema(
    name="experience",
    description="A list of experience entries, each with role, company, and years."
)

resume_response_schemas = [
    full_name_schema, email_schema, phone_schema, linkedin_schema,
    education_schema, skills_schema, experience_schema
]
resume_output_parser = StructuredOutputParser.from_response_schemas(resume_response_schemas)
resume_format_instructions = resume_output_parser.get_format_instructions()

resume_parser_template = """
You are an expert HR assistant that extracts structured candidate profile data from resumes.

Extract the following fields:
- full_name (string)
- email (string)
- phone (string)
- linkedin (string)
- education (list of objects, each object has: degree, institution, year)
- skills (list of strings)
- experience (list of objects, each object has: role, company, years)

Respond ONLY in JSON format as follows:
{format_instructions}

Now extract from the following input:
"{resume_text}"
"""

def resume_extraction(resume_text, model="mistralai/mistral-7b-instruct"):
    prompt = resume_parser_template.format(
        resume_text=resume_text[:5000],
        format_instructions=resume_format_instructions
    )
    response = or_generate(prompt, model=model, max_tokens=800)
    json_text = extract_json_block(response)
    try:
        return resume_output_parser.parse(json_text)
    except:
        return {"raw_response": response}

# ------------------------------
# Resume Job Match
# ------------------------------
job_match_prompt = """
You are an expert career coach and recruiter.

Compare the following resume data with the provided job description.
Evaluate and return ONLY a valid JSON object with this structure:

{{
  "score": (integer 0–100),
  "strengths": [list of strings],
  "weaknesses": [list of strings],
  "recommendations": [list of strings]
}}

Resume:
{resume_data}

Job Description:
{job_description}

IMPORTANT: Return ONLY valid JSON. No explanation, no markdown, no commentary.
"""



def resume_job_match(resume_data, job_description, model="mistralai/mistral-7b-instruct"):
    prompt = job_match_prompt.format(
        resume_data=json.dumps(resume_data, indent=2),
        job_description=job_description
    )
    response = or_generate(prompt, model=model, max_tokens=700)
    json_text = extract_json_block(response)

    try:
        return json.loads(json_text)
    except Exception as e:
        # fallback: return raw text so you can debug
        return {"raw_response": response, "error": str(e)}



# ------------------------------
# RAG Q&A
# ------------------------------
def rag_qa(doc_text, question, model="mistralai/mistral-7b-instruct"):
    prompt = rag_prompt.format(context=doc_text[:4000], question=question)
    response = or_generate(prompt, model=model, max_tokens=500)
    return response

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Smart AI Hub", page_icon="🤖", layout="wide")
st.title("📚 Smart AI Knowledge & Productivity Hub")

# Sidebar - Model picker
model_choice = st.sidebar.selectbox(
    "🧠 Choose Model",
    ["mistralai/mistral-7b-instruct", "anthropic/claude-3-sonnet", "openai/gpt-4o", "openai/gpt-4o-mini"]
)

mode = st.sidebar.radio(
    "🔍 Select a Tool",
    ["💡 Idea Generator", "🧾 Expense Extraction", "📄 Resume Extraction", "📚 PDF Q&A (RAG)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Author:** Abdelrahman Ahmed")
st.sidebar.caption("AI Engineer | Data Scientist")

# ------------------------------
# Modes
# ------------------------------
if mode == "💡 Idea Generator":
    st.header("💡 Creative App Idea Generator")
    topic = st.text_input("Enter a topic (e.g., health, education, finance)")
    col1, col2 = st.columns(2)

    if col1.button("Generate One Idea"):
        ideas = idea_generator(topic, n=1, model=model_choice)
        idea = ideas[0]
        st.subheader("App Idea")
        st.write(idea["idea"])
        st.subheader("Features")
        st.write(idea["features"])
        st.subheader("Tagline")
        st.success(idea["tagline"])

    if col2.button("Generate 3 Ideas"):
        ideas = idea_generator(topic, n=3, model=model_choice)
        for i, idea in enumerate(ideas, 1):
            st.markdown(f"### 🚀 Idea {i}:")
            st.write(idea["idea"])
            st.write(idea["features"])
            st.success(idea["tagline"])
        st.download_button(
            "⬇️ Download Ideas (JSON)",
            data=json.dumps(ideas, indent=2),
            file_name="ideas.json",
            mime="application/json"
        )

elif mode == "🧾 Expense Extraction":
    st.header("🧾 Structured Info Extraction (Expenses)")
    user_input = st.text_area("Describe your day with purchases")
    if st.button("Extract"):
        result = structured_extraction(user_input, model=model_choice)
        df = pd.DataFrame(result["purchases"])
        st.dataframe(df)
        st.metric("💰 Total Spent", f"${result['total']}")
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), "expenses.csv")


elif mode == "📄 Resume Extraction":
    st.header("📄 Resume Info Extraction")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file:
        if st.button("Extract"):
            resume_text = load_resume(uploaded_file)
            result = resume_extraction(resume_text, model=model_choice)
            st.session_state["resume_data"] = result  # ✅ save in session_state

        # ------------------------------
        # Display extracted resume info (if available)
        # ------------------------------
        if "resume_data" in st.session_state:
            result = st.session_state["resume_data"]

            st.subheader("📌 Candidate Profile")
            st.write(f"**Name:** {result.get('full_name','')}")
            st.write(f"**Email:** {result.get('email','')}")
            st.write(f"**Phone:** {result.get('phone','')}")
            st.write(f"**LinkedIn:** {result.get('linkedin','')}")

            if result.get("skills"):
                st.subheader("🛠 Skills")
                st.write(", ".join(result["skills"]))

            if result.get("education"):
                st.subheader("🎓 Education")
                df_edu = pd.DataFrame(result["education"])
                st.table(df_edu)

            if result.get("experience"):
                st.subheader("💼 Experience")
                df_exp = pd.DataFrame(result["experience"])
                st.table(df_exp)

            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(result, indent=2),
                file_name="resume_data.json",
                mime="application/json"
            )

            # ------------------------------
            # Job Matching Section
            # ------------------------------
            st.markdown("---")
            st.subheader("📊 Job Matching")
            job_desc = st.text_area("Paste Job Description")

            if st.button("Compare with Job Description"):
                if job_desc.strip():
                    match_result = resume_job_match(result, job_desc, model=model_choice)
                    st.metric("Match Score", f"{match_result.get('score', 'N/A')}/100")

                    st.write("✅ **Strengths:**")
                    st.write("\n".join(match_result.get("strengths", [])))

                    st.write("⚠️ **Weaknesses:**")
                    st.write("\n".join(match_result.get("weaknesses", [])))

                    st.write("💡 **Recommendations:**")
                    st.write("\n".join(match_result.get("recommendations", [])))
                else:
                    st.warning("⚠️ Please paste a job description before comparing.")


elif mode == "📚 PDF Q&A (RAG)":
    st.header("📚 Ask Questions About Your PDF")
    uploaded_doc = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_doc:
        doc_text = extract_text_from_pdf(uploaded_doc)
        question = st.text_input("Ask a question about the document")
        if st.button("Ask"):
            answer = rag_qa(doc_text, question, model=model_choice)
            st.write("💡 Answer:", answer)

