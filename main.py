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
from langchain.document_loaders import PyPDFLoader
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ==============================
# Config & Look
# ==============================
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

# ==============================
# Load API Key
# ==============================
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
if not OPENROUTER_KEY:
    st.error("❌ OPENROUTER_KEY not found in .env file!")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "HTTP-Referer": "http://localhost:8501",  # replace with your domain if deployed
    "X-Title": "Smart AI Hub"
}

# ==============================
# Language detection & model choice
# ==============================
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return "ar" if lang == "ar" else "en"
    except Exception:
        return "en"

# ==============================
# Rendering helper (RTL for Arabic)
# ==============================
def render_answer(text: str):
    """Render text RTL if Arabic detected, else normal."""
    lang = detect_language(text)
    if lang == "ar":
        st.markdown(
            f"<div style='direction: rtl; text-align: right;'>{text}</div>",
            unsafe_allow_html=True
        )
    else:
        st.write(text)


# Preferred models by language (all via OpenRouter)
LANG_MODEL_PREFERENCE = {
    "ar": ["openai/gpt-4o-mini", "qwen/Qwen2-72B-Instruct"],  # Arabic-strong
    "en": ["mistralai/mistral-7b-instruct", "openai/gpt-4o-mini"]
}

def choose_model_by_text(text: str, user_override: str | None) -> str:
    """
    If user selects a specific model (not Auto), honor it.
    Otherwise detect language and pick the first preferred available.
    """
    if user_override and user_override != "Auto (detect language)":
        return user_override

    lang = detect_language(text)
    # just return the first preferred (we can’t probe availability here)
    return LANG_MODEL_PREFERENCE.get(lang, LANG_MODEL_PREFERENCE["en"])[0]

# ==============================
# HTTP utils
# ==============================
def or_generate(prompt, model="mistralai/mistral-7b-instruct", max_tokens=500, temperature=0.4):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]
    return ""

# ==============================
# Text / JSON helpers
# ==============================
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
EASTERN_PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
WESTERN = "0123456789"
ARABIC_MAP = {ord(a): w for a, w in zip(ARABIC_INDIC, WESTERN)}
PERSIAN_MAP = {ord(a): w for a, w in zip(EASTERN_PERSIAN, WESTERN)}

def normalize_digits(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.translate(ARABIC_MAP).translate(PERSIAN_MAP)

def extract_json_block(text: str) -> str:
    """
    1) Try fenced ```json ... ```
    2) Else take the first {...} block
    3) Else return "{}"
    """
    if not text:
        return "{}"
    fenced = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0).strip() if m else "{}"

def safe_json_loads(s: str):
    """
    Try to parse JSON; on failure, attempt minimal cleanup.
    """
    try:
        return json.loads(s)
    except Exception:
        # remove trailing text after last }
        last = s.rfind("}")
        if last != -1:
            try:
                return json.loads(s[: last + 1])
            except Exception:
                pass
    return None

# ==============================
# PDF utils
# ==============================
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    buf = []
    for page in pdf.pages:
        page_text = page.extract_text() or ""
        buf.append(page_text)
    return "\n".join(buf)

def load_resume(file):
    if file.type == "application/pdf":
        with open("temp_resume.pdf", "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader("temp_resume.pdf")
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    else:
        st.warning("⚠️ Please upload a PDF file only.")
        return ""

# ==============================
# Prompts (Arabic-aware)
# ==============================
idea_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
أنت مولّد أفكار تطبيقات احترافي. الإدخال قد يكون بالعربية أو الإنجليزية.
القواعد:
- إذا كان الإدخال بالعربية → أجب بالعربية.
- إذا كان الإدخال بالإنجليزية → أجب بالإنجليزية.
- أعطِ فكرة مبتكرة (غير إنشائية) خلال 2–3 جُمل فقط.

Topic / الموضوع: {topic}
"""
)

feature_prompt = PromptTemplate(
    input_variables=["app_idea"],
    template="""
The input can be Arabic or English. Match the language of the input idea.
List exactly 3 UNIQUE, practical features (bulleted). Do not repeat the idea.

Idea:
{app_idea}
"""
)

tagline_prompt = PromptTemplate(
    input_variables=["app_idea", "features"],
    template="""
Write a catchy tagline (≤ 12 words). Match the language (Arabic/English) of the idea.

Idea:
{app_idea}

Features:
{features}
"""
)

# RAG answer language matches the question language
rag_prompt = """
You are a helpful assistant. Answer ONLY from this document.
If the question is in Arabic, answer in Arabic. If English, answer in English.

Document:
{context}

Question:
{question}
"""

# ==============================
# Structured extraction (Expenses)
# ==============================
def structured_extraction(user_input: str, user_model_choice: str):
    purchase_schema = ResponseSchema(
        name="purchases",
        description="List of purchased items with fields: name, amount (number), currency (string, optional)"
    )
    total_schema = ResponseSchema(
        name="total",
        description="Total amount of money spent (number)"
    )
    response_schemas = [purchase_schema, total_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """
You are a smart assistant that extracts purchases from a description of someone’s day.
Input may be Arabic or English. Respond in JSON ONLY.

Rules:
- Extract (name, amount, [currency]) for each purchase.
- Convert Arabic-Indic numbers (e.g., ٥٠, ۳۵) to standard digits.
- Ignore returned/refunded items when calculating totals.

Format:
{format_instructions}

Input:
"{user_input}"
"""
    prompt = PromptTemplate(
        template=template, input_variables=["user_input", "format_instructions"]
    ).format(user_input=user_input, format_instructions=format_instructions)

    model = choose_model_by_text(user_input, user_model_choice)
    raw = or_generate(prompt, model=model, max_tokens=600)
    json_text = extract_json_block(raw)
    try:
        data = output_parser.parse(json_text)
    except Exception:
        data = safe_json_loads(json_text) or {"raw_response": raw}

    # post-normalize amounts if they came back as strings with Arabic digits
    if isinstance(data, dict) and "purchases" in data and isinstance(data["purchases"], list):
        for p in data["purchases"]:
            if isinstance(p, dict) and "amount" in p:
                if isinstance(p["amount"], str):
                    s = normalize_digits(p["amount"])
                    try:
                        p["amount"] = float(re.sub(r"[^\d.]+", "", s))
                    except Exception:
                        pass
    if isinstance(data, dict) and "total" in data and isinstance(data["total"], str):
        s = normalize_digits(data["total"])
        try:
            data["total"] = float(re.sub(r"[^\d.]+", "", s))
        except Exception:
            pass
    return data

# ==============================
# Resume extraction (Structured JSON)
# ==============================
# Schemas
full_name_schema = ResponseSchema(name="full_name", description="Candidate full name")
email_schema = ResponseSchema(name="email", description="Email address")
linkedin_schema = ResponseSchema(name="linkedin", description="LinkedIn URL")
phone_schema = ResponseSchema(name="phone", description="Phone number")
education_schema = ResponseSchema(
    name="education", description="List of objects: {degree, institution, year}"
)
skills_schema = ResponseSchema(name="skills", description="List of skills (strings)")
experience_schema = ResponseSchema(
    name="experience", description="List of objects: {role, company, years}"
)
resume_response_schemas = [
    full_name_schema, email_schema, phone_schema, linkedin_schema,
    education_schema, skills_schema, experience_schema
]
resume_output_parser = StructuredOutputParser.from_response_schemas(resume_response_schemas)
resume_format_instructions = resume_output_parser.get_format_instructions()

resume_parser_template = """
You are an expert HR assistant. The resume can be Arabic or English.

Extract the following fields and return ONLY JSON:
- full_name (string)
- email (string)
- phone (string)
- linkedin (string)
- education (list of {degree, institution, year})
- skills (list of strings)
- experience (list of {role, company, years})

Format:
{format_instructions}

Resume Text:
"{resume_text}"
"""

def resume_extraction(resume_text: str, user_model_choice: str):
    model = choose_model_by_text(resume_text, user_model_choice)
    prompt = resume_parser_template.format(
        resume_text=resume_text[:7000],
        format_instructions=resume_format_instructions
    )
    raw = or_generate(prompt, model=model, max_tokens=900)
    json_text = extract_json_block(raw)
    try:
        return resume_output_parser.parse(json_text)
    except Exception:
        # fall back to raw JSON if possible
        fallback = safe_json_loads(json_text)
        return fallback if fallback else {"raw_response": raw}

# ==============================
# Resume ↔ Job match (strict JSON)
# ==============================
job_match_prompt = """
You are an expert recruiter.

Compare the resume data with the job description and return ONLY a valid JSON object:
{
  "score": <integer 0-100>,
  "strengths": [strings],
  "weaknesses": [strings],
  "recommendations": [strings]
}

Resume:
{resume_data}

Job Description:
{job_description}

IMPORTANT:
- Return ONLY JSON. No markdown, no commentary.
- If inputs are Arabic, use Arabic in lists. The JSON keys remain in English.
"""

def resume_job_match(resume_data, job_description: str, user_model_choice: str):
    # detect language preference from job_description primarily
    model = choose_model_by_text(job_description or json.dumps(resume_data), user_model_choice)
    prompt = job_match_prompt.format(
        resume_data=json.dumps(resume_data, ensure_ascii=False, indent=2),
        job_description=job_description
    )
    raw = or_generate(prompt, model=model, max_tokens=700)
    json_text = extract_json_block(raw)
    parsed = safe_json_loads(json_text)
    return parsed if parsed else {"raw_response": raw}

# ==============================
# RAG Q&A
# ==============================
def chunk_document(text: str, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

def find_relevant_chunks(chunks, question, top_k=3):
    """Use TF-IDF similarity to select top chunks."""
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    doc_vecs = vectorizer.transform(chunks)
    q_vec = vectorizer.transform([question])
    sims = (doc_vecs @ q_vec.T).toarray().ravel()
    top_idx = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_idx]

def rag_qa(doc_text: str, question: str, user_model_choice: str):
    chunks = chunk_document(doc_text)
    relevant = find_relevant_chunks(chunks, question)
    context = "\n\n".join(relevant)

    model = choose_model_by_text(question, user_model_choice)
    prompt = rag_prompt.format(context=context, question=question)
    return or_generate(prompt, model=model, max_tokens=600)


# ==============================
# UI
# ==============================
st.title("📚 Smart AI Knowledge & Productivity Hub")

# Sidebar
st.sidebar.markdown("### 🧠 Model")
model_choice = st.sidebar.selectbox(
    "Model selection",
    ["Auto (detect language)", "mistralai/mistral-7b-instruct", "openai/gpt-4o-mini", "qwen/Qwen2-72B-Instruct", "anthropic/claude-3-sonnet"],
    index=0
)

mode = st.sidebar.radio(
    "🔍 Select a Tool",
    ["💡 Idea Generator", "🧾 Expense Extraction", "📄 Resume Extraction", "📚 PDF Q&A (RAG)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Author:** Abdelrahman Ahmed")
st.sidebar.caption("AI Engineer | Data Scientist")

# ==============================
# Modes
# ==============================
if mode == "💡 Idea Generator":
    st.header("💡 Creative App Idea Generator")
    topic = st.text_input("Enter a topic / اكتب موضوع الفكرة (مثال: الصحة، التعليم، التمويل)")
    c1, c2 = st.columns(2)

    if c1.button("Generate One Idea / فكرة واحدة"):
        with st.spinner("Generating..."):
            model = choose_model_by_text(topic, model_choice)
            app_idea = or_generate(idea_prompt.format(topic=topic), model=model, max_tokens=150)
            features = or_generate(feature_prompt.format(app_idea=app_idea), model=model, max_tokens=150)
            tagline = or_generate(tagline_prompt.format(app_idea=app_idea, features=features), model=model, max_tokens=40)
        st.subheader("App Idea / فكرة التطبيق")
        render_answer(app_idea.strip())

        st.subheader("Features / المزايا")
        render_answer(features.strip())

        st.subheader("Tagline")
        render_answer(tagline.strip())


    if c2.button("Generate 3 Ideas / ٣ أفكار"):
        with st.spinner("Generating..."):
            ideas = []
            model = choose_model_by_text(topic, model_choice)
            for _ in range(3):
                app_idea = or_generate(idea_prompt.format(topic=topic), model=model, max_tokens=150)
                features = or_generate(feature_prompt.format(app_idea=app_idea), model=model, max_tokens=150)
                tagline = or_generate(tagline_prompt.format(app_idea=app_idea, features=features), model=model, max_tokens=40)
                ideas.append({"idea": app_idea.strip(), "features": features.strip(), "tagline": tagline.strip()})
        for i, idea in enumerate(ideas, 1):
            st.markdown(f"### 🚀 Idea {i}")
            render_answer(idea["idea"])
            render_answer(idea["features"])
            render_answer(idea["tagline"])


        st.download_button(
            "⬇️ Download Ideas (JSON)",
            data=json.dumps(ideas, ensure_ascii=False, indent=2),
            file_name="ideas.json",
            mime="application/json"
        )

elif mode == "🧾 Expense Extraction":
    st.header("🧾 Structured Info Extraction (Expenses) / استخراج المصروفات")
    user_input = st.text_area("Describe your day with purchases / صِف مشترياتك اليوم")
    if st.button("Extract / استخراج"):
        with st.spinner("Extracting..."):
            result = structured_extraction(user_input, model_choice)
        if isinstance(result, dict) and "purchases" in result:
            df = pd.DataFrame(result["purchases"])
            st.dataframe(df)
            total = result.get("total", "N/A")
            st.metric("💰 Total Spent / الإجمالي", f"{total}")
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), "expenses.csv")
        else:
            st.warning("Couldn't parse structured data. Here is the raw response:")
            st.code(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2))

elif mode == "📄 Resume Extraction":
    st.header("📄 Resume Info Extraction / استخراج معلومات السيرة الذاتية")
    uploaded_file = st.file_uploader("Upload Resume (PDF) / ارفع السيرة الذاتية (PDF)", type=["pdf"])

    if uploaded_file and st.button("Extract / استخراج"):
        with st.spinner("Parsing resume..."):
            resume_text = load_resume(uploaded_file)
            result = resume_extraction(resume_text, model_choice)
            st.session_state["resume_data"] = result

    if "resume_data" in st.session_state:
        result = st.session_state["resume_data"]
        st.subheader("📌 Candidate Profile / بيانات المرشح")
        st.write(f"**Name / الاسم:** {result.get('full_name','')}")
        st.write(f"**Email / البريد:** {result.get('email','')}")
        st.write(f"**Phone / الهاتف:** {result.get('phone','')}")
        st.write(f"**LinkedIn:** {result.get('linkedin','')}")

        if result.get("skills"):
            st.subheader("🛠 Skills / المهارات")
            try:
                st.write(", ".join(result["skills"]))
            except Exception:
                st.write(result["skills"])

        if result.get("education"):
            st.subheader("🎓 Education / التعليم")
            try:
                df_edu = pd.DataFrame(result["education"])
                st.table(df_edu)
            except Exception:
                st.write(result["education"])

        if result.get("experience"):
            st.subheader("💼 Experience / الخبرات")
            try:
                df_exp = pd.DataFrame(result["experience"])
                st.table(df_exp)
            except Exception:
                st.write(result["experience"])

        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="resume_data.json",
            mime="application/json"
        )

        st.markdown("---")
        st.subheader("📊 Job Matching / مطابقة الوظيفة")
        job_desc = st.text_area("Paste Job Description / الصِق الوصف الوظيفي")
        if st.button("Compare / مقارنة"):
            if job_desc.strip():
                with st.spinner("Scoring..."):
                    match_result = resume_job_match(result, job_desc, model_choice)
                score = (match_result or {}).get("score", "N/A")
                st.metric("Match Score", f"{score}/100")

                st.write("✅ **Strengths:**")
                strengths = (match_result or {}).get("strengths", [])
                render_answer("\n".join(strengths) if strengths else "—")

                st.write("⚠️ **Weaknesses:**")
                weaknesses = (match_result or {}).get("weaknesses", [])
                render_answer("\n".join(weaknesses) if weaknesses else "—")

                st.write("💡 **Recommendations:**")
                recs = (match_result or {}).get("recommendations", [])
                render_answer("\n".join(recs) if recs else "—")


                if "raw_response" in (match_result or {}):
                    st.caption("⚠️ Raw model output (for debugging):")
                    st.code(match_result["raw_response"])
            else:
                st.warning("⚠️ Please paste a job description before comparing.")

elif mode == "📚 PDF Q&A (RAG)":
    st.header("📚 Ask Questions About Your PDF / اسأل عن ملف PDF")
    uploaded_doc = st.file_uploader("Upload PDF / ارفع ملف PDF", type=["pdf"])
    if uploaded_doc:
        doc_text = extract_text_from_pdf(uploaded_doc)
        question = st.text_input("Ask your question / اكتب سؤالك")
        if st.button("Ask / اسأل"):
            with st.spinner("Thinking..."):
                answer = rag_qa(doc_text, question, model_choice)
            st.write("💡 Answer / الإجابة:")
            render_answer(answer.strip() if answer else "—")

