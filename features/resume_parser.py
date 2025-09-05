import streamlit as st
import pandas as pd
import json
from core.llm_utils import or_generate
from core.utils import extract_json_block, safe_json_loads
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from prompts.resume_prompts import resume_parser_template
from core.pdf_utils import extract_text_from_pdf_filelike, load_with_langchain

def run(model_choice: str):
    st.header("📄 Resume Info Extraction / استخراج معلومات السيرة الذاتية")
    uploaded_file = st.file_uploader("Upload Resume (PDF) / ارفع السيرة الذاتية (PDF)", type=["pdf"])
    if uploaded_file and st.button("Extract / استخراج"):
        with st.spinner("Parsing resume..."):
            # extract text
            try:
                resume_text = extract_text_from_pdf_filelike(uploaded_file)
            except Exception:
                with open("temp_resume.pdf", "wb") as f:
                    f.write(uploaded_file.read())
                resume_text = load_with_langchain("temp_resume.pdf")

            # build parser schema
            full_name_schema = ResponseSchema(name="full_name", description="Candidate full name")
            email_schema = ResponseSchema(name="email", description="Email address")
            phone_schema = ResponseSchema(name="phone", description="Phone number")
            linkedin_schema = ResponseSchema(name="linkedin", description="LinkedIn URL")
            education_schema = ResponseSchema(name="education", description="List of education objects")
            skills_schema = ResponseSchema(name="skills", description="List of skills")
            experience_schema = ResponseSchema(name="experience", description="List of experience objects")

            output_parser = StructuredOutputParser.from_response_schemas(
                [full_name_schema, email_schema, phone_schema, linkedin_schema, education_schema, skills_schema, experience_schema]
            )
            format_instructions = output_parser.get_format_instructions()
            prompt = resume_parser_template.format(resume_text=resume_text[:7000], format_instructions=format_instructions)
            model = model_choice if model_choice != "Auto (detect language)" else "mistralai/mistral-7b-instruct"
            raw = or_generate(prompt, model=model, max_tokens=900)
            json_text = extract_json_block(raw)
            try:
                result = output_parser.parse(json_text)
            except Exception:
                result = safe_json_loads(json_text) or {"raw_response": raw}

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
            st.write(", ".join(result["skills"]) if isinstance(result["skills"], list) else result["skills"])

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

        st.download_button("⬇️ Download JSON", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="resume_data.json", mime="application/json")
