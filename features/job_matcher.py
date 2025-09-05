import streamlit as st
import json
from core.llm_utils import or_generate
from core.utils import extract_json_block, safe_json_loads
from prompts.job_prompts import job_match_prompt

def run(model_choice: str):
    st.header("📊 Job Matching / مطابقة الوظيفة")
    if "resume_data" not in st.session_state:
        st.warning("Please extract a resume first.")
        return
    resume_data = st.session_state["resume_data"]
    job_desc = st.text_area("Paste Job Description / الصِق الوصف الوظيفي")
    if st.button("Compare / مقارنة"):
        if not job_desc.strip():
            st.warning("Please paste a job description.")
            return
        with st.spinner("Scoring..."):
            prompt = job_match_prompt.format(resume_data=json.dumps(resume_data, ensure_ascii=False, indent=2), job_description=job_desc)
            model = model_choice if model_choice != "Auto (detect language)" else "mistralai/mistral-7b-instruct"
            raw = or_generate(prompt, model=model, max_tokens=700)
            json_text = extract_json_block(raw)
            parsed = safe_json_loads(json_text)
            if not parsed:
                st.warning("Couldn't parse JSON reliably. Showing raw output:")
                st.code(raw)
                return
        score = parsed.get("score", "N/A")
        st.metric("Match Score", f"{score}/100")

        st.write("✅ **Strengths:**")
        strengths = parsed.get("strengths", [])
        st.write("\n".join(strengths) if strengths else "—")

        st.write("⚠️ **Weaknesses:**")
        weaknesses = parsed.get("weaknesses", [])
        st.write("\n".join(weaknesses) if weaknesses else "—")

        st.write("💡 **Recommendations:**")
        recs = parsed.get("recommendations", [])
        st.write("\n".join(recs) if recs else "—")
