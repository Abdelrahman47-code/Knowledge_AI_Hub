import streamlit as st
from core.pdf_utils import extract_text_from_pdf_filelike
from core.llm_utils import or_generate
from prompts.rag_prompts import rag_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def chunk_document(text: str, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def find_relevant_chunks(chunks, question, top_k=3):
    if not chunks:
        return []
    try:
        vectorizer = TfidfVectorizer().fit(chunks + [question])
        doc_vecs = vectorizer.transform(chunks)
        q_vec = vectorizer.transform([question])
        sims = (doc_vecs @ q_vec.T).toarray().ravel()
        top_idx = sims.argsort()[::-1][:top_k]
        return [chunks[i] for i in top_idx]
    except Exception:
        # return first top_k chunks
        return chunks[:top_k]

def run(model_choice: str):
    st.header("📚 Ask Questions About Your PDF (RAG) / اسأل عن ملف PDF")
    uploaded_doc = st.file_uploader("Upload PDF / ارفع ملف PDF", type=["pdf"])
    if uploaded_doc:
        doc_text = extract_text_from_pdf_filelike(uploaded_doc)
        question = st.text_input("Ask your question / اكتب سؤالك")
        if st.button("Ask / اسأل"):
            if not question.strip():
                st.warning("Please enter a question.")
                return
            with st.spinner("Thinking..."):
                chunks = chunk_document(doc_text)
                relevant = find_relevant_chunks(chunks, question, top_k=3)
                context = "\n\n".join(relevant)
                model = model_choice if model_choice != "Auto (detect language)" else "mistralai/mistral-7b-instruct"
                prompt = rag_prompt.format(context=context, question=question)
                ans = or_generate(prompt, model=model, max_tokens=600)
            st.write("💡 Answer / الإجابة:")
            # simple directionality: if question detected as Arabic then render RTL
            from core.lang_utils import detect_language
            if detect_language(question) == "ar":
                st.markdown(f"<div style='direction: rtl; text-align: right;'>{ans}</div>", unsafe_allow_html=True)
            else:
                st.write(ans)
