import streamlit as st
from core.pdf_utils import extract_text_from_pdf_filelike
from core.llm_utils import or_generate
from prompts.rag_prompts import rag_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from core.lang_utils import detect_language

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
        return chunks[:top_k]

def run(model_choice: str):
    st.header("📚 Chat with Your PDF (RAG) / تحدث مع ملف PDF")

    uploaded_doc = st.file_uploader("📄 Upload PDF / ارفع ملف PDF", type=["pdf"])

    if uploaded_doc:
        doc_text = extract_text_from_pdf_filelike(uploaded_doc)
        chunks = chunk_document(doc_text)

        # Initialize chat history
        if "rag_chat_history" not in st.session_state:
            st.session_state.rag_chat_history = []

        # Reset button
        if st.button("🗑️ New Chat / محادثة جديدة"):
            st.session_state.rag_chat_history = []
            st.rerun()

        # Display chat history with icons
        for role, content in st.session_state.rag_chat_history:
            icon = "🤵" if role == "user" else "🤖"
            with st.chat_message(role, avatar=icon):
                st.markdown(content)

        # User input
        if question := st.chat_input("💬 Ask about the document / اسأل عن المستند"):
            st.session_state.rag_chat_history.append(("user", question))
            with st.chat_message("user", avatar="🤵"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🤔 Thinking..."):
                    relevant = find_relevant_chunks(chunks, question, top_k=3)
                    context = "\n\n".join(relevant)

                    # Add last few messages from history
                    history_context = "\n".join(
                        [f"{role.upper()}: {msg}" for role, msg in st.session_state.rag_chat_history[-5:]]
                    )

                    model = (
                        model_choice
                        if model_choice != "Auto (detect language)"
                        else "mistralai/mistral-7b-instruct"
                    )
                    prompt = rag_prompt.format(
                        context=context + "\n\nConversation:\n" + history_context,
                        question=question,
                    )
                    ans = or_generate(prompt, model=model, max_tokens=600)

                # Handle Arabic/English direction
                if detect_language(question) == "ar":
                    st.markdown(
                        f"<div style='direction: rtl; text-align: right; font-size: 1.05em;'>{ans}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"<div style='font-size: 1.05em;'>{ans}</div>", unsafe_allow_html=True)

            # Save assistant response
            st.session_state.rag_chat_history.append(("assistant", ans))
