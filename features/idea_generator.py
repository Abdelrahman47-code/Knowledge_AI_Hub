import streamlit as st
from core.llm_utils import or_generate
from core.lang_utils import is_arabic
from prompts.idea_prompts import idea_prompt, feature_prompt, tagline_prompt
import json

def run(model_choice: str):
    st.header("💡 Creative App Idea Generator")
    
    topic = st.text_input("Enter a topic / اكتب موضوع الفكرة (مثال: الصحة، التعليم، التمويل)")
    num_ideas = st.number_input("Number of ideas / عدد الأفكار", min_value=1, max_value=10, value=1)
    num_sentences = st.number_input("Max sentences per idea / عدد الجمل للفكرة الواحدة", min_value=1, max_value=10, value=5)
    num_features = st.number_input("Number of features / عدد المزايا", min_value=1, max_value=10, value=3)
    
    if st.button("Generate Ideas / توليد الأفكار"):
        if not topic.strip():
            st.warning("Please enter a topic.")
            return
        
        with st.spinner("Generating ideas..."):
            model = model_choice if model_choice != "Auto (detect language)" else (
                "qwen/Qwen2-72B-Instruct" if is_arabic(topic) else "mistralai/mistral-7b-instruct"
            )
            ideas = []
            
            for _ in range(num_ideas):
                # Generate app idea
                app_idea = or_generate(
                    idea_prompt.format(topic=topic, num_sentences=str(num_sentences)),
                    model=model,
                    max_tokens=200
                )
                
                # Generate features
                features = or_generate(
                    feature_prompt.format(app_idea=app_idea, num_features=str(num_features)),
                    model=model,
                    max_tokens=200
                )
                
                # Generate tagline
                tagline = or_generate(
                    tagline_prompt.format(app_idea=app_idea, features=features),
                    model=model,
                    max_tokens=40
                )

                ideas.append({
                    "idea": app_idea.strip(),
                    "features": features.strip(),
                    "tagline": tagline.strip()
                })
        
        # Show results
        for i, idea in enumerate(ideas, 1):
            st.markdown(f"### 🚀 Idea {i}")
            st.markdown(idea["idea"])
            st.markdown(idea["features"])
            st.success(idea["tagline"])
        
        # Download JSON
        st.download_button(
            "⬇️ Download Ideas (JSON)",
            data=json.dumps(ideas, ensure_ascii=False, indent=2),
            file_name="ideas.json",
            mime="application/json"
        )
