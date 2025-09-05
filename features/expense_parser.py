import streamlit as st
import pandas as pd
from core.llm_utils import or_generate
from core.utils import extract_json_block, normalize_digits, safe_json_loads
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from prompts.expense_prompts import expense_template

def run(model_choice: str):
    st.header("🧾 Structured Info Extraction (Expenses) / استخراج المصروفات")
    user_input = st.text_area("Describe your day with purchases / صِف مشترياتك اليوم")
    if st.button("Extract / استخراج"):
        if not user_input.strip():
            st.warning("Please enter text.")
            return
        with st.spinner("Extracting..."):
            purchase_schema = ResponseSchema(name="purchases", description="List of purchased items with fields name, amount, currency(optional)")
            total_schema = ResponseSchema(name="total", description="Total amount")
            output_parser = StructuredOutputParser.from_response_schemas([purchase_schema, total_schema])
            format_instructions = output_parser.get_format_instructions()
            prompt = expense_template.format(user_input=user_input, format_instructions=format_instructions)
            model = model_choice if model_choice != "Auto (detect language)" else "mistralai/mistral-7b-instruct"
            raw = or_generate(prompt, model=model, max_tokens=600)
            json_text = extract_json_block(raw)
            try:
                data = output_parser.parse(json_text)
            except Exception:
                data = safe_json_loads(json_text) or {"raw_response": raw}

        if isinstance(data, dict) and "purchases" in data:
            df = pd.DataFrame(data["purchases"])
            st.dataframe(df)
            total = data.get("total", "N/A")
            st.metric("💰 Total Spent / الإجمالي", f"{total}")
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), "expenses.csv")
        else:
            st.warning("Couldn't parse structured data. Raw output:")
            st.code(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2))
