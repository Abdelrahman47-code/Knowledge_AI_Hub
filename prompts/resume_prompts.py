from langchain.prompts import PromptTemplate

resume_parser_template = PromptTemplate(
    input_variables=["resume_text", "format_instructions"],
    template= """
You are an expert HR assistant. The resume can be Arabic or English.

Extract the following fields and return ONLY JSON:
- full_name (string)
- email (string)
- phone (string)
- linkedin (string)
- education (list of {{degree, institution, year}})
- skills (list of strings)
- experience (list of {{role, company, years}})

Format:
{format_instructions}

Resume Text:
"{resume_text}"
"""
)
