# from langchain.prompts import PromptTemplate

# expense_template = PromptTemplate(
#     input_variables=["user_input", "format_instructions"],
#     template="""
# You are a smart assistant that extracts purchases from a description of someone’s day.
# Input may be Arabic or English. Respond in JSON ONLY.

# Rules:
# - Extract (name, amount, [currency]) for each purchase.
# - Ignore returned/refunded items when calculating totals.

# Format:
# {format_instructions}

# Input:
# "{user_input}"
# """
# )



from langchain.prompts import PromptTemplate

expense_template = PromptTemplate(
    input_variables=["user_input", "format_instructions"],
    template="""
You are a precise assistant that extracts **purchases** from a daily description.  
The input may be in Arabic or English. Respond in **valid JSON ONLY**, following the exact schema.

Rules:
- Each purchase must include:
  - name (string)
  - amount (number, if missing → null)
  - currency (string, optional, if missing → null)
- If the text mentions free items, gifts, or no purchase → skip them.
- Ignore returned/refunded items in both purchases and totals.
- Do not guess amounts or currencies if not explicitly mentioned.
- Always include a total value if calculable, otherwise null.

Format (must strictly follow this schema):
{format_instructions}

User Input:
"{user_input}"

JSON Output:
"""
)
