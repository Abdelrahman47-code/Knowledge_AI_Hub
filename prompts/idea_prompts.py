# from langchain.prompts import PromptTemplate

# idea_prompt = PromptTemplate(
#     input_variables=["topic", "num_sentences"],
#     template="""
# أنت مولّد أفكار تطبيقات احترافي. الإدخال قد يكون بالعربية أو الإنجليزية.
# القواعد:
# - إذا كان الإدخال بالعربية → أجب بالعربية.
# - إذا كان الإدخال بالإنجليزية → أجب بالإنجليزية.
# - أعطِ فكرة مبتكرة (غير إنشائية) خلال {num_sentences} جُمل.

# Topic / الموضوع: {topic}
# """
# )


# feature_prompt = PromptTemplate(
#     input_variables=["app_idea"],
#     template="""
# List exactly 3 UNIQUE, practical features (bulleted). Match language of the idea.
# Idea:
# {app_idea}
# """
# )

# tagline_prompt = PromptTemplate(
#     input_variables=["app_idea", "features"],
#     template="""
# Write a catchy tagline (max 12 words). Match the language of the idea.

# Idea:
# {app_idea}

# Features:
# {features}
# """
# )

from langchain.prompts import PromptTemplate

idea_prompt = PromptTemplate(
    input_variables=["topic", "num_sentences"],
    template="""
You are a professional **app idea generator**. The input can be in Arabic or English.
Rules:
- If the input is in Arabic → respond in Arabic.
- If the input is in English → respond in English.
- Provide one innovative (non-generic) idea explained in about {num_sentences} sentences.

Topic: {topic}
"""
)

feature_prompt = PromptTemplate(
    input_variables=["app_idea", "num_features"],
    template="""
List {num_features} UNIQUE, practical features (bulleted). 
Match the language of the idea.

Idea:
{app_idea}
"""
)

tagline_prompt = PromptTemplate(
    input_variables=["app_idea", "features"],
    template="""
Write a catchy tagline (max 12 words). 
Match the language of the idea.

Idea:
{app_idea}

Features:
{features}
"""
)
