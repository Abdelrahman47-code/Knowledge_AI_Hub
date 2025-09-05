# Knowledge AI Hub

An AI-powered productivity and knowledge assistant that combines idea generation, expense tracking, resume parsing, job matching, and RAG-based Q&A into one unified hub.

## 🚀 Features

- **💡 Creative Idea Generator** – Generate innovative app/product ideas across domains.
- **🧾 Expense Parser** – Extract structured information from natural language expense descriptions.
- **📄 Resume Parser** – Extract key details (skills, education, experience) from resumes.
- **🤝 Job Matcher** – Match resumes against job descriptions with strengths, weaknesses, and recommendations.
- **🔎 RAG Q&A** – Ask questions directly from PDF documents using Retrieval-Augmented Generation.

## 📸 App Demo Images
![image](demo_images/1.png)
---
![image](demo_images/2.png)
---
![image](demo_images/3.png)
---
![image](demo_images/4.png)
---
![image](demo_images/5.png)
---
![image](demo_images/6.png)
---
![image](demo_images/7.png)
---

## 🛠️ Tech Stack

- **Python** 3.10+
- **Streamlit** – Frontend UI
- **LangChain** – Document processing & chaining
- **OpenRouter API** – LLM access (replace with your own provider if needed)
- **PyPDFLoader** – PDF handling
- **dotenv** – Environment management

## 📂 Project Structure

```
knowledge_ai_hub/
├─ app.py                  # Streamlit entry point
├─ requirements.txt        # Dependencies
├─ .env.example            # Example env vars
├─ README.md               # Documentation
├─ core/                   # Core utilities
│  ├─ config.py
│  ├─ llm_utils.py
│  ├─ utils.py
│  ├─ pdf_utils.py
│  └─ lang_utils.py
├─ features/               # Main features
│  ├─ idea_generator.py
│  ├─ expense_parser.py
│  ├─ resume_parser.py
│  ├─ job_matcher.py
│  └─ rag_qa.py
└─ prompts/                # LLM prompt templates
   ├─ idea_prompts.py
   ├─ expense_prompts.py
   ├─ resume_prompts.py
   ├─ job_prompts.py
   └─ rag_prompts.py
```

## ⚙️ Setup Instructions

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-username/knowledge-ai-hub.git
   cd knowledge-ai-hub
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate      # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**

   Copy `.env.example` → `.env` and add your keys:
   ```
   OPENROUTER_KEY=your_api_key_here
   MODEL_NAME=openai/gpt-4o-mini
   ```

5. **Run App**
   ```bash
   streamlit run app.py
   ```

## 🧑‍💻 Development Workflow

- Keep feature-specific logic in `features/`.
- Store reusable utilities in `core/`.
- Keep all prompt templates in `prompts/`.
- Use `.env` for API keys and config.

## ✅ Roadmap

- Add multi-model support (Anthropic, Gemini, etc.)
- Improve resume-job scoring system
- Export results (PDF, Excel)
- Add database persistence for user history
- Deploy on Streamlit Cloud / Docker

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss.

## 📜 License

MIT License © 2025 [Your Name]