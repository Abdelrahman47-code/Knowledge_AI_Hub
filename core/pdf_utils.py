from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf_filelike(file) -> str:
    """
    Extract text from uploaded file-like pdf using PyPDF2.
    """
    pdf = PdfReader(file)
    pages = []
    for page in pdf.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)

def load_with_langchain(path: str) -> str:
    """
    Use LangChain loader (file path) to get document text.
    """
    loader = PyPDFLoader(path)
    docs = loader.load()
    return "\n".join([d.page_content for d in docs])
