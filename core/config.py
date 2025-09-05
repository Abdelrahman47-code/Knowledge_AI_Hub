import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

if not OPENROUTER_KEY:
    OPENROUTER_KEY = None

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}" if OPENROUTER_KEY else "",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Smart AI Hub"
}
