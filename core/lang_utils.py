from langdetect import detect

def detect_language(text: str) -> str:
    try:
        lang = detect(text or "en")
        return "ar" if lang == "ar" else "en"
    except Exception:
        return "en"

def is_arabic(text: str) -> bool:
    return detect_language(text) == "ar"
