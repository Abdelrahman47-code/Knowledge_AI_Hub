import re
import json

ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
EASTERN_PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
WESTERN = "0123456789"
ARABIC_MAP = {ord(a): w for a, w in zip(ARABIC_INDIC, WESTERN)}
PERSIAN_MAP = {ord(a): w for a, w in zip(EASTERN_PERSIAN, WESTERN)}

def normalize_digits(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.translate(ARABIC_MAP).translate(PERSIAN_MAP)

def extract_json_block(text: str) -> str:
    if not text:
        return "{}"
    fenced = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0).strip() if m else "{}"

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        last = s.rfind("}")
        if last != -1:
            try:
                return json.loads(s[: last + 1])
            except Exception:
                pass
    return None
