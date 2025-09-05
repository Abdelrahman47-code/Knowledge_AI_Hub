import requests
from .config import HEADERS

def or_generate(prompt: str, model: str = "mistralai/mistral-7b-instruct", max_tokens: int = 500, temperature: float = 0.4, timeout: int = 60) -> str:
    """
    Call OpenRouter chat/completions.
    Returns the assistant content or raises on HTTP errors.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"]
    return str(data)
