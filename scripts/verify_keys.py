import os
import requests
from dotenv import load_dotenv
load_dotenv()

def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
    try:
        res = requests.post(url, json=payload, timeout=10)
        print(f"Gemini Test: Status {res.status_code}")
        if res.status_code != 200:
            print(f"Gemini Error: {res.text}")
    except Exception as e:
        print(f"Gemini Exception: {e}")

def test_stability():
    api_key = os.getenv("STABILITY_API_KEY", "").strip()
    url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    headers = {"authorization": f"Bearer {api_key}", "accept": "image/*"}
    # Just check if we can reach it or get an auth error
    try:
        # We don't want to spend credits, so we'll just do a GET or a HEAD if possible
        # Actually, Stability only supports POST for generation.
        # Let's try to list engines (legacy API) just to check the key.
        url_check = "https://api.stability.ai/v1/user/account"
        res = requests.get(url_check, headers={"authorization": f"Bearer {api_key}"})
        print(f"Stability Test (Account Check): Status {res.status_code}")
        if res.status_code != 200:
            print(f"Stability Error: {res.text}")
    except Exception as e:
        print(f"Stability Exception: {e}")

test_gemini()
test_stability()
