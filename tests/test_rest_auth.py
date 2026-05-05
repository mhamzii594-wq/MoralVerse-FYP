import os
import requests
import json

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = os.getenv("GEMINI_STORY_MODEL") or "gemini-2.0-flash"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

payload = {
    "contents": [{"parts": [{"text": "Say hello."}]}]
}

try:
    print(f"Testing REST Auth for Stories with key: {api_key[:5]}...")
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=10
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Success! REST API is working for stories.")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")
