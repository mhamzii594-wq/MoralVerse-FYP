import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = "gemini-1.5-flash" # Use a stable model
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

payload = {
    "contents": [{"parts": [{"text": "This is a test of the Gemini 1.5 Flash REST audio output."}]}],
    "generationConfig": {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": "Puck"
                }
            }
        }
    }
}

try:
    print(f"Calling Gemini REST API (v1beta) with model {model}...")
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=15
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        candidates = result.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for i, part in enumerate(parts):
                if "inlineData" in part:
                    print(f"Success! Found audio data in part {i}, length: {len(part['inlineData']['data'])}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")
