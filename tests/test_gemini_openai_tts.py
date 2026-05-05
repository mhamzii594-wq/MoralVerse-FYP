import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip()
url = "https://generativelanguage.googleapis.com/v1beta/openai/v1/audio/speech"

payload = {
    "model": "tts-1",
    "input": "This is a test of the OpenAI-compatible Gemini TTS endpoint.",
    "voice": "alloy"
}

try:
    print("Calling Gemini OpenAI-compatible TTS endpoint...")
    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        data=json.dumps(payload),
        timeout=10
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:100]}")
except Exception as e:
    print(f"Error: {e}")
