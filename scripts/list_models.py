import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

api_key = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

print("Listing available models...")
try:
    for model in client.models.list():
        print(model)
except Exception as e:
    print(f"Error: {e}")
