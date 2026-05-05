import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

api_key = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

model = "gemini-2.5-flash-preview-tts"
text = "This is a test of the Gemini TTS system to see if it hangs or returns data."

print(f"Calling Gemini SDK for TTS with model {model}...")
try:
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Puck"
                    )
                )
            ),
        ),
    )
    print("Response received!")
    if response.candidates:
        print(f"Parts count: {len(response.candidates[0].content.parts)}")
        for i, part in enumerate(response.candidates[0].content.parts):
            if part.inline_data:
                print(f"Part {i} has inline_data, length: {len(part.inline_data.data)}")
except Exception as e:
    print(f"Error: {e}")
