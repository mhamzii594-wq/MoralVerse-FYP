import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_url = os.getenv("LOCAL_IMAGE_API_URL")
print(f"Testing API URL: {api_url}")

payload = {
    "prompt": "Studio Ghibli style, a small cat in a garden, vibrant colors",
    "width": 512,
    "height": 512,
    "num_inference_steps": 2
}

try:
    response = requests.post(api_url, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success! Image data received.")
        data = response.json()
        if "image" in data:
            print(f"Image string length: {len(data['image'])}")
        else:
            print(f"Response keys: {data.keys()}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Connection failed: {e}")
