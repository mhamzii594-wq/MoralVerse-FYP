import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'moralverse.settings'
import django; django.setup()

from pathlib import Path
from google import genai
from google.genai import types

client = genai.Client(api_key='AIzaSyAJStqGTQfchlO-Kir1SGli8c8FQs8M4mM')

avatars_dir = Path('media/avatars')
recent = sorted(avatars_dir.glob('*_avatar_*.jpg'), key=lambda p: p.stat().st_mtime, reverse=True)
recent = [f for f in recent if '_final' not in f.name and '_ghibli' not in f.name]
print('Using avatar:', recent[0].name, 'size:', recent[0].stat().st_size)

with open(recent[0], 'rb') as f:
    img_bytes = f.read()

print('Calling Gemini Image Generation...')
try:
    resp = client.models.generate_content(
        model='gemini-2.0-flash-preview-image-generation',
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
            'Convert this photo into a Studio Ghibli anime-style portrait. Keep exactly the same face shape, hair color, hair style, eye color, and facial features as the person in the photo. Use Hayao Miyazaki hand-drawn illustration style: large expressive eyes, soft warm lighting, clean thick outlines, vibrant saturated colors, gentle smile, pure white background. Output only the image.',
        ],
        config=types.GenerateContentConfig(response_modalities=['IMAGE']),
    )
    found_image = False
    for part in (resp.candidates or [{}])[0].content.parts:
        if hasattr(part, 'inline_data') and part.inline_data:
            data = part.inline_data.data
            print(f'Got image data: {len(data)} bytes, mime: {part.inline_data.mime_type}')
            out_path = avatars_dir / 'test_ghibli_gemini.jpg'
            out_path.write_bytes(data)
            print('Saved to:', out_path)
            found_image = True
            break
    if not found_image:
        print('No image returned. Text:', resp.text[:200] if resp.text else 'None')
except Exception as e:
    print('Error:', e)
