import os, base64, json, requests
api_key = 'Xv2Nfydhxxt6JKyHlMfXiI762n3r2vRLWDPkYAMsxdRPp1UAAO7LYSE8ZUUi'
from pathlib import Path
avatars_dir = Path('media/avatars')
recent = sorted(avatars_dir.glob('*_avatar_*.jpg'), key=lambda p: p.stat().st_mtime, reverse=True)
recent = [f for f in recent if '_final' not in f.name and '_ghibli' not in f.name]
print('Using avatar:', recent[0].name)
with open(recent[0], 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')
payload = {'key': api_key, 'model_id': 'flux-kontext-dev', 'prompt': 'Studio Ghibli anime portrait', 'init_image': img_b64, 'base64': True, 'width': '512', 'height': '512', 'samples': '1', 'num_inference_steps': '20'}
resp = requests.post('https://modelslab.com/api/v6/images/img2img', json=payload, timeout=30)
print('Status:', resp.status_code)
data = resp.json()
print('Response status:', data.get('status'))
print('Message:', data.get('message', '')[:200])
print('Full (truncated):', str(data)[:500])
