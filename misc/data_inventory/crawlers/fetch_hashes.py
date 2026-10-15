"""Fetch OSF's stored checksums (md5 + sha256) for every file in data/osf_all.json -> data/osf_hashes.json (read-only)."""
import json
import os
import time
import urllib.error
import urllib.request

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
nodes = json.load(open(os.path.join(DATA, 'osf_all.json'), encoding='utf-8'))
out_path = os.path.join(DATA, 'osf_hashes.json')
out = json.load(open(out_path)) if os.path.isfile(out_path) else {}
todo = [f['id'] for n in nodes for f in n['files'] if f['kind'] == 'file' and f['id'] not in out]
print(len(todo), 'files to fetch', flush=True)
for i, fid in enumerate(todo):
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(f'https://api.osf.io/v2/files/{fid}/', timeout=60))
            out[fid] = (d['data']['attributes'].get('extra') or {}).get('hashes') or {}
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(30 * (attempt + 1)); continue
            out[fid] = {'error': e.code}; break
        except Exception:
            time.sleep(5)
    if i % 50 == 0:
        print(i, flush=True); json.dump(out, open(out_path, 'w'), indent=0)
json.dump(out, open(out_path, 'w'), indent=0)
print('done', len(out))
