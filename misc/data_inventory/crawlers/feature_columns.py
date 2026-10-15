"""Read the column header of one training CSV per dataset (OSF zips, OSF loose files, Google Drive) with HTTP range
requests -> data/feature_columns.json. Only the first few KB of each file are fetched."""
import csv
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_zip_contents import MultiPartRangeFile  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
nodes = json.load(open(os.path.join(DATA, 'osf_all.json'), encoding='utf-8'))
drive = json.load(open(os.path.join(DATA, 'gdrive_inventory.json'), encoding='utf-8'))
contents = {}
for f in ('osf_split_zip_contents.json', 'osf_zip_contents.json'):
    contents.update(json.load(open(os.path.join(DATA, f), encoding='utf-8')))


def header_from_text(text):
    return next(csv.reader(io.StringIO(text)))


def range_head(url, n=200000):
    req = urllib.request.Request(url, headers={'Range': f'bytes=0-{n}', 'User-Agent': 'Mozilla/5.0'})
    return urllib.request.urlopen(req, timeout=120).read().decode('utf-8', 'ignore')


def osf_parts(node_id, path):
    n = next(x for x in nodes if x['id'] == node_id)
    files = [f for f in n['files'] if f['kind'] == 'file']
    single = [f for f in files if f['path'] == path]
    if single:
        return [(single[0]['download'], single[0]['size'])]
    return [(f['download'], f['size']) for f in sorted([f for f in files if re.match(re.escape(path) + r'\.\d{3}$', f['path'])], key=lambda f: f['name'])]


def zip_member_header(node_id, path, member):
    z = zipfile.ZipFile(io.BufferedReader(MultiPartRangeFile(osf_parts(node_id, path)), buffer_size=1 << 16))
    with z.open(member) as fh:
        return header_from_text(fh.read(200000).decode('utf-8', 'ignore'))


TARGETS = []
# OSF archives with training CSVs: first CSV inside (the example project: its targets_inserted file)
for key, members in contents.items():
    if not isinstance(members, list):
        continue
    node_id, path = key.split(':', 1)
    if 'features' in path.lower() and 'annotation' in path.lower():
        member = next(m['name'] for m in members if m['name'].lower().endswith('.csv'))
        TARGETS.append(('osf-zip', node_id, path, member))
    if 'Example_project' in path:
        member = next((m['name'] for m in members if 'targets_inserted' in m['name'] and m['name'].endswith('.csv')), None)
        if member:
            TARGETS.append(('osf-zip', node_id, path, member))
# Tutorial Data: a loose targets_inserted CSV
tut = next(x for x in nodes if x['id'] == 'dg385')
f = next(f for f in tut['files'] if f['kind'] == 'file' and '/targets_inserted/' in f['path'])
TARGETS.append(('osf-file', 'dg385', f['path'], f['download']))
# Google Drive: a loose annotation CSV
d = next(i for i in drive['items'] if i['name'] == 'Video14.csv')
TARGETS.append(('drive-file', 'gdrive', d['path'], f"https://drive.usercontent.google.com/download?id={d['id']}&export=download&confirm=t"))

out = {}
for kind, node_id, path, member in TARGETS:
    try:
        cols = zip_member_header(node_id, path, member) if kind == 'osf-zip' else header_from_text(range_head(member))
        key = f'{node_id}:{path}'
        out[key] = {'example_file': member if kind == 'osf-zip' else path.rsplit('/', 1)[-1], 'columns': cols}
        print(key.split('/')[-1], len(cols), 'columns | last:', cols[-4:], flush=True)
    except Exception as e:
        print(node_id, path, 'ERROR', repr(e)[:150], flush=True)
json.dump(out, open(os.path.join(DATA, 'feature_columns.json'), 'w', encoding='utf-8'), indent=1)
