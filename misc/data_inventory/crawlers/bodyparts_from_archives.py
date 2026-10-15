"""Read the body-part names from the DeepLabCut CollectedData_*.csv files inside the labelled-image archives on OSF,
without downloading the archives (HTTP range requests) -> data/archive_bodyparts.json

For each archive: the body parts (and individuals, for multi-animal projects) of every distinct CSV header found
in up to MAX_CSV CollectedData files, so differing label schemes inside one archive are also caught."""
import csv
import io
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_zip_contents import MultiPartRangeFile  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MAX_CSV = 6

nodes = json.load(open(os.path.join(DATA, 'osf_all.json'), encoding='utf-8'))
contents = {}
for f in ('osf_split_zip_contents.json', 'osf_zip_contents.json'):
    contents.update(json.load(open(os.path.join(DATA, f), encoding='utf-8')))


def parts_for(node, key_path):
    """[(url, size)] of the archive parts (one entry for a single zip)."""
    files = [f for f in node['files'] if f['kind'] == 'file']
    single = [f for f in files if f['path'] == key_path]
    if single:
        return [(single[0]['download'], single[0]['size'])]
    split = sorted([f for f in files if re.match(re.escape(key_path) + r'\.\d{3}$', f['path'])], key=lambda f: f['name'])
    return [(f['download'], f['size']) for f in split]


def parse_header(text):
    rows = list(csv.reader(io.StringIO(text)))[:4]
    out = {}
    for r in rows:
        if r and r[0].strip().lower() in ('scorer', 'individuals', 'bodyparts', 'coords'):
            out[r[0].strip().lower()] = r[1:]
    bps = list(dict.fromkeys(b for b in out.get('bodyparts', []) if b))
    inds = list(dict.fromkeys(i for i in out.get('individuals', []) if i))
    return {'bodyparts': bps, 'individuals': inds, 'scorer': (out.get('scorer') or [''])[0]}


result = {}
by_id = {n['id']: n for n in nodes}
for key, members in contents.items():
    if not isinstance(members, list):
        continue
    csvs = [m['name'] for m in members if re.search(r'CollectedData[^/]*\.csv$', m['name'])]
    if not csvs:
        continue
    node_id, path = key.split(':', 1)
    parts = parts_for(by_id[node_id], path)
    headers = {}
    try:
        z = zipfile.ZipFile(io.BufferedReader(MultiPartRangeFile(parts), buffer_size=1 << 16))
        step = max(1, len(csvs) // MAX_CSV)
        for name in csvs[::step][:MAX_CSV]:
            with z.open(name) as fh:
                text = fh.read(20000).decode('utf-8', 'ignore')
            h = parse_header(text)
            sig = (tuple(h['individuals']), tuple(h['bodyparts']))
            headers.setdefault(sig, {'example_file': name, **h})
        result[key] = {'n_collecteddata_csv': len(csvs), 'schemes': list(headers.values())}
        print(key.split('/')[-1], len(csvs), 'csv |', [(len(s['individuals']), len(s['bodyparts'])) for s in headers.values()], flush=True)
    except Exception as e:
        result[key] = {'error': repr(e)[:200]}
        print(key, 'ERROR', repr(e)[:150], flush=True)
json.dump(result, open(os.path.join(DATA, 'archive_bodyparts.json'), 'w', encoding='utf-8'), indent=1)
