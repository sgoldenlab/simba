"""List data files tracked in the SimBA git repo (size, last commit date, where they are referenced) -> data/repo_data_files.json"""
import json
import os
import re
import subprocess
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OWN = ('docs/data_inventory.rst', 'docs/_static/data_inventory/', 'misc/data_inventory/')   # this page and its generator
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'repo_data_files.json')
DATA_EXT = ('.csv', '.h5', '.hdf5', '.mp4', '.avi', '.mov', '.sav', '.parquet', '.slp', '.zip', '.pickle', '.pkl', '.npy', '.npz', '.xlsx', '.feather', '.nwb', '.json', '.pt', '.onnx')
TEXT_EXT = ('.py', '.md', '.rst', '.txt', '.ipynb', '.html', '.yml', '.yaml', '.cfg', '.ini', '.js')
SKIP_JSON_DIRS = ('docs/_static', '.github', 'simba/assets/lookups')   # json there is site/app config, not data


def git(*args):
    return subprocess.run(['git', '-C', REPO] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8', 'ignore')


tracked = git('ls-files').splitlines()
tracked = [p for p in tracked if not p.startswith(OWN)]
data_files = [p for p in tracked if p.lower().endswith(DATA_EXT)
              and not (p.lower().endswith('.json') and (p.startswith(SKIP_JSON_DIRS) or '/' not in p or 'package' in p))]

# date each file last changed in one pass over the log
last_date = {}
for block in git('log', '--format=@@%ad', '--date=short', '--name-only', '--', *data_files).split('@@')[1:]:
    lines = [l for l in block.splitlines() if l.strip()]
    for p in lines[1:]:
        last_date.setdefault(p, lines[0])

# where each basename is mentioned in tracked text files (docs, code, notebooks)
names = defaultdict(list)
for p in data_files:
    names[os.path.basename(p)].append(p)
pat = re.compile('|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True)))
mentions = defaultdict(list)
for t in tracked:
    if not t.lower().endswith(TEXT_EXT) or t.startswith('simba/sandbox/'):
        continue
    try:
        lines = open(os.path.join(REPO, t), encoding='utf-8', errors='ignore').read().splitlines()
    except OSError:
        continue
    for i, l in enumerate(lines, 1):
        for m in set(pat.findall(l)):
            mentions[m].append({'file': t, 'line': i})

out = []
for p in data_files:
    full = os.path.join(REPO, p)
    out.append({'path': p, 'size': os.path.getsize(full) if os.path.isfile(full) else None,
                'last_changed': last_date.get(p, ''), 'mentions': mentions.get(os.path.basename(p), [])[:20],
                'n_mentions': len(mentions.get(os.path.basename(p), []))})
json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
print(len(out), 'data files ->', OUT)
