"""Find every OSF / data-Drive link in the SimBA repo (file, line, url) -> repo_refs.json."""
import json
import os
import re
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OWN = ('docs/data_inventory.rst', 'docs/_static/data_inventory/', 'misc/data_inventory/')   # this page and its generator
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'repo_refs.json')
PAT = re.compile(r'https?://(?:www\.)?osf\.io/[^\s)\]"\'<>`|]+|https?://drive\.google\.com/drive/folders/17-Bi0D78Qn[^\s)\]"\'<>`]*')
SKIP = {'.git', '_build', 'node_modules', 'build', 'sandbox', '__pycache__', '.idea', '.claude'}
EXT = ('.md', '.rst', '.py', '.txt', '.html', '.ipynb', '.yml', '.yaml', '.cfg')

refs = []
for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for fn in files:
        if not fn.endswith(EXT):
            continue
        p = os.path.join(root, fn)
        if os.path.relpath(p, REPO).replace(os.sep, '/').startswith(OWN):
            continue
        try:
            lines = open(p, encoding='utf-8', errors='ignore').read().splitlines()
        except OSError:
            continue
        for i, l in enumerate(lines, 1):
            for m in PAT.findall(l):
                refs.append({'file': os.path.relpath(p, REPO).replace(os.sep, '/'), 'line': i, 'url': m.rstrip('.,;:\\')})
json.dump(refs, open(OUT, 'w', encoding='utf-8'), indent=1)
print(len(refs), 'refs in', len({r['file'] for r in refs}), 'files')
for f, c in Counter(r['file'] for r in refs).most_common(30):
    print(c, f)
