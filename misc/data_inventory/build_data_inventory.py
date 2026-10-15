"""
Build ``docs/data_inventory.rst`` and ``docs/_static/data_inventory/``: a page listing every piece of public SimBA data (OSF + Google Drive) and where it lives.

Inputs (made by the read-only crawlers in this folder, see ``README.md``):
    osf_all.json                 every sgoldenlab OSF project, wiki and file
    gdrive_inventory.json        the public Google Drive folder with the mouse resident-intruder training data
    *_zip_contents.json          member lists of every archive (read from the zip index with HTTP range requests)
    repo_refs.json               every OSF / Drive link in the SimBA repo (file + line)
"""
import collections
import csv
import gzip
import html
import io
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, 'docs', 'data_inventory.rst')
SNAPSHOT_DATE = '2026-10-15'
GITHUB_BLOB = 'https://github.com/sgoldenlab/simba/blob/master/'
FULL_LIST_MAX = 300          # archives with at most this many files are listed in full, larger ones per folder
ROOT_ORDER = ['tmu6y', 'd69jt', 'uhjzf', '5t4y9', 'dg385', 'fvbuh']
ROOT_BLURB = {
    'tmu6y': 'The main SimBA project. Holds the behavioral classifiers project, the tracking models project (DeepLabCut, YOLO, Mask R-CNN, DeepPoseKit, videos) and slides/talks.',
    'd69jt': 'An older (Dec 2019) standalone project with random forest classifiers for mouse, rat and CRIM13 resident-intruder.',
    'uhjzf': 'An older (Dec 2019) standalone project with DeepLabCut labelled images (PNG + H5/CSV annotations), one sub-project per setup.',
    '5t4y9': 'An older (Dec 2019) standalone project with DeepLabCut tracking weights, one sub-project per setup.',
    'dg385': 'A small, complete tutorial set (Apr 2021): 3 videos, their tracking data, targets_inserted annotations and two classifiers.',
    'fvbuh': 'Not SimBA: installers for the WholeBrain (R) brain-registration package.',
}


def load(name):
    return json.load(open(os.path.join(DATA, name), encoding='utf-8'))


def hsize(b):
    if b is None:
        return ''
    for unit, div in (('GB', 1e9), ('MB', 1e6), ('KB', 1e3)):
        if b >= div:
            return f'{b / div:.2f} {unit}' if unit == 'GB' else f'{b / div:.1f} {unit}'
    return f'{b} B'


def parse_drive_size(s):
    if not s:
        return None
    m = re.match(r'([\d.]+)\s*(KB|MB|GB|bytes?)', s)
    if not m:
        return None
    return int(float(m.group(1)) * {'KB': 1e3, 'MB': 1e6, 'GB': 1e9}.get(m.group(2), 1))


def drive_date(s):
    # Drive's folder view gives m/d/yy
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2})$', s or '')
    return f'20{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}' if m else (s or '')


def e(s):
    return html.escape(str(s), quote=True)


def a(url, text):
    return f'<a href="{e(url)}" target="_blank" rel="noopener">{e(text)}</a>'


def osf_file_url(node_id, rec):
    return f'https://osf.io/{rec["guid"]}/' if rec.get('guid') else f'https://osf.io/{node_id}/files/osfstorage/{rec["id"]}'


def raw(lines, indent='   '):
    # every physical line must stay indented, or RST ends the raw block early
    return ['.. raw:: html', ''] + [indent + part for l in lines for part in l.split('\n')] + ['']


def wiki_html(md):
    """Small Markdown -> HTML for the OSF wikis (headings, pipe tables, links, paragraphs)."""
    out, table = [], []

    def inline(t):
        t = e(t)
        return re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener">{m.group(1)}</a>', t)

    def flush():
        if table:
            rows = [[c.strip() for c in r.strip().strip('|').split('|')] for r in table if not re.match(r'^\s*\|?\s*-{3,}', r)]
            out.append('<table class="inv-table inv-inner"><thead><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in rows[0]) + '</tr></thead><tbody>'
                       + ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in rows[1:]) + '</tbody></table>')
            table.clear()
    for line in md.replace('\r', '').split('\n'):
        if line.strip().startswith('|'):
            table.append(line); continue
        flush()
        if line.strip().startswith('#'):
            out.append(f'<p><b>{inline(line.strip().lstrip("#").strip())}</b></p>')
        elif line.strip():
            out.append(f'<p>{inline(line.strip())}</p>')
    flush()
    return ''.join(out)


def heading(text, ch):
    return [text, ch * max(len(text), 4), '']


# ---------------------------------------------------------------------------------------------------------- load
nodes = load('osf_all.json')
node_by_id = {n['id']: n for n in nodes}
drive = load('gdrive_inventory.json')
zips = {}
for f in ('osf_split_zip_contents.json', 'osf_zip_contents.json'):
    zips.update(load(f))
drive_zips = load('gdrive_zip_contents.json')
refs = load('repo_refs.json')
hashes = load('osf_hashes.json')     # OSF's stored md5/sha256 per file id

# ---------------------------------------------------------------------------------- items: one row per "thing"
# split archives (name.zip.001, .002 ...) become one item, everything else one item per file
items = []      # dict(source, node, path, name, kind, size, parts, url, archive_key, modified, downloads)
for n in nodes:
    split = collections.OrderedDict()
    for f in n['files']:
        if f['kind'] != 'file':
            continue
        m = re.match(r'(.+?)\.(\d{3})$', f['name'])
        if m:
            split.setdefault(f['path'].rsplit('/', 1)[0] + '/' + m.group(1), []).append(f)
            continue
        items.append(dict(source='osf', node=n['id'], path=f['path'], name=f['name'], size=f['size'], parts=1,
                          url=osf_file_url(n['id'], f), modified=(f['modified'] or '')[:10], downloads=f['downloads'], sha=hashes.get(f['id'], {}).get('sha256'),
                          archive_key=f"{n['id']}:{f['path']}" if f['name'].lower().endswith('.zip') else None))
    for path, parts in split.items():
        parts.sort(key=lambda r: r['name'])
        folder = path.rsplit('/', 1)[0] + '/'
        folder_rec = next((f for f in n['files'] if f['kind'] == 'folder' and f['path'] == folder), None)
        dl = [p['downloads'] for p in parts if p['downloads'] is not None]
        items.append(dict(source='osf', node=n['id'], path=path, name=path.rsplit('/', 1)[1], size=sum(p['size'] or 0 for p in parts),
                          parts=len(parts), url=osf_file_url(n['id'], folder_rec) if folder_rec else f'https://osf.io/{n["id"]}/files/osfstorage',
                          modified=max((p['modified'] or '')[:10] for p in parts), downloads=min(dl) if dl else None,
                          archive_key=f"{n['id']}:{path}", part_names=[p['name'] for p in parts],
                          sha='+'.join(hashes.get(p['id'], {}).get('sha256') or '?' for p in parts)))
for d in drive['items']:
    if d['kind'] != 'file':
        continue
    items.append(dict(source='gdrive', node='gdrive', path=d['path'], name=d['name'], size=parse_drive_size(d['size']), parts=1,
                      url=d['url'], modified=drive_date(d['modified']), downloads=None,
                      archive_key=d['path'] if d['name'].lower().endswith('.zip') else None))


def archive_members(item):
    if not item.get('archive_key'):
        return None
    m = (drive_zips if item['source'] == 'gdrive' else zips).get(item['archive_key'])
    return m if isinstance(m, list) else None


# DeepLabCut weights come as 3 TensorFlow snapshot files per folder: show them as one model
def is_snapshot(it):
    return it['name'].startswith('snapshot-')


# --------------------------------------------------------------------------------------------- categories
CATS = collections.OrderedDict([
    ('dlc_model', ('DeepLabCut models (trained weights)', 'TensorFlow snapshot files (.data, .index, .meta) from DeepLabCut 2.x.')),
    ('other_model', ('Other pose / detection models', 'YOLO, Mask R-CNN and DeepPoseKit model files.')),
    ('labelled', ('Labelled images for pose-estimation training', 'Frames plus body-part annotations (DeepLabCut: PNG + CollectedData H5/CSV; YOLO: PNG + XML; Mask R-CNN: JPG; DeepPoseKit: H5 annotation set).')),
    ('videos', ('Videos', 'Raw experiment videos (MP4).')),
    ('classifier', ('Behavior classifiers (.sav)', 'Pickled scikit-learn random forest models, saved with an older scikit-learn. Tested 2026-10-15 on one file from OSF '
                   '(<code>3mc7g/lateral_threat.sav</code>) and one from Google Drive (<code>Model_Files/Allogroom_vigorous.sav</code>): both load and predict in '
                   'Python 3.6.13 with scikit-learn 0.22.2, and both fail in Python 3.10 with scikit-learn 1.4.2 '
                   '(<code>No module named &#x27;sklearn.ensemble.forest&#x27;</code>). On newer installs, retrain from the training data '
                   '(<a href="Scenario3.html">Scenario 3</a>). The mouse resident-intruder classifiers come in two generations that are not interchangeable: '
                   'the OSF set expects 490 features per frame, the Google Drive set 498.')),
    ('training', ('Classifier training data (features + behavior annotations)', 'Per-video CSVs with pose features and 0/1 behavior columns (SimBA <code>targets_inserted</code> format).')),
    ('project', ('Example / tutorial SimBA projects', 'Ready-made SimBA projects and tutorial material.')),
    ('meta', ('Classifier settings (SimBA meta files)', 'One CSV per classifier with the hyperparameters used to train it.')),
    ('importance', ('Feature importance tables', 'Random forest feature importances (mean decrease in impurity and permutation importance) per classifier.')),
    ('shap', ('SHAP explainability data', 'SHAP values for classifier explainability.')),
    ('scripts', ('Feature extraction scripts', 'Standalone Python feature extraction scripts per body-part configuration (2020; superseded by the SimBA package).')),
    ('talks', ('Talks, slides and QR codes', 'Presentations, the overview talk video and printable QR codes.')),
    ('other', ('Other files', '')),
    ('not_simba', ('Not SimBA', 'Files on the sgoldenlab OSF account that belong to other software.')),
])


def categorise(it):
    p, nm, low = it['path'], it['name'], it['name'].lower()
    if it['node'] == 'fvbuh':
        return 'not_simba'
    if it['node'] == '32vtd':
        return 'talks'
    if it['node'] == 'emxyw':
        return 'scripts'
    if it['node'] == 'dg385':
        return 'project'
    if is_snapshot(it):
        return 'dlc_model'
    if low.endswith('.sav'):
        return 'classifier'
    if '/YOLO/Models/' in p or '/mask RCNN/Models/' in p or '/DeepPoseKit/models/' in p:
        return 'other_model'
    if 'SHAP' in p:
        return 'shap'
    if 'SimBA meta files' in p:
        return 'meta'
    if 'feature importance' in p.lower():
        return 'importance'
    if 'Example_project' in p:
        return 'project'
    if it['source'] == 'gdrive':
        return 'training' if p.startswith('/Annotations/') else 'other'
    if '/Videos/' in p or low.endswith('.mp4'):
        return 'videos'
    if 'features' in p.lower() and 'annotation' in p.lower():
        return 'training'
    if 'abel' in p or 'annotation_sets' in p or it['node'] in ('7fgwn', 'm862q', 'puw5a', 'zs2hq', 'kym42', 'w7fs4', 'ds2g4') \
            or '/YOLO/Annotations/' in p or '/DeepPoseKit/skeleton/' in p:
        return 'labelled'
    return 'other'


for it in items:
    it['cat'] = categorise(it)

# ------------------------------------------------------------------------------------------ repo references
def ref_targets(url):
    """Map a repo URL to (node_id, storage_id_or_guid) keys."""
    m = re.match(r'https?://(?:www\.)?osf\.io/([a-z0-9]{5})/files/osfstorage/([0-9a-f]{24})', url)
    if m:
        return ('node', m.group(1)), ('storage', m.group(2))
    m = re.match(r'https?://(?:www\.)?osf\.io/([a-z0-9]{5})/files/([a-z0-9]{5})', url)
    if m:
        return ('node', m.group(1)), ('guid', m.group(2))
    m = re.match(r'https?://(?:www\.)?osf\.io/([a-z0-9]{5})/?$', url)
    if m:
        k = m.group(1)
        return (('node', k),) if k in node_by_id else (('guid', k),)
    if 'drive.google.com' in url:
        return (('node', 'gdrive'),)
    return ()


refs_by_key = collections.defaultdict(list)
unresolved_refs = []
guid_to_file = {(f['guid']): (n['id'], f) for n in nodes for f in n['files'] if f.get('guid')}
storage_to_file = {f['id']: (n['id'], f) for n in nodes for f in n['files']}
for r in refs:
    keys = ref_targets(r['url'])
    hit = False
    for k in keys:
        if k[0] == 'node' and (k[1] in node_by_id or k[1] == 'gdrive'):
            refs_by_key[k].append(r); hit = True
        elif k[0] == 'guid' and k[1] in guid_to_file:
            refs_by_key[k].append(r); hit = True
        elif k[0] == 'storage' and k[1] in storage_to_file:
            refs_by_key[k].append(r); hit = True
    if not hit:
        unresolved_refs.append(r)


def node_refs(node_id):
    return refs_by_key.get(('node', node_id), [])


def ref_html(rs, limit=6):
    seen, out = set(), []
    for r in rs:
        key = (r['file'], r['line'])
        if key in seen:
            continue
        seen.add(key)
        out.append(a(f"{GITHUB_BLOB}{r['file']}#L{r['line']}", f"{r['file']}:{r['line']}"))
    if len(out) > limit:
        out = out[:limit] + [f'+{len(out) - limit} more']
    return ', '.join(out)


def node_linked_anywhere(node_id):
    if node_refs(node_id):
        return True
    n = node_by_id[node_id]
    return any(node_linked_anywhere(c) for c in n['children'])


def file_refs(it):
    out = []
    if it['source'] != 'osf':
        return out
    n = node_by_id[it['node']]
    for f in n['files']:
        if it['path'].startswith(f['path']) or f['path'] == it['path']:
            if f.get('guid'):
                out += refs_by_key.get(('guid', f['guid']), [])
            out += refs_by_key.get(('storage', f['id']), [])
    return out


# ---------------------------------------------------------------------------------------------- findings
def path_of(node_id):
    chain, cur = [], node_by_id.get(node_id)
    while cur:
        chain.append(cur)
        parent = next((p for p in nodes if cur['id'] in p['children']), None)
        cur = parent
    return list(reversed(chain))


def location_html(it, with_link=True):
    if it['source'] == 'gdrive':
        crumbs = [a(drive['root_url'], 'Google Drive')] + [e(p) for p in it['path'].strip('/').split('/')[:-1]]
    else:
        crumbs = [a(n['url'], f"{n['title']} ({n['id']})") for n in path_of(it['node'])] + [e(p) for p in it['path'].strip('/').split('/')[:-1]]
    return ' › '.join(crumbs)


findings = []
# 1. empty folders
for n in nodes:
    for f in n['files']:
        if f['kind'] == 'folder' and not any(g['path'].startswith(f['path']) and g['kind'] == 'file' for g in n['files']):
            findings.append(('Empty folder', f"{a(osf_file_url(n['id'], f), n['title'] + ' › ' + f['path'].strip('/'))} contains no files."))
# 2. folder names that are notes to the reader
for n in nodes:
    for f in n['files']:
        if f['kind'] == 'folder' and re.search(r'SEE GOOGLE DRIVE|OUT OF DATE|NO LONGER UPDATED|NEED REFINEMENT', f['name']):
            findings.append(('Status note in a folder name', f"{a(osf_file_url(n['id'], f), n['title'] + ' › ' + f['path'].strip('/'))}"
                             + (' — the Google Drive folder it refers to is not linked anywhere on OSF.' if 'GOOGLE DRIVE' in f['name'] else '.')))
# 3. projects not linked from the SimBA repo
for n in nodes:
    if not node_linked_anywhere(n['id']) and not any(file_refs(it) for it in items if it['node'] == n['id']):
        findings.append(('Not linked from SimBA docs/README', f"{a(n['url'], n['title'] + ' (' + n['id'] + ')')} — {len([it for it in items if it['node'] == n['id']])} item(s)."))
if not node_refs('gdrive'):
    findings.append(('Not linked from SimBA docs/README', f"{a(drive['root_url'], 'Google Drive: ' + drive['title'])} — the mouse resident-intruder training data and 10 classifiers."))
# 4. byte-identical copies in more than one place (OSF sha256 checksums; split archives compare every part,
#    DeepLabCut models compare all three snapshot files, so same-architecture models of equal size are not mixed up)
units = collections.defaultdict(list)
for it in items:
    if it['source'] != 'osf' or not it.get('sha') or '?' in it['sha'] or not it['size'] or it['size'] < 1e6 and not is_snapshot(it):
        continue
    if is_snapshot(it):
        continue
    units[it['sha']].append(it)
models = collections.defaultdict(list)
for node_id, folder in sorted({(it['node'], it['path'].rsplit('/', 1)[0]) for it in items if is_snapshot(it)}):
    group = sorted([it for it in items if it['node'] == node_id and is_snapshot(it) and it['path'].rsplit('/', 1)[0] == folder], key=lambda g: g['name'])
    models['|'.join(g['sha'] or '?' for g in group)].append(dict(group[0], name=f"model {group[0]['name'].split('.')[0]} (3 files)", size=sum(g['size'] or 0 for g in group)))
for key, group in list(units.items()) + list(models.items()):
    if len(group) > 1:
        where = '; '.join(f"{location_html(g)} › <code>{e(g['name'])}</code>" for g in group)
        labels = [{t for t in ('RGB', 'GREYSCALE', 'CLAHE') if t in (location_html(g) + g['name']).upper().replace('GRAYSCALE', 'GREYSCALE')} for g in group]
        clash = len({frozenset(l) for l in labels if l}) > 1
        findings.append(('Byte-identical copies in more than one place', f"{hsize(group[0]['size'])}: {where}"
                         + (f' <span class="inv-flag">labelled {" vs. ".join("/".join(sorted(l)) for l in labels if l)}</span>' if clash else '')))
# 4b. two incompatible generations of mouse resident-intruder classifiers (checked by loading one file of each)
findings.append(('Two generations of mouse resident-intruder classifiers',
                 'The OSF mouse classifiers (Random forest behavioral classifiers › Mouse resident intruder, 3mc7g; tested <code>lateral_threat.sav</code>) expect '
                 '<b>490</b> features per frame; the Google Drive classifiers (tested <code>Allogroom_vigorous.sav</code>) expect <b>498</b>. Data made for one set will not run through the other.'))
# 5. DeepLabCut weights without config files
dlc_folders = sorted({(it['node'], it['path'].rsplit('/', 1)[0]) for it in items if is_snapshot(it)})
has_cfg = any(it['name'].lower() in ('config.yaml', 'pose_cfg.yaml') for it in items)
if dlc_folders and not has_cfg:
    findings.append(('DeepLabCut models are weights only', f'All {len(dlc_folders)} DeepLabCut model folders contain only the three TensorFlow snapshot files; '
                     'no <code>config.yaml</code> / <code>pose_cfg.yaml</code> is stored anywhere on OSF, neither as a file nor inside any of the archives, '
                     'so the network settings are not recorded. This page fills in the body parts from the annotation files of the matching '
                     'labelled-image archives; adding each model\'s <code>config.yaml</code> to OSF would make them self-contained.'))
# 6. archive read errors
for k, v in list(zips.items()) + list(drive_zips.items()):
    if not isinstance(v, list):
        findings.append(('Archive could not be read', f'<code>{e(k)}</code>: {e(v.get("error"))}'))

# --------------------------------------------------------------------------------------------- page parts
MANIFEST_DIR = os.path.join(REPO, 'docs', '_static', 'data_inventory')   # generated, every file in it is rewritten
MANIFEST_URL = '_static/data_inventory'
_manifest_rows, _manifest_written = [], {}


def write_manifest(it, files):
    """One CSV per archive (cached: archives appear in two sections of the page), plus rows for the combined CSV."""
    key = (it['node'], it['path'])
    if key in _manifest_written:
        return _manifest_written[key]
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', f"{it['node']}_{it['name']}").strip('_')
    name = f'{stem}.csv'
    with open(os.path.join(MANIFEST_DIR, name), 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file_in_archive', 'size_bytes', 'date'])
        for m in files:
            w.writerow([m['name'], m['size'], m['date']])
            _manifest_rows.append([it['source'], it['node'], it['path'], it['url'], m['name'], m['size'], m['date']])
    _manifest_written[key] = name
    return name


def archive_detail_html(it):
    members = archive_members(it)
    if members is None:
        return ''
    files = [m for m in members if not m['name'].endswith('/')]
    ext = collections.Counter((os.path.splitext(m['name'])[1].lower().lstrip('.') or 'no extension') for m in files)
    summary = ', '.join(f'{c:,} {x.upper()}' for x, c in ext.most_common())
    total = sum(m['size'] for m in files)
    csv_name = write_manifest(it, files)
    head = (f'<details class="inv-zip"><summary>Inside: {len(files):,} files ({summary}), {hsize(total)} unpacked</summary>'
            f'<p class="inv-note">Complete file list: <a href="{MANIFEST_URL}/{e(csv_name)}" download>{e(csv_name)}</a> (CSV, every file with size and date)</p>')
    if len(files) <= FULL_LIST_MAX:
        rows = ''.join(f'<tr><td>{e(m["name"])}</td><td class="num">{hsize(m["size"])}</td><td class="num">{m["date"]}</td></tr>' for m in files)
        return head + f'<table class="inv-table inv-inner"><thead><tr><th>File inside the archive</th><th>Size</th><th>Date</th></tr></thead><tbody>{rows}</tbody></table></details>'
    folders = collections.OrderedDict()
    for m in files:
        d = m['name'].rsplit('/', 1)[0] if '/' in m['name'] else '(archive root)'
        folders.setdefault(d, collections.Counter())[os.path.splitext(m['name'])[1].lower().lstrip('.') or '-'] += 1
    rows = ''.join(f'<tr><td>{e(d)}</td><td>{", ".join(f"{c} {x.upper()}" for x, c in cnt.most_common())}</td></tr>' for d, cnt in folders.items())
    return head + f'<p class="inv-note">Too many files to list one by one; showing the {len(folders)} folders and what each holds.</p>' \
                  f'<table class="inv-table inv-inner"><thead><tr><th>Folder inside the archive</th><th>Contents</th></tr></thead><tbody>{rows}</tbody></table></details>'


def item_row(it, show_location=False, show_cat=False):
    name = a(it['url'], it['name'])
    if it['parts'] > 1:
        name += f'<div class="inv-sub">split archive: {it["parts"]} parts (<code>.{it["part_names"][0].rsplit(".", 1)[1]}</code>–<code>.{it["part_names"][-1].rsplit(".", 1)[1]}</code>); download all parts, then open <code>.001</code> with 7-Zip</div>'
    name += archive_detail_html(it)
    cells = [name]
    if show_location:
        cells.append(location_html(it))
    if show_cat:
        cells.append(e(CATS[it['cat']][0]))
    cells += [f'<span class="num">{hsize(it["size"])}</span>', f'<span class="num">{e(it["modified"])}</span>', '' if it['downloads'] is None else str(it['downloads'])]
    return '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'


def items_table(rows_html, show_location=False, show_cat=False):
    hdr = ['Item'] + (['Where'] if show_location else []) + (['Type'] if show_cat else []) + ['Size', 'Modified', 'Downloads*']
    return ['<div class="inv-wrap"><table class="inv-table">', '<thead><tr>' + ''.join(f'<th>{h}</th>' for h in hdr) + '</tr></thead><tbody>'] + rows_html + ['</tbody></table></div>']


def dlc_model_rows():
    rows = []
    for node_id, folder in dlc_folders:
        group = [it for it in items if it['node'] == node_id and is_snapshot(it) and it['path'].rsplit('/', 1)[0] == folder]
        n = node_by_id[node_id]
        label = folder.strip('/').split('/')[-1] if folder.strip('/') else n['title']
        it0 = group[0]
        iters = sorted({re.match(r'snapshot-(\d+)', g['name']).group(1) for g in group})
        files = ' '.join(a(g['url'], g['name'].split('.', 1)[1]) for g in sorted(group, key=lambda g: g['name']))
        dl = [g['downloads'] for g in group if g['downloads'] is not None]
        cell = f'{e(label)}<div class="inv-sub">snapshot {", ".join(iters)} — files: {files}</div>'
        rows.append(f'<tr><td>{cell}</td><td>{location_html(it0)}</td><td class="num">{hsize(sum(g["size"] or 0 for g in group))}</td>'
                    f'<td class="nw">{e(max(g["modified"] for g in group))}</td><td>{max(dl) if dl else ""}</td></tr>')
    return rows


STYLE = """<style>
.inv-intro p { font-size:14.5px; line-height:1.6; }
.inv-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0; }
@media (max-width:820px){ .inv-stats { grid-template-columns:repeat(2,1fr); } }
.inv-stats div { border:1px solid #e1e4e8; border-radius:12px; padding:12px 14px; background:#fff; box-shadow:0 2px 8px rgba(0,0,0,.05); }
.inv-stats b { display:block; font-size:22px; color:#21567a; line-height:1.2; }
.inv-stats span { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:#2f8f9d; }
.inv-note { font-size:12.5px !important; color:#6a7884; line-height:1.5; }
.inv-wrap { overflow-x:auto; margin:6px 0 22px; border:1px solid #e1e4e8; border-radius:10px; }
table.inv-table { width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.45; margin:0 !important; }
table.inv-table th { background:#f4f8fb; color:#21567a; text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.03em;
  padding:7px 10px; border-bottom:1px solid #e1e4e8; white-space:nowrap; position:sticky; top:0; }
table.inv-table td { padding:6px 10px; border-bottom:1px solid #eef2f5; vertical-align:top; word-break:break-word; }
table.inv-table tr:hover > td { background:#fafcfd; }
table.inv-table td .num, table.inv-table td.num, table.inv-table td.nw { white-space:nowrap; font-variant-numeric:tabular-nums; }
table.inv-table a { color:#21567a; font-weight:600; }
tr.inv-folder > td { background:#f8fafb; color:#21567a; font-weight:600; }
.inv-sub { font-size:11.5px; color:#6a7884; font-weight:400; margin-top:2px; }
.inv-sub a { font-weight:400 !important; }
.inv-flag { display:inline-block; font-size:10.5px; font-weight:700; padding:1px 7px; border-radius:8px; background:#fdf0e6; color:#983412; margin-left:4px; }
/* project boxes reuse the docs' own <details> style (custom.css); only spacing is tuned here */
.rst-content details.inv-loc { margin:0 0 8px; }
.rst-content details.inv-loc > summary { font-weight:400; font-size:13.5px; }
.rst-content details.inv-loc.inv-child { margin-left:22px; }
.rst-content details.inv-loc .inv-wrap { margin-bottom:10px; }
/* small in-cell toggles ('Inside: ...'): same chevron, no box */
.rst-content details.inv-zip { margin:4px 0 0; border:none; background:none; box-shadow:none; border-radius:0; overflow:visible; }
.rst-content details.inv-zip > summary { padding:1px 0 1px 15px; background:none; font-size:11.5px; font-weight:600; color:#2f8f9d; }
.rst-content details.inv-zip > summary:hover { background:none; text-decoration:underline; }
.rst-content details.inv-zip > summary::before { left:2px; width:5px; height:5px; border-color:#2f8f9d; }
.rst-content details.inv-zip > :not(summary) { padding-left:0; padding-right:0; }
.rst-content details.inv-zip > :not(summary):last-child { padding-bottom:0; }
table.inv-inner { font-size:11.5px; border:1px solid #eef2f5; }
table.inv-inner th { position:static; }
.inv-wikibox { border-left:4px solid #21567a; background:#f4f8fb; border-radius:0 10px 10px 0; padding:8px 14px; margin:8px 0 12px; font-size:13px; }
.inv-wikibox p { margin:4px 0 !important; }
.inv-findings h4 { color:#21567a; margin:16px 0 6px; font-size:15px; }
.inv-findings li { font-size:13px; line-height:1.5; margin-bottom:5px; list-style:disc; }
.inv-findings ul { margin-left:18px !important; }
.inv-nav { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 4px; }
.inv-nav a { font-size:12.5px; font-weight:700; padding:5px 12px; border-radius:16px; background:#e8f4fb; color:#21567a !important; text-decoration:none !important; }
.inv-nav a:hover { background:#d3e9f6; }
.inv-callout { border-left:4px solid #c77c1b; background:#fdf6ec; border-radius:0 12px 12px 0; padding:10px 14px; margin:10px 0 18px; font-size:13px; line-height:1.55; color:#2c3e50; }
.inv-filter { position:sticky; top:0; z-index:3; background:#fcfcfc; padding:10px 0 6px; border-bottom:1px solid #eef2f5; margin-bottom:8px; }
#inv-q { width:100%; box-sizing:border-box; font-size:14px; padding:8px 12px; border:1px solid #c9d6e0; border-radius:10px; margin-bottom:8px; }
.inv-chips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:6px; }
.inv-chips button { font-size:11.5px; font-weight:600; padding:3px 10px; border-radius:12px; border:1px solid #c9d6e0; background:#fff; color:#21567a; cursor:pointer; }
.inv-chips button i { font-style:normal; color:#8b95a1; font-weight:400; margin-left:2px; }
.inv-chips button.on { background:#21567a; color:#fff; border-color:#21567a; }
.inv-chips button.on i { color:#cfe3f0; }
.inv-title { font-weight:600; color:#21567a; }
.inv-tag { display:inline-block; font-size:10.5px; font-weight:700; padding:1px 7px; border-radius:8px; background:#e8f4fb; color:#21567a; margin:3px 4px 0 0; }
.inv-credit { font-style:italic; }
.inv-type { font-size:11.5px; font-weight:700; color:#2f8f9d; white-space:nowrap; }
table.inv-start td:first-child { width:32%; }
a.inv-jump { font-weight:600; }
tr.inv-hit > td { background:#fff6d6 !important; transition:background .3s; }
</style>"""

# --------------------------------------------------------------------------------------------- page data
# data files committed to the SimBA GitHub repository
REPO_FOLDERS = collections.OrderedDict([
    ('tests/data', 'Small test projects and inputs used by the automated tests.'),
    ('misc', 'Example input and output files that the tutorials and docs link to (ROI, cue-light, BORIS/Ethovision/Observer imports, directionality, ...).'),
    ('docs/tables', 'CSV tables that are rendered as tables inside docs pages.'),
    ('docs/_static', 'Media and data embedded in the docs site.'),
    ('docs/images', 'Example outputs and videos shown in the docs (older location).'),
    ('images', 'Example outputs and videos from the original GitHub tutorials.'),
    ('simba/assets', 'Lookup tables and assets shipped with the SimBA package.'),
    ('simba/pose_configurations', 'Body-part configuration files shipped with the SimBA package.'),
    ('simba/feature_extractors', 'Feature-extractor support files shipped with the SimBA package.'),
    ('simba/sandbox', 'Tracked files inside the gitignored sandbox folder (committed before the folder was ignored).'),
    ('docs', 'Other data files directly under docs/.'),
])
repo_files = load('repo_data_files.json') if os.path.isfile(os.path.join(DATA, 'repo_data_files.json')) else []
if repo_files:
    def repo_folder(p):
        return next((f for f in REPO_FOLDERS if p.startswith(f + '/')), p.split('/')[0] if '/' in p else '(repo root)')
    groups = collections.OrderedDict()
    for f in sorted(repo_files, key=lambda r: r['path'].lower()):
        groups.setdefault(repo_folder(f['path']), []).append(f)
    order = [g for g in REPO_FOLDERS if g in groups] + [g for g in groups if g not in REPO_FOLDERS]
else:
    groups, order = {}, []

# other labs' OSF projects linked from the docs
other = collections.OrderedDict()
for r in unresolved_refs:
    m = re.match(r'https?://(?:www\.)?osf\.io/([a-z0-9]{5})', r['url'])
    if m and m.group(1) != 'user':
        other.setdefault(m.group(1), []).append(r)

# --------------------------------------------------------------------------------------------- write page
exec(open(os.path.join(HERE, 'page_layout.py'), encoding='utf-8').read())

# gzipped: ~21 MB as plain CSV, every row is also in the per-archive CSVs
# mtime=0 so an unchanged inventory gives a byte-identical file (no git churn on rebuilds)
with open(os.path.join(MANIFEST_DIR, 'all_archive_contents.csv.gz'), 'wb') as raw, \
        io.TextIOWrapper(gzip.GzipFile(fileobj=raw, mode='wb', mtime=0), encoding='utf-8', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['source', 'project', 'archive_path', 'archive_url', 'file_in_archive', 'size_bytes', 'date'])
    for it in items:   # every archive, including identical copies the page folds into one row
        for m in archive_members(it) or []:
            if not m['name'].endswith('/'):
                w.writerow([it['source'], it['node'], it['path'], it['url'], m['name'], m['size'], m['date']])
page = '\n'.join(L) + '\n'
open(OUT, 'w', encoding='utf-8').write(page)
# the folder only holds this page's downloads: drop files the page no longer links (renamed / removed archives)
for fn in os.listdir(MANIFEST_DIR):
    if f'{MANIFEST_URL}/{e(fn)}' not in page:
        os.remove(os.path.join(MANIFEST_DIR, fn))
        print('removed unlinked', fn)
print('wrote', OUT, '| items', len(items), '| findings', len(findings), '| unresolved refs', len(unresolved_refs))
for k, v in grouped.items():
    print(f'  {k}: {len(v)}')
