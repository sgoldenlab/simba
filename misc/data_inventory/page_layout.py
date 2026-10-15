# Page layout for build_data_inventory.py (exec'd at the end of it, so every name defined there is available).
# One page, ordered for finding things: jump menu -> start here -> one searchable table -> browse by location
# -> repo files -> maintainer notes -> about.

SHORT_TYPE = {'dlc_model': 'DeepLabCut model', 'other_model': 'Other pose model', 'labelled': 'Labelled images', 'videos': 'Videos',
              'classifier': 'Classifier', 'training': 'Training data', 'project': 'Example project', 'meta': 'Classifier settings',
              'importance': 'Feature importance', 'shap': 'SHAP', 'scripts': 'Scripts', 'talks': 'Talks & slides', 'other': 'Other',
              'not_simba': 'Not SimBA'}
GROUPABLE = {'meta', 'importance', 'scripts', 'training', 'videos', 'other', 'project', 'labelled', 'not_simba'}
bodyparts = load('archive_bodyparts.json') if os.path.isfile(os.path.join(DATA, 'archive_bodyparts.json')) else {}
feature_cols = load('feature_columns.json') if os.path.isfile(os.path.join(DATA, 'feature_columns.json')) else {}
current_cols = load('current_feature_columns.json') if os.path.isfile(os.path.join(DATA, 'current_feature_columns.json')) else {}
BOOKKEEPING = {'video_no', 'frames'}   # row bookkeeping in older SimBA files, not features
CRIM13_CREDIT = ('CRIM13 videos come from the Caltech Resident-Intruder Mouse dataset (Burgos-Artizzu, Doll&aacute;r, Lin, Anderson &amp; Perona, '
                 '&ldquo;Social behavior recognition in continuous video&rdquo;, CVPR 2012); please cite it when you use this data.')


# ---------------------------------------------------------------- classifier training settings (SimBA meta files)
def clf_key(name):
    k = os.path.splitext(name.lower())[0]
    for suffix in ('_prediction', '_rat', '_crim', '_sniffing', '_sniff'):
        k = k.replace(suffix, '')
    return k


def is_number(s):
    try:
        float(s); return True
    except (TypeError, ValueError):
        return False


META_DATASET = {'Mouse resident-intruder': 'mouse_drive', 'Rat resident intruder': 'rat', 'CRIM13': 'crim13'}
clf_settings = {}   # (dataset, key) -> dict
for n in nodes:
    for f in n['files']:
        if f['kind'] != 'file' or 'SimBA meta files' not in f['path']:
            continue
        p = os.path.join(DATA, 'meta_files', f"{n['id']}__{f['path'].strip('/').replace('/', '__')}")
        if not os.path.isfile(p):
            continue
        row = next(csv.DictReader(open(p, encoding='utf-8', errors='ignore')))
        shifted = is_number(row.get('under_sample_setting')) and (row.get('over_sample_ratio') or '').lower().startswith('random')
        under_ratio = row.get('under_sample_setting') if shifted else row.get('under_sample_ratio')
        under_on = shifted or (row.get('under_sample_setting') or '').lower().startswith('random')
        dataset = META_DATASET[f['path'].strip('/').split('/')[1]]
        clf_settings[(dataset, clf_key(f['name']))] = {
            'url': osf_file_url(n['id'], f), 'file': f['name'], 'name_inside': row.get('Classifier_name', ''), 'shifted': shifted,
            'trees': row.get('RF_n_estimators'), 'max_features': row.get('RF_max_features'), 'criterion': row.get('RF_criterion'),
            'test': row.get('train_test_size'), 'leaf': row.get('RF_min_sample_leaf'),
            'under': f'random undersample, ratio {under_ratio}' if under_on and is_number(under_ratio) and float(under_ratio) > 0 else 'no undersampling'}


def classifier_dataset(it):
    if it['source'] == 'gdrive':
        return 'mouse_drive'
    p = (plain_location(it)).lower()
    if it['node'] == '3mc7g':
        return 'mouse_osf'
    return 'rat' if 'rat' in p.split() or 'rat resident' in p else ('crim13' if 'crim13' in p else None)


def settings_html(it):
    ds = classifier_dataset(it)
    s = clf_settings.get((ds, clf_key(it['name'])))
    if not s:
        why = ' (none were uploaded for this older 490-feature set)' if ds == 'mouse_osf' else ''
        return f'<div class="inv-sub">Training settings: no settings file on OSF{why}.</div>'
    pct = f'{float(s["test"]) * 100:.0f}% test' if is_number(s['test']) else ''
    txt = ' · '.join(x for x in [f'{int(float(s["trees"])):,} trees' if is_number(s['trees']) else '', f'max features {s["max_features"]}', s['criterion'],
                                 pct, f'min leaf {s["leaf"]}', s['under']] if x)
    flags = ''
    if clf_key(s['name_inside']) != clf_key(it['name']):
        flags += f' <span class="inv-flag">file names the classifier &ldquo;{e(s["name_inside"])}&rdquo;</span>'
    if s['shifted']:
        flags += ' <span class="inv-flag">columns shifted in file</span>'
    return f'<div class="inv-sub"><b>Training settings</b> ({a(s["url"], s["file"])}): {e(txt)}{flags}</div>'


# ---------------------------------------------------------------- column lists of the training data
def column_kinds(cols):
    """index / pose / bookkeeping / feature / behavior for each column of a SimBA targets_inserted CSV.
    Pose = <bodypart>_x/_y/_p triplets; behaviors = the columns after the last Low_prob_detections* feature."""
    kinds = []
    last_feat = max((i for i, c in enumerate(cols) if c.startswith('Low_prob_detections')), default=None)
    for i, c in enumerate(cols):
        if i == 0 and (c == '' or c.lower().startswith('unnamed') or c.lower() in ('frame', 'index', 'scorer')):
            kinds.append('index')
        elif re.search(r'_(x|y|p)$', c, re.I) and all(f'{c[:-2]}_{s}' in cols or f'{c[:-2]}_{s.upper()}' in cols for s in 'xyp'):
            kinds.append('pose')
        elif c in BOOKKEEPING:
            kinds.append('bookkeeping')
        elif last_feat is not None and i > last_feat:
            kinds.append('behavior')
        else:
            kinds.append('feature')
    return kinds


def compare_with_current(cols, kinds, n_bp):
    """Pose + feature columns vs. what this repo's feature extractor writes for the same body-part count.
    Returns (renamed pairs at the same position, columns current SimBA adds, columns only in this file) or None."""
    cur = current_cols.get(str(n_bp))
    if not cur:
        return None
    old = [c for c, k in zip(cols, kinds) if k in ('pose', 'feature')]
    renamed, added, removed = [], [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, cur, autojunk=False).get_opcodes():
        if op == 'replace' and i2 - i1 == j2 - j1:
            renamed += list(zip(old[i1:i2], cur[j1:j2]))
        elif op != 'equal':
            removed += old[i1:i2]
            added += cur[j1:j2]
    return renamed, added, removed


def comparison_html(cmp):
    renamed, added, removed = cmp
    if not (renamed or added or removed):
        return 'same feature columns, names and order as current SimBA.'
    parts = []
    if renamed:
        parts.append(f'{len(renamed)} columns have older names but sit in the same positions ('
                     + ', '.join(f'<code>{e(a)}</code> &rarr; <code>{e(b)}</code>' for a, b in renamed[:2]) + (', ...' if len(renamed) > 2 else '') + ')')
    if added:
        parts.append(f'current SimBA adds {len(added)} features this file lacks (' + ', '.join(f'<code>{e(c)}</code>' for c in added[:4]) + (', ...' if len(added) > 4 else '') + ')')
    if removed:
        parts.append(f'{len(removed)} columns current SimBA no longer computes (' + ', '.join(f'<code>{e(c)}</code>' for c in removed[:4]) + (', ...' if len(removed) > 4 else '') + ')')
    return '; '.join(parts) + '.'


_fc_written = {}


def feature_columns_link(u):
    """Link to a CSV listing the columns of this training dataset (written once per dataset)."""
    it = u['items'][0]
    if u['cat'] not in ('training', 'project'):
        return ''
    if it['source'] == 'gdrive':
        key = next((k for k in feature_cols if k.startswith('gdrive:')), None)
    elif u['kind'] == 'group' and it['node'] == 'dg385':
        key = next((k for k in feature_cols if k.startswith('dg385:')), None)
    else:
        key = f"{it['node']}:{it['path']}" if f"{it['node']}:{it['path']}" in feature_cols else None
    if not key:
        return ''
    if key not in _fc_written:
        cols = feature_cols[key]['columns']
        kinds = column_kinds(cols)
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', key.split(':')[0] + '_' + os.path.splitext(key.rsplit('/', 1)[-1])[0]).strip('_')
        name = f'columns_{stem}.csv'
        os.makedirs(MANIFEST_DIR, exist_ok=True)
        with open(os.path.join(MANIFEST_DIR, name), 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['column_number', 'column_name', 'kind'])
            for i, (c, k) in enumerate(zip(cols, kinds)):
                w.writerow([i, c, k])
        cnt = collections.Counter(kinds)
        _fc_written[key] = (name, cnt, feature_cols[key]['example_file'])
    name, cnt, example = _fc_written[key]
    cols = feature_cols[key]['columns']
    kinds = column_kinds(cols)
    behaviors = [c for c, k in zip(cols, kinds) if k == 'behavior']
    bps = list(dict.fromkeys(c[:-2] for c, k in zip(cols, kinds) if k == 'pose'))
    keep = [c for c, k in zip(cols, kinds) if k == 'bookkeeping']
    out = (f'<div class="inv-sub"><b>Columns</b> (header of <code>{e(example.rsplit("/", 1)[-1])}</code>): {len(bps)} body parts ({cnt["pose"]} pose columns), '
           f'<b>{cnt["feature"]} features</b>, '
           + (f'{len(keep)} bookkeeping columns ({e(", ".join(keep))}), ' if keep else '')
           + f'behavior column(s): {e(", ".join(behaviors))} &mdash; '
           f'<a href="{MANIFEST_URL}/{e(name)}" download>full column list (CSV)</a></div>')
    m = pose_match(bps)
    if m:
        nm, same_order, exact = m[0]
        also = f' (or &ldquo;{e(m[1][0])}&rdquo; for multi-animal trackers)' if len(m) > 1 else ''
        how = 'same names and order' if exact else ('same body parts and order, spelling differs' if same_order else 'same body parts, different order')
        out += f'<div class="inv-sub"><b>SimBA body-part setting:</b> &ldquo;{e(nm)}&rdquo;{also} &mdash; {how}.</div>'
    cmp = compare_with_current(cols, kinds, len(bps))
    if cmp:
        out += f'<div class="inv-sub"><b>Compared with current SimBA feature extraction:</b> {comparison_html(cmp)}</div>'
    return out


# ---------------------------------------------------------------- body parts -> SimBA's built-in pose settings
POSE_DIR = os.path.join(REPO, 'simba', 'pose_configurations')
pose_configs = []
try:
    names = [l.strip() for l in open(os.path.join(POSE_DIR, 'configuration_names', 'pose_config_names.csv'), encoding='utf-8') if l.strip()]
    bps = [[b for b in l.strip().split(',') if b] for l in open(os.path.join(POSE_DIR, 'bp_names', 'bp_names.csv'), encoding='utf-8')]
    pose_configs = list(zip(names, bps))
except OSError:
    pass


def bp_norm(b):
    toks = b.lower().replace('lateral', 'lat').split('_')
    return '_'.join(sorted(toks))


def pose_match(bodyparts_list):
    want = [bp_norm(b) for b in bodyparts_list]
    out = []
    for name, cfg in pose_configs:
        have = [bp_norm(b) for b in cfg]
        if sorted(have) == sorted(want):
            exact = [b.lower() for b in cfg] == [b.lower() for b in bodyparts_list]
            out.append((name, have == want, exact))
    return out


def plain_location(it):
    if it['source'] == 'gdrive':
        return 'Google Drive mouse resident-intruder ' + it['path']   # the folder holds only mouse RI data (checked)
    return ' '.join(n['title'] for n in path_of(it['node'])) + ' ' + it['path']


def tags_for(text):
    t = text.lower()
    species = 'Rat' if re.search(r'(?<![a-z])rat(?![a-z])', t) else ('Mouse (CRIM13)' if 'crim' in t else
               ('Mouse' if re.search(r'mouse|mice|c57', t) else ''))   # explicit words only: file names like RI_02 are used for mice and rats
    setup = ('resident-intruder' if re.search(r'resident|(?<![a-z])ri(?![a-z])|csds|crim', t) else
             'open field' if re.search(r'open.?field|openfield|_of_', t) else      # not a bare "of": snapshot files are named data-00000-of-00001
             'operant' if 'operant' in t else '')
    imaging = [x for x, p in (('RGB', r'rgb'), ('greyscale', r'grey|gray|(?<![a-z])bw(?![a-z])'), ('CLAHE', r'clahe')) if re.search(p, t)]
    return species, setup, imaging


# ------------------------------------------------------------------ units: one table row per dataset
units = []
done = set()
# DeepLabCut models (3 snapshot files each)
for node_id, folder in dlc_folders:
    group = sorted([it for it in items if it['node'] == node_id and is_snapshot(it) and it['path'].rsplit('/', 1)[0] == folder], key=lambda g: g['name'])
    label = folder.strip('/').split('/')[-1] if folder.strip('/') else node_by_id[node_id]['title']
    units.append({'kind': 'model', 'cat': 'dlc_model', 'title': label, 'items': group, 'sig': '|'.join(g['sha'] or '?' for g in group)})
    done.update(id(g) for g in group)
# the Tutorial Data project is one complete set: one row
tut_items = [it for it in items if it['node'] == 'dg385']
if tut_items:
    units.append({'kind': 'group', 'cat': 'project', 'title': f'Tutorial Data: 3 videos, tracking, annotations, 2 classifiers ({len(tut_items)} files)',
                  'items': sorted(tut_items, key=lambda g: g['path'].lower()), 'sig': None})
    done.update(id(g) for g in tut_items)
# folders with many small files of one type -> one row
by_folder = collections.defaultdict(list)
for it in items:
    if id(it) in done or it.get('archive_key') or it['cat'] not in GROUPABLE:
        continue
    by_folder[(it['node'], it['path'].rsplit('/', 1)[0], it['cat'])].append(it)
for (node_id, folder, cat), group in by_folder.items():
    if len(group) >= 4:
        parts = folder.strip('/').split('/') if folder.strip('/') else [node_by_id[node_id]['title'] if node_id != 'gdrive' else 'Google Drive']
        # generic folder names (e.g. "Permutation importance") get their parent in front so the row says which dataset
        label = ' › '.join(parts[-2:]) if len(parts) > 1 and cat in ('importance', 'videos') else parts[-1]
        if node_id == 'gdrive' and cat == 'training':
            label = parts[-1].replace('_', ' ') + ' annotations'
        units.append({'kind': 'group', 'cat': cat, 'title': f'{label} ({len(group)} files)', 'items': sorted(group, key=lambda g: g['name'].lower()), 'sig': None})
        done.update(id(g) for g in group)
for it in items:
    if id(it) not in done:
        units.append({'kind': 'item', 'cat': it['cat'], 'title': it['name'], 'items': [it], 'sig': it.get('sha') if it['source'] == 'osf' else None})
        if it['source'] == 'gdrive' and it['path'].startswith('/Annotations/') and it.get('archive_key'):
            units[-1]['behavior'] = it['path'].split('/')[2]   # Drive annotation zips are named after a video, not the behavior

# fold byte-identical copies into one row; the copy inside the main SimBA project is the one shown first
merged = collections.OrderedDict()
for u in units:
    key = u['sig'] if u['sig'] and '?' not in u['sig'] else id(u)
    merged.setdefault(key, []).append(u)
rows_units = []
for key, us in merged.items():
    us.sort(key=lambda u: (node_by_id[u['items'][0]['node']]['root'] != 'tmu6y' if u['items'][0]['source'] == 'osf' else False))
    primary = dict(us[0]); primary['copies'] = us[1:]
    rows_units.append(primary)
CAT_ORDER = list(CATS)
rows_units.sort(key=lambda u: (CAT_ORDER.index(u['cat']), plain_location(u['items'][0]).lower(), u['title'].lower()))
for i, u in enumerate(rows_units):
    u['id'] = f'd{i + 1}'


def unit_size(u):
    return sum(i['size'] or 0 for i in u['items'])


def unit_downloads(u):
    vals = []
    for uu in [u] + u['copies']:
        d = [i['downloads'] for i in uu['items'] if i['downloads'] is not None]
        if d:
            vals.append(min(d) if uu['kind'] == 'model' else (sum(d) if uu['kind'] == 'group' else d[0]))
    return sum(vals) if vals else None


def unit_refs(u):
    out = []
    for uu in [u] + u['copies']:
        for it in uu['items']:
            out += file_refs(it)
            if it['source'] == 'osf':
                n = node_by_id[it['node']]
                if not n['children'] and len([x for x in items if x['node'] == n['id']]) <= 6:
                    out += node_refs(n['id'])
            elif it['source'] == 'gdrive':
                out += node_refs('gdrive')
    return out


def archive_bodyparts(it):
    key = (it['archive_key'] or '')
    v = bodyparts.get(key)
    return v.get('schemes') if isinstance(v, dict) else None


def model_bodyparts(u):
    """Body parts of a DeepLabCut model, from the CollectedData header of the labelled images for the same setup."""
    it = u['items'][0]
    setup = it['path'].rsplit('/', 1)[0].strip('/').split('/')[-1] if it['path'].strip('/').count('/') else node_by_id[it['node']]['title']
    for uu in [u] + u['copies']:
        f = uu['items'][0]
        name = f['path'].rsplit('/', 1)[0].strip('/').split('/')[-1] if f['path'].strip('/').count('/') else ''
        if name:
            setup = name
    for key, v in bodyparts.items():
        if isinstance(v, dict) and v.get('schemes') and key.startswith('sr3ck:') and f'/{setup}/' in key:
            return v['schemes'], key
    return None, None


def bodyparts_html(schemes, source_note):
    if not schemes:
        return ''
    out = []
    for s in schemes:
        who = f'{len(s["individuals"])} animals ({", ".join(s["individuals"])}) × ' if s['individuals'] else ''
        out.append(f'{who}{len(s["bodyparts"])} body parts: {", ".join(s["bodyparts"])}')
    html_ = f'<div class="inv-sub"><b>Body parts</b> ({source_note}): ' + ' <i>or</i> '.join(e(x) for x in out) + '</div>'
    for s in schemes:
        m = pose_match(s['bodyparts'])
        if m:
            name, same_order, exact = m[0]
            also = f' (or &ldquo;{e(m[1][0])}&rdquo; for multi-animal trackers)' if len(m) > 1 else ''
            how = 'same names and order' if exact else ('same body parts and order; names differ only in spelling, e.g. Lateral/Lat' if same_order
                                                         else 'same body parts in a different order')
            html_ += f'<div class="inv-sub"><b>SimBA body-part setting:</b> &ldquo;{e(name)}&rdquo;{also} &mdash; {how}.</div>'
        else:
            html_ += f'<div class="inv-sub"><b>SimBA body-part setting:</b> no built-in match for these {len(s["bodyparts"])} body parts; create a user-defined pose configuration.</div>'
    return html_


def unit_row(u):
    it = u['items'][0]
    text_for_tags = ' '.join(plain_location(x) + ' ' + x['name'] for uu in [u] + u['copies'] for x in uu['items'][:3])
    species, setup, imaging = tags_for(text_for_tags)
    own_imaging = tags_for(' '.join([u['title']] + [x['name'] for x in u['items'][:3]]))[2]
    if own_imaging:   # a file's own name beats its folder's (e.g. "RGB & CLAHE & BW" folder holding one RGB archive)
        imaging = own_imaging
    tag_html = ''.join(f'<span class="inv-tag">{e(t)}</span>' for t in [species, setup] + imaging if t)
    if u['kind'] == 'model':
        title = e(u['title'])
        iters = sorted({re.match(r'snapshot-(\d+)', g['name']).group(1) for g in u['items']})
        sub = f'<div class="inv-sub">TensorFlow snapshot {", ".join(iters)}: ' + ' · '.join(a(g['url'], g['name'].split('.', 1)[1]) for g in u['items']) + '</div>'
        schemes, src = model_bodyparts(u)
        sub += bodyparts_html(schemes, 'from the labelled images for this setup') if schemes else '<div class="inv-sub">Body parts: not recorded (no config file on OSF)</div>'
    elif u['kind'] == 'group':
        title = e(u['title'])
        lis = ''.join(f'<tr><td>{a(g["url"], g["name"])}</td><td class="num">{hsize(g["size"])}</td><td class="num">{e(g["modified"])}</td></tr>' for g in u['items'])
        sub = feature_columns_link(u) + (f'<details class="inv-zip"><summary>Show the {len(u["items"])} files</summary><table class="inv-table inv-inner"><thead><tr><th>File</th><th>Size</th><th>Modified</th></tr></thead>'
                                         f'<tbody>{lis}</tbody></table></details>')
    else:
        title = a(it['url'], it['name'])
        if u['cat'] == 'other_model':   # file names like model_0.h5 say nothing; lead with the tool and setup folder
            parts = it['path'].strip('/').split('/')
            title = f'{e(parts[0])} model' + (f': {e(parts[-2])}' if len(parts) > 3 else '') + f' <span class="inv-sub">({a(it["url"], it["name"])})</span>'
        if u.get('behavior'):
            title = f'{a(it["url"], u["behavior"].replace("_", " ") + " annotations")} <span class="inv-sub">({e(it["name"])})</span>'
        sub = ''
        if it['parts'] > 1:
            sub += (f'<div class="inv-sub">Split archive, {it["parts"]} parts (<code>.{it["part_names"][0].rsplit(".", 1)[1]}</code>–<code>.{it["part_names"][-1].rsplit(".", 1)[1]}</code>): '
                    'download all parts into one folder, then open <code>.001</code> with 7-Zip. '
                    f'To check the download, compare each part with its MD5/SHA-256 in <a href="{MANIFEST_URL}/checksums.csv" download>checksums.csv</a>.</div>')
        if u['cat'] == 'classifier':
            sub += settings_html(it)
        fc = feature_columns_link(u)
        if fc:
            sub += fc
        schemes = archive_bodyparts(it) if it.get('archive_key') else None
        if schemes:
            sub += bodyparts_html(schemes, 'from the CollectedData annotation files inside')
        sub += archive_detail_html(it)
    if species == 'Mouse (CRIM13)' and u['cat'] in ('videos', 'labelled', 'training', 'dlc_model'):
        sub += f'<div class="inv-sub inv-credit">{CRIM13_CREDIT}</div>'
    where = location_html(it)
    if u['copies']:
        where += '<div class="inv-sub">Identical copy also at: ' + '; '.join(location_html(c['items'][0]) for c in u['copies']) + '</div>'
    refs_ = unit_refs(u)
    if refs_:
        where += f'<div class="inv-sub">Linked from: {ref_html(refs_, limit=4)}</div>'
    dl = unit_downloads(u)
    search = (u['title'] + ' ' + u.get('behavior', '') + ' ' + text_for_tags + ' ' + SHORT_TYPE[u['cat']] + ' ' + ' '.join([species, setup] + imaging)).lower()
    return (f'<tr id="{u["id"]}" data-type="{u["cat"]}" data-species="{e(species)}" data-search="{e(search)}">'
            f'<td><div class="inv-title">{title}</div>{tag_html}{sub}</td><td><span class="inv-type">{e(SHORT_TYPE[u["cat"]])}</span></td>'
            f'<td>{where}</td><td class="num">{hsize(unit_size(u))}</td><td class="num">{"" if dl is None else dl}</td></tr>')


# ------------------------------------------------------------------ page
L = []
L += [':orphan:', '',
      '.. GENERATED FILE, do not edit by hand. Rebuild with misc/data_inventory/build_data_inventory.py',
      '   (see misc/data_inventory/README.md). Hand edits here are overwritten by the next build.', '',
      '.. _data_inventory:', '', '.. meta::', '   :description: Every public SimBA dataset, model, classifier, video and annotation file on OSF and Google Drive, and where to find it.', '']
L += heading('SimBA data inventory', '=')
drive_items = [it for it in items if it['source'] == 'gdrive']
n_archive_files = sum(len([m for m in (archive_members(it) or []) if not m['name'].endswith('/')]) for it in items)
raw_osf_files = sum(1 for n in nodes for f in n['files'] if f['kind'] == 'file')
total = sum(it['size'] or 0 for it in items)
NAV = [('start-here', 'Start here'), ('all-data', 'All data (search)'), ('browse-by-location', 'Browse by location'),
       ('data-files-in-the-simba-github-repository', 'Files in the GitHub repo'), ('for-maintainers', 'For maintainers'), ('about-this-page', 'About this page')]
L += raw(STYLE.splitlines() + [
    '<div class="inv-intro">',
    f'<p>Every piece of data the SimBA team has shared publicly, and where it is: pose-estimation models, labelled images, videos, '
    f'behavior classifiers, training data, example projects and talks. It covers {len(nodes)} OSF projects on the '
    f'{a("https://osf.io/user/mutws", "sgoldenlab OSF account")}, one Google Drive folder and the data files inside the SimBA GitHub repository.</p>',
    '<div class="inv-stats">',
    f'<div><b>{len(rows_units)}</b><span>datasets &amp; files</span></div>',
    f'<div><b>{n_archive_files:,}</b><span>files inside archives</span></div>',
    f'<div><b>{total / 1e9:,.0f} GB</b><span>on OSF + Drive</span></div>',
    f'<div><b>{SNAPSHOT_DATE}</b><span>snapshot date</span></div>',
    '</div>',
    '<nav class="inv-nav">' + ''.join(f'<a href="#{k}">{e(v)}</a>' for k, v in NAV) + '</nav>',
    '</div>'])

# ---- start here
L += heading('Start here', '-')


def first_unit(pred):
    return next((u for u in rows_units if pred(u)), None)


def jump(u, text=None):
    return f'<a href="#{u["id"]}" class="inv-jump">{e(text or u["title"])}</a>' if u else ''


def chip_link(cat, text):
    return f'<a href="#all-data" class="inv-jump" data-set-type="{cat}">{e(text)}</a>'


tut = node_by_id.get('dg385')
ex_proj = first_unit(lambda u: u['cat'] == 'project' and 'Example_project' in u['items'][0]['path'])
mouse_models = [u for u in rows_units if u['cat'] == 'dlc_model' and '16BP' in u['items'][0]['path']]
of_model = first_unit(lambda u: u['cat'] == 'dlc_model' and 'open-field' in u['items'][0]['path'])
op_model = first_unit(lambda u: u['cat'] == 'dlc_model' and 'operant environment' in u['items'][0]['path'])
rat_model = first_unit(lambda u: u['cat'] == 'dlc_model' and 'Rat resident' in u['items'][0]['path'])
rat_train = first_unit(lambda u: u['cat'] == 'training' and 'Rat_RI' in u['title'])
crim_train = first_unit(lambda u: u['cat'] == 'training' and 'CRIM13' in u['title'])
start_rows = [
    ('Try SimBA end to end on a small dataset',
     f'{a(tut["url"], "Tutorial Data")}: 3 short videos with their tracking data, annotated <code>targets_inserted</code> files and two classifiers (attack, defensive), 2.7 GB.' if tut else ''),
    ('Open a finished SimBA project', f'{jump(ex_proj, "Mouse resident-intruder example project (attack)")}, 2.3 GB.' if ex_proj else ''),
    ('Track two mice (white + black) in a home cage', 'DeepLabCut models, 16 body parts: ' + ', '.join(jump(u, re.sub(r"^Mouse resident-intruder: home-cage, ", "", u["title"])) for u in mouse_models) + '.'),
    ('Track one mouse in an open field or operant box', ', '.join(x for x in [jump(of_model, 'open field model'), jump(op_model, 'operant model')] if x) + '.'),
    ('Track two rats', f'{jump(rat_model, "rat resident-intruder model")}.' if rat_model else ''),
    ('Score mouse resident-intruder behaviors', f'Classifiers and per-behavior annotations on {a(drive["root_url"], "Google Drive")} (16 body parts, 498 features; see the note below the table). '
     'Older OSF classifiers use 490 features and are not interchangeable with these.'),
    ('Score rat resident-intruder or CRIM13 behaviors', ', '.join(x for x in [jump(rat_train, 'rat training data'), jump(crim_train, 'CRIM13 training data')] if x)
     + ', and the matching classifiers: ' + chip_link('classifier', 'show all classifiers') + '. '
     + '<span class="inv-sub">' + CRIM13_CREDIT + '</span>'),
    ('Train your own pose model', chip_link('labelled', 'Labelled images') + ' (PNG frames + body-part annotations).'),
    ('Get raw experiment videos', chip_link('videos', 'Videos') + '.'),
]
L += raw(['<div class="inv-wrap"><table class="inv-table inv-start"><thead><tr><th>I want to…</th><th>Use</th></tr></thead><tbody>']
         + [f'<tr><td><b>{e(k)}</b></td><td>{v}</td></tr>' for k, v in start_rows if v] + ['</tbody></table></div>',
         '<div class="inv-callout"><b>Before you download a classifier (.sav):</b> they were saved with scikit-learn 0.22 and only load in SimBA&rsquo;s '
         'Python 3.6 install (tested). On Python 3.9+ installs, retrain from the training data instead (<a href="Scenario3.html">Scenario 3</a>).</div>'])

# ---- all data
L += heading('All data', '-')
type_counts = collections.Counter(u['cat'] for u in rows_units)
species_counts = collections.Counter(tags_for(' '.join(plain_location(x) + ' ' + x['name'] for uu in [u] + u['copies'] for x in uu['items'][:3]))[0] for u in rows_units)
chips_type = ''.join(f'<button type="button" data-type="{c}">{e(SHORT_TYPE[c])} <i>{type_counts[c]}</i></button>' for c in CATS if type_counts[c])
chips_species = ''.join(f'<button type="button" data-species="{e(s)}">{e(s)} <i>{species_counts[s]}</i></button>' for s in ('Mouse', 'Mouse (CRIM13)', 'Rat') if species_counts[s])
L += raw([
    '<p>One row per dataset. Identical copies stored in several places are shown once, with the other locations under "Identical copy also at". '
    'Type in the box or click the buttons to filter; click a row&rsquo;s "Inside" line to see what an archive contains.</p>',
    '<div class="inv-filter">',
    '<input id="inv-q" type="search" placeholder="Search, e.g. attack, rat, CLAHE, open field, YOLO" aria-label="Search the data table">',
    f'<div class="inv-chips" data-group="type"><button type="button" data-type="" class="on">All types</button>{chips_type}</div>',
    f'<div class="inv-chips" data-group="species"><button type="button" data-species="" class="on">All animals</button>{chips_species}</div>',
    '<div id="inv-count" class="inv-note"></div>',
    '</div>',
    '<div class="inv-wrap"><table class="inv-table" id="inv-all"><thead><tr><th>Dataset / file</th><th>Type</th><th>Where</th><th>Size</th><th>Downloads*</th></tr></thead><tbody>']
    + [unit_row(u) for u in rows_units] + ['</tbody></table></div>',
    '<p class="inv-note">*OSF&rsquo;s own download counter for each file since upload, summed over identical copies. For a split archive it is the lowest count among its parts '
    '(a download needs every part). OSF does not say whether repeat or automated downloads are filtered, so read it as rough popularity. Google Drive publishes no counts. '
    f'Complete file lists of every archive in one file: <a href="{MANIFEST_URL}/all_archive_contents.csv.gz" download>all_archive_contents.csv.gz</a> '
    '(gzip-compressed CSV; unpack with 7-Zip or <code>gunzip</code>). '
    f'MD5 and SHA-256 of every OSF file, to verify downloads: <a href="{MANIFEST_URL}/checksums.csv" download>checksums.csv</a>.</p>',
    """<script>
(function(){
  var q=document.getElementById('inv-q'), rows=[].slice.call(document.querySelectorAll('#inv-all > tbody > tr')), cnt=document.getElementById('inv-count');
  var state={type:'',species:''};
  function apply(){
    var words=(q.value||'').toLowerCase().split(/\\s+/).filter(Boolean), shown=0;
    rows.forEach(function(r){
      var ok=(!state.type||r.dataset.type===state.type)&&(!state.species||r.dataset.species===state.species)&&
             words.every(function(w){return r.dataset.search.indexOf(w)>-1;});
      r.style.display=ok?'':'none'; if(ok)shown++;
    });
    cnt.textContent='Showing '+shown+' of '+rows.length+' rows';
  }
  function setChip(group,val){
    state[group]=val;
    [].forEach.call(document.querySelectorAll('.inv-chips[data-group="'+group+'"] button'),function(b){b.classList.toggle('on',(b.dataset[group]||'')===val);});
    apply();
  }
  [].forEach.call(document.querySelectorAll('.inv-chips button'),function(b){
    b.addEventListener('click',function(){var g=b.parentNode.dataset.group; setChip(g,b.dataset[g]||'');});
  });
  [].forEach.call(document.querySelectorAll('a[data-set-type]'),function(a){
    a.addEventListener('click',function(){q.value=''; setChip('species',''); setChip('type',a.dataset.setType);});
  });
  [].forEach.call(document.querySelectorAll('a.inv-jump[href^="#d"]'),function(a){
    a.addEventListener('click',function(){q.value=''; setChip('species',''); setChip('type','');
      var t=document.querySelector(a.getAttribute('href')); if(t){t.classList.add('inv-hit'); setTimeout(function(){t.classList.remove('inv-hit');},2500);}});
  });
  q.addEventListener('input',apply); apply();
})();
</script>"""])

# mouse RI training data note (the Drive folder people ask about)
L += raw(['<div class="inv-callout" id="mouse-ri-note"><b>About the mouse resident-intruder training data (Google Drive).</b> '
          'Checked from the column headers of two annotation CSVs: each has 548 columns = 1 index + 48 pose columns (16 body parts × x, y, probability) '
          '+ 498 feature columns + 1 behavior column (0/1). The body parts are Nose, Ear_left, Ear_right, Center, Lat_left, Lat_right, Tail_base and Tail_end '
          'for animal 1 and animal 2. The Drive classifiers also expect 498 features (tested <code>Allogroom_vigorous.sav</code>). To reuse this data, '
          'your pose estimation must track the same 8 body parts per animal; otherwise the feature columns will not line up.</div>'])

# ---- browse by location
L += heading('Browse by location', '-')
L += raw(['<p>The same data laid out as it sits on OSF and Google Drive: every project, folder and file, with each project&rsquo;s description and wiki. Click a project to open it.</p>'])


def location_rows(node_id, files_records, node_items):
    rows, folders = [], sorted({f['path'] for f in files_records if f['kind'] == 'folder'} | {'/'})
    for folder in folders:
        here = [it for it in node_items if it['path'].rsplit('/', 1)[0] + '/' == folder]
        depth = folder.strip('/').count('/') + (1 if folder != '/' else 0)
        if folder != '/':
            rec = next(f for f in files_records if f['path'] == folder)
            label = a(osf_file_url(node_id, rec), folder.strip('/').split('/')[-1] + '/')
            note = '' if any(f['path'].startswith(folder) and f['kind'] == 'file' for f in files_records) else ' <span class="inv-flag">empty</span>'
            rows.append(f'<tr class="inv-folder"><td colspan="4" style="padding-left:{0.6 + depth * 1.1:.1f}em">📁 {label}{note}</td></tr>')
        for it in sorted(here, key=lambda i: i['name'].lower()):
            extra = f' <span class="inv-sub">({it["parts"]} parts)</span>' if it['parts'] > 1 else ''
            rows.append(f'<tr><td style="padding-left:{0.6 + (depth + 1) * 1.1:.1f}em">{a(it["url"], it["name"])}{extra}</td>'
                        f'<td class="num">{hsize(it["size"])}</td><td class="num">{e(it["modified"])}</td><td class="num">{"" if it["downloads"] is None else it["downloads"]}</td></tr>')
    return rows


def loc_table(rows):
    return ['<div class="inv-wrap"><table class="inv-table"><thead><tr><th>Folder / file</th><th>Size</th><th>Modified</th><th>Downloads</th></tr></thead><tbody>'] + rows + ['</tbody></table></div>']


loc_html = []
for r in ROOT_ORDER:
    for n in [x for x in nodes if x['root'] == r]:
        n_items = [it for it in items if it['node'] == n['id']]
        indent = ' inv-child' if n['depth'] else ''
        if n_items:
            amount = f'{len(n_items)} items · {hsize(sum(i["size"] or 0 for i in n_items))}'
        else:   # container project: say what is underneath
            below = [it for it in items if it['source'] == 'osf' and node_by_id[it['node']]['root'] == r and n['id'] in [p['id'] for p in path_of(it['node'])]]
            amount = f'{len(n["children"])} sub-projects · {hsize(sum(i["size"] or 0 for i in below))} in total'
        summary = (f'<b>{e(n["title"])}</b> <span class="inv-sub">osf.io/{n["id"]} · {amount}'
                   + (f' · {e(ROOT_BLURB[r])}' if n['depth'] == 0 else '') + '</span>')
        body = [f'<p>{a(n["url"], n["url"])} · created {n["created"][:10]} · last modified {n["modified"][:10]}'
                + (f' · part of {a(path_of(n["id"])[-2]["url"], path_of(n["id"])[-2]["title"])}' if n['depth'] else '') + '</p>']
        if n['description']:
            body.append(f'<p><i>Description:</i> {e(n["description"])}</p>')
        for w in n['wikis']:
            if w['content'].strip():
                body.append(f'<div class="inv-wikibox"><div class="inv-sub">Project wiki</div>{wiki_html(w["content"])}</div>')
        nr = node_refs(n['id'])
        if nr:
            body.append(f'<p class="inv-sub">Linked from: {ref_html(nr, limit=12)}</p>')
        body += loc_table(location_rows(n['id'], n['files'], n_items)) if n_items else ['<p class="inv-note">No files at this level' + ('; see the sub-projects below.' if n['children'] else '.') + '</p>']
        loc_html += [f'<details class="inv-loc{indent}"><summary>{summary}</summary>'] + body + ['</details>']
drive_rows = []
for folder in sorted({d['path'] for d in drive['items'] if d['kind'] == 'folder'} | {'/'}):
    depth = folder.strip('/').count('/') + (1 if folder != '/' else 0)
    if folder != '/':
        rec = next(d for d in drive['items'] if d['path'] == folder)
        drive_rows.append(f'<tr class="inv-folder"><td colspan="4" style="padding-left:{0.6 + depth * 1.1:.1f}em">📁 {a(rec["url"], folder.strip("/").split("/")[-1] + "/")}</td></tr>')
    for it in sorted([i for i in drive_items if i['path'].rsplit('/', 1)[0] + '/' == folder], key=lambda i: i['name'].lower()):
        drive_rows.append(f'<tr><td style="padding-left:{0.6 + (depth + 1) * 1.1:.1f}em">{a(it["url"], it["name"])}</td><td class="num">{hsize(it["size"])}</td><td class="num">{e(it["modified"])}</td><td></td></tr>')
loc_html += [f'<details class="inv-loc"><summary><b>Google Drive: {e(drive["title"])}</b> <span class="inv-sub">{len(drive_items)} files · {hsize(sum(i["size"] or 0 for i in drive_items))} · '
             'mouse resident-intruder annotations and classifiers</span></summary>',
             f'<p>{a(drive["root_url"], drive["root_url"])} · shared as &ldquo;anyone with the link can view&rdquo;. See the <a href="#mouse-ri-note">note on its format</a>.</p>']
loc_html += loc_table(drive_rows) + ['</details>']
L += raw(loc_html)

# ---- repo files
if repo_files:
    L += heading('Data files in the SimBA GitHub repository', '-')
    total_repo = sum(f['size'] or 0 for f in repo_files)
    repo_html = [f'<p>Small data files that ship inside the repository itself ({len(repo_files)} files, {hsize(total_repo)}): example inputs and outputs, test projects, '
                 'tables rendered in the docs. Images used only as figures are left out. &ldquo;Mentioned in&rdquo; lists the docs, code and notebooks that use the file name.</p>']
    for g in order:
        fs = groups[g]
        body = []
        for f in fs:
            rel = f['path'][len(g) + 1:] if f['path'].startswith(g + '/') else f['path']
            seen, ment = set(), []
            for m in f['mentions']:
                if m['file'] not in seen:
                    seen.add(m['file']); ment.append(a(f"{GITHUB_BLOB}{m['file']}#L{m['line']}", m['file']))
            ment_html = ', '.join(ment[:4]) + (f' +{len(ment) - 4} more' if len(ment) > 4 else '') if ment else '<span class="inv-flag">not mentioned</span>'
            body.append(f'<tr><td>{a(GITHUB_BLOB + f["path"], rel)}</td><td class="num">{hsize(f["size"])}</td><td class="num">{e(f["last_changed"])}</td><td>{ment_html}</td></tr>')
        unref_n = sum(1 for f in fs if not f['n_mentions'])
        repo_html += [f'<details class="inv-loc"><summary><b><code>{e(g)}/</code></b> <span class="inv-sub">{len(fs)} files · {hsize(sum(f["size"] or 0 for f in fs))} · '
                      f'{e(REPO_FOLDERS.get(g, ""))}' + (f' · {unref_n} not mentioned anywhere' if unref_n else '') + '</span></summary>',
                      '<div class="inv-wrap"><table class="inv-table"><thead><tr><th>File</th><th>Size</th><th>Last changed</th><th>Mentioned in</th></tr></thead><tbody>']
        repo_html += body + ['</tbody></table></div></details>']
    L += raw(repo_html)

# ---- maintainers
L += heading('For maintainers', '-')
findings.append(('No licence on the data', f'None of the {len(nodes)} OSF projects has a licence set, so the terms for reusing the data are not stated. '
                 'The CRIM13 videos are third-party data (Caltech) and need their own attribution.'))
for (ds, key), s in sorted(clf_settings.items()):
    if clf_key(s['name_inside']) != key:
        findings.append(('Classifier settings file names a different classifier',
                         f'{a(s["url"], s["file"])} contains <code>Classifier_name = {e(s["name_inside"])}</code>; unclear whether its settings belong to this classifier.'))
shifted_files = [s for s in clf_settings.values() if s['shifted']]
if shifted_files:
    findings.append(('Classifier settings files with shifted columns',
                     f'{len(shifted_files)} files ({", ".join(a(s["url"], s["file"]) for s in shifted_files)}) hold the undersampling ratio in the '
                     '<code>under_sample_setting</code> column and the text &ldquo;Random undersample&rdquo; in <code>over_sample_ratio</code>; this page reads them as the ratio.'))
no_settings = [u for u in rows_units if u['cat'] == 'classifier' and not clf_settings.get((classifier_dataset(u['items'][0]), clf_key(u['items'][0]['name'])))]
if no_settings:
    findings.append(('Classifiers without a settings file', f'{len(no_settings)} classifiers: ' + ', '.join(f'<a href="#{u["id"]}">{e(u["title"])}</a> ({e(node_by_id[u["items"][0]["node"]]["title"] if u["items"][0]["source"] == "osf" else "Google Drive")})' for u in no_settings) + '.'))
for key, v in feature_cols.items():
    kinds = column_kinds(v['columns'])
    n_feat = kinds.count('feature')
    label = re.search(r'(\d{3}) features', key)
    if label and int(label.group(1)) != n_feat:
        findings.append(('Feature count differs from the folder name', f'<code>{e(key.split(":", 1)[1])}</code> sits in a folder named &ldquo;{label.group(1)} features&rdquo; but its CSVs have {n_feat} feature columns (checked <code>{e(v["example_file"])}</code>).'))
    cmp = compare_with_current(v['columns'], kinds, len({c[:-2] for c, k in zip(v['columns'], kinds) if k == 'pose'}))
    if cmp and any(cmp):
        findings.append(('Feature columns differ from current SimBA', f'<code>{e(key.split(":", 1)[1].rsplit("/", 1)[-1])}</code>: {comparison_html(cmp)} '
                         'Classifiers read features by position, so renamed columns still predict; training on a mix of old and new files, or picking features by name, does not line up.'))
    behaviors = [c for c, k in zip(v['columns'], kinds) if k == 'behavior']
    if behaviors and all(b.endswith('_prediction') for b in behaviors):
        findings.append(('Behavior columns named *_prediction', f'In <code>{e(key.split(":", 1)[1].rsplit("/", 1)[-1])}</code> all {len(behaviors)} behavior columns end in <code>_prediction</code>, '
                         'so it is not clear from the names whether they hold human annotations or classifier output.'))

# checksums for every OSF file (lets people verify large multi-part downloads)
with open(os.path.join(MANIFEST_DIR, 'checksums.csv'), 'w', encoding='utf-8', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['project', 'path', 'size_bytes', 'md5', 'sha256', 'url'])
    for n in nodes:
        for f in n['files']:
            if f['kind'] == 'file':
                h = hashes.get(f['id'], {})
                w.writerow([n['id'], f['path'], f['size'], h.get('md5', ''), h.get('sha256', ''), osf_file_url(n['id'], f)])
grouped = collections.OrderedDict()
for kind, text in findings:
    grouped.setdefault(kind, []).append(text)
L += raw(['<p>Things found while building this page that make the data harder to find or use. Click a heading to expand it.</p>', '<div class="inv-findings">']
         + [f'<details class="inv-loc"><summary><b>{e(k)}</b> <span class="inv-sub">({len(v)})</span></summary><ul>' + ''.join(f'<li>{t}</li>' for t in v) + '</ul></details>'
            for k, v in grouped.items()] + ['</div>'])
if other:
    titles = load('other_osf_titles.json') if os.path.isfile(os.path.join(DATA, 'other_osf_titles.json')) else {}
    L += raw(['<p class="inv-note">OSF projects by other labs that the SimBA docs link to (not SimBA data): '
              + '; '.join(f'{a("https://osf.io/" + k + "/", titles.get(k, "osf.io/" + k))} (linked from {ref_html(v, limit=3)})' for k, v in other.items()) + '.</p>'])

# ---- about
L += heading('About this page', '-')
L += [f'Built on {SNAPSHOT_DATE} from read-only crawls: the OSF API (every project, sub-project, wiki and file of the sgoldenlab account, '
      'including each file\'s download counter and SHA-256 checksum), the public Google Drive folder view, and the index at the end of each zip '
      'archive, fetched with HTTP range requests so no archive was downloaded. Body-part lists come from the ``CollectedData`` annotation files '
      'inside the labelled-image archives. Links into the SimBA repository were found by scanning the docs, README and code. '
      'Identical copies are matched by checksum. Classifier training settings come from the SimBA meta files on OSF, matched to classifiers by file name. '
      'Column lists come from the header row of one CSV per training dataset. SimBA body-part settings are matched against '
      '``simba/pose_configurations`` by comparing body-part names (ignoring Lateral/Lat spelling and word order within a name). '
      'Data files inside the GitHub repository were listed with ``git ls-files``. '
      'Re-run the crawlers and ``build_data_inventory.py`` to refresh it.', '']
L += ['**What this page cannot see:** private OSF projects or data under other OSF accounts; Google Drive, Dropbox or other folders '
      'that are not linked from the SimBA repository; and whether every file inside an archive extracts cleanly (only the archive '
      'indexes were read; checking contents would mean downloading about 300 GB).', '']
