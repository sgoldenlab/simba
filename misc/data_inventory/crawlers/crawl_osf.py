"""Read-only crawl of the public SimBA OSF tree (root project tmu6y): every node, wiki and file -> osf_inventory.json."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = 'tmu6y'
API = 'https://api.osf.io/v2'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'osf_inventory.json')
N_REQ = 0


def get(url):
    global N_REQ
    for attempt in range(6):
        try:
            N_REQ += 1
            with urllib.request.urlopen(urllib.request.Request(url, headers={'Accept': 'application/vnd.api+json'}), timeout=60) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(30 * (attempt + 1)); continue
            if e.code in (401, 403, 404, 410):
                return None
            time.sleep(5)
        except Exception:
            time.sleep(5)
    return None


def get_all(url):
    out = []
    sep = '&' if '?' in url else '?'
    url = f'{url}{sep}page[size]=100'
    while url:
        d = get(url)
        if d is None:
            break
        out.extend(d.get('data', []))
        url = (d.get('links') or {}).get('next')
    return out


def get_text(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode('utf-8', 'ignore')
    except Exception:
        return ''


def crawl_files(url, path, files):
    for f in get_all(url):
        a = f['attributes']
        rec = {'name': a['name'], 'kind': a['kind'], 'path': path + a['name'] + ('/' if a['kind'] == 'folder' else ''),
               'size': a.get('size'), 'modified': a.get('date_modified'), 'created': a.get('date_created'),
               'guid': a.get('guid'), 'id': f['id'],
               'html': (f.get('links') or {}).get('html'), 'download': (f.get('links') or {}).get('download'),
               'downloads': (a.get('extra') or {}).get('downloads')}
        files.append(rec)
        if a['kind'] == 'folder':
            crawl_files(f['relationships']['files']['links']['related']['href'], rec['path'], files)


def crawl_node(node_id, depth, nodes):
    d = get(f'{API}/nodes/{node_id}/')
    if d is None:
        nodes.append({'id': node_id, 'private_or_missing': True, 'depth': depth}); return
    a = d['data']['attributes']
    node = {'id': node_id, 'depth': depth, 'title': a['title'], 'description': a.get('description') or '',
            'category': a.get('category'), 'created': a.get('date_created'), 'modified': a.get('date_modified'),
            'tags': a.get('tags') or [], 'url': f'https://osf.io/{node_id}/', 'wikis': [], 'files': [], 'children': []}
    for w in get_all(f'{API}/nodes/{node_id}/wikis/'):
        node['wikis'].append({'name': w['attributes']['name'], 'content': get_text(w['links']['download'])})
    crawl_files(f'{API}/nodes/{node_id}/files/osfstorage/', '/', node['files'])
    nodes.append(node)
    print(f'{"  " * depth}{node_id} {a["title"]}: {len(node["files"])} entries (requests so far {N_REQ})', flush=True)
    for c in get_all(f'{API}/nodes/{node_id}/children/'):
        node['children'].append(c['id'])
        crawl_node(c['id'], depth + 1, nodes)


if __name__ == '__main__':
    nodes = []
    crawl_node(sys.argv[1] if len(sys.argv) > 1 else ROOT, 0, nodes)
    json.dump(nodes, open(OUT, 'w', encoding='utf-8'), indent=1)
    print('saved', OUT, '| nodes', len(nodes), '| requests', N_REQ)
