"""Read-only crawl of the public Google Drive folder (training data / classifiers) -> gdrive_inventory.json.
Structure, ids and dates come from the embedded folder view; file sizes from the normal folder page's aria-labels."""
import html
import json
import os
import re
import sys
import urllib.request

ROOT = '17-Bi0D78Qn--tyR_zWBxy7bIpawCuWL4'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gdrive_inventory.json')
ENTRY = re.compile(r'<div class="flip-entry" id="entry-([\w-]+)".*?<a href="([^"]+)".*?<div class="flip-entry-title">(.*?)</div>.*?<div class="flip-entry-last-modified"><div>(.*?)</div>', re.S)


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60) as r:
        return r.read().decode('utf-8', 'ignore')


def sizes(folder_id):
    s = fetch(f'https://drive.google.com/drive/folders/{folder_id}')
    labels = [html.unescape(l) for l in re.findall(r'aria-label="([^"]+)"', s)]
    out, last = {}, None
    for l in labels:
        m = re.match(r'(.+?) (?:Binary|CSV|Compressed archive|Text|PDF|Image|Video|Unknown file|Microsoft Excel|Google \w+|[\w ]+?) Shared$', l)
        if m and not l.endswith('Shared folder'):
            last = m.group(1)
        elif l.startswith('Size: ') and last:
            out[last] = l.split('\n')[0].replace('Size: ', ''); last = None
    return out


def crawl(folder_id, path, items, depth=0):
    page = fetch(f'https://drive.google.com/embeddedfolderview?id={folder_id}')
    title = html.unescape(re.search(r'<title>(.*?)</title>', page).group(1))
    size_map = sizes(folder_id)
    for fid, href, name, modified in ENTRY.findall(page):
        name = html.unescape(name)
        is_folder = '/folders/' in href
        rec = {'name': name, 'kind': 'folder' if is_folder else 'file', 'path': path + name + ('/' if is_folder else ''),
               'id': fid, 'url': href.split('?')[0] if is_folder else f'https://drive.google.com/file/d/{fid}/view',
               'modified': modified, 'size': None if is_folder else size_map.get(name)}
        items.append(rec)
        print('  ' * depth + rec['path'], rec['size'] or '', flush=True)
        if is_folder:
            crawl(fid, rec['path'], items, depth + 1)
    return title


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    items = []
    title = crawl(root, '/', items)
    json.dump({'root_id': root, 'root_url': f'https://drive.google.com/drive/folders/{root}', 'title': title, 'items': items},
              open(OUT, 'w', encoding='utf-8'), indent=1)
    print('saved', OUT, '| items', len(items))
