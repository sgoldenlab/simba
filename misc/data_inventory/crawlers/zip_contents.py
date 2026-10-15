"""List the members of remote zip files using HTTP range requests (reads only the zip index, not the data).
Usage: python zip_contents.py <inventory.json> <source: gdrive|osf> -> <source>_zip_contents.json"""
import io
import json
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, size=None):
        self.url, self.pos, self.cache = url, 0, {}
        if size is None:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0', 'User-Agent': 'Mozilla/5.0'}), timeout=60)
            size = int(r.headers['Content-Range'].split('/')[-1])
        self.size = size

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b''
        end = self.pos + n - 1
        r = urllib.request.urlopen(urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}', 'User-Agent': 'Mozilla/5.0'}), timeout=120)
        b = r.read()
        self.pos += len(b)
        return b

    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)


def list_zip(url):
    f = io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20)
    with zipfile.ZipFile(f) as z:
        return [{'name': i.filename, 'size': i.file_size, 'compressed': i.compress_size,
                 'date': '%04d-%02d-%02d' % i.date_time[:3]} for i in z.infolist()]


if __name__ == '__main__':
    inv_path, source = sys.argv[1], sys.argv[2]
    inv = json.load(open(inv_path, encoding='utf-8'))
    targets = []
    if source == 'gdrive':
        targets = [(i['path'], f"https://drive.usercontent.google.com/download?id={i['id']}&export=download&confirm=t")
                   for i in inv['items'] if i['name'].lower().endswith('.zip')]
    else:
        for node in inv:
            for fl in node.get('files', []):
                if fl['kind'] == 'file' and fl['name'].lower().endswith('.zip') and fl.get('download'):
                    targets.append((f"{node['id']}:{fl['path']}", fl['download']))
    out = {}
    for key, url in targets:
        try:
            out[key] = list_zip(url)
            print(key, len(out[key]), 'members', flush=True)
        except Exception as e:
            out[key] = {'error': repr(e)[:200]}
            print(key, 'ERROR', repr(e)[:120], flush=True)
    json.dump(out, open(os.path.join(HERE, f'{source}_zip_contents.json'), 'w', encoding='utf-8'), indent=1)
