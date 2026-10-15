"""List members of split zip archives on OSF (name.zip.001, .002, ...) by treating the parts as one virtual file and
reading only the zip index with HTTP range requests -> osf_split_zip_contents.json"""
import bisect
import io
import json
import os
import re
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


class MultiPartRangeFile(io.RawIOBase):
    def __init__(self, parts):  # parts: [(url, size)] in order
        self.urls = [u for u, _ in parts]
        self.starts, total = [], 0
        for _, s in parts:
            self.starts.append(total); total += s
        self.size, self.pos = total, 0

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
        out = b''
        while n > 0:
            i = bisect.bisect_right(self.starts, self.pos) - 1
            part_end = self.starts[i + 1] if i + 1 < len(self.starts) else self.size
            local, take = self.pos - self.starts[i], min(n, part_end - self.pos)
            req = urllib.request.Request(self.urls[i], headers={'Range': f'bytes={local}-{local + take - 1}', 'User-Agent': 'Mozilla/5.0'})
            b = urllib.request.urlopen(req, timeout=120).read()
            if not b:
                break
            out += b; self.pos += len(b); n -= len(b)
        return out

    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)


if __name__ == '__main__':
    nodes = json.load(open(os.path.join(HERE, 'osf_all.json'), encoding='utf-8'))
    out = {}
    for n in nodes:
        groups = {}
        for f in n['files']:
            m = re.match(r'(.+?)\.(\d{3})$', f['name']) if f['kind'] == 'file' else None
            if m:
                groups.setdefault(f['path'].rsplit('/', 1)[0] + '/' + m.group(1), []).append((int(m.group(2)), f))
        for key, parts in groups.items():
            parts.sort()
            nums = [p[0] for p in parts]
            k = f"{n['id']}:{key}"
            if nums != list(range(nums[0], nums[0] + len(nums))):
                out[k] = {'error': f'missing part numbers: have {nums}'}; print(k, 'GAP', nums); continue
            try:
                vf = io.BufferedReader(MultiPartRangeFile([(f['download'], f['size']) for _, f in parts]), buffer_size=1 << 20)
                with zipfile.ZipFile(vf) as z:
                    out[k] = [{'name': i.filename, 'size': i.file_size, 'date': '%04d-%02d-%02d' % i.date_time[:3]} for i in z.infolist()]
                print(k, len(out[k]), 'members', flush=True)
            except Exception as e:
                out[k] = {'error': repr(e)[:200]}; print(k, 'ERROR', repr(e)[:150], flush=True)
    json.dump(out, open(os.path.join(HERE, 'osf_split_zip_contents.json'), 'w', encoding='utf-8'), indent=1)
