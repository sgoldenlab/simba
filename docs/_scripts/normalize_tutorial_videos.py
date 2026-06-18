"""
Normalize video embeds in a Markdown tutorial for Sphinx/MyST publishing.

GitHub auto-embeds bare ``[name.webm](https://github.com/user-attachments/assets/<id>)``
links as players, but Sphinx renders them as plain links. This script:

  1. Finds bare user-attachments video links (text ends in .webm/.mp4).
  2. Downloads each asset to a local image dir.
  3. Probes the codec (re-encode left to the user if exotic; vp8/vp9/h264 play natively).
  4. Rewrites the link to a raw <video controls src="..."> tag, which renders as a
     player BOTH on GitHub and in Sphinx — so the .md stays single-source.

Usage:
    python normalize_tutorial_videos.py <tutorial.md> <image_subdir>
    e.g. python normalize_tutorial_videos.py ../Scenario1.md images/scenario1
"""
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0"
# Bare link on its own, text ends in .webm/.mp4, pointing at a user-attachments asset.
LINK_RE = re.compile(
    r"\[([^\]]+?\.(?:webm|mp4))\]\((https://github\.com/user-attachments/assets/[0-9a-fA-F-]+)\)"
)
PLAYABLE = {"vp8", "vp9", "av1", "h264"}


def codec_of(path: Path) -> str:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "?"


def main(md_path: str, img_subdir: str) -> int:
    md = Path(md_path).resolve()
    docs_root = md.parent                      # tutorial lives at docs root
    out_dir = docs_root / img_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    text = md.read_text(encoding="utf-8")

    seen, rewrites = {}, 0

    def replace(m):
        nonlocal rewrites
        fname, url = m.group(1), m.group(2)
        fname = re.sub(r"[^A-Za-z0-9._-]", "_", fname)          # sanitize
        dest = out_dir / fname
        if url not in seen:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
                f.write(r.read())
            codec = codec_of(dest)
            flag = "OK" if codec in PLAYABLE else f"!! re-encode ({codec})"
            print(f"  downloaded {fname:40s} {dest.stat().st_size//1024:>6} KB  codec={codec} {flag}")
            seen[url] = fname
        rewrites += 1
        src = f"{img_subdir}/{fname}"
        return (f'<p align="center">\n'
                f'  <video src="{src}" width="600" controls>Your browser does not support the video tag.</video>\n'
                f'</p>')

    new_text = LINK_RE.sub(replace, text)
    if new_text != text:
        md.write_text(new_text, encoding="utf-8")
    print(f"Rewrote {rewrites} video embed(s) in {md.name}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
