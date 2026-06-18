"""
Normalize image references in a Markdown tutorial for Sphinx/MyST publishing,
and stage the referenced image files into docs/images/ so they are served.

Tutorials reference images inconsistently and often in ways that break in Sphinx:
  - GitHub ``blob/`` URLs  -> serve an HTML page, not image bytes (broken)
  - absolute ``/images/x`` -> breaks on Read the Docs (served under /en/<ver>/)
  - relative ``images/x``  -> fine (page published at repo-root depth)

This rewrites the first two to the relative ``images/x`` form, then copies any
referenced file that isn't already in docs/images/ from the repo-root images/
tree (where most originals live). Only files actually referenced are copied, so
docs/images/ stays small instead of cloning the ~1 GB repo-root tree.

Usage:
    python normalize_tutorial_images.py <tutorial.md>
"""
import os
import re
import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent          # docs/
REPO_IMAGES = DOCS.parent / "images"                   # repo-root images/
DOCS_IMAGES = DOCS / "images"

# github.com/<org>/<repo>/blob|raw/<branch>/images/...  ->  images/...
BLOB_RE = re.compile(r"https?://github\.com/[^/]+/[^/]+/(?:blob|raw)/[^/]+/(images/)")
# absolute /images/ inside src="..." or ](...)  ->  images/
ABS_RE = re.compile(r'(src=["\']|\]\()/images/')
# any local images/<path>.<ext> reference (after rewriting)
REF_RE = re.compile(r'images/[^)"\'\s>]+\.(?:png|jpe?g|webp|gif|svg)', re.I)


def main(md_path):
    md = Path(md_path).resolve()
    text = md.read_text(encoding="utf-8")

    text = BLOB_RE.sub(r"\1", text)
    text = ABS_RE.sub(r"\1images/", text)
    md.write_text(text, encoding="utf-8")

    refs = sorted(set(REF_RE.findall(text)))
    copied = missing = present = 0
    for rel in refs:
        dest = DOCS / rel                              # docs/images/...
        if dest.is_file():
            present += 1
            continue
        src = DOCS.parent / rel                        # repo-root images/...
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
        else:
            missing += 1
            print(f"  MISSING (no source anywhere): {rel}")
    print(f"{md.name}: {len(refs)} image refs | {present} already in docs | "
          f"{copied} copied from repo-root | {missing} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1]))
