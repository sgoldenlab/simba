# -*- coding: utf-8 -*-
"""
Check that the "Tools from papers citing SimBA" table actually renders on the Related Software page.
Run by .github/workflows/related_tools.yml; exits non-zero (failing the run) when a check fails.

  1. docutils parses docs/simba.related_software.rst and the table HTML ends up inside a raw node,
     i.e. the ``.. raw:: html`` directive is still a directive and not swallowed into a paragraph.
  2. The live Read the Docs page contains the table.

Run:  python misc/check_related_tools_page.py [--skip-live]
"""
import os, sys, urllib.request

from docutils import nodes
from docutils.core import publish_doctree

HERE = os.path.dirname(os.path.abspath(__file__))
RST = os.path.join(HERE, "..", "docs", "simba.related_software.rst")
LIVE_URL = "https://simba-uw-tf-dev.readthedocs.io/en/latest/simba.related_software.html"
MARKER = 'class="simba-rt"'


def check_rst():
    with open(RST, encoding="utf-8") as f:
        src = f.read()
    # Sphinx-only roles and emoji substitutions are unknown to plain docutils; keep parsing past them.
    tree = publish_doctree(src, source_path=RST,
                           settings_overrides={"report_level": 5, "halt_level": 5, "file_insertion_enabled": True})
    return any(MARKER in n.astext() for n in tree.findall(nodes.raw))


def check_live():
    req = urllib.request.Request(LIVE_URL, headers={"User-Agent": "simba-related-tools-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return MARKER in r.read().decode("utf-8", "replace")


def main():
    failures = []
    if not check_rst():
        failures.append(f"{os.path.normpath(RST)}: the related tools table is not inside a '.. raw:: html' directive "
                        "(is the blank line before the directive missing?)")
    if "--skip-live" not in sys.argv and not check_live():
        failures.append(f"{LIVE_URL}: the live page does not contain the related tools table")
    for f in failures:
        print("FAIL:", f)
    if not failures:
        print("OK: the related tools table renders in the RST source and on the live page")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
