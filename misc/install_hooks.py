# -*- coding: utf-8 -*-
"""Install the repo's git hooks from misc/hooks/ into .git/hooks/.

.git/ is not version-controlled, so the hooks live in misc/hooks/ and are copied
into place by this script. Re-run it after pulling a change to misc/hooks/.

Run:  python misc/install_hooks.py
      python misc/install_hooks.py --status   # report only, install nothing
"""
import os, sys, stat, shutil, filecmp, subprocess

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hooks")


def git_dir():
    out = subprocess.check_output(["git", "rev-parse", "--git-dir"])
    return out.decode("utf-8").strip()


def main():
    status_only = "--status" in sys.argv
    dst_dir = os.path.join(git_dir(), "hooks")
    if not os.path.isdir(SRC_DIR):
        print("[hooks] no source dir %s" % SRC_DIR)
        return 1
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    names = sorted(n for n in os.listdir(SRC_DIR) if not n.endswith(".sample"))
    if not names:
        print("[hooks] nothing to install in %s" % SRC_DIR)
        return 0

    for name in names:
        src, dst = os.path.join(SRC_DIR, name), os.path.join(dst_dir, name)
        if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
            print("[hooks] %-10s up to date" % name)
            continue
        if status_only:
            state = "differs from" if os.path.exists(dst) else "missing from"
            print("[hooks] %-10s %s %s" % (name, state, dst_dir))
            continue
        shutil.copyfile(src, dst)
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("[hooks] %-10s installed -> %s" % (name, dst))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
