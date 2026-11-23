#!/usr/bin/env python3
"""Install the diff-visibility hooks for this checkout.

The hook runs after every commit, writes that commit's diff to
``.git/pic_diff_last.txt`` and echoes it on stderr, so the diff shows up even
when stdout is suppressed.  It is written in plain POSIX shell and uses only git,
so it works wherever git runs - including Git for Windows, whose bundled shell
has no python3 on its PATH.  Installing is idempotent.

Usage:
    python3 tools/install_diff_hooks.py
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys

HOOK = """#!/bin/sh
# Installed by tools/install_diff_hooks.py - keep this shell-only and git-only.
out="${GIT_DIR:-.git}/pic_diff_last.txt"
{
  echo "=== diff of $(git log -1 --format='%h %s') ==="
  git diff --stat HEAD~1 HEAD
  echo
  git diff --unified=3 HEAD~1 HEAD
} >"$out" 2>&1 || true
cat "$out" >&2
exit 0
"""


def git_dir() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--git-dir"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return out


def main() -> int:
    hooks = os.path.join(git_dir(), "hooks")
    os.makedirs(hooks, exist_ok=True)
    path = os.path.join(hooks, "post-commit")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HOOK)
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
