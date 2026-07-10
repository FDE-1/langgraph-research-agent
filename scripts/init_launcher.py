"""Hand off to init.sh through the *right* bash.

`bash` on a Windows PATH is usually C:\\...\\WindowsApps\\bash.exe — the WSL launcher.
WSL has its own git, no uv, and sees the tree through /mnt/c, so init.sh dies there.
Git ships a real bash next to itself; find it via `git --exec-path` rather than PATH.
"""

import os
import subprocess
import sys
from pathlib import Path

BASH = "bash.exe" if os.name == "nt" else "bash"


def find_bash() -> Path:
    """`<prefix>/libexec/git-core` -> `<prefix>/bin/bash`, walking up to whichever exists."""
    exec_path = subprocess.run(
        ["git", "--exec-path"], capture_output=True, text=True, check=True
    ).stdout.strip()
    for parent in Path(exec_path).parents:
        candidate = parent / "bin" / BASH
        if candidate.is_file():
            return candidate
    raise SystemExit("error: no bash found next to git — run `bash init.sh <name>` yourself")


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1]:
        raise SystemExit("usage: make init NAME=my-new-project")

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    bash = find_bash()
    # execv, not run(): this process must not keep a handle on a directory init.sh renames.
    # argv[0] stays a bare name — Windows rebuilds a command line from argv without
    # quoting it, so "C:\Program Files\..." would reach bash split on the space.
    os.execv(str(bash), ["bash", "init.sh", sys.argv[1]])


if __name__ == "__main__":
    main()
