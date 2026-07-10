"""Build livrable.zip from committed files only. Pure stdlib — no shell, no du/cut."""

import subprocess
import sys
from pathlib import Path


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.rstrip()


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    zip_path = root / "livrable.zip"

    dirty = git("-C", str(root), "status", "--porcelain", "--", ".")
    if dirty:
        print("error: uncommitted changes — they would be missing from the zip", file=sys.stderr)
        print(dirty, file=sys.stderr)
        return 1

    git(
        "-C",
        str(root),
        "archive",
        "--format=zip",
        f"--prefix={root.name}/",
        "HEAD",
        "-o",
        str(zip_path),
        "--",
        ".",
    )
    size_kb = zip_path.stat().st_size / 1024
    print(f"livrable.zip ({size_kb:.0f}K) — committed files only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
