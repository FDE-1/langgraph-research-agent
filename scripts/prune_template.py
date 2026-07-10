"""Delete the one-shot scaffolding: init.sh, its launcher, examples/, and this script.

Run once, after `make init`. Everything goes through `git rm`, so a mistake is one
`git checkout` away — which is exactly why an untracked target aborts instead.
"""

import subprocess
import sys
from pathlib import Path

TARGETS: tuple[str, ...] = (
    "init.sh",
    "examples",
    "scripts/init_launcher.py",
    "scripts/prune_template.py",
)

# The Makefile keeps its scaffolding targets between these two lines.
BEGIN = "# >>> template-scaffolding"
END = "# <<< template-scaffolding"


def is_tracked(root: Path, path: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", path],
            capture_output=True,
        ).returncode
        == 0
    )


def strip_scaffolding(makefile: Path) -> bool:
    lines = makefile.read_text(encoding="utf-8").splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith(BEGIN)]
    ends = [i for i, line in enumerate(lines) if line.startswith(END)]
    if not starts or not ends:
        return False
    kept = lines[: starts[0]] + lines[ends[-1] + 1 :]

    # Leave no dangling names: `.PHONY: … init prune` makes `make init` answer
    # "nothing to be done" instead of failing on an unknown target.
    for i, line in enumerate(kept):
        if line.startswith(".PHONY:"):
            names = [n for n in line[len(".PHONY:") :].split() if n not in {"init", "prune"}]
            kept[i] = ".PHONY: " + " ".join(names) + "\n"
            break

    makefile.write_text("".join(kept), encoding="utf-8")
    return True


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    present = [t for t in TARGETS if (root / t).exists()]
    if not present:
        print("Nothing to prune — already done.")
        return 0

    untracked = [t for t in present if not is_tracked(root, t)]
    if untracked:
        print("error: these are untracked, `git rm` cannot undo their removal:", file=sys.stderr)
        for t in untracked:
            print(f"  {t}", file=sys.stderr)
        print("commit them first, or delete them by hand.", file=sys.stderr)
        return 1

    subprocess.run(["git", "-C", str(root), "rm", "-rq", *present], check=True)
    if strip_scaffolding(root / "Makefile"):
        subprocess.run(["git", "-C", str(root), "add", "Makefile"], check=True)
        print("Removed the init/prune targets from the Makefile.")

    for t in present:
        print(f"Removed {t}")
    print("\nStaged. Review with `git diff --cached`, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
