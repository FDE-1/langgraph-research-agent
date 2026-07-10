"""Remove generated artifacts. Pure stdlib so `make clean` needs no POSIX shell."""

import shutil
from pathlib import Path

CLEAN_PATHS: tuple[str, ...] = (
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    ".claude",
    "htmlcov",
    "build",
    "dist",
    "livrable.zip",
)
PRUNED: frozenset[str] = frozenset({".venv", ".git"})


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def iter_pycache(root: Path) -> list[Path]:
    """Walk once, skipping pruned dirs — `rglob` would descend into .venv."""
    found: list[Path] = []
    stack = [root]
    while stack:
        for child in stack.pop().iterdir():
            if not child.is_dir() or child.name in PRUNED:
                continue
            if child.name == "__pycache__":
                found.append(child)
            else:
                stack.append(child)
    return found


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    for name in CLEAN_PATHS:
        remove(root / name)
    for egg_info in (root / "src").glob("*.egg-info"):
        remove(egg_info)
    for cache in iter_pycache(root):
        remove(cache)
    print("Cleaned")


if __name__ == "__main__":
    main()
