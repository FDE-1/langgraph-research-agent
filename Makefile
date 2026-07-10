.PHONY: install fix lint typecheck test all clean init prune dist

install:
	uv sync

fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/ tests/

test:
	uv run pytest

clean:
	@uv run python scripts/clean.py

all: lint typecheck test
	@"$(MAKE)" --no-print-directory clean
	@echo All checks passed

# >>> template-scaffolding
# Used once on a fresh project, then deleted by `make prune`. Do not add real targets
# below this line: `prune` drops everything down to the closing marker.

# --no-project: never build a .venv here, init.sh renames the directory out from under it.
init:
	@uv run --no-project python scripts/init_launcher.py "$(NAME)"

# Deletes init.sh, examples/, the launcher, the pruner, and this block.
prune:
	@uv run --no-project python scripts/prune_template.py
# <<< template-scaffolding

dist: clean
	@uv run python scripts/dist.py
