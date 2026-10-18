PYTHON := uv run python
ARGS   ?=

install:
	uv sync

run:
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb -m src $(ARGS)

clean:
	find . \( -path ./.venv -o -path ./data \) -prune -o \
		-type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" \) \
		-exec rm -rf {} +

lint:
	uv run flake8 src --exclude=__pycache__
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs


.PHONY: install run debug clean lint lint-strict