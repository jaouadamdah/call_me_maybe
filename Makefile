MAIN = src

all: run


install:
	uv sync

run:
	@uv run python3 -m $(MAIN)

debug:
	@uv run python3 -m pdb -m $(MAIN)


clean:
	@rm -fr ./__pycache__ ./.mypy_cache
	@rm -fr $(MAIN)/__pycache__


lint:
	@flake8 $(MAIN)
	@mypy $(MAIN) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@flake8 $(MAIN)
	@mypy $(MAIN) --strict

PHONY: all install run debug clean lint lint-strict