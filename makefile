SRC = src
SDK = llm_sdk/llm_sdk
PYTHON = python3

run:
	@uv run $(PYTHON) -m $(SRC)

install:
	@uv sync

debug:
	@uv run $(PYTHON) -m pdb -m $(SRC)


clean:
	@rm -fr .mypy_cache
	@rm -fr $(SRC)/__pycache__ 
	@rm -fr $(SDK)/__pycache__


lint:
	@flake8 $(SRC)
	@mypy $(SRC) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@flake8 $(SRC)
	@mypy $(SRC) --strict

PHONY: install run debug clean lint lint-strict