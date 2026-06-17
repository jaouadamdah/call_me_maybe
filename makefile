SRC = src
SDK = llm_sdk/llm_sdk
PYTHON = python3
UV = uv run
RM = rm -fr

run:
	@uv run $(PYTHON) -m $(SRC)

install:
	@uv sync

debug:
	@uv run $(PYTHON) -m pdb -m $(SRC)


clean:
	@$(RM) .mypy_cache
	@$(RM) $(SRC)/__pycache__ 
	@$(RM) $(SDK)/__pycache__


lint:
	@$(UV) flake8 $(SRC)
	@$(UV) mypy $(SRC) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@$(UV) flake8 $(SRC)
	@$(UV) mypy $(SRC) --strict

PHONY: install run debug clean lint lint-strict