PYTHON  := uv run python
MODULE  := src

CHUNK_SIZE ?= 2000
K          ?= 10
DATASET    ?= dataset_docs_public
ARGS       ?=

SEARCH_DIR := data/output/search_results/UnansweredQuestions
ANSWER_DIR := data/output/search_results_and_answer/UnansweredQuestions

install:
	uv sync

# Usage: make run ARGS="search 'How to configure the OpenAI server?' --k 5"
run:
	$(PYTHON) -m $(MODULE) $(ARGS)

# Usage: make debug ARGS="search 'some query' --k 5"
debug:
	$(PYTHON) -m pdb -m $(MODULE) $(ARGS)

clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" \) \
		-not -path "./.venv/*" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

# ---- Pipeline shortcuts (not mandatory) ----

index:
	$(PYTHON) -m $(MODULE) index --max_chunk_size $(CHUNK_SIZE)

search_dataset:
	$(PYTHON) -m $(MODULE) search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/$(DATASET).json \
		--k $(K) \
		--save_directory $(SEARCH_DIR)

answer_dataset:
	$(PYTHON) -m $(MODULE) answer_dataset \
		--student_search_results_path $(SEARCH_DIR)/$(DATASET).json \
		--save_directory $(ANSWER_DIR)

evaluate:
	$(PYTHON) -m $(MODULE) evaluate \
		--student_search_results_path $(SEARCH_DIR)/$(DATASET).json \
		--dataset_path data/datasets/AnsweredQuestions/$(DATASET).json

.PHONY: install run debug clean lint lint-strict \
	index search_dataset answer_dataset evaluate pipeline

pipeline: index search_dataset evaluate answer_dataset