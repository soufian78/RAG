PYTHON  := uv run python
ARGS    ?=
K       ?= 10
DATASET ?= dataset_docs_public
SEARCH_DIR := data/output/search_results/UnansweredQuestions

install:
	uv sync

run:
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb -m src $(ARGS)

pipeline:
	$(PYTHON) -m src index --max_chunk_size 2000
	$(PYTHON) -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/$(DATASET).json \
		--k $(K) --save_directory $(SEARCH_DIR)
	$(PYTHON) -m src evaluate \
		--student_search_results_path $(SEARCH_DIR)/$(DATASET).json \
		--dataset_path data/datasets/AnsweredQuestions/$(DATASET).json

.PHONY: install run debug pipeline