import fire
import time

from .retriever import run_search_dataset, run_search
from .indexer import run_index
from .generator import run_generation_dataset


class CLI:
    """Provide command-line commands to index, search, and answer questions."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index raw files by splitting their content into chunks."""
        run_index(raw_path="data/raw", max_chunk=max_chunk_size)

    def search(self, query: str, k: int = 5) -> None:
        """Search the index and display the most relevant chunks."""
        result = run_search(query=query, k=k)
        print(result.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search for relevant chunks for every question in a dataset."""
        output_path = run_search_dataset(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k,
        )
        print(f"Saved to: {output_path}")

    def answer_dataset(
        self,
        student_search_results_path: str = (
            "data/datasets/UnansweredQuestions/dataset_docs_public.json"
        ),
        save_directory: str = "data/output/search_results_and_answer",
        k: int = 5,
    ) -> None:
        """Generate answers for questions using retrieved search results."""


        output_path = run_generation_dataset(
            dataset_path=student_search_results_path,
            save_directory=save_directory,
            k=k,
        )

        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except Exception as e:
        print(e)
        exit(1)
    except InterruptedError as e:
        print(e)
