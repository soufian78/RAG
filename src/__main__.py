try:
    import fire
    from .retriever import run_search_dataset, run_search
    from .indexer import run_index
    from .generator import run_generation_dataset, answer_question
    from .evaluator import evaluate_search_results
    import sys
except (KeyboardInterrupt, ImportError):
    print("error ")
    exit(1)


class CLI:
    """Provide command-line commands to index, search, and answer questions."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index raw files by splitting their content into chunks."""
        try:
            int(max_chunk_size)
        except ValueError:
            print("max_chunk_size must be integer")
            sys.exit(1)
        run_index(raw_path="data/raw", max_chunk=max_chunk_size)

    def search(self, query: str, k: int = 5) -> None:
        """Search the index and display the most relevant chunks."""
        try:
            int(k)
            if k <= 0:
                raise ValueError
        except ValueError:
            print("k must be possitive integer")
            sys.exit(1)
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
        try:
            int(k)
            if k <= 0:
                raise ValueError
        except ValueError:
            print("k must be possitive integer")
            sys.exit(1)
        print(f"Saved to: {output_path}")

    def answer(self, query: str, k: int = 3) -> None:
        """Generate an answer for a single question."""
        result = answer_question(question=query, k=k)
        print(result)

    def answer_dataset(
        self,
        student_search_results_path: str = (
            "data/datasets/UnansweredQuestions/dataset_docs_public.json"
        ),
        save_directory: str = "data/output/search_results_and_answer",
        k: int = 5,
    ) -> None:
        """Generate answers for questions using retrieved search results."""
        try:
            int(k)
            if k <= 0:
                raise ValueError
        except ValueError:
            print("k must be possitive integer")
            sys.exit(1)
        output_path = run_generation_dataset(
            dataset_path=student_search_results_path,
            save_directory=save_directory,
            k=k,
        )

        print(f"Saved to: {output_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate retrieval quality using Recall@k."""

        try:
            scores = evaluate_search_results(
                student_search_results_path=student_search_results_path,
                dataset_path=dataset_path,
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"Evaluation failed: {error}")
            return

        print("\nEvaluation Results")
        print("=" * 30)

        for cutoff, score in scores.items():
            print(f"Recall@{cutoff}: {score:.3f}")


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except Exception as e:
        print(e)
        sys.exit(1)
    except (KeyboardInterrupt, InterruptedError) as e:
        print(e)
        sys.exit(1)
