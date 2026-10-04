import fire


from .retriever import run_search_dataset

class CLI:
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
    ) -> None:

        output_path = run_search_dataset(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k,
        )

        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(CLI)