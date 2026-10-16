import fire


from .retriever import run_search_dataset,run_search
from .indexer import run_index
from .generator import run_generation_dataset

class CLI:
    def index(self,
              max_chunk_size: int = 2000)->None:
        run_index(raw_path= "data/raw", max_chunk=max_chunk_size)
        
        
    def search(self,
               query: str,
               k: int = 5)->None:
        result = run_search(query=query,
                           k=k)
        print(result.model_dump_json(indent=2))
        
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
    def answer_dataset(self, student_search_results_path=, save_directory ):
        run_generation_dataset(dataset_path=student_search_results_path, save_directory=save_directory)
        


if __name__ == "__main__":
    fire.Fire(CLI)
