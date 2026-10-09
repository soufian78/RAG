import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from .models import RagDataset, Chunk, MinimalSource, MinimalSearchResults, StudentSearchResults

import uuid
from functools import lru_cache




def tokenize(text: str) -> list[str]:
    return text.lower().split()

def search(
    query: str,
    chunks: list[Chunk],
    bm25: BM25Okapi,
    k: int = 5,
) -> list[Chunk]:

    if k <= 0 or not query.strip():
        return []

    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: (
            float(scores[i])
            + (
                0.5
                if "/docs/" in chunks[i].file_path
                else 0.0
            )
        ),
        reverse=True,
    )

    return [chunks[i] for i in ranked_indices[:k]]


def chunk_to_source(chunk: Chunk) -> MinimalSource:
    return MinimalSource(
        file_path=chunk.file_path,
        first_character_index=chunk.start,
        last_character_index=chunk.end,
    )


def search_source(
    query: str,
    chunks: list[Chunk],
    bm25: BM25Okapi,
    k: int = 5,
) -> list[MinimalSource]:

    found_chunks = search(
        query=query,
        chunks=chunks,
        bm25=bm25,
        k=k,
    )

    return [
        chunk_to_source(chunk)
        for chunk in found_chunks
    ]


@lru_cache(maxsize=1)
def load_index() -> tuple[list[Chunk], BM25Okapi]:
    input_dir = Path("data/processed")

    with open(
        input_dir / "chunks.json",
        "r",
        encoding="utf-8"
    ) as f:
        chunks_data = json.load(f)

    chunks = [
        Chunk(**data)
        for data in chunks_data
    ]

    with open(input_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    return chunks, bm25

def retrieve(
    question_id: str,
    query: str,
    chunks: list[Chunk],
    bm25: BM25Okapi,
    k: int = 5,
) -> MinimalSearchResults:

    sources = search_source(
        query=query,
        chunks=chunks,
        bm25=bm25,
        k=k,
    )

    return MinimalSearchResults(
        question_id=question_id,
        question=query,
        retrieved_sources=sources,
    )


def search_dataset(dataset_path: str, chunks: list[Chunk], bm25: BM25Okapi, k: int = 5)->StudentSearchResults:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = RagDataset(**data)
        
    results: list[MinimalSearchResults] = []
    for question in dataset.rag_questions:
        result = retrieve(question_id=question.question_id,
                          query=question.question,
                          chunks=chunks,
                          bm25=bm25,
                          k = k)
        results.append(result)
        
    return StudentSearchResults(search_results = results, k=k)



def save_search_results(results: StudentSearchResults,
                        dataset_path: str,
                        save_dir) -> Path:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(dataset_path).name
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results.model_dump(),
            f,
            indent=2,
            ensure_ascii=False
        )
    return output_path

def run_search_dataset(dataset_path: str,
                       save_directory: str,
                       k: int = 5)->Path:
    chunks, bm25 = load_index()
    results = search_dataset(dataset_path=dataset_path, chunks=chunks, bm25=bm25, k=k)
    output_path = save_search_results(results=results, dataset_path=dataset_path, save_dir=save_directory)
    return output_path
    

def run_search(
    query: str,
    k: int = 5) -> MinimalSearchResults:
    chunks, bm25 = load_index()
    return retrieve(question_id=str(uuid.uuid4()),
                    query=query,
                    chunks=chunks,
                    bm25=bm25,
                    k=k)




if __name__ == "__main__":
    output_path = run_search_dataset(
        dataset_path="data/datasets/test.json",
        save_directory="data/output/search_results/test",
        k=3,
    )

    print("Saved to:", output_path)