import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from models import RagDataset, Chunk, MinimalSource, MinimalSearchResults, StudentSearchResults
def tokenize(text: str) -> list[str]:
    return text.lower().split()


def search(
    query: str,
    chunks: list[Chunk],
    bm25: BM25Okapi,
    k: int = 5,
) -> list[Chunk]:

    tokens = tokenize(query)

    scores = bm25.get_scores(tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    top_indices = ranked_indices[:k]

    return [chunks[i] for i in top_indices]


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


def load_index() -> tuple[list[Chunk], BM25Okapi]:
    input_dir = Path("data/processed")

    with open(
        input_dir / "chunks.json",
        "r",
        encoding="utf-8",
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


def search_dataset(dataset_path: str, chunks: list[Chunk], bm25: BM25Okapi, k: int = 5)->list[StudentSearchResults]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        dataset = RagDataset(**data)
        
        results: list[MinimalSearchResults] = []
        for question in dataset.rag_questions:
            result = retrieve(question_id=question.question_id,
                              query=question.question,
                              chunks=chunks,
                              k = k)
            results.append(result)
        
        return StudentSearchResults(search_results = results, k=k)
            

def save_search_results(results: StudentSearchResults,
                        dataset_path: str,
                        save_dir) -> Path:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
        
    






if __name__ == "__main__":
    chunks, bm25 = load_index()

    result = retrieve(
        question_id="q1",
        query="hello function",
        chunks=chunks,
        bm25=bm25,
        k=3,
    )

    print(result.model_dump_json(indent=2))