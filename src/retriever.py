from rank_bm25 import BM25Okapi
from models import Chunk, MinimalSource

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from models import Chunk


def tokenize(text: str):
    return text.lower().split()

def search(
        query: str,
        chunks: list[Chunk],
        bm25: BM25Okapi,
        k: int = 5,
    ) -> list[MinimalSource]:
    tokens = tokenize(query)
    
    scores = bm25.get_scores(tokens)
    
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = ranked_indices[:k]
    result = []
    
    for i in top_indices:
        chunk = chunks[i]
        result.append(MinimalSource(file_path = chunk.file_path,
                                    first_character_index = chunk.start,
                                    last_character_index = chunk.end))
    return result




def load_index() -> tuple[list[Chunk], BM25Okapi]:
    input_dir = Path("data/processed")

    with open(input_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = [
        Chunk(**data)
        for data in chunks_data
    ]

    with open(input_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    return chunks, bm25