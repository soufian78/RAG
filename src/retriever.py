from rank_bm25 import BM25Okapi
from models import Chunk, MinimalSource



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
    