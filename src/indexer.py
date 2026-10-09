from .chunker import chunk_python, chunk_markdown
from .models import Chunk
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import re

def index_files(
    raw_path: str,
    max_chunk: int
) -> list[Chunk]:

    chunks: list[Chunk] = []

    for path in Path(raw_path).rglob("*"):

        if not path.is_file():
            continue

        if path.suffix not in [".py", ".md"]:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        if path.suffix == ".py":
            file_chunks = chunk_python(
                text,
                str(path),
                max_chunk
            )

        else:
            file_chunks = chunk_markdown(
                text,
                str(path),
                max_chunk
            )

        chunks.extend(file_chunks)

    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())

import re
from pathlib import Path
from rank_bm25 import BM25Okapi


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    tokenized_corpus = []
    current_headings = {}

    for chunk in chunks:
        tokens = tokenize(chunk.content)

        # Garder la pondération du nom de fichier
        filename = Path(chunk.file_path).stem
        tokens += tokenize(filename) * 3

        # Chercher les titres Markdown dans le chunk
        if chunk.file_path.endswith(".md"):
            headings = re.findall(
                r"^#{1,6}\s+(.+)$",
                chunk.content,
                flags=re.MULTILINE
            )

            if headings:
                current_headings[chunk.file_path] = headings[-1]

            heading = current_headings.get(chunk.file_path, "")

            tokens += tokenize(heading) * 2

        tokenized_corpus.append(tokens)

    return BM25Okapi(tokenized_corpus)

def save_index(chunks: list[Chunk], bm25: BM25Okapi):
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_data = [chunk.model_dump() for chunk in chunks]
    with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False)

    with open(output_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
        
def run_index(raw_path: str = "data/raw", max_chunk: int = 2000)-> None:
    chunks = index_files(raw_path, max_chunk)
    
    if not chunks:
        print("No chnks found.")
        return
    bm25 = build_bm25(chunks)
    save_index(chunks, bm25)
    print(f"Indexing complete: {len(chunks)} chunks created.")
     
    
if __name__ == "__main__":
    run_index()
