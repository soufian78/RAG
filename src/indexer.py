from pathlib import Path

from chunker import chunk_python, chunk_markdown
from models import Chunk


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
    return text.lower().split()


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    tokenized_corpus = [
        tokenize(chunk.content)
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_corpus)