from .chunker import chunk_python, chunk_markdown
from .models import Chunk
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import re
import ast

def index_files(raw_path: str, max_chunk: int) -> list[Chunk]:
    """Read files and split them into chunks."""

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
            file_chunks = chunk_python(text, str(path), max_chunk)

        else:
            file_chunks = chunk_markdown(text, str(path), max_chunk)

        chunks.extend(file_chunks)

    return chunks


def tokenize(text: str) -> list[str]:
    """Split text into searchable tokens, handling underscores and CamelCase."""

    # Split CamelCase: loadIndex -> load Index
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)

    # Split acronyms: HTTPResponse -> HTTP Response
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)

    # Split underscores
    text = text.replace("_", " ")

    # Lowercase and extract tokens
    tokens = re.findall(r"[a-z0-9]+", text.lower())

    # Keep the complete normalized term and its parts.
    return tokens


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    """Build the BM25 index from chunks."""

    tokenized_corpus = []
    current_headings = {}

    for chunk in chunks:
        tokens = tokenize(chunk.content)

        # Give more weight to the filename.
        filename = Path(chunk.file_path).stem
        tokens += tokenize(filename) * 4

        # Add Python class and function names.
        if chunk.file_path.endswith(".py"):
            try:
                tree = ast.parse(chunk.content)

                for node in ast.walk(tree):
                    if isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    ):
                        tokens += tokenize(node.name) * 2

            except SyntaxError:
                pass

        # Keep Markdown headings as searchable terms.
        if chunk.file_path.endswith(".md"):
            heading_re = re.compile(
                r"^#{1,6}\s+(.+)$",
                re.MULTILINE,
            )
            headings = heading_re.findall(chunk.content)

            if headings:
                current_headings[chunk.file_path] = headings[-1]

            heading = current_headings.get(chunk.file_path, "")
            tokens += tokenize(heading)

        tokenized_corpus.append(tokens)

    return BM25Okapi(
        tokenized_corpus,
        k1=1,
        b=0.75,
    )



def save_index(chunks: list[Chunk], bm25: BM25Okapi) -> None:
    """Save chunks and the BM25 index to disk."""

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_data = [chunk.model_dump() for chunk in chunks]

    with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False)

    with open(output_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)


def run_index(raw_path: str = "data/raw", max_chunk: int = 2000) -> None:
    """Run the complete indexing process."""

    if max_chunk > 2000:
        print("max_chunk is too high!")
        return

    if max_chunk < 100:
        print("max_chunk is too low!")
        return

    chunks = index_files(raw_path, max_chunk)

    if not chunks:
        print("No chunks found.")
        return

    bm25 = build_bm25(chunks)
    save_index(chunks, bm25)

    print(f"Indexing complete: {len(chunks)} chunks created.")


if __name__ == "__main__":
    run_index()
