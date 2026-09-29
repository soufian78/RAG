from pathlib import Path
from chunker import chunk_markdown,chunk_python
from models import Chunk

def index_files(raw_path: str, max_chunk:int):
    chunks = []
    for path in Path(raw_path).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in [".py", ".md"]:
            continue
        try:
            text = path.read_text(encoding="uts-8")
        except (UnicodeDecodeError, OSError):
            continue
        if path.suffix == ".py":
            file_chunks = chunk_python(text,
                                       str(path),
                                       max_chunk)
        elif path.suffix == ".md":
            file_chunks = chunk_markdown(text,
                                         str(path),
                                         max_chunk)
        chunks.extend(file_chunks)
    return chunks