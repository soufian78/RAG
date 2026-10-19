
import ast
import re
from .models import Chunk


def split_with_overlap(
    text: str,
    file_path: str,
    start_offset: int,
    max_chunk: int,
    overlap: int = 100,
) -> list[Chunk]:
    """Split text into overlapping chunks with source offsets."""
    if max_chunk <= 0 or overlap < 0 or overlap >= max_chunk:
        return []

    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk, len(text))

        chunks.append(
            Chunk(
                file_path=file_path,
                start=start_offset + start,
                end=start_offset + end,
                content=text[start:end],
            )
        )

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_python(text: str, file_path: str, max_chunk: int) -> list[Chunk]:
    """Split Python code into overlapping chunks."""
    if max_chunk <= 0:
        return []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines(keepends=True)
    offset_lines = [0]

    for line in lines:
        offset_lines.append(offset_lines[-1] + len(line))

    chunks: list[Chunk] = []

    for node in tree.body:
        if not hasattr(node, "lineno") or node.end_lineno is None:
            continue

        start = offset_lines[node.lineno - 1]
        end = offset_lines[node.end_lineno]
        node_text = text[start:end]

        if len(node_text) <= max_chunk:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    start=start,
                    end=end,
                    content=node_text,
                )
            )
        else:
            chunks.extend(
                split_with_overlap(
                    node_text,
                    file_path,
                    start,
                    max_chunk,
                    overlap=100,
                )
            )

    return chunks


def chunk_markdown(text: str, file_path: str, max_chunk: int) -> list[Chunk]:
    """Split Markdown into overlapping chunks while preserving offsets."""
    if max_chunk <= 0:
        return []

    chunks: list[Chunk] = []
    sections = re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)

    current_pos = 0

    for section in sections:
        if not section:
            continue

        section_start = current_pos
        section_end = section_start + len(section)

        # Keep paragraph boundaries when possible.
        local_start = 0

        while local_start < len(section):
            local_end = min(local_start + max_chunk, len(section))

            if local_end < len(section):
                split_pos = section.rfind("\n\n", local_start, local_end)
                if split_pos > local_start:
                    local_end = split_pos + 2

            if local_end <= local_start:
                local_end = min(local_start + max_chunk, len(section))

            chunks.append(
                Chunk(
                    file_path=file_path,
                    start=section_start + local_start,
                    end=section_start + local_end,
                    content=section[local_start:local_end],
                )
            )

            if local_end == len(section):
                break

            local_start = max(local_end - 100, local_start + 1)

        current_pos = section_end

    return chunks
