import ast
import re
from models import Chunk


def chunk_python(
    text: str,
    file_path: str,
    max_chunk: int
) -> list[Chunk]:

    chunks: list[Chunk] = []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines(keepends=True)

    offset_lines: list[int] = [0]

    for line in lines:
        offset_lines.append(
            offset_lines[-1] + len(line)
        )

    for node in tree.body:

        if not hasattr(node, "lineno"):
            continue

        start_line = node.lineno - 1

        if node.end_lineno is None:
            continue

        end_line = node.end_lineno

        start = offset_lines[start_line]
        end = offset_lines[end_line]

        chunk_text = text[start:end]

        if len(chunk_text) <= max_chunk:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    start=start,
                    end=end,
                    content=chunk_text,
                )
            )

        else:
            for i in range(0, len(chunk_text), max_chunk):

                part = chunk_text[i:i + max_chunk]

                part_start = start + i
                part_end = part_start + len(part)

                chunks.append(
                    Chunk(
                        file_path=file_path,

                        start=part_start,

                        end=part_end,
                        content=part,
                    )
                )

    return chunks


def chunk_markdown(text: str, filr_path: str, max_chunk: int):
    chunks = [] 
    sections = re.split(
            r"(?=^#{1,6}\s)",
            text,
            flags=re.MULTILINE
        )
    current_pos = 0
    for section in sections:
        if not section:
            continue
        start = text.find(section, current_pos)
        end = start + len(section)
        if (len(section) <= max_chunk):
            chunks.append(Chunk(file_path = filr_path,
                                start = star,
                                end = end,
                                content = section))
        else:
            for i in range(0, len(section),max_chunk ):
                part = section[i: i+max_chunk]
                part_start = start +i
                part_end = part_start + len(part)
                chunks.append(Chunk(file_path = filr_path,
                                    start = part_start,
                                    end = part_end,
                                    content = part))
        current_pos = end
    return chunks

