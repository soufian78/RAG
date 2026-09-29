from pydantic import BaseModel


class Chunk(BaseModel):
    file_path:str
    start: int
    end: int
    content: str
    
