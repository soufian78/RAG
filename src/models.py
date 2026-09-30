from pydantic import BaseModel


class Chunk(BaseModel):
    file_path:str
    start: int
    end: int
    content: str
    
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    
