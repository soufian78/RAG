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
    
class UnansweredQuestion(BaseModel):
    question_id: str
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str 


class RagDataset(BaseModel):
    rag_questions: list[
        AnsweredQuestion | UnansweredQuestion
    ]

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]
    
    
class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int
