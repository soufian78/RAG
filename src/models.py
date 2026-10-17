from pydantic import BaseModel


class Chunk(BaseModel):
    """Represent a text chunk from a file."""

    file_path: str
    start: int
    end: int
    content: str


class MinimalSource(BaseModel):
    """Represent the location of a source in a file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question without an answer."""

    question_id: str
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its answer and sources."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset of RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent the search results for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]
    
class MinimalAnswer(MinimalSearchResults):
    """Represent search results for one question with its generated answer."""

    answer: str




class StudentSearchResults(BaseModel):
    """Represent search results for a dataset."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent search results and generated answers for a dataset."""

    search_results: list[MinimalAnswer]
    k: int