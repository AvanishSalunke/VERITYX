from typing import Any
from pydantic import BaseModel


class QueryRequest(BaseModel):
    collection_name: str
    query: str


class TraceStep(BaseModel):
    """Exactly the four keys the organizers require."""
    step_name: str
    description: str
    input: Any
    output: Any


class Chunk(BaseModel):
    text: str
    document: str
    page: int


class QueryResponse(BaseModel):
    answer: str
    chunks: list[Chunk]
    trace: list[TraceStep]
