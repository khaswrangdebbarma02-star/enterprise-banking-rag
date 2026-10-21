
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_pipeline import ask_banking_question


app = FastAPI(
    title="Enterprise Banking Knowledge Engine",
    description=(
        "A RAG API for querying sample banking policy documents."
    ),
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question about the indexed banking documents",
    )


class SourceDetail(BaseModel):
    document: str
    chunk_id: int
    content: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceDetail]


@app.get("/")
def home():
    return {
        "message": "Enterprise Banking Knowledge Engine API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    try:
        result = ask_banking_question(request.question)

        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
        )

    except Exception as exc:
        print(f"RAG request failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Please try again.",
        ) from exc
