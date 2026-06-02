from __future__ import annotations

from fastapi import FastAPI, HTTPException

from domain_slm_guardrails.api.rag import answer_query
from domain_slm_guardrails.api.schemas import QueryRequest, QueryResponse
from domain_slm_guardrails.core.domain_registry import list_domains


app = FastAPI(
    title="Domain SLM Guardrails RAG API",
    version="0.2.0",
    description="Week 2 baseline RAG API with citation-bearing responses.",
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "domains": list_domains()}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return answer_query(
            domain=request.domain,
            query=request.query,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

