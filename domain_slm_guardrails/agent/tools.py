from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, Optional

from domain_slm_guardrails.api.rag import _expand_query, answer_query
from domain_slm_guardrails.retrieval.hybrid import load_hybrid_retriever
from domain_slm_guardrails.retrieval.preprocessor import QueryPreprocessor
from domain_slm_guardrails.retrieval.reranker import CrossEncoderReranker
from domain_slm_guardrails.guardrails.guardrails_manager import GuardrailsManager


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for agent tools."""

    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments and return a result."""
        pass


class ToolRegistry:
    """Registry to manage all available tools for the agent."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a new tool in the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool with name {tool.name} already exists.")
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool from the registry by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> list[BaseTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_tool_descriptions(self) -> str:
        """Get a JSON string of all tool descriptions for LLM prompt."""
        descriptions = []
        for tool in self.list_tools():
            descriptions.append({
                "name": tool.name,
                "description": tool.description
            })
        return json.dumps(descriptions, indent=2)


class SearchRetrieverTool(BaseTool):
    name = "search_retriever"
    description = "Search the RAG system for documents relevant to a query. Takes query, domain, and top_k."

    def __init__(self, domain: str):
        self.domain = domain
        self.retriever = load_hybrid_retriever(domain)

    def execute(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        try:
            results = self.retriever.search(query, top_k=top_k)
            return ToolResult(
                success=True,
                data=[{
                    "text": r.chunk.get("text", ""),
                    "source_id": r.chunk.get("source_id", ""),
                    "score": r.score
                } for r in results]
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class QueryExpanderTool(BaseTool):
    name = "query_expander"
    description = "Expand a query to improve retrieval. Takes query as input."

    def __init__(self):
        self.preprocessor = QueryPreprocessor()

    def execute(self, query: str, **kwargs) -> ToolResult:
        try:
            processed_obj = self.preprocessor.preprocess(query)
            processed = processed_obj.cleaned
            expanded_term = _expand_query(query)
            if expanded_term:
                expanded_query = f"{processed} {expanded_term}"
            else:
                expanded_query = processed_obj.expanded
            return ToolResult(
                success=True,
                data={
                    "original_query": query,
                    "processed_query": processed,
                    "expanded_query": expanded_query
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class RerankResultsTool(BaseTool):
    name = "rerank_results"
    description = "Rerank retrieved results using a cross-encoder. Takes query and results list."

    def __init__(self):
        try:
            self.reranker = CrossEncoderReranker()
        except Exception:
            self.reranker = None

    def execute(self, query: str, results: list[dict], **kwargs) -> ToolResult:
        try:
            if not self.reranker:
                return ToolResult(success=True, data=results)  # fallback to original
            reranked = self.reranker.rerank(query, results)
            return ToolResult(success=True, data=reranked)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CheckGuardrailsTool(BaseTool):
    name = "check_guardrails"
    description = "Check if a response is safe and grounded in context. Takes query, generated_response, and retrieved_chunks."

    def __init__(self):
        self.guardrails_manager = GuardrailsManager()

    def execute(
        self,
        query: str,
        generated_response: str,
        retrieved_chunks: list[dict],
        **kwargs
    ) -> ToolResult:
        try:
            result = self.guardrails_manager.apply_guardrails(
                query=query,
                generated_response=generated_response,
                retrieved_chunks=retrieved_chunks
            )
            return ToolResult(
                success=True,
                data={
                    "overall_pass": result.overall_pass,
                    "warnings": result.warnings,
                    "suggestions": result.suggestions,
                    "confidence_score": result.confidence_score
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
