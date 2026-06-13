from __future__ import annotations

import ast
import html
import json
import math
import operator
import os
import re
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Sequence

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


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "max": max,
    "min": min,
    "round": round,
    "sqrt": math.sqrt,
}
_CONSTANTS = {"e": math.e, "pi": math.pi}


@dataclass(frozen=True)
class DosageCalculation:
    dose_per_kg: float
    weight_kg: float
    administrations_per_day: int
    single_dose: float
    daily_dose: float
    unit: str
    capped_by: list[str]


class SafeExpressionEvaluator:
    """Evaluate arithmetic expressions without allowing Python code execution."""

    max_power: float = 10_000

    def evaluate(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        value = self._eval_node(tree.body)
        if not math.isfinite(value):
            raise ValueError("calculation result is not finite")
        return float(value)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return float(_CONSTANTS[node.id])
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = _BINARY_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
            if isinstance(node.op, ast.Pow) and abs(right) > self.max_power:
                raise ValueError("exponent is too large")
            return float(op(left, right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
            return float(op(self._eval_node(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = _FUNCTIONS.get(node.func.id)
            if func is None:
                raise ValueError(f"function {node.func.id!r} is not allowed")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
            return float(func(*[self._eval_node(arg) for arg in node.args]))
        raise ValueError(f"expression node {type(node).__name__} is not allowed")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Safely evaluate arithmetic and weight-based dosage calculations."

    def __init__(self, evaluator: SafeExpressionEvaluator | None = None):
        self.evaluator = evaluator or SafeExpressionEvaluator()

    def execute(self, **kwargs) -> ToolResult:
        try:
            mode = str(kwargs.get("mode", "expression"))
            if mode == "dosage":
                output = self.calculate_dosage(
                    dose_per_kg=float(kwargs["dose_per_kg"]),
                    weight_kg=float(kwargs["weight_kg"]),
                    administrations_per_day=int(kwargs.get("administrations_per_day", 1)),
                    unit=str(kwargs.get("unit", "mg")),
                    max_single_dose=self._optional_float(kwargs.get("max_single_dose")),
                    max_daily_dose=self._optional_float(kwargs.get("max_daily_dose")),
                )
                return ToolResult(success=True, data=asdict(output))
            expression = str(kwargs["expression"])
            return ToolResult(
                success=True,
                data={"expression": expression, "result": self.evaluator.evaluate(expression)},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def calculate_dosage(
        self,
        dose_per_kg: float,
        weight_kg: float,
        administrations_per_day: int = 1,
        unit: str = "mg",
        max_single_dose: float | None = None,
        max_daily_dose: float | None = None,
    ) -> DosageCalculation:
        if dose_per_kg <= 0:
            raise ValueError("dose_per_kg must be positive")
        if weight_kg <= 0:
            raise ValueError("weight_kg must be positive")
        if administrations_per_day <= 0:
            raise ValueError("administrations_per_day must be positive")
        single_dose = dose_per_kg * weight_kg
        daily_dose = single_dose * administrations_per_day
        capped_by: list[str] = []
        if max_single_dose is not None and single_dose > max_single_dose:
            single_dose = max_single_dose
            capped_by.append("max_single_dose")
        if max_daily_dose is not None and daily_dose > max_daily_dose:
            daily_dose = max_daily_dose
            capped_by.append("max_daily_dose")
        elif "max_single_dose" in capped_by:
            daily_dose = single_dose * administrations_per_day
        return DosageCalculation(
            dose_per_kg=dose_per_kg,
            weight_kg=weight_kg,
            administrations_per_day=administrations_per_day,
            single_dose=round(single_dose, 4),
            daily_dose=round(daily_dose, 4),
            unit=unit,
            capped_by=capped_by,
        )

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)


@dataclass(frozen=True)
class CitationVerification:
    valid: bool
    cited_ids: list[str]
    invalid_ids: list[str]
    unused_ids: list[str]
    unsupported_sentences: list[dict[str, Any]]


class CitationVerificationTool(BaseTool):
    name = "citation_verifier"
    description = "Verify citation markers and basic sentence support against retrieved citations."

    _citation_re = re.compile(r"\[(C\d+)\]")
    _sentence_re = re.compile(r"(?<=[.!?])\s+|\n+")
    _word_re = re.compile(r"[A-Za-z0-9_]+")
    _stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "in", "is", "it", "its", "may", "of", "on", "or", "that", "the",
        "this", "to", "was", "were", "when", "with",
    }

    def __init__(self, min_overlap: float = 0.18):
        self.min_overlap = min_overlap

    def execute(self, answer: str, citations: Sequence[Any], **kwargs) -> ToolResult:
        try:
            return ToolResult(success=True, data=asdict(self.verify(answer, citations)))
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def verify(self, answer: str, citations: Sequence[Any]) -> CitationVerification:
        citation_map = {self._citation_id(c): c for c in citations}
        citation_ids = set(citation_map)
        cited_ids = list(dict.fromkeys(self._citation_re.findall(answer)))
        invalid_ids = [cid for cid in cited_ids if cid not in citation_ids]
        unused_ids = sorted(citation_ids.difference(cited_ids))
        unsupported: list[dict[str, Any]] = []
        for sentence in self._sentences(answer):
            sentence_citations = self._citation_re.findall(sentence)
            if not sentence_citations:
                continue
            valid_sentence_citations = [cid for cid in sentence_citations if cid in citation_map]
            if not valid_sentence_citations:
                unsupported.append({
                    "sentence": sentence,
                    "citation_ids": sentence_citations,
                    "best_overlap": 0.0,
                })
                continue
            sentence_terms = self._terms(self._citation_re.sub("", sentence))
            best_overlap = max(
                self._overlap(sentence_terms, self._terms(self._citation_text(citation_map[cid])))
                for cid in valid_sentence_citations
            )
            if best_overlap < self.min_overlap:
                unsupported.append({
                    "sentence": sentence,
                    "citation_ids": valid_sentence_citations,
                    "best_overlap": round(best_overlap, 4),
                })
        return CitationVerification(
            valid=not invalid_ids and not unsupported,
            cited_ids=cited_ids,
            invalid_ids=invalid_ids,
            unused_ids=unused_ids,
            unsupported_sentences=unsupported,
        )

    @staticmethod
    def _citation_id(citation: Any) -> str:
        return str(citation.get("citation_id") if isinstance(citation, dict) else citation.citation_id)

    @staticmethod
    def _citation_text(citation: Any) -> str:
        return str(citation.get("text", "") if isinstance(citation, dict) else citation.text)

    @classmethod
    def _sentences(cls, answer: str) -> list[str]:
        parts = [part.strip() for part in cls._sentence_re.split(answer) if part.strip()]
        merged: list[str] = []
        for part in parts:
            if cls._citation_re.match(part) and merged:
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)
        return merged

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        return {
            term.lower()
            for term in cls._word_re.findall(text)
            if len(term) > 1 and term.lower() not in cls._stopwords
        }

    @staticmethod
    def _overlap(answer_terms: set[str], citation_terms: set[str]) -> float:
        if not answer_terms:
            return 0.0
        return len(answer_terms & citation_terms) / len(answer_terms)


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Optional external web search for information outside the local corpus."

    def __init__(
        self,
        enabled: bool | None = None,
        fetcher: Callable[[str, float], str] | None = None,
        timeout: float = 5.0,
    ):
        if enabled is None:
            enabled = os.getenv("AGENTIC_RAG_ENABLE_WEB_SEARCH", "").lower() in {
                "1",
                "true",
                "yes",
            }
        self.enabled = enabled
        self.fetcher = fetcher or self._fetch
        self.timeout = timeout

    def execute(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        try:
            if not self.enabled:
                return ToolResult(
                    success=True,
                    data=[],
                    error="web_search_disabled",
                )
            params = urllib.parse.urlencode({"q": query})
            page = self.fetcher(f"https://duckduckgo.com/html/?{params}", self.timeout)
            return ToolResult(
                success=True,
                data=[asdict(r) for r in self._parse_duckduckgo_html(page, max_results)],
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def _fetch(url: str, timeout: float) -> str:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "domain-slm-guardrails-agent/0.1"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    @staticmethod
    def _parse_duckduckgo_html(page: str, max_results: int) -> list[WebSearchResult]:
        item_re = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>'
            r"(?P<title>.*?)</a>.*?"
            r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
            re.DOTALL,
        )
        results: list[WebSearchResult] = []
        for match in item_re.finditer(page):
            raw_url = html.unescape(match.group("url"))
            parsed = urllib.parse.urlparse(raw_url)
            if parsed.path == "/l/":
                raw_url = urllib.parse.parse_qs(parsed.query).get("uddg", [raw_url])[0]
            results.append(
                WebSearchResult(
                    title=WebSearchTool._clean_html(match.group("title")),
                    url=raw_url,
                    snippet=WebSearchTool._clean_html(match.group("snippet")),
                )
            )
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _clean_html(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        return html.unescape(re.sub(r"\s+", " ", text)).strip()
