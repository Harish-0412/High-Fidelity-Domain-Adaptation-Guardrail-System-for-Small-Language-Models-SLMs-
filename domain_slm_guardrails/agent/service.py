from __future__ import annotations

import re
import time
from typing import Any

from domain_slm_guardrails.agent.tools import (
    CalculatorTool,
    CitationVerificationTool,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
)
from domain_slm_guardrails.api.rag import answer_query
from domain_slm_guardrails.api.schemas import (
    AgentExecutionStep,
    AgentPlanResponse,
    AgentToolInfo,
    GuardrailStatus,
    QueryResponse,
)


_ARITHMETIC_RE = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?\s*[-+*/%^]\s*)+\d+(?:\.\d+)?")
_DOSE_PER_KG_RE = re.compile(
    r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml)?\s*/\s*kg",
    re.IGNORECASE,
)
_WEIGHT_RE = re.compile(r"(?P<weight>\d+(?:\.\d+)?)\s*kg\b", re.IGNORECASE)
_FREQUENCY_RE = re.compile(
    r"(?P<count>\d+)\s*(?:times|x|doses|administrations)\s*(?:per|a)?\s*day",
    re.IGNORECASE,
)
_WEB_SEARCH_TERMS = {
    "latest",
    "current",
    "recent",
    "today",
    "new",
    "updated",
    "external",
    "web",
}


class AgenticRAGService:
    """Phase 3/4 deterministic planner and executor for the API path."""

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or self._build_registry()

    @staticmethod
    def _build_registry() -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WebSearchTool())
        registry.register(CitationVerificationTool())
        return registry

    def list_tools(self) -> list[AgentToolInfo]:
        return [
            AgentToolInfo(name=tool.name, description=tool.description)
            for tool in self.registry.list_tools()
        ]

    def plan(self, domain: str, query: str, top_k: int = 5) -> list[AgentExecutionStep]:
        steps: list[AgentExecutionStep] = []
        calculation = self._extract_calculation(query)
        if calculation:
            steps.append(
                AgentExecutionStep(
                    step_id="S1",
                    tool_name="calculator",
                    action=calculation["action"],
                    reason=calculation["reason"],
                    inputs=calculation["inputs"],
                )
            )
        if self._should_search_web(query):
            steps.append(
                AgentExecutionStep(
                    step_id=f"S{len(steps) + 1}",
                    tool_name="web_search",
                    action="search_external_context",
                    reason="Query asks for current or external information beyond the local corpus.",
                    inputs={"query": query, "max_results": 3},
                )
            )
        steps.append(
            AgentExecutionStep(
                step_id=f"S{len(steps) + 1}",
                tool_name="rag_answer",
                action="retrieve_and_generate_grounded_answer",
                reason="Use the domain corpus as the primary evidence source.",
                inputs={"domain": domain, "query": query, "top_k": top_k},
            )
        )
        steps.append(
            AgentExecutionStep(
                step_id=f"S{len(steps) + 1}",
                tool_name="citation_verifier",
                action="verify_answer_citations",
                reason="Validate citation markers and evidence overlap.",
                inputs={},
            )
        )
        steps.append(
            AgentExecutionStep(
                step_id=f"S{len(steps) + 1}",
                tool_name="response_synthesizer",
                action="compose_final_response",
                reason="Combine RAG answer with transparent tool observations.",
                inputs={},
            )
        )
        return steps

    def describe_plan(
        self,
        domain: str,
        query: str,
        top_k: int = 5,
        include_outputs: bool = False,
    ) -> AgentPlanResponse:
        if include_outputs:
            execution = self.run(domain=domain, query=query, top_k=top_k)
            return AgentPlanResponse(
                domain=domain,
                query=query,
                tools_available=self.list_tools(),
                steps=execution.steps,
                final_answer=execution.response.answer,
                citations=execution.response.citations,
                latency_ms=execution.response.latency_ms,
            )
        return AgentPlanResponse(
            domain=domain,
            query=query,
            tools_available=self.list_tools(),
            steps=self.plan(domain=domain, query=query, top_k=top_k),
        )

    def run(self, domain: str, query: str, top_k: int = 5) -> "AgenticExecution":
        started = time.perf_counter()
        steps = self.plan(domain=domain, query=query, top_k=top_k)
        notes: list[str] = []
        response: QueryResponse | None = None

        for step in steps:
            if step.tool_name in {"calculator", "web_search"}:
                result = self._execute_tool(step.tool_name, **step.inputs)
                self._mark_step(step, result)
                if result.success:
                    notes.extend(self._format_tool_notes(step.tool_name, result))
                continue

            if step.tool_name == "rag_answer":
                response = answer_query(domain=domain, query=query, top_k=top_k)
                step.status = "completed"
                step.output = {
                    "answer_preview": response.answer[:300],
                    "citation_count": len(response.citations),
                    "fallback_used": response.guardrail_status.fallback_used,
                }
                continue

            if step.tool_name == "citation_verifier":
                if response is None:
                    step.status = "skipped"
                    step.error = "RAG answer did not complete."
                    continue
                result = self._execute_tool(
                    "citation_verifier",
                    answer=response.answer,
                    citations=response.citations,
                )
                self._mark_step(step, result)
                if result.success and not result.data.get("valid", False):
                    self._downgrade_for_failed_verification(response, result.data)
                continue

            if step.tool_name == "response_synthesizer":
                if response is None:
                    step.status = "skipped"
                    continue
                response.answer = self._compose_answer(response.answer, notes)
                response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
                step.status = "completed"
                step.output = {
                    "answer_preview": response.answer[:300],
                    "tool_note_count": len(notes),
                }

        if response is None:
            response = QueryResponse(
                domain=domain,
                query=query,
                answer="I could not complete the agentic RAG workflow.",
                citations=[],
                guardrail_status=GuardrailStatus(
                    rag_grounded=False,
                    fallback_used=True,
                    reason="agent_execution_failed",
                ),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        return AgenticExecution(response=response, steps=steps)

    def _execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(success=False, error=f"unknown tool: {tool_name}")
        return tool.execute(**kwargs)

    @staticmethod
    def _extract_calculation(query: str) -> dict[str, Any] | None:
        dose_match = _DOSE_PER_KG_RE.search(query)
        weight_match = _WEIGHT_RE.search(query)
        if dose_match and weight_match:
            frequency_match = _FREQUENCY_RE.search(query)
            return {
                "action": "calculate_weight_based_dosage",
                "reason": "Query contains a dose-per-kg expression and patient weight.",
                "inputs": {
                    "mode": "dosage",
                    "dose_per_kg": float(dose_match.group("dose")),
                    "weight_kg": float(weight_match.group("weight")),
                    "administrations_per_day": int(frequency_match.group("count"))
                    if frequency_match
                    else 1,
                    "unit": dose_match.group("unit") or "mg",
                },
            }
        arithmetic_match = _ARITHMETIC_RE.search(query)
        if arithmetic_match:
            return {
                "action": "calculate_arithmetic_expression",
                "reason": "Query contains an explicit arithmetic expression.",
                "inputs": {"expression": arithmetic_match.group(0).replace("^", "**")},
            }
        return None

    @staticmethod
    def _should_search_web(query: str) -> bool:
        terms = {term.lower() for term in re.findall(r"[A-Za-z]+", query)}
        return bool(terms & _WEB_SEARCH_TERMS)

    @staticmethod
    def _mark_step(step: AgentExecutionStep, result: ToolResult) -> None:
        step.status = "completed" if result.success else "failed"
        step.output = result.data
        step.error = result.error

    @staticmethod
    def _format_tool_notes(tool_name: str, result: ToolResult) -> list[str]:
        if tool_name == "calculator" and isinstance(result.data, dict):
            if {"single_dose", "daily_dose"}.issubset(result.data):
                return [
                    "Calculation: "
                    f"{result.data['dose_per_kg']:g} {result.data['unit']}/kg x "
                    f"{result.data['weight_kg']:g} kg = "
                    f"{result.data['single_dose']:g} {result.data['unit']} per dose; "
                    f"{result.data['daily_dose']:g} {result.data['unit']}/day."
                ]
            return [f"Calculation: {result.data['expression']} = {result.data['result']:g}."]
        if tool_name == "web_search":
            if result.error == "web_search_disabled":
                return ["External web search was planned but is disabled in this environment."]
            if not result.data:
                return ["External web search returned no usable results."]
            return [
                f"External result: {item['title']} - {item['url']}"
                for item in result.data[:3]
            ]
        return []

    @staticmethod
    def _downgrade_for_failed_verification(
        response: QueryResponse,
        verification: dict[str, Any],
    ) -> None:
        response.guardrail_status.rag_grounded = False
        response.guardrail_status.fallback_used = True
        response.guardrail_status.reason = "citation_verification_failed"
        response.answer = (
            f"{response.answer}\n\n"
            "Citation verification warning: the agent detected "
            f"{len(verification.get('unsupported_sentences', []))} unsupported cited "
            f"sentence(s) and {len(verification.get('invalid_ids', []))} invalid citation id(s)."
        )

    @staticmethod
    def _compose_answer(answer: str, notes: list[str]) -> str:
        if not notes:
            return answer
        return f"{answer}\n\nAgent tool notes:\n" + "\n".join(f"- {note}" for note in notes)


class AgenticExecution:
    def __init__(self, response: QueryResponse, steps: list[AgentExecutionStep]):
        self.response = response
        self.steps = steps
