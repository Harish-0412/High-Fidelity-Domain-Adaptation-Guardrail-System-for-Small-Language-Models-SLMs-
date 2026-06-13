from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from domain_slm_guardrails.agent.tools import (
    ToolRegistry,
    ToolResult,
    SearchRetrieverTool,
    QueryExpanderTool,
    RerankResultsTool,
    CheckGuardrailsTool
)
from domain_slm_guardrails.llm import OllamaClient, LLMConfig
from domain_slm_guardrails.llm.prompts import PromptTemplates


@dataclass
class AgentStep:
    step_number: int
    tool_name: Optional[str]
    tool_arguments: dict
    tool_result: Optional[ToolResult]
    reasoning: str


@dataclass
class AgentResponse:
    answer: str
    steps: list[AgentStep]
    confidence: float


class MedicalRAGAgent:
    """Agent that uses tools to answer medical prescription questions."""

    def __init__(
        self,
        domain: str = "medical_prescription",
        llm_config: Optional[LLMConfig] = None
    ):
        self.domain = domain
        self.llm = OllamaClient(llm_config) if llm_config else OllamaClient()
        self.logger = logging.getLogger(__name__)
        self.registry = ToolRegistry()

        # Register all tools
        self.registry.register(SearchRetrieverTool(domain))
        self.registry.register(QueryExpanderTool())
        self.registry.register(RerankResultsTool())
        self.registry.register(CheckGuardrailsTool())

    def _decide_next_step(
        self,
        query: str,
        conversation_history: list[AgentStep]
    ) -> Optional[tuple[str, dict, str]]:
        """
        Ask the LLM what the next step should be.
        Returns a tuple of (tool_name, tool_arguments, reasoning), or None if done.
        """
        # First: always follow fixed plan to avoid loops
        if not conversation_history:
            return ("query_expander", {"query": query}, "Expand query first")
        elif len(conversation_history) == 1:
            last_step = conversation_history[-1]
            if last_step.tool_result and last_step.tool_result.success and last_step.tool_result.data:
                expanded_query = last_step.tool_result.data.get("expanded_query", query)
            else:
                expanded_query = query
            return (
                "search_retriever",
                {"query": expanded_query, "top_k": 5},
                "Search for relevant documents"
            )
        elif len(conversation_history) == 2:
            # After search, we have retrieved chunks
            return None
        else:
            return None

    def _execute_step(
        self,
        tool_name: str,
        tool_args: dict
    ) -> ToolResult:
        tool = self.registry.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool {tool_name} not found"
            )
        return tool.execute(**tool_args)

    def answer_query(
        self,
        query: str,
        max_steps: int = 5
    ) -> AgentResponse:
        steps: list[AgentStep] = []
        retrieved_chunks: list[dict] = []
        current_query = query
        final_answer = ""
        confidence = 0.0

        for step_num in range(1, max_steps + 1):
            decision = self._decide_next_step(current_query, steps)
            if decision is None:
                break

            tool_name, tool_args, reasoning = decision

            # Inject domain/query into arguments if needed
            if tool_name == "search_retriever" and "domain" not in tool_args:
                tool_args["domain"] = self.domain
            if "query" not in tool_args and tool_name in [
                "search_retriever",
                "query_expander",
                "rerank_results"
            ]:
                tool_args["query"] = current_query
            if tool_name == "check_guardrails" and "retrieved_chunks" not in tool_args:
                tool_args["retrieved_chunks"] = retrieved_chunks

            result = self._execute_step(tool_name, tool_args)
            step = AgentStep(
                step_number=step_num,
                tool_name=tool_name,
                tool_arguments=tool_args,
                tool_result=result,
                reasoning=reasoning
            )
            steps.append(step)

            # Capture data
            if tool_name == "search_retriever" and result.success:
                retrieved_chunks = result.data or []
            elif tool_name == "query_expander" and result.success and result.data:
                current_query = result.data.get("expanded_query", current_query)
            elif tool_name == "rerank_results" and result.success:
                retrieved_chunks = result.data or []

        # Generate final answer
        if retrieved_chunks:
            context = "\n".join([c.get("text", "") for c in retrieved_chunks[:3]])
            final_answer_prompt = f"""You are a medical prescription assistant. Answer the query using only this context:
{context}

Query: {query}

Your answer must be grounded in the context, cite your sources, and end with a safety disclaimer.
"""
            final_answer = self.llm.generate(
                final_answer_prompt,
                PromptTemplates.SYSTEM_PROMPT
            )
            confidence = 0.8
        else:
            final_answer = "I couldn't find any relevant documents to answer that query."
            confidence = 0.0

        # Check guardrails one last time
        guardrails_tool = CheckGuardrailsTool()
        guardrails_result = guardrails_tool.execute(
            query=query,
            generated_response=final_answer,
            retrieved_chunks=retrieved_chunks
        )
        if guardrails_result.success and guardrails_result.data:
            confidence = guardrails_result.data.get("confidence_score", confidence)
            if not guardrails_result.data.get("overall_pass", True):
                warnings = guardrails_result.data.get("warnings", [])
                final_answer += "\n\nNote: " + " ".join(warnings)

        return AgentResponse(
            answer=final_answer,
            steps=steps,
            confidence=confidence
        )
