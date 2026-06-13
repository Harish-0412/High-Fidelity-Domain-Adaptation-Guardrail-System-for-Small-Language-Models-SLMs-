from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from domain_slm_guardrails.agent.tools import (
    ToolRegistry,
    ToolResult,
    SearchRetrieverTool,
    QueryExpanderTool,
    RerankResultsTool,
    CheckGuardrailsTool
)
from domain_slm_guardrails.agent.planning import (
    AgentPlan,
    PlanGenerator,
    Reflector
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
class AgentState:
    """State tracked during the agent's execution."""
    query: str
    current_plan: Optional[AgentPlan] = None
    plan_step_index: int = 0
    retrieved_chunks: list[dict] = field(default_factory=list)
    revision_count: int = 0


@dataclass
class AgentResponse:
    answer: str
    steps: list[AgentStep]
    confidence: float
    plan: Optional[AgentPlan] = None


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

        # Initialize planning and reflection modules
        self.plan_generator = PlanGenerator(self.llm)
        self.reflector = Reflector(self.llm)

    def _decide_next_step(
        self,
        state: AgentState,
        conversation_history: list[AgentStep]
    ) -> Optional[tuple[str, dict, str]]:
        """
        Use the plan to decide the next step.
        Returns a tuple of (tool_name, tool_arguments, reasoning), or None if done.
        """
        if not state.current_plan:
            tools_desc = self.registry.get_tool_descriptions()
            state.current_plan = self.plan_generator.generate_plan(
                state.query,
                tools_desc
            )
            state.plan_step_index = 0

        if state.plan_step_index >= len(state.current_plan.steps):
            return None

        current_plan_step = state.current_plan.steps[state.plan_step_index]

        # Inject any missing arguments
        arguments = current_plan_step.arguments.copy()
        if not arguments.get("query"):
            arguments["query"] = state.query

        reasoning = f"Plan step {current_plan_step.step_number}: {current_plan_step.description}"

        return (
            current_plan_step.action,
            arguments,
            reasoning
        )

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
        max_steps: int = 10
    ) -> AgentResponse:
        # Initialize agent state
        state = AgentState(query=query)
        steps: list[AgentStep] = []
        final_answer = ""
        confidence = 0.0
        current_query = query

        for step_num in range(1, max_steps + 1):
            decision = self._decide_next_step(state, steps)
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
                tool_args["retrieved_chunks"] = state.retrieved_chunks

            result = self._execute_step(tool_name, tool_args)
            step = AgentStep(
                step_number=step_num,
                tool_name=tool_name,
                tool_arguments=tool_args,
                tool_result=result,
                reasoning=reasoning
            )
            steps.append(step)
            state.plan_step_index += 1

            # Capture data
            if tool_name == "search_retriever" and result.success:
                state.retrieved_chunks = result.data or []
            elif tool_name == "query_expander" and result.success and result.data:
                current_query = result.data.get("expanded_query", current_query)
                if "query" in tool_args:
                    tool_args["query"] = current_query
            elif tool_name == "rerank_results" and result.success:
                state.retrieved_chunks = result.data or []

            # Reflection after each step
            if state.current_plan:
                reflect_action, reflect_msg = self.reflector.reflect(
                    state.plan_step_index,
                    state.current_plan,
                    steps,
                    state.retrieved_chunks
                )
                if reflect_action == "revise" and state.revision_count < 2:
                    state.revision_count += 1
                    self.logger.info(f"Revising plan: {reflect_msg}")
                    state.current_plan = None
                    state.plan_step_index = 0

        # Generate final answer
        if state.retrieved_chunks:
            context = "\n".join([c.get("text", "") for c in state.retrieved_chunks[:3]])
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
            retrieved_chunks=state.retrieved_chunks
        )
        if guardrails_result.success and guardrails_result.data:
            confidence = guardrails_result.data.get("confidence_score", confidence)
            if not guardrails_result.data.get("overall_pass", True):
                warnings = guardrails_result.data.get("warnings", [])
                final_answer += "\n\nNote: " + " ".join(warnings)

        return AgentResponse(
            answer=final_answer,
            steps=steps,
            confidence=confidence,
            plan=state.current_plan
        )
