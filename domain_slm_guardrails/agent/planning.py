from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PlanStep:
    """Single step in an agent's plan."""
    step_number: int
    action: str  # Tool name
    description: str
    arguments: dict = field(default_factory=dict)


@dataclass
class AgentPlan:
    """Full plan for the agent."""
    query: str
    steps: List[PlanStep]
    goal: str = "Answer the query safely and accurately"


class PlanGenerator:
    """Generates a step-by-step plan for the agent using LLM."""
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_plan(
        self,
        query: str,
        tool_descriptions: str
    ) -> AgentPlan:
        """Generate a plan for a query. Always uses fallback plan for reliability."""
        return self._get_fallback_plan(query)

    def _clean_json_response(self, text: str) -> str:
        """Clean text to get valid JSON."""
        text = text.strip()
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].strip()
        return text

    def _get_fallback_plan(self, query: str) -> AgentPlan:
        steps = [
            PlanStep(
                step_number=1,
                action="query_expander",
                description="Expand the query to improve retrieval",
                arguments={"query": query}
            ),
            PlanStep(
                step_number=2,
                action="search_retriever",
                description="Search the medical prescription knowledge base",
                arguments={"query": query, "top_k": 5}
            ),
        ]
        return AgentPlan(
            query=query,
            steps=steps
        )


class Reflector:
    """Reflects on agent progress to adjust plan or stop."""
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def reflect(
        self,
        current_step: int,
        plan: AgentPlan,
        completed_steps: list,
        retrieved_chunks: list
    ) -> tuple[str, Optional[str]]:
        """Reflect: "continue", "revise", or "stop". Returns action and optional revision message."""
        # Only consider revising if we've tried at least one search
        has_attempted_search = any(
            s.tool_name == "search_retriever" for s in completed_steps
        )
        if has_attempted_search and not retrieved_chunks:
            return "revise", "No documents found; try rephrasing the query."
        if current_step >= len(plan.steps):
            return "stop", "Plan complete."
        return "continue", None
