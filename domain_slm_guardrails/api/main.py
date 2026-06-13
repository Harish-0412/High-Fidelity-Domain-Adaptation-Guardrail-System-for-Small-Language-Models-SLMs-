from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException

from domain_slm_guardrails.api.rag import answer_query
from domain_slm_guardrails.api.schemas import (
    QueryRequest,
    QueryResponse,
    ThresholdUpdateRequest,
    VoiceAgentConfig,
)
from domain_slm_guardrails.core.domain_registry import list_domains, get_domain_config


app = FastAPI(
    title="Domain SLM Guardrails RAG API",
    version="0.2.0",
    description="Week 2 baseline RAG API with citation-bearing responses.",
)


AGENT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "agent.json"


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "domains": list_domains()}


@app.get("/voice-agent", response_model=VoiceAgentConfig)
def get_voice_agent() -> VoiceAgentConfig:
    if not AGENT_CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="agent.json was not found")

    try:
        with AGENT_CONFIG_PATH.open("r", encoding="utf-8") as agent_file:
            agent_config = json.load(agent_file)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="agent.json is not valid JSON") from exc

    conversation_config = agent_config.get("conversation_config", {})
    agent_details = conversation_config.get("agent", {})
    widget_settings = agent_config.get("platform_settings", {}).get("widget", {})

    return VoiceAgentConfig(
        agent_id=agent_config["agent_id"],
        name=agent_config.get("name", "Voice assistant"),
        first_message=agent_details.get("first_message"),
        supports_text_only=bool(widget_settings.get("supports_text_only", True)),
        widget={
            "variant": widget_settings.get("variant"),
            "placement": widget_settings.get("placement"),
            "avatar": widget_settings.get("avatar"),
            "bg_color": widget_settings.get("bg_color"),
            "text_color": widget_settings.get("text_color"),
            "btn_color": widget_settings.get("btn_color"),
            "btn_text_color": widget_settings.get("btn_text_color"),
            "border_color": widget_settings.get("border_color"),
        },
    )


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


@app.get("/guardrail/thresholds")
def get_thresholds() -> dict[str, float]:
    from domain_slm_guardrails.api.rag import enforcer
    res = {}
    for d in list_domains():
        if d in enforcer.runtime_threshold_overrides:
            res[d] = enforcer.runtime_threshold_overrides[d]
        else:
            try:
                res[d] = get_domain_config(d).critic_hallucination_threshold
            except Exception:
                res[d] = 0.5
    return res


@app.post("/guardrail/thresholds")
def update_threshold(req: ThresholdUpdateRequest) -> dict[str, object]:
    from domain_slm_guardrails.api.rag import enforcer
    try:
        get_domain_config(req.domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    
    enforcer.runtime_threshold_overrides[req.domain] = req.threshold
    return {
        "status": "success",
        "domain": req.domain,
        "threshold": req.threshold,
    }


@app.get("/guardrail/metrics")
def get_metrics() -> dict[str, object]:
    from domain_slm_guardrails.api.rag import enforcer
    return enforcer.get_metrics()


@app.post("/guardrail/metrics/reset")
def reset_metrics() -> dict[str, object]:
    from domain_slm_guardrails.api.rag import enforcer
    enforcer.reset_metrics()
    return {"status": "success", "message": "Metrics reset successfully"}


@app.get("/guardrail/logs")
def get_logs(limit: int = 50) -> list[dict[str, object]]:
    from domain_slm_guardrails.api.rag import enforcer
    log_path = enforcer.audit_log_path
    if not log_path.exists():
        return []
    
    entries = []
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read audit logs: {e}")
    
    return entries[-limit:]


