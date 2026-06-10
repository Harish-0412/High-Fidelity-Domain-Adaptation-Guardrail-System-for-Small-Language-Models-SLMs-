# Generic Guardrailed SLM Platform With Medical Prescription Adapter

## Summary
Build a reusable domain-adaptation and guarded-inference platform for Small Language Models, with **medical prescription** as the first production adapter. The core system will support future domains like legal compliance, insurance policy, finance, and internal enterprise SOPs without rewriting the whole stack.

The v1 goal is a working deployable system that can ingest domain documents, fine-tune or align an SLM, retrieve evidence, generate source-grounded answers, detect hallucination risk, and return structured auditable outputs.

## Core Platform
- Build a generic pipeline with reusable modules:
  - `ingestion`: accepts PDFs, markdown, text, CSV/JSON knowledge files.
  - `chunking`: converts cleaned documents into overlapping evidence chunks.
  - `retrieval`: dense vector search with Qdrant plus BM25 hybrid retrieval.
  - `training`: QLoRA SFT and DPO alignment pipelines.
  - `critic`: hidden-state capture, token labeling, probe training, hallucination scoring.
  - `inference`: RAG prompt assembly, constrained decoding, critic guardrail, fallback handling.
  - `api`: FastAPI service exposing generic domain-aware endpoints.
  - `evaluation`: grounding, citation accuracy, hallucination rate, JSON validity, latency.
- Use a domain registry so each domain can define:
  - corpus location
  - prompt templates
  - output schemas
  - retrieval filters
  - evaluation datasets
  - fallback behavior
  - domain-specific labels and terminology

## Medical Prescription Adapter
- Implement `medical_prescription` as the first domain adapter.
- Ingest medical prescription documents such as CPT/ICD/HCPCS references, payer policies, Medicare manuals, compliance docs, and internal billing SOPs.
- Define medical prescription prompt templates for:
  - billing Q&A
  - code/modifier explanation
  - claim review
  - documentation sufficiency check
  - compliance/risk assessment
- Define structured output schemas such as:
  - `answer_with_citations`
  - `claim_review`
  - `coding_recommendation`
  - `missing_documentation`
  - `risk_assessment`
- Build a medical prescription benchmark set to measure factual grounding, citation quality, hallucination rate, and safe refusal behavior.

## Inputs And Outputs
- Supported input types:
  - raw domain documents for ingestion
  - supervised fine-tuning examples
  - DPO preference pairs
  - user questions
  - structured enterprise case data such as notes, codes, policies, or claim details
- Runtime query example:
```json
{
  "domain": "medical_prescription",
  "task": "claim_review",
  "query": "Can CPT 99214 be billed with modifier 25?",
  "output_format": "claim_review"
}
```
- Runtime output example:
```json
{
  "answer": "Modifier 25 may be appropriate only when a significant, separately identifiable E/M service is documented.",
  "decision": "needs_review",
  "risk_level": "medium",
  "citations": [
    {
      "source_id": "medicare_manual",
      "page": 42,
      "chunk_id": "medicare_manual_p42_c03"
    }
  ],
  "guardrail_status": {
    "rag_grounded": true,
    "json_valid": true,
    "critic_score": 0.18,
    "fallback_used": false
  }
}
```
- If hallucination risk is high, return a safe fallback:
```json
{
  "answer": "I could not verify this confidently from the available source documents.",
  "fallback_used": true,
  "reason": "high_hallucination_risk",
  "citations": []
}
```

## Model And Guardrail Plan
- Use `Llama-3-8B` as the first base model.
- Train with QLoRA using domain data plus 10-20% general language data to reduce catastrophic forgetting.
- Run DPO alignment using preference pairs where chosen answers are cited, factual, and concise, while rejected answers are vague, unsupported, or slightly hallucinated.
- Train the hallucination critic using hidden states from middle-to-late transformer layers.
- Use the critic during generation through a custom guardrail hook.
- Use Outlines for JSON/schema-constrained decoding where structured output is required.
- Trigger fallback when the critic score crosses the configured domain threshold.

## Deployment Plan
- Deploy v1 as a single-GPU API service.
- Use Docker Compose for:
  - FastAPI inference service
  - Qdrant vector database
  - optional worker for ingestion/training jobs
- Store artifacts separately:
  - base model reference
  - LoRA adapter
  - DPO adapter
  - critic checkpoint
  - domain retrieval index
  - domain config
- Keep Triton Inference Server as a later scaling option after the single-GPU API is stable.

## Test Plan
- Unit tests for ingestion, chunking, retrieval, prompt assembly, schema validation, and fallback logic.
- Integration tests for full medical prescription flow: document ingest → retrieval → answer → citations → guardrail status.
- Training smoke tests for QLoRA, DPO, hidden-state capture, and critic training on small datasets.
- Evaluation benchmarks:
  - citation coverage
  - hallucination rate before/after critic
  - JSON validity
  - fallback precision
  - latency overhead per generated token
  - API p95 response time
- Acceptance targets:
  - critic AUC >= 0.85
  - constrained JSON validity >= 99%
  - measurable hallucination reduction versus RAG-only baseline
  - critic latency overhead target < 5 ms/token

## Timeline
- **Week 1:** scaffold generic platform, domain registry, ingestion, chunking, Qdrant, BM25 retrieval.
- **Week 2:** medical prescription adapter, RAG API baseline, citation-bearing responses, initial evaluation set.
- **Week 3:** QLoRA training pipeline, medical prescription SFT dataset, adapter export.
- **Week 4:** DPO preference generation, DPO training, groundedness comparison against SFT baseline.
- **Week 5:** hidden-state collection, token labeling, critic dataset generation.
- **Week 6:** critic model training, AUC evaluation, threshold tuning.
- **Week 7:** live guardrail hook, constrained decoding, fallback path, Docker deployment, final benchmarks.

## Assumptions
- The platform is generic, but v1 proof domain is medical prescription.
- Qdrant is the default vector store.
- Outlines is the default constrained decoding library.
- FastAPI is the serving layer.
- Llama-3-8B is the first supported model.
- Medical billing is used to prove production viability before adding more domain adapters.
