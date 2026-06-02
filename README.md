# Generic Domain SLM Guardrails

Week 1 implements the generic retrieval foundation for a guardrailed small
language model platform, with `medical_billing` as the first domain adapter.

## Week 1 Commands

```bash
python scripts/ingest_domain.py --domain medical_billing
python scripts/build_index.py --domain medical_billing --no-qdrant
python scripts/query_retrieval.py --domain medical_billing --query "When is modifier 25 used?"
```

## Week 2 RAG API

Run the citation-bearing API:

```bash
uvicorn domain_slm_guardrails.api.main:app --reload --port 8000
```

Query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"domain\":\"medical_billing\",\"query\":\"When should modifier 25 be used?\",\"top_k\":5}"
```

Run the initial evaluation set:

```bash
python scripts/run_rag_eval.py
```

To use Qdrant, start it first:

```bash
docker compose -f docker/docker-compose.yml up -d
python scripts/build_index.py --domain medical_billing
```

The local dense index is always written, so retrieval remains testable even
when Qdrant or sentence-transformers are not installed.
