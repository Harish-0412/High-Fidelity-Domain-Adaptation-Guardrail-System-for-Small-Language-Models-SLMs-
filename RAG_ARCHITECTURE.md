# Complete RAG Architecture: From Scratch

## Executive Summary

This document defines a production-grade Retrieval-Augmented Generation (RAG) system that transforms the current hybrid-retrieval platform into a **complete end-to-end RAG engine**. The system will:

1. **Ingest** documents at scale with advanced preprocessing
2. **Index** vectors in production-grade vector databases
3. **Retrieve** semantically relevant context using multi-modal retrieval
4. **Generate** grounded answers via LLM with citations
5. **Evaluate** quality and hallucination risk
6. **Deploy** as a scalable, containerized service

---

## Part 1: Complete RAG Architecture

### 1.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────────┘

Phase 1: INGESTION & PREPROCESSING
├── Document Loader (PDFs, TXT, Markdown, HTML)
├── Advanced Text Cleaner (OCR-aware, table detection)
├── Smart Chunker (semantic + token-based, overlap)
├── Metadata Extractor (source, author, date, doc_type)
└── Chunk Validator & Storage (JSONL/Parquet)

Phase 2: EMBEDDING & INDEXING
├── Multi-Model Embeddings (BGE, E5, MiniLM)
├── Embedding Caching Layer
├── Vector Normalization & Dimensionality Reduction
├── Qdrant Vector DB Integration
├── BM25+ Sparse Index (Elasticsearch/Milvus alternative)
├── Hybrid Index Management
└── Index Versioning & Rollback

Phase 3: RETRIEVAL & RERANKING
├── Query Parser & Expansion
├── Dense Retrieval (Qdrant)
├── Sparse Retrieval (BM25+)
├── Cross-Encoder Reranking
├── MMR-based Diversity
├── Metadata Filtering
└── Top-K Selection & Scoring

Phase 4: GENERATION & GROUNDING
├── LLM Integration (Ollama/OpenAI/Hugging Face)
├── Prompt Template Management
├── Context Assembly
├── Token-level Citation Tracking
├── Response Validation
├── Fallback & Refusal Handling
└── Confidence Scoring

Phase 5: EVALUATION & MONITORING
├── Groundedness Metrics
├── Citation Coverage
├── Hallucination Detection
├── Latency Monitoring
├── Quality Dashboards
└── A/B Testing Framework

Phase 6: DEPLOYMENT & SERVING
├── API Layer (FastAPI)
├── Rate Limiting & Auth
├── Load Balancing
├── Health Checks
├── Logging & Telemetry
├── Docker Containerization
└── Kubernetes Ready
```

---

### 1.2 Data Flow Architecture

```
INGESTION FLOW:
Documents → Loader → Cleaner → Chunker → Metadata → Validator → Storage
                                                           ↓
                                                    data/processed/[domain]/chunks.parquet
                                                    data/processed/[domain]/metadata.jsonl

EMBEDDING FLOW:
Chunks → Embedding Model → Cache Check → Embed → Normalize → Store in Qdrant
                                                                    ↓
                                                    data/indexes/[domain]/qdrant/

RETRIEVAL FLOW:
Query → Parse → Expand → Dense Retrieve (Qdrant) ──┐
                          Sparse Retrieve (BM25) ──┼→ Rerank → Merge → Filter → Top-K
                                                     ↓
                                        Scored Results with Metadata

GENERATION FLOW:
Query + Retrieved Chunks → Prompt Template → LLM → Response → Extract Citations → 
Confidence Score → Guardrail Check → Return with Metadata

EVALUATION FLOW:
Ground Truth Cases → Query Expansion → Retrieve → Generate → Evaluate Metrics →
Report Dashboard
```

---

### 1.3 Detailed Component Specifications

#### **A. Document Ingestion**

```yaml
INGESTION_CONFIG:
  supported_formats:
    - pdf: "PyMuPDF, pdfplumber"
    - txt: "UTF-8 text files"
    - markdown: "Markdown with code blocks"
    - html: "Beautiful Soup parsing"
    - docx: "python-docx"
  
  preprocessing:
    ocr_fallback: true
    table_detection: true
    code_block_preservation: true
    language_detection: true
  
  chunking_strategy:
    method: "semantic + token-based"
    token_size: 512
    overlap: 128
    min_chunk_size: 50
    max_chunk_size: 2048
    semantic_threshold: 0.5
  
  metadata_extraction:
    source_document_id
    page_number
    section_hierarchy
    doc_type: "policy, manual, code_reference, etc"
    creation_date
    author_if_available
```

#### **B. Embedding & Vector Storage**

```yaml
EMBEDDING_CONFIG:
  models:
    primary: "BAAI/bge-large-en-v1.5"  # 1024-dim, excellent for retrieval
    fallback: "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight
    
  embedding_settings:
    batch_size: 32
    normalize_embeddings: true
    cache_strategy: "redis"  # For repeated queries
    dimension_reduction: false  # Keep full 1024-dim for Qdrant
    
  vector_storage:
    backend: "qdrant"
    host: "localhost"
    port: 6333
    index_type: "hnsw"
    hnsw_config:
      m: 16
      ef_construct: 200
      ef_search: 100
    distance_metric: "cosine"
    
  sparse_index:
    backend: "bm25"
    tokenizer: "stemming"
    parameters:
      k1: 1.5
      b: 0.75
      delta: 1.0
```

#### **C. Retrieval Pipeline**

```yaml
RETRIEVAL_CONFIG:
  query_preprocessing:
    lowercase: true
    remove_stopwords: true
    expand_abbreviations: true
    handle_typos: true
    
  dense_retrieval:
    top_k: 10
    score_threshold: 0.3
    similarity_metric: "cosine"
    
  sparse_retrieval:
    top_k: 10
    score_threshold: 0.01
    
  reranking:
    model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
    top_k_input: 20  # Rerank top 20 from both retrievers
    top_k_output: 5  # Return top 5 after reranking
    
  fusion:
    method: "rrf"  # Reciprocal Rank Fusion
    rrf_k: 60
    
  filtering:
    metadata_filters:
      - doc_type: "policy"
      - date_after: "2023-01-01"
    min_relevance_score: 0.4
    
  diversity:
    use_mmr: true
    lambda: 0.6  # 0.6 = balance relevance & diversity
```

#### **D. Generation & Grounding**

```yaml
GENERATION_CONFIG:
  llm:
    provider: "ollama"  # local, openai, huggingface
    model: "neural-chat"  # or llama2, mistral, etc
    temperature: 0.7
    max_tokens: 512
    top_p: 0.95
    
  prompt_templates:
    medical_billing:
      system: |
        You are an expert medical billing specialist. Answer questions using
        ONLY the provided documents. Always cite sources. If unsure, decline.
      user: |
        {query}
        
        Available context (with source IDs):
        {context}
        
        Rules:
        1. Use citations: [Source: chunk_id]
        2. Be precise
        3. Decline if unsure
        4. Explain reasoning
    
  citation_tracking:
    extract_method: "span_extraction"  # Identify which spans correspond to which chunks
    confidence_threshold: 0.7
    citation_format: "[Source: {source_id}, Page: {page}]"
    
  guardrails:
    min_confidence: 0.6
    hallucination_detector: true
    refusal_threshold: 0.3
    max_citations_per_response: 5
    
  fallback_strategy:
    low_confidence: "Return safe fallback response"
    no_relevant_docs: "Polite refusal with explanation"
    guardrail_triggered: "Escalate to human review"
```

#### **E. Quality Evaluation**

```yaml
EVALUATION_METRICS:
  retrieval_metrics:
    - NDCG@5: "Normalized Discounted Cumulative Gain"
    - MRR: "Mean Reciprocal Rank"
    - MAP: "Mean Average Precision"
    - Recall@K: "Proportion of relevant docs retrieved"
    
  generation_metrics:
    - Bleu/Rouge: "Text similarity to reference"
    - Citation_Accuracy: "Citations match retrieved chunks"
    - Factuality: "Automatic fact verification"
    - Grounding_Rate: "% answers supported by context"
    
  hallucination_metrics:
    - Hallucination_Rate: "% answers with unsupported claims"
    - Contradiction_Rate: "% answers contradicting sources"
    - Confidence_Calibration: "Agreement between confidence & accuracy"
    
  user_experience:
    - Latency_p50, p95, p99
    - Token_Throughput
    - Cost_per_Query
```

---

### 1.4 Database Schema

#### **Chunks Table (Vector DB)**
```json
{
  "chunk_id": "medical_billing_001_chunk_05",
  "document_id": "medical_billing_001",
  "document_name": "Medicare_Billing_Manual_2026.pdf",
  "page_number": 42,
  "section": "Modifier 25 Rules",
  "text": "Modifier 25 may only be appended when a significant, separately identifiable...",
  "source_type": "policy_manual",
  "date_processed": "2026-06-12",
  "embedding": [0.123, 0.456, ...],  // 1024-dim vector
  "metadata": {
    "doc_type": "policy",
    "domain": "medical_billing",
    "relevance_hints": ["modifier", "cpt", "billing"],
    "created_date": "2024-01-01"
  }
}
```

#### **Index Metadata**
```json
{
  "index_version": "v1.0.0",
  "domain": "medical_billing",
  "total_chunks": 2543,
  "embedding_model": "BAAI/bge-large-en-v1.5",
  "embedding_dimension": 1024,
  "created_at": "2026-06-12T10:00:00Z",
  "last_updated": "2026-06-12T10:00:00Z",
  "documents_count": 45,
  "average_chunk_tokens": 487,
  "vector_store_backend": "qdrant",
  "bm25_indexed": true
}
```

#### **Query Log & Analytics**
```json
{
  "query_id": "q_20260612_001",
  "domain": "medical_billing",
  "user_id": "user_123",
  "timestamp": "2026-06-12T15:30:00Z",
  "query_text": "Can CPT 99214 be billed with modifier 25?",
  "query_embedding": [...],
  "retrieved_chunks": ["chunk_001", "chunk_042", "chunk_103"],
  "top_k": 5,
  "retrieval_latency_ms": 245,
  "generation_latency_ms": 1230,
  "response": "Modifier 25 may be appropriate only when...",
  "citations": ["chunk_001"],
  "confidence_score": 0.87,
  "user_satisfaction": 0.9,
  "fallback_used": false
}
```

---

## Part 2: Detailed Implementation Phases

### PHASE 1: Foundation & Setup (1 Week)

**Goal**: Establish robust data infrastructure and local development environment.

#### Phase 1.1: Development Environment
- [ ] Install Qdrant locally (Docker: `docker run -p 6333:6333 qdrant/qdrant`)
- [ ] Set up vector database monitoring dashboard
- [ ] Configure environment variables (`.env` template)
- [ ] Set up Ollama with neural-chat model: `ollama pull neural-chat`
- [ ] Create Docker Compose with Qdrant, Ollama, Redis (caching)

#### Phase 1.2: Project Structure Refactoring
```
domain_slm_guardrails/
├── ingestion/
│   ├── advanced_loader.py        [NEW] Multi-format document loading
│   ├── advanced_cleaner.py       [NEW] OCR, table detection, language detection
│   ├── smart_chunker.py          [NEW] Semantic + token-based chunking
│   ├── metadata_extractor.py     [NEW] Extract doc metadata
│   ├── chunk_validator.py        [NEW] QA for chunks
│   └── ingestion_pipeline.py     [REFACTOR] Orchestrate full pipeline
│
├── embedding/                     [NEW]
│   ├── embedding_models.py        Multi-model support
│   ├── embedding_cache.py         Redis-backed cache
│   └── embedding_loader.py        Load & normalize embeddings
│
├── retrieval/
│   ├── dense_retriever.py         [NEW] Qdrant integration
│   ├── sparse_retriever.py        [REFACTOR] Improve BM25
│   ├── reranker.py                [NEW] Cross-encoder reranking
│   ├── retriever_hybrid.py        [REFACTOR] Orchestrate fusion
│   └── retriever_filters.py       [NEW] Metadata filtering
│
├── generation/                    [NEW]
│   ├── llm_interface.py           LLM provider abstraction
│   ├── prompt_manager.py          Template management
│   ├── response_generator.py      Generation pipeline
│   ├── citation_extractor.py      Extract & validate citations
│   └── guardrail_checker.py       Confidence & hallucination checks
│
├── evaluation/
│   ├── retrieval_metrics.py       [NEW] NDCG, MRR, etc.
│   ├── generation_metrics.py      [NEW] Factuality, grounding
│   ├── hallucination_detector.py  [NEW] Automatic hallucination detection
│   └── eval_runner.py             [REFACTOR] Full evaluation pipeline
│
├── serving/                       [NEW]
│   ├── api_v2.py                  Refactored FastAPI
│   ├── auth.py                    API authentication
│   ├── rate_limiter.py            Rate limiting
│   └── monitoring.py              Prometheus metrics
│
└── config/
    ├── ingestion_config.yaml      [NEW]
    ├── embedding_config.yaml      [NEW]
    ├── retrieval_config.yaml      [NEW]
    ├── generation_config.yaml     [NEW]
    └── eval_config.yaml           [NEW]
```

#### Phase 1.3: Configuration Management
- [ ] Create unified config loader with YAML + environment variable override
- [ ] Define config schemas for each component
- [ ] Create domain-specific configs (medical_billing, etc.)
- [ ] Set up config versioning

#### Phase 1.4: Utilities & Helpers
- [ ] Create logger factory with structured logging
- [ ] Build metrics collector (Prometheus-compatible)
- [ ] Set up timing decorators for latency tracking
- [ ] Create error handling standardization

---

### PHASE 2: Advanced Document Ingestion (1 Week)

**Goal**: Build production-grade document processing with quality validation.

#### Phase 2.1: Multi-Format Loader
```python
# pseudo-code
class UniversalDocumentLoader:
    def load(filepath: str) -> List[Page]:
        if filepath.endswith('.pdf'):
            return load_pdf_with_pdfplumber(filepath)
        elif filepath.endswith('.docx'):
            return load_docx(filepath)
        elif filepath.endswith('.html'):
            return load_html(filepath)
        # ... etc
    
    def detect_language(text: str) -> str
    def detect_encoding(bytes: bytes) -> str
    def handle_ocr(image_region: bytes) -> str
```

**Tasks**:
- [ ] Implement PDF loader (PyMuPDF + pdfplumber for tables)
- [ ] Implement DOCX loader
- [ ] Implement HTML loader
- [ ] Add OCR fallback for scanned PDFs (Tesseract)
- [ ] Create table detection & extraction
- [ ] Add language detection (textract)
- [ ] Write unit tests (fixtures with real documents)

#### Phase 2.2: Advanced Text Cleaner
```python
class AdvancedTextCleaner:
    def clean(text: str) -> str:
        # 1. Fix encoding issues
        text = fix_encoding(text)
        
        # 2. Remove boilerplate (headers, footers, page numbers)
        text = remove_boilerplate(text)
        
        # 3. Normalize whitespace
        text = normalize_whitespace(text)
        
        # 4. Preserve structure (section markers, lists)
        text = preserve_structure(text)
        
        # 5. Handle special characters
        text = normalize_special_chars(text)
        
        return text
```

**Tasks**:
- [ ] Implement encoding fix (chardet)
- [ ] Implement boilerplate removal (regex patterns for domain)
- [ ] Implement whitespace normalization
- [ ] Implement structure preservation (headers, bullets)
- [ ] Add domain-specific cleanup rules
- [ ] Write tests with messy real-world documents

#### Phase 2.3: Smart Chunker
```python
class SmartChunker:
    def chunk(text: str, config: ChunkingConfig) -> List[Chunk]:
        # Strategy: Semantic boundaries first, then token limit
        
        # 1. Split by semantic boundaries (paragraphs, sections)
        semantic_chunks = split_by_semantics(text)
        
        # 2. Further split if exceeds token limit
        final_chunks = []
        for semantic in semantic_chunks:
            if token_count(semantic) > config.max_tokens:
                sub_chunks = split_by_tokens(semantic, config)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(semantic)
        
        # 3. Add overlap
        overlapped = add_overlap(final_chunks, config.overlap_tokens)
        
        return overlapped
```

**Tasks**:
- [ ] Implement semantic splitting (sentence/paragraph level)
- [ ] Implement token-based splitting (transformers.utils.length_grouping)
- [ ] Implement overlap addition
- [ ] Create ChunkingConfig dataclass
- [ ] Add chunking quality metrics (coverage, overlap ratio)
- [ ] Write tests with various text types

#### Phase 2.4: Metadata Extraction
```python
class MetadataExtractor:
    def extract(filepath: str, pages: List[Page]) -> Dict:
        return {
            "document_id": generate_id(filepath),
            "source_path": filepath,
            "source_type": detect_document_type(filepath),
            "date_processed": now(),
            "created_date": extract_creation_date(filepath),
            "author": extract_author_if_available(filepath),
            "total_pages": len(pages),
            "language": detect_language(pages),
            "domain_hints": extract_domain_keywords(pages),
            "checksum": compute_checksum(filepath),  # For deduplication
        }
```

**Tasks**:
- [ ] Create MetadataExtractor class
- [ ] Extract document type (policy, manual, etc.)
- [ ] Extract creation/modification dates
- [ ] Extract author information
- [ ] Compute document checksum
- [ ] Extract domain-specific keywords
- [ ] Write tests

#### Phase 2.5: Chunk Storage & Versioning
```python
class ChunkStore:
    def save(chunks: List[Chunk], version: str):
        # Save to Parquet for efficiency
        df = chunks_to_dataframe(chunks)
        df.to_parquet(f"data/processed/{domain}/chunks_v{version}.parquet")
        
        # Also save metadata
        metadata_to_jsonl(chunks_metadata, f"...metadata_v{version}.jsonl")
        
        # Track version
        save_index_manifest({
            "version": version,
            "chunk_count": len(chunks),
            "created_at": now(),
            "source_documents": list_documents(),
        })
```

**Tasks**:
- [ ] Implement Parquet-based storage
- [ ] Create version management
- [ ] Implement deduplication (by checksum)
- [ ] Create rollback mechanism
- [ ] Write chunk validation tests
- [ ] Create data migration utilities

---

### PHASE 3: Embedding & Vector Indexing (1 Week)

**Goal**: Production-grade embeddings with multiple models and Qdrant integration.

#### Phase 3.1: Multi-Model Embedding System
```python
class EmbeddingModel:
    def __init__(self, model_name: str, cache: Optional[RedisCache] = None):
        self.model = load_model(model_name)
        self.cache = cache
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str], batch_size: int = 32) -> List[np.array]:
        # Check cache first
        results = []
        to_embed = []
        indices = []
        
        for i, text in enumerate(texts):
            cached = self.cache.get(hash(text)) if self.cache else None
            if cached:
                results.append(cached)
            else:
                to_embed.append(text)
                indices.append(i)
        
        # Embed uncached
        if to_embed:
            embeddings = self.model.encode(to_embed, batch_size=batch_size, 
                                          normalize_embeddings=True)
            
            # Cache results
            for text, embedding in zip(to_embed, embeddings):
                if self.cache:
                    self.cache.set(hash(text), embedding)
                results.insert(indices[len(results)], embedding)
        
        return results
```

**Tasks**:
- [ ] Implement multi-model support (BGE, E5, MiniLM)
- [ ] Add Redis caching layer
- [ ] Implement batch processing
- [ ] Add embedding normalization
- [ ] Create model lazy-loading
- [ ] Write performance tests
- [ ] Document model comparison (quality vs speed)

#### Phase 3.2: Qdrant Vector Database Integration
```python
class QdrantVectorStore:
    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
    
    def create_collection(self, vector_size: int, distance_metric: str):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        )
    
    def index_chunks(self, chunks_with_embeddings: List[Tuple]):
        points = [
            PointStruct(
                id=int(chunk['chunk_id']),
                vector=embedding,
                payload=chunk['metadata'],
            )
            for chunk, embedding in chunks_with_embeddings
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def search(self, query_embedding: np.array, top_k: int = 5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_vectors=False,
            with_payload=True,
        )
        return results
```

**Tasks**:
- [ ] Set up Qdrant Python client
- [ ] Create collection management
- [ ] Implement batch indexing
- [ ] Implement search interface
- [ ] Add metadata filtering
- [ ] Implement vector normalization
- [ ] Create index monitoring
- [ ] Write integration tests
- [ ] Create Qdrant Docker Compose config

#### Phase 3.3: Sparse Index (BM25+) Upgrade
- [ ] Refactor existing BM25Index for domain-specific tuning
- [ ] Add Elasticsearch integration option
- [ ] Implement index persistence (pickle → SQLite)
- [ ] Add index update mechanisms
- [ ] Create BM25 performance benchmarks

#### Phase 3.4: Embedding Pipeline Orchestration
```python
class EmbeddingPipeline:
    def run(self, domain: str, force_reembed: bool = False):
        # Load chunks
        chunks = load_chunks_jsonl(f"data/processed/{domain}/chunks.jsonl")
        
        # Check if embeddings exist
        if not force_reembed and embeddings_exist(domain):
            return load_embeddings(domain)
        
        # Embed
        embedder = EmbeddingModel("BAAI/bge-large-en-v1.5")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedder.embed(texts)
        
        # Index in Qdrant
        vector_store = QdrantVectorStore("localhost", 6333, f"{domain}_chunks")
        vector_store.index_chunks(list(zip(chunks, embeddings)))
        
        # Index in BM25
        bm25_index = BM25Index(chunks)
        bm25_index.save(f"data/indexes/{domain}/bm25.pkl")
        
        # Save metadata
        save_index_manifest({...})
        
        return {"qdrant": vector_store, "bm25": bm25_index}
```

**Tasks**:
- [ ] Create EmbeddingPipeline orchestrator
- [ ] Implement parallel embedding (with multiprocessing)
- [ ] Add progress tracking (tqdm)
- [ ] Implement resumable embedding
- [ ] Create embedding health checks
- [ ] Write end-to-end tests

---

### PHASE 4: Advanced Retrieval System (1.5 Weeks)

**Goal**: Multi-stage retrieval with reranking and filtering.

#### Phase 4.1: Query Preprocessing & Expansion
```python
class QueryPreprocessor:
    def preprocess(self, query: str) -> ProcessedQuery:
        # 1. Clean
        cleaned = clean_query(query)
        
        # 2. Expand (synonyms, abbreviations)
        expanded = expand_query(cleaned)
        
        # 3. Detect intent
        intent = detect_intent(cleaned)  # "factual", "comparative", etc.
        
        # 4. Extract entities
        entities = extract_entities(cleaned)
        
        # 5. Embed
        query_embedding = embed_query(expanded)
        
        return ProcessedQuery(
            original=query,
            cleaned=cleaned,
            expanded=expanded,
            intent=intent,
            entities=entities,
            embedding=query_embedding,
        )
```

**Tasks**:
- [ ] Create QueryPreprocessor class
- [ ] Implement query cleaning (lowercase, remove stop words)
- [ ] Implement query expansion (synonyms, abbreviations)
- [ ] Add intent detection
- [ ] Add entity extraction (spaCy)
- [ ] Implement query embedding
- [ ] Write tests with various query types

#### Phase 4.2: Dense Retriever (Qdrant)
```python
class DenseRetriever:
    def retrieve(self, query_embedding: np.array, top_k: int = 10) -> List[DenseResult]:
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return [
            DenseResult(
                chunk_id=result.id,
                chunk=result.payload,
                score=result.score,
                rank=i,
            )
            for i, result in enumerate(results)
        ]
```

**Tasks**:
- [ ] Wrap Qdrant search
- [ ] Return DenseResult objects
- [ ] Add score normalization
- [ ] Add result filtering
- [ ] Implement caching
- [ ] Write performance tests

#### Phase 4.3: Sparse Retriever (BM25+)
- [ ] Refactor BM25Index to return standardized results
- [ ] Implement result ranking
- [ ] Add score normalization
- [ ] Create SparseResult dataclass

#### Phase 4.4: Cross-Encoder Reranking
```python
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, candidates: List[Chunk], top_k: int = 5):
        # Score each candidate
        pairs = [[query, chunk['text']] for chunk in candidates]
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in ranked[:top_k]]
```

**Tasks**:
- [ ] Implement CrossEncoderReranker
- [ ] Add batch processing
- [ ] Implement caching
- [ ] Write performance benchmarks
- [ ] Create reranker evaluation

#### Phase 4.5: Metadata Filtering & Hybrid Fusion
```python
class HybridRetriever:
    def retrieve(self, query: str, filters: Optional[Dict] = None, top_k: int = 5):
        # Preprocess query
        processed_query = self.query_preprocessor.preprocess(query)
        
        # Dense retrieval
        dense_results = self.dense_retriever.retrieve(
            processed_query.embedding, 
            top_k=10
        )
        
        # Sparse retrieval
        sparse_results = self.sparse_retriever.retrieve(query, top_k=10)
        
        # Apply metadata filters
        dense_results = apply_filters(dense_results, filters)
        sparse_results = apply_filters(sparse_results, filters)
        
        # Rerank
        candidates = deduplicate(dense_results + sparse_results)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        
        # Apply MMR for diversity
        final = apply_mmr(reranked, lambda_param=0.6)
        
        return final
```

**Tasks**:
- [ ] Implement HybridRetriever orchestrator
- [ ] Add RRF fusion
- [ ] Implement metadata filtering
- [ ] Add MMR diversity
- [ ] Implement result deduplication
- [ ] Create comprehensive tests

#### Phase 4.6: Retrieval Evaluation
```python
class RetrievalEvaluator:
    def evaluate(self, queries_and_ground_truth: List[Dict]) -> Dict:
        results = {
            "ndcg": [],
            "mrr": [],
            "map": [],
            "recall": [],
        }
        
        for query_data in queries_and_ground_truth:
            retrieved = self.retriever.retrieve(query_data['query'])
            ground_truth = query_data['relevant_chunk_ids']
            
            results['ndcg'].append(ndcg_score(retrieved, ground_truth))
            results['mrr'].append(mrr_score(retrieved, ground_truth))
            # ... etc
        
        return {
            "ndcg_mean": mean(results['ndcg']),
            "mrr_mean": mean(results['mrr']),
            # ... summary stats
        }
```

**Tasks**:
- [ ] Implement RetrievalEvaluator
- [ ] Add NDCG metric
- [ ] Add MRR metric
- [ ] Add MAP metric
- [ ] Add Recall@K metric
- [ ] Create evaluation benchmarks

---

### PHASE 5: LLM Integration & Generation (1.5 Weeks)

**Goal**: Complete response generation with citations and confidence scoring.

#### Phase 5.1: LLM Provider Abstraction
```python
class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        pass

class OllamaLLM(LLMInterface):
    def generate(self, prompt: str, **kwargs):
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        })
        return response.json()['response']

class OpenAILLM(LLMInterface):
    def generate(self, prompt: str, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
```

**Tasks**:
- [ ] Create LLMInterface abstract class
- [ ] Implement OllamaLLM provider
- [ ] Implement OpenAI provider
- [ ] Implement HuggingFace provider
- [ ] Add token counting
- [ ] Add error handling & retries
- [ ] Create provider tests

#### Phase 5.2: Prompt Management
```python
class PromptManager:
    def __init__(self, domain: str):
        self.domain = domain
        self.templates = load_prompt_templates(domain)
    
    def build_prompt(self, query: str, context: List[Chunk], output_format: str) -> str:
        template = self.templates[output_format]
        
        return template.format(
            query=query,
            context=format_context(context),
            instructions=build_instructions(self.domain),
        )

# Templates per domain
MEDICAL_BILLING_TEMPLATES = {
    "qa": """You are a medical billing expert. Answer using ONLY the provided documents.
{instructions}

Question: {query}

Context:
{context}

Answer:""",
    
    "claim_review": """Review the claim using the provided policies.
{instructions}

Claim Details: {query}

Relevant Policies:
{context}

Assessment:""",
}
```

**Tasks**:
- [ ] Create PromptManager class
- [ ] Create domain-specific templates
- [ ] Implement context formatting
- [ ] Add few-shot examples
- [ ] Create prompt versioning
- [ ] Write prompt quality tests

#### Phase 5.3: Response Generation Pipeline
```python
class ResponseGenerator:
    def generate(self, query: str, context: List[Chunk], config: GenerationConfig) -> Response:
        # Build prompt
        prompt = self.prompt_manager.build_prompt(query, context, config.output_format)
        
        # Generate
        raw_response = self.llm.generate(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        
        # Parse response
        parsed = parse_response(raw_response, config.output_format)
        
        # Extract citations
        citations = extract_citations(parsed['answer'], context)
        
        # Validate
        if not citations and not config.allow_fallback:
            return fallback_response(query)
        
        return Response(
            answer=parsed['answer'],
            citations=citations,
            confidence=compute_confidence(parsed, citations),
            reasoning=parsed.get('reasoning'),
        )
```

**Tasks**:
- [ ] Implement ResponseGenerator
- [ ] Add response parsing
- [ ] Create output formatters (JSON, structured)
- [ ] Add streaming support
- [ ] Implement caching
- [ ] Write generation tests

#### Phase 5.4: Citation Extraction & Validation
```python
class CitationExtractor:
    def extract(self, response_text: str, context_chunks: List[Chunk]) -> List[Citation]:
        citations = []
        
        # Method 1: Explicit citation markers [Source: chunk_id]
        explicit = extract_explicit_citations(response_text)
        
        # Method 2: Span-based: find sentences and match to chunks
        sentences = split_into_sentences(response_text)
        for sent_idx, sentence in enumerate(sentences):
            best_chunk = find_best_matching_chunk(sentence, context_chunks)
            if similarity_score(sentence, best_chunk) > 0.7:
                citations.append(Citation(
                    sentence=sentence,
                    chunk_id=best_chunk['chunk_id'],
                    confidence=similarity_score(sentence, best_chunk),
                ))
        
        # Method 3: Attention-based (for advanced models)
        # attention_citations = extract_from_attention_weights(response_text, context)
        
        return citations
```

**Tasks**:
- [ ] Implement CitationExtractor
- [ ] Add explicit citation parsing
- [ ] Implement span-matching
- [ ] Add confidence scoring
- [ ] Create citation validation
- [ ] Write extraction tests

#### Phase 5.5: Confidence & Guardrails
```python
class ConfidenceScorer:
    def score(self, response: Response, context: List[Chunk]) -> float:
        # Factors:
        # 1. Citation coverage (how much of answer is cited)
        citation_coverage = compute_citation_coverage(response)
        
        # 2. Context relevance (retrieval scores)
        context_relevance = mean([c['score'] for c in context])
        
        # 3. Response coherence (language model perplexity)
        coherence = compute_coherence(response.answer)
        
        # 4. Factuality score (automatic verification)
        factuality = verify_factuality(response.answer, context)
        
        # Combined score
        score = (0.4 * citation_coverage + 
                0.3 * context_relevance + 
                0.2 * coherence + 
                0.1 * factuality)
        
        return min(1.0, max(0.0, score))

class GuardrailChecker:
    def check(self, response: Response, config: GuardrailConfig) -> GuardrailStatus:
        status = GuardrailStatus()
        
        if response.confidence < config.min_confidence:
            status.low_confidence = True
            status.recommendation = "FALLBACK"
        
        if len(response.citations) == 0:
            status.no_citations = True
            status.recommendation = "REFUSAL"
        
        if has_contradictions(response.answer, response.citations):
            status.contradiction_detected = True
            status.recommendation = "ESCALATE"
        
        return status
```

**Tasks**:
- [ ] Implement ConfidenceScorer
- [ ] Create guardrail checker
- [ ] Implement fallback responses
- [ ] Add hallucination detection
- [ ] Create confidence calibration
- [ ] Write guardrail tests

---

### PHASE 6: API & Serving (1 Week)

**Goal**: Production-ready API with monitoring and deployment.

#### Phase 6.1: FastAPI Refactoring
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Optional

app = FastAPI(
    title="RAG API v2",
    version="2.0.0",
    description="Production RAG with retrieval, generation, citations",
)

class QueryRequest(BaseModel):
    query: str
    domain: str
    top_k: int = 5
    temperature: float = 0.7
    stream: bool = False

class Citation(BaseModel):
    text: str
    source_id: str
    page: Optional[int] = None
    confidence: float

class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: List[Citation]
    confidence: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    guardrail_status: GuardrailStatus

@app.post("/v2/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    query_id = generate_query_id()
    start_time = time.time()
    
    try:
        # Retrieve
        retrieval_start = time.time()
        context = retriever.retrieve(request.query, top_k=request.top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Generate
        generation_start = time.time()
        response = generator.generate(request.query, context, request.temperature)
        generation_time = (time.time() - generation_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        # Log
        log_query_analytics(query_id, request, response, {
            'retrieval_ms': retrieval_time,
            'generation_ms': generation_time,
            'total_ms': total_time,
        })
        
        return QueryResponse(
            query_id=query_id,
            answer=response.answer,
            citations=response.citations,
            confidence=response.confidence,
            retrieval_latency_ms=retrieval_time,
            generation_latency_ms=generation_time,
            total_latency_ms=total_time,
            guardrail_status=response.guardrail_status,
        )
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/v2/health")
async def health_endpoint() -> dict:
    return {
        "status": "ok",
        "services": {
            "qdrant": check_qdrant(),
            "llm": check_llm(),
            "embedding": check_embedding(),
        }
    }

@app.post("/v2/stream")
async def stream_endpoint(request: QueryRequest):
    """Streaming response generation"""
    def generate():
        context = retriever.retrieve(request.query, top_k=request.top_k)
        
        # Stream from LLM
        for chunk in generator.generate_streaming(request.query, context):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Tasks**:
- [ ] Refactor main.py to v2 endpoints
- [ ] Implement streaming responses
- [ ] Add query ID tracking
- [ ] Implement latency tracking
- [ ] Add error handling
- [ ] Create comprehensive API tests

#### Phase 6.2: Authentication & Rate Limiting
```python
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from slowapi import Limiter

security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

@app.post("/v2/query", dependencies=[Depends(security)])
@limiter.limit("100/minute")
async def query_endpoint(request: QueryRequest, credentials: HTTPAuthCredentials):
    # Validate API key
    if not validate_api_key(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # ... rest of endpoint
```

**Tasks**:
- [ ] Implement API key authentication
- [ ] Add rate limiting
- [ ] Create user/domain quota system
- [ ] Write auth tests

#### Phase 6.3: Monitoring & Observability
```python
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Metrics
query_counter = Counter('rag_queries_total', 'Total queries')
latency_histogram = Histogram('rag_latency_ms', 'Query latency')
confidence_gauge = Gauge('rag_confidence', 'Average confidence')
error_counter = Counter('rag_errors_total', 'Total errors')

# Structured logging
logger = structlog.get_logger()

def log_query_analytics(query_id: str, request, response, metrics):
    logger.info(
        "query_completed",
        query_id=query_id,
        domain=request.domain,
        confidence=response.confidence,
        citations_count=len(response.citations),
        latency_ms=metrics['total_ms'],
        fallback_used=response.guardrail_status.fallback,
    )
    
    query_counter.inc()
    latency_histogram.observe(metrics['total_ms'])
    confidence_gauge.set(response.confidence)
```

**Tasks**:
- [ ] Add Prometheus metrics
- [ ] Implement structured logging
- [ ] Create monitoring dashboard (Grafana)
- [ ] Add performance alerting
- [ ] Write monitoring tests

#### Phase 6.4: Docker & Deployment
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e ".[api,retrieval]"

# Copy code
COPY domain_slm_guardrails/ domain_slm_guardrails/
COPY configs/ configs/
COPY data/ data/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health || exit 1

# Run
CMD ["uvicorn", "domain_slm_guardrails.api.api_v2:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_storage:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  rag-api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
      - ollama
      - redis
    environment:
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_URL=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DOMAIN=medical_billing
    volumes:
      - ./data:/app/data

volumes:
  qdrant_storage:
  ollama_storage:
```

**Tasks**:
- [ ] Create Dockerfile
- [ ] Update docker-compose.yml
- [ ] Add environment configuration
- [ ] Test Docker build and run
- [ ] Create health check endpoints

---

### PHASE 7: Evaluation & Quality Assurance (1 Week)

**Goal**: Comprehensive evaluation framework for quality tracking.

#### Phase 7.1: Evaluation Metrics
```python
class EvaluationMetrics:
    """Comprehensive evaluation suite"""
    
    def evaluate_retrieval(self, queries_and_gt):
        """NDCG, MRR, MAP, Recall"""
        return {
            'ndcg@5': compute_ndcg(queries_and_gt),
            'mrr': compute_mrr(queries_and_gt),
            'map': compute_map(queries_and_gt),
            'recall@10': compute_recall(queries_and_gt),
        }
    
    def evaluate_generation(self, queries_and_references):
        """BLEU, ROUGE, factuality"""
        return {
            'bleu': compute_bleu(queries_and_references),
            'rouge1': compute_rouge1(queries_and_references),
            'factuality': compute_factuality(queries_and_references),
        }
    
    def evaluate_hallucination(self, responses_and_context):
        """Hallucination rate, contradiction rate"""
        return {
            'hallucination_rate': detect_hallucinations(responses_and_context),
            'contradiction_rate': detect_contradictions(responses_and_context),
        }
    
    def evaluate_citations(self, responses_and_context):
        """Citation accuracy, coverage"""
        return {
            'citation_accuracy': evaluate_citation_accuracy(responses_and_context),
            'citation_coverage': evaluate_citation_coverage(responses_and_context),
        }
```

**Tasks**:
- [ ] Implement RetrievalMetrics
- [ ] Implement GenerationMetrics
- [ ] Implement HallucinationMetrics
- [ ] Create CitationMetrics
- [ ] Write metric calculation tests
- [ ] Create metric aggregation

#### Phase 7.2: Automated Hallucination Detection
```python
class HallucinationDetector:
    def detect(self, response: Response, context: List[Chunk]) -> float:
        """Return hallucination probability 0-1"""
        
        # Factor 1: Citation coverage
        coverage = len(response.citations) / count_claims(response.answer)
        
        # Factor 2: Contradiction detection
        contradiction_score = detect_contradictions(response.answer, context)
        
        # Factor 3: Factuality verification (external API)
        factuality_score = verify_factuality(response.answer)
        
        # Factor 4: Confidence calibration
        expected_confidence = estimate_expected_confidence(context)
        calibration_score = abs(response.confidence - expected_confidence)
        
        hallucination_prob = (
            (1 - coverage) * 0.3 +
            contradiction_score * 0.3 +
            (1 - factuality_score) * 0.2 +
            calibration_score * 0.2
        )
        
        return min(1.0, max(0.0, hallucination_prob))
```

**Tasks**:
- [ ] Implement HallucinationDetector
- [ ] Add contradiction detection
- [ ] Add factuality checking
- [ ] Create calibration scoring
- [ ] Write hallucination tests

#### Phase 7.3: Evaluation Dataset Management
```python
class EvaluationDatasetManager:
    def load_benchmark(self, domain: str, benchmark_name: str) -> EvaluationDataset:
        """Load ground-truth evaluation datasets"""
        path = f"data/evaluation/{domain}/{benchmark_name}.jsonl"
        
        cases = []
        with open(path) as f:
            for line in f:
                case = EvaluationCase(**json.loads(line))
                cases.append(case)
        
        return EvaluationDataset(cases)
    
    def run_benchmark(self, domain: str, benchmark_name: str) -> BenchmarkResults:
        """Run full evaluation"""
        dataset = self.load_benchmark(domain, benchmark_name)
        
        results = []
        for case in dataset.cases:
            response = self.rag_system.query(case.query)
            
            result = EvaluationResult(
                query_id=case.id,
                query=case.query,
                response=response,
                ground_truth=case.ground_truth,
                metrics=self.compute_metrics(response, case.ground_truth),
            )
            results.append(result)
        
        return BenchmarkResults(results)
```

**Tasks**:
- [ ] Create EvaluationDatasetManager
- [ ] Build medical_billing benchmark dataset
- [ ] Implement benchmark runner
- [ ] Create result reporting
- [ ] Write dataset management tests

#### Phase 7.4: Continuous Evaluation & Regression Testing
```python
class ContinuousEvaluator:
    def run_regression_suite(self, domain: str) -> RegressionReport:
        """Run all tests to catch regressions"""
        
        benchmarks = ["retrieval", "generation", "citations", "hallucination"]
        
        results = {}
        for benchmark in benchmarks:
            prev_results = load_baseline(domain, benchmark)
            current_results = self.run_benchmark(domain, benchmark)
            
            diff = compare_results(prev_results, current_results)
            results[benchmark] = {
                'current': current_results,
                'previous': prev_results,
                'diff': diff,
                'passed': check_threshold(diff),
            }
        
        return RegressionReport(results)
    
    def generate_report(self, report: RegressionReport) -> str:
        """Generate markdown report"""
        md = "# Regression Test Report\n\n"
        
        for benchmark, result in report.results.items():
            md += f"## {benchmark}\n"
            md += f"Status: {'PASSED' if result['passed'] else 'FAILED'}\n"
            md += f"Previous: {result['previous'].average_score}\n"
            md += f"Current: {result['current'].average_score}\n"
            md += f"Delta: {result['diff'].delta_score}\n\n"
        
        return md
```

**Tasks**:
- [ ] Implement ContinuousEvaluator
- [ ] Create regression test suite
- [ ] Build baseline tracking
- [ ] Add report generation
- [ ] Write CI/CD integration

---

### PHASE 8: Documentation & Final Integration (1 Week)

**Goal**: Complete documentation and project cleanup.

#### Phase 8.1: API Documentation
- [ ] Auto-generate OpenAPI/Swagger docs
- [ ] Create API usage guide
- [ ] Document all endpoints
- [ ] Create SDK examples

#### Phase 8.2: Architecture Documentation
- [ ] Document system design decisions
- [ ] Create deployment guide
- [ ] Document configuration options
- [ ] Create troubleshooting guide

#### Phase 8.3: Integration & Testing
- [ ] End-to-end integration tests
- [ ] Performance benchmarks
- [ ] Load testing
- [ ] Failure mode testing

#### Phase 8.4: Cleanup & Optimization
- [ ] Remove old unused code
- [ ] Optimize hot paths
- [ ] Reduce dependencies
- [ ] Security audit

---

## Part 3: Implementation Checklist

### Pre-Phase Checklist
- [ ] Set up repository structure
- [ ] Create .env templates
- [ ] Set up git workflow
- [ ] Create development guidelines

### Phase-by-Phase Milestones

**Phase 1**: Setup & Infra
- [ ] Development environment ready
- [ ] Docker Compose with Qdrant, Ollama, Redis
- [ ] Config system in place
- [ ] Logging framework

**Phase 2**: Ingestion
- [ ] Multi-format document loading
- [ ] Advanced text cleaning
- [ ] Smart chunking
- [ ] Metadata extraction
- [ ] All ingestion tests passing

**Phase 3**: Embeddings
- [ ] Multi-model embedding support
- [ ] Qdrant integration working
- [ ] Embedding cache functional
- [ ] Full pipeline tested

**Phase 4**: Retrieval
- [ ] Query preprocessing
- [ ] Dense + sparse retrieval
- [ ] Cross-encoder reranking
- [ ] Metadata filtering
- [ ] End-to-end retrieval tests

**Phase 5**: Generation
- [ ] LLM provider abstraction
- [ ] Prompt templates defined
- [ ] Response generation pipeline
- [ ] Citation extraction
- [ ] Guardrail system
- [ ] Generation tests passing

**Phase 6**: API
- [ ] FastAPI v2 endpoints
- [ ] Auth & rate limiting
- [ ] Monitoring & metrics
- [ ] Docker deployment ready
- [ ] API tests passing

**Phase 7**: Evaluation
- [ ] Evaluation metrics implemented
- [ ] Hallucination detection working
- [ ] Benchmark dataset loaded
- [ ] Regression tests passing
- [ ] Reports generating

**Phase 8**: Documentation
- [ ] API docs complete
- [ ] Deployment docs complete
- [ ] Architecture docs complete
- [ ] README updated
- [ ] All code commented

---

## Time Estimation

- **Phase 1**: 3-4 days (Setup)
- **Phase 2**: 4-5 days (Ingestion)
- **Phase 3**: 4-5 days (Embeddings)
- **Phase 4**: 5-6 days (Retrieval)
- **Phase 5**: 5-6 days (Generation)
- **Phase 6**: 4-5 days (API)
- **Phase 7**: 4-5 days (Evaluation)
- **Phase 8**: 3-4 days (Docs)

**Total**: 7-8 weeks for full production-ready system

---

## Risk Mitigations

1. **Embedding Latency**: Use smaller models initially, scale up later
2. **LLM Availability**: Implement fallback to simpler responses
3. **Vector DB Performance**: Start with smaller datasets, scale infrastructure
4. **Citation Accuracy**: Implement human verification loop
5. **Hallucination**: Conservative confidence thresholds, extensive testing

---

## Success Criteria

✓ All phases completed on schedule
✓ 95%+ test coverage
✓ NDCG@5 >= 0.75 on retrieval
✓ Citation accuracy >= 90%
✓ Hallucination rate <= 5%
✓ P95 latency <= 2 seconds
✓ 99.9% uptime SLA
✓ Production deployment successful
