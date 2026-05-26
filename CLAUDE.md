# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LightRAG is a Retrieval-Augmented Generation (RAG) framework that uses graph-based knowledge representation for enhanced information retrieval. The system extracts entities and relationships from documents, builds a knowledge graph, and uses multi-modal retrieval (local, global, hybrid, mix, naive) for queries.

## Core Architecture

### Key Components

- **lightrag.py**: Main orchestrator class (`LightRAG`) that coordinates document insertion, query processing, and storage management. Critical: Always call `await rag.initialize_storages()` after instantiation.

- **operate.py**: Core extraction and query operations including entity/relation extraction, chunking, and multi-mode retrieval logic. Contains `kg_query` and `naive_query` entry points.

- **base.py**: Abstract base classes for storage backends (`BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, `BaseDocStatusStorage`) and the `QueryParam` dataclass.

- **kg/**: Storage implementations (JSON, NetworkX, Neo4j, PostgreSQL, MongoDB, Redis, Milvus, Qdrant, Faiss, Memgraph, OpenSearch). Each storage type provides different trade-offs for production vs. development use.

- **llm/**: LLM provider bindings (OpenAI, Ollama, Azure, Gemini, Bedrock, Anthropic, etc.). All use async patterns with caching support.

- **api/**: FastAPI server (`lightrag_server.py`) with REST endpoints and Ollama-compatible API, plus React 19 + TypeScript WebUI.

- **rerank.py**: Reranking abstraction with support for Jina, Cohere, and Aliyun rerank APIs. Includes document chunking for context window limits and score aggregation strategies.

- **utils.py**: Core utilities — caching (`compute_args_hash`, `handle_cache`, `save_to_cache`), embedding function wrapping (`wrap_embedding_func_with_attrs`), output formatting (`convert_to_user_format`), async helpers, and logger setup.

- **constants.py**: All default values (`DEFAULT_TOP_K`, `DEFAULT_CHUNK_TOP_K`, `DEFAULT_MAX_ASYNC`, etc.), entity types, and storage configuration constants.

### Storage Layer

LightRAG uses 4 storage types with pluggable backends:
- **KV_STORAGE**: LLM response cache, text chunks, document info
- **VECTOR_STORAGE**: Entity/relation/chunk embeddings
- **GRAPH_STORAGE**: Entity-relation graph structure
- **DOC_STATUS_STORAGE**: Document processing status tracking

Workspace isolation is implemented differently per storage type (subdirectories for file-based, prefixes for collections, fields for relational DBs).

### Query Modes

- **local**: Context-dependent retrieval focused on specific entities
- **global**: Community/summary-based broad knowledge retrieval
- **hybrid**: Combines local and global
- **naive**: Direct vector search without graph
- **mix**: Integrates KG and vector retrieval (recommended with reranker)

### Query Retrieval Flow

When a query is executed (e.g. via `aquery`), the internal pipeline proceeds as:

1. **Keyword extraction** — LLM generates high-level (for relationships) and low-level (for entities) keywords from the query.
2. **KG search** (`_perform_kg_search`) — entities are retrieved via `entities_vdb.query(top_k=param.top_k)`, relationships via `relationships_vdb.query(top_k=param.top_k)`, and optionally vector chunks via `chunks_vdb.query(top_k=search_top_k)`.
3. **Token truncation** (`_apply_token_truncation`) — entities and relations are trimmed to `max_entity_tokens` / `max_relation_tokens` budgets.
4. **Chunk merging** (`_merge_all_chunks`) — chunks from entities, relations, and vector search are deduplicated and round-robin merged.
5. **Unified chunk processing** (`process_chunks_unified` in utils.py) — reranking, score filtering, `chunk_top_k` limiting, and token-based truncation.
6. **LLM generation** — final context string is built and sent to the LLM.

### Reranking

The rerank module (`rerank.py`) provides:
- `generic_rerank_api(query, documents, model, base_url, api_key, ...)` — HTTP client for Jina/Cohere/Aliyun rerank APIs with exponential backoff and retries.
- `chunk_documents_for_rerank(documents, max_tokens=480, overlap_tokens=32)` — splits long documents to fit reranker context windows.
- `aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs, aggregation="max")` — strategies: `max`, `mean`, `first`.

Configured via `enable_rerank` in `QueryParam` and `rerank_model_func` in `LightRAG` constructor. Default min rerank score threshold is `0.5` (`DEFAULT_MIN_RERANK_SCORE`).

### API Architecture

The API server uses a **router factory pattern** — each router module exports a factory function that receives the `rag` instance:

| Router | Factory | Key Endpoints |
|--------|---------|---------------|
| `routers/query_routes.py` | `create_query_routes(rag, api_key, top_k)` | `POST /query`, `POST /query/data`, `POST /query/stream` |
| `routers/document_routes.py` | `create_document_routes(rag, doc_manager, api_key)` | `POST /documents/upload`, `POST /documents/scan`, list/delete |
| `routers/graph_routes.py` | `create_graph_routes(rag, api_key)` | `GET /graph/label/list`, `GET /graph/label/popular` |
| `routers/ollama_api.py` | `OllamaAPI` class | Ollama-compatible `/api/chat`, `/api/tags` |

Authentication is handled by `api/auth.py` — supports bcrypt password verification, JWT tokens with auto-renewal, and API key auth. Configure via `AUTH_ACCOUNTS` and `TOKEN_SECRET` in `.env`.

## Development Commands

### Setup
```bash
# Install core package (development mode)
uv sync
source .venv/bin/activate  # Or: .venv\Scripts\activate on Windows

# Install with API support
uv sync --extra api

# Install specific extras
uv sync --extra offline-storage  # Storage backends
uv sync --extra offline-llm      # LLM providers
uv sync --extra test             # Testing dependencies
```

### API Server
```bash
# Copy and configure environment
cp env.example .env  # Edit with your LLM/embedding configs

# Build WebUI
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

# Run server
lightrag-server                                           # Production
uvicorn lightrag.api.lightrag_server:app --reload        # Development
lightrag-gunicorn                                         # Multi-worker (gunicorn)
```

### Docker
```bash
# Basic deployment
docker compose up -d

# Full production stack (requires NVIDIA GPU) — includes vLLM embedding, vLLM rerank, PostgreSQL, Neo4j, Milvus
docker compose -f docker-compose-full.yml up -d

# Podman compatibility
docker compose -f docker-compose.podman.yml up -d

# Multi-platform build and push
bash docker-build-push.sh
```

Dockerfiles use multi-stage builds: `oven/bun` for frontend, `uv` for Python deps, final image `python:3.12-slim`. Port 9621.

### Testing
```bash
# Run offline tests (default)
python -m pytest tests

# Run integration tests (requires external services)
python -m pytest tests --run-integration
# Or set: LIGHTRAG_RUN_INTEGRATION=true

# Run specific test file
python test_graph_storage.py

# Keep artifacts for debugging
python -m pytest tests --keep-artifacts

# Run with custom workers
python -m pytest tests --test-workers 4
```

### Linting
```bash
ruff check .
```

## Key Implementation Patterns

### LightRAG Initialization (Critical)

The most common error is forgetting to initialize storages:

```python
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )

    # REQUIRED: Initialize storage backends
    await rag.initialize_storages()

    # Now safe to use
    await rag.ainsert("Your text here")
    result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))

    # Cleanup
    await rag.finalize_storages()

asyncio.run(main())
```

### Custom Embedding Functions

Use `@wrap_embedding_func_with_attrs` decorator and call `.func` when wrapping:

```python
from lightrag.utils import wrap_embedding_func_with_attrs

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def custom_embed(texts: list[str]) -> np.ndarray:
    # Call underlying function, not wrapped version
    return await openai_embed.func(texts, model="text-embedding-3-large")
```

### Storage Configuration

Configure via environment variables or constructor params:

```python
# Environment-based (recommended for production)
# See env.example for full list

# Constructor-based
rag = LightRAG(
    working_dir="./storage",
    workspace="project_name",  # For data isolation
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="Neo4JStorage",
    doc_status_storage="PGDocStatusStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2
    }
)
```

### Document Insertion

```python
# Single document
await rag.ainsert("Text content")

# Batch insertion
await rag.ainsert(["Text 1", "Text 2", ...])

# With custom IDs
await rag.ainsert("Text", ids=["doc-123"])

# With file paths (for citation)
await rag.ainsert(["Text 1", "Text 2"], file_paths=["doc1.pdf", "doc2.pdf"])

# Configure batch size
rag = LightRAG(..., max_parallel_insert=4)  # Default: 2, max recommended: 10
```

### Query Configuration

```python
from lightrag import QueryParam

result = await rag.aquery(
    "Your question",
    param=QueryParam(
        mode="mix",                    # Recommended with reranker
        top_k=60,                      # KG entities/relations to retrieve
        chunk_top_k=20,                # Text chunks to retrieve
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        max_total_tokens=30000,
        enable_rerank=True,
        user_prompt="Additional instructions for LLM",
        stream=False
    )
)
```

### Cache System

The caching layer uses `compute_args_hash(*args)` to generate MD5 hashes for cache keys and `handle_cache(hashing_kv, args_hash, prompt, mode, cache_type)` for lookups. Cache is stored in the KV storage backend. Configure cache behavior via globals:
- `enable_llm_cache` — controls query response caching
- `enable_llm_cache_for_entity_extract` — controls entity extraction caching

### Extending the API

To add a new API endpoint, create a router module in `api/routers/` following the factory pattern:

```python
def create_my_routes(rag, api_key=None):
    router = APIRouter(tags=["my-feature"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post("/my-endpoint", dependencies=[Depends(combined_auth)])
    async def handler(request: MyRequest):
        ...
    return router
```

Then register it in `api/lightrag_server.py` via `app.include_router(create_my_routes(rag, api_key))`.

## WebUI Development

### Structure
- `lightrag_webui/src/`: React components (TypeScript)
- Uses Vite + Bun build system
- Tailwind CSS for styling
- React 19 with functional components and hooks

### Commands
```bash
cd lightrag_webui
bun install --frozen-lockfile  # Install dependencies
bun run dev                    # Development server (Node + Vite)
bun run dev:bun                # Development server (Bun native)
bun run build                  # Production build
bun run preview                # Preview production build locally

# Linting (ESLint with TypeScript, React hooks, Stylistic rules)
bun run lint                   # Run ESLint on all *.ts/tsx/js/jsx files

# Testing (Bun built-in test runner)
bun test                       # Run all tests
bun test --watch               # Watch mode
bun test --coverage            # With coverage report
bun test src/api/lightrag.test.ts  # Run a single test file
```

### Lint Rules
ESLint is configured with TypeScript-ESLint, React Hooks plugin, Prettier integration, and `@stylistic` rules:
- 2-space indentation, single quotes enforced
- `@typescript-eslint/no-explicit-any` is disabled (allowed)

## Evaluation

The `lightrag/evaluation/` directory provides a RAGAS-based evaluation framework:

```bash
pip install -e ".[evaluation]"
python lightrag/evaluation/eval_rag_quality.py
```

Key components:
- **`eval_rag_quality.py`** — Main evaluator: queries a running LightRAG API and scores with RAGAS metrics (context precision, faithfulness, answer relevancy).
- **`offline_retrieval_check.py`** — Lightweight lexical retrieval check without model calls. Verifies sample questions can retrieve expected documents.
- **`sample_dataset.json`** / **`sample_documents/`** — Test data for proof-of-concept.

Key environment variables: `EVAL_LLM_MODEL` (gpt-4o-mini), `EVAL_EMBEDDING_MODEL` (text-embedding-3-large), `EVAL_QUERY_TOP_K` (10), `EVAL_MAX_CONCURRENT` (2). See `README_EVALUASTION_RAGAS.md` for full documentation and troubleshooting.

## Tools

The `lightrag/tools/` directory contains standalone utilities, each with its own README:

| Tool | Command | Purpose |
|------|---------|---------|
| `check_initialization.py` | `python -m lightrag.tools.check_initialization --demo` | Diagnose LightRAG setup after `initialize_storages()` |
| `clean_llm_query_cache.py` | `python -m lightrag.tools.clean_llm_query_cache` | Interactive cache cleaner by mode/type across 5 KV backends |
| `download_cache.py` | (import) | Pre-download tiktoken models for offline deployment |
| `hash_password.py` | `python -m lightrag.tools.hash_password --username admin` | Generate bcrypt hashes for `AUTH_ACCOUNTS` |
| `migrate_llm_cache.py` | `python -m lightrag.tools.migrate_llm_cache` | Migrate extraction/summary cache between KV backends |
| `prepare_qdrant_legacy_data.py` | (import) | Copy Qdrant collections for backward-compat testing |
| `lightrag_visualizer/` | `lightrag-viewer` (install with `[tools]` extra) | 3D graph viewer (ModernGL + imgui_bundle) |

## Common Issues

### 1. Storage Not Initialized
**Error**: `AttributeError: __aenter__` or `KeyError: 'history_messages'`
**Solution**: Always call `await rag.initialize_storages()` after creating LightRAG instance

### 2. Embedding Model Changes
When switching embedding models, you MUST clear the data directory (except optionally `kv_store_llm_response_cache.json` for LLM cache).

### 3. Nested Embedding Functions
Cannot wrap already-decorated embedding functions. Use `.func` to access underlying function:
```python
# Wrong: EmbeddingFunc(func=openai_embed)
# Right: EmbeddingFunc(func=openai_embed.func)
```

### 4. Context Length for Ollama
Ollama models default to 8k context; LightRAG requires 32k+. Configure via:
```python
llm_model_kwargs={"options": {"num_ctx": 32768}}
```

## Configuration Files

### .env Configuration
Primary configuration file for API server. Key sections:
- Server settings (HOST, PORT, CORS)
- Storage backends (connection strings via environment variables)
- Query parameters (TOP_K, MAX_TOTAL_TOKENS, etc.)
- Reranking configuration (RERANK_BINDING, RERANK_MODEL)
- Authentication (AUTH_ACCOUNTS, LIGHTRAG_API_KEY)

See `env.example` for comprehensive template.

### Workspace Isolation
Each LightRAG instance can use a `workspace` parameter for data isolation. Implementation varies by storage type:
- File-based: subdirectories
- Collection-based: collection name prefixes
- Relational DB: workspace column filtering
- Qdrant: payload-based partitioning

## Testing Guidelines

### Test Structure
- `tests/`: Main test suite (mirrors feature folders)
- `test_*.py` in root: Specific integration tests
- Markers: `offline`, `integration`, `requires_db`, `requires_api`

### Running Tests
```bash
# Default: runs only offline tests
pytest tests

# Include integration tests
pytest tests --run-integration

# Keep test artifacts for debugging
pytest tests --keep-artifacts

# Configure test workers
pytest tests --test-workers 4
```

### Environment Variables for Tests
Set `LIGHTRAG_*` variables for integration tests:
- `LIGHTRAG_RUN_INTEGRATION=true`
- `LIGHTRAG_KEEP_ARTIFACTS=true`
- `LIGHTRAG_TEST_WORKERS=4`
- Plus storage-specific connection strings

## Code Style

### Language
- Comment Language - Use English for comments and documentation
- Backend Language - Use English for backend code and messages
- Frontend Internationalization: i18next for multi-language support

### Python
- Follow PEP 8 with 4-space indentation
- Use type annotations
- Prefer dataclasses for state management
- Use `lightrag.utils.logger` instead of print
- Async/await patterns throughout
- Keep storage implementations in `kg/` with consistent base class inheritance

### TypeScript/React
- Functional components with hooks
- 2-space indentation
- PascalCase for components
- Tailwind utility-first styling

## Important Architectural Notes

### LLM Requirements
- Minimum 32B parameters recommended
- 32KB context minimum (64KB recommended)
- Avoid reasoning models during indexing
- Stronger models for query stage than indexing stage

### Embedding Models
- Must be consistent across indexing and querying
- Recommended: `BAAI/bge-m3`, `text-embedding-3-large`
- Changing models requires clearing vector storage and recreating with new dimensions

### Reranker Configuration
- Significantly improves retrieval quality
- Recommended models: `BAAI/bge-reranker-v2-m3`, Jina rerankers
- Use "mix" mode when reranker is enabled
