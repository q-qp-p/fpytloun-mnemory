# AGENTS.md — Coding Agent Instructions for mnemory

## Project Overview

**mnemory** is a self-hosted, two-tier memory system for AI agents, exposed as an MCP (Model Context Protocol) server over Streamable HTTP. It provides persistent memory with intelligent fact extraction, deduplication, and semantic search.

- **Language**: Python 3.11+
- **Framework**: MCP SDK (`mcp.server.fastmcp.FastMCP`) with Starlette for HTTP
- **Core dependency**: [mem0ai](https://github.com/mem0ai/mem0) for vector memory operations
- **License**: MIT
- **Repository**: https://github.com/fpytloun/mnemory

## Architecture

```
mnemory/
├── server.py              # MCP server entry point, 12 tool definitions, health endpoint, auth middleware
├── config.py              # Configuration from environment variables (dataclasses)
├── categories.py          # Predefined category registry, validation, matching logic
├── memory.py              # Business logic layer (orchestrates vector + artifact stores)
├── instructions.py        # MCP server instructions text (system prompt for LLMs)
└── storage/
    ├── vector.py          # Vector store abstraction (mem0 wrapper + direct Qdrant helpers)
    └── artifact.py        # Artifact store abstraction (S3 and filesystem backends)
```

### Layer responsibilities

| Layer | File | Responsibility |
|---|---|---|
| **Transport** | `server.py` | MCP tool definitions, HTTP routing, auth, serialization |
| **Business Logic** | `memory.py` | Validation, reranking, core memory assembly, artifact lifecycle |
| **Categories** | `categories.py` | Category validation, prefix matching, counting |
| **Vector Storage** | `storage/vector.py` | mem0 wrapper + direct Qdrant client for advanced queries |
| **Artifact Storage** | `storage/artifact.py` | S3/MinIO and filesystem backends for binary artifacts |
| **Configuration** | `config.py` | Environment variable parsing, mem0 config construction |

### Key design decisions

1. **mem0 for core operations**: mem0 handles fact extraction (LLM-driven), deduplication, contradiction resolution, embedding, and semantic search. We wrap it, not replace it.

2. **Direct Qdrant access for gaps**: mem0 doesn't support date filtering, metadata-only updates, or null field queries. For these, we access Qdrant directly via `qdrant-client`. The `VectorStore` class encapsulates both.

3. **Two-tier memory**: Fast memory (vector store, max 1000 chars, searchable) + slow memory (artifact store, up to 100KB, retrieved on demand). Artifacts are always attached to a parent fast memory.

4. **Configurable backends**: Vector store supports Qdrant (production) or Chroma (local dev). Artifact store supports S3/MinIO (production) or local filesystem (dev). Selected via `VECTOR_BACKEND` and `ARTIFACT_BACKEND` env vars.

5. **Official MCP SDK**: Uses `from mcp.server.fastmcp import FastMCP` (the official `mcp` package), NOT the standalone `fastmcp` package. Starlette is used directly for custom routes and middleware.

6. **Stateless HTTP**: The server runs in stateless mode (`stateless_http=True, json_response=True`) for Kubernetes compatibility.

## Build / Run / Test

### Local development

```bash
# Install with all optional dependencies
pip install -e ".[all,dev]"

# Run with minimal config (Chroma + filesystem)
export LLM_API_KEY=sk-your-key
export VECTOR_BACKEND=chroma
export ARTIFACT_BACKEND=filesystem
export CHROMA_PATH=./data/chroma
export ARTIFACT_PATH=./data/artifacts
export HISTORY_DB_PATH=./data/history.db
mnemory

# Or run the module directly
python -m mnemory.server
```

### Docker build

```bash
# Build for linux/amd64 (for Kubernetes deployment)
docker buildx build --platform linux/amd64 -t genunix/mnemory:latest .

# Push
docker push genunix/mnemory:latest
```

### Tests

```bash
pytest tests/
```

### Linting

```bash
ruff check mnemory/
ruff format mnemory/
```

## Code Conventions

### Style

- Python 3.11+ features (type unions with `|`, `from __future__ import annotations`)
- Type hints on all function signatures
- Docstrings on all public classes and methods
- `logging` module for all output (never `print()`)
- f-strings for string formatting
- `json.dumps()` for all MCP tool return values (tools return `str`)

### Error handling

- MCP tools catch all exceptions and return JSON error objects: `{"error": true, "message": "..."}`
- Validation errors (`ValueError`) are caught separately from internal errors
- Internal errors are logged with `logger.exception()` for stack traces
- Never let exceptions propagate to the MCP framework

### Configuration

- All config via environment variables (no config files)
- Dataclass-based config objects in `config.py`
- `load_config()` validates required fields at startup
- Defaults are sensible for local development

### Memory metadata

Custom metadata is stored as flat fields in the Qdrant payload alongside mem0's standard fields. Our custom fields:

| Field | Type | Description |
|---|---|---|
| `memory_type` | str | preference, fact, episodic, procedural, context |
| `categories` | list[str] | Category tags |
| `importance` | str | low, normal, high, critical |
| `pinned` | bool | Whether to include in core memories |
| `artifacts` | list[dict] | Artifact metadata (id, filename, content_type, size, created_at) |
| `created_at_utc` | str | Our own UTC timestamp (mem0 uses US/Pacific) |

### Adding a new MCP tool

1. Define the tool function in `server.py` with `@mcp.tool()` decorator
2. Write a detailed docstring (this becomes the tool description for LLMs)
3. Add business logic in `memory.py` (keep `server.py` thin)
4. Return JSON string via `json.dumps()`
5. Wrap in try/except returning error JSON on failure
6. Update `instructions.py` if the tool changes the usage workflow
7. Update README.md tool table

### Adding a new storage backend

1. Implement the backend class in the appropriate `storage/` file
2. Follow the existing protocol/interface pattern
3. Add configuration fields to `config.py`
4. Add the backend selection logic in the store's `__init__`
5. Add optional dependency to `pyproject.toml`
6. Update README.md configuration table

## Important Notes

- **mem0 limitations**: The public `update()` method drops custom metadata. We work around this by calling `update()` for content changes and then `set_payload()` on Qdrant for metadata. See `VectorStore.update_metadata()`.

- **Date filtering**: mem0 stores timestamps as ISO strings but its filter builder only supports numeric ranges. We use Qdrant's `DatetimeRange` directly for date queries. See `VectorStore.get_recent_memories()`.

- **Category filtering**: mem0 can filter by exact metadata match but not prefix matching. Category prefix matching (e.g., "project" matches "project:foo") is done as post-filtering in Python. See `matches_category_filter()`.

- **Chroma fallback**: When using Chroma backend, advanced features (date filtering, metadata-only updates) fall back to less efficient implementations (get_all + post-filter, full rewrite). Qdrant is recommended for production.

- **Artifact metadata**: Artifact references (id, filename, size, etc.) are stored in the fast memory's metadata in the vector store. The actual content is in S3/filesystem. Deleting a memory should also delete its artifacts.
