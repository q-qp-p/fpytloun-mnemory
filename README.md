<p align="center">
  <img src="files/banner.png" alt="mnemory banner" />
</p>

# mnemory

A self-hosted, two-tier memory system for AI agents and assistants, exposed as an [MCP](https://modelcontextprotocol.io/) server over Streamable HTTP.

Built on [mem0](https://github.com/mem0ai/mem0) for intelligent fact extraction and deduplication, with an artifact store for detailed content. Works with any MCP-compatible client: Open WebUI, Claude Code, Opencode, Cursor, VS Code, and more.

## Features

- **Two-tier memory**: Fast memory (vector-searchable facts in Qdrant/Chroma) + slow memory (detailed artifacts in S3/filesystem)
- **Intelligent storage**: mem0-powered fact extraction, deduplication, and contradiction resolution
- **Structured memory model**: Memory types (preference, fact, episodic, procedural, context), predefined categories with project namespacing, importance levels
- **Core memories**: Pinned memories + recent context loaded at conversation start for agent identity and user context
- **Artifact store**: Attach detailed content (research reports, analysis, code, images) to memories, retrieved on demand with pagination
- **Multi-user & multi-agent**: User isolation via `user_id`, agent-specific memories via `agent_id`
- **Configurable backends**: Qdrant or Chroma for vectors, S3/MinIO or local filesystem for artifacts
- **MCP native**: Streamable HTTP transport, works with Open WebUI (v0.6.31+), Claude Code, and any MCP client
- **Session-level identity**: API key → user_id mapping + X-Agent-Id header, so LLMs don't need to pass identity per tool call
- **API key authentication**: Optional Bearer token or X-API-Key header auth, with multi-key support
- **Health endpoint**: `/health` for Kubernetes probes

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  MCP Clients                      │
│  Open WebUI, Claude Code, Opencode, Cursor, ...  │
└──────────────────────┬───────────────────────────┘
                       │ Streamable HTTP (/mcp)
                       ▼
┌──────────────────────────────────────────────────┐
│                    mnemory                        │
│                                                   │
│  13 MCP Tools:                                    │
│  add_memory, add_memories, search_memories,       │
│  get_core_memories, list_memories, update_memory,  │
│  delete_memory, delete_all_memories,              │
│  list_categories, save_artifact, get_artifact,    │
│  list_artifacts, delete_artifact                  │
│                                                   │
│  ┌─────────────────┐  ┌───────────────────────┐  │
│  │  Fast Memory     │  │  Slow Memory          │  │
│  │  (mem0 + Qdrant) │  │  (S3/MinIO or FS)     │  │
│  │  Searchable      │  │  Detailed artifacts   │  │
│  │  facts/summaries │  │  retrieved on demand   │  │
│  └────────┬─────────┘  └───────────┬───────────┘  │
└───────────┼────────────────────────┼──────────────┘
            │                        │
            ▼                        ▼
     ┌────────────┐          ┌──────────────┐
     │   Qdrant   │          │  S3 / MinIO  │
     │ or Chroma  │          │  or local FS │
     └────────────┘          └──────────────┘
            │
            ▼
     ┌────────────┐
     │  LiteLLM   │
     │ or OpenAI  │
     │(LLM+embed) │
     └────────────┘
```

## Quick Start

```bash
git clone https://github.com/fpytloun/mnemory.git
cd mnemory
export OPENAI_API_KEY=sk-your-key
docker-compose up -d
```

The MCP endpoint is available at `http://localhost:8050/mcp`.

### Alternative: Docker run

```bash
docker run -d \
  -p 8050:8050 \
  -e LLM_API_KEY=$OPENAI_API_KEY \
  -e VECTOR_BACKEND=chroma \
  -e ARTIFACT_BACKEND=filesystem \
  -v mnemory-data:/data \
  genunix/mnemory:latest
```

### Alternative: Local development

```bash
pip install -e ".[all]"
export LLM_API_KEY=$OPENAI_API_KEY
export VECTOR_BACKEND=chroma
export ARTIFACT_BACKEND=filesystem
mnemory
```

### Production (Qdrant + S3)

```bash
docker run -d \
  -p 8050:8050 \
  -e LLM_API_KEY=sk-your-key \
  -e LLM_BASE_URL=https://your-litellm-proxy/v1 \
  -e LLM_MODEL=gpt-5-mini \
  -e VECTOR_BACKEND=qdrant \
  -e QDRANT_HOST=qdrant.example.com \
  -e QDRANT_PORT=6333 \
  -e ARTIFACT_BACKEND=s3 \
  -e S3_ENDPOINT=http://minio.example.com:9000 \
  -e S3_ACCESS_KEY=admin \
  -e S3_SECRET_KEY=secret \
  -e S3_BUCKET=mnemory \
  -e MCP_API_KEYS='{"your-api-key": "your-username"}' \
  -v mnemory-data:/data \
  genunix/mnemory:latest
```

## Configuration

All configuration is via environment variables:

### LLM & Embeddings

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | (required) | API key for LLM provider |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `gpt-5-mini` | LLM model for fact extraction and deduplication |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBED_BASE_URL` | (falls back to `LLM_BASE_URL`) | Separate base URL for embeddings |
| `EMBED_DIMS` | `1536` | Embedding dimensions |

### Vector Store

| Variable | Default | Description |
|---|---|---|
| `VECTOR_BACKEND` | `qdrant` | Vector store backend: `qdrant` or `chroma` |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_API_KEY` | | Qdrant API key (optional) |
| `QDRANT_COLLECTION` | `mnemory` | Qdrant collection name |
| `CHROMA_PATH` | `/data/chroma` | Chroma local storage path |

### Artifact Store

| Variable | Default | Description |
|---|---|---|
| `ARTIFACT_BACKEND` | `s3` | Artifact backend: `s3` or `filesystem` |
| `S3_ENDPOINT` | `http://localhost:9000` | S3/MinIO endpoint URL |
| `S3_ACCESS_KEY` | (required for s3) | S3 access key |
| `S3_SECRET_KEY` | (required for s3) | S3 secret key |
| `S3_BUCKET` | `mnemory` | S3 bucket name |
| `S3_REGION` | | S3 region (optional) |
| `ARTIFACT_PATH` | `/data/artifacts` | Local filesystem path for artifacts |

### Server

| Variable | Default | Description |
|---|---|---|
| `MCP_HOST` | `0.0.0.0` | Listen host |
| `MCP_PORT` | `8050` | Listen port |
| `MCP_API_KEY` | | Single API key for authentication (empty = no auth) |
| `MCP_API_KEYS` | | JSON dict mapping API keys to user IDs (see below) |
| `INSTRUCTION_MODE` | `proactive` | LLM behavioral instructions: `passive`, `proactive`, or `personality` (see below) |
| `ENABLE_DELETE_ALL` | `false` | Enable the `delete_all_memories` tool (destructive, disabled by default) |
| `LOG_LEVEL` | `INFO` | Logging level |

#### Instruction Modes (`INSTRUCTION_MODE`)

Controls how aggressively the LLM uses memory tools. These instructions are sent via the MCP protocol and injected into the LLM's system prompt by supporting clients.

| Mode | Description |
|---|---|
| `passive` | Soft guidance — use memory when asked or clearly relevant. Minimal behavioral directives. |
| `proactive` | **Default.** Always search before answering, proactively store new information, treat memory as primary context. Plug-and-play magic. |
| `personality` | Proactive + identity development. The agent can develop and maintain its own personality, knowledge, and "soul" through `role=assistant` memories. |

For most setups, `proactive` (default) is the right choice — it makes memory work automatically without any system prompt configuration. Use `personality` when all connected agents should develop their own identity. Use `passive` for manual control.

To activate personality behavior for a **specific agent** while keeping the server in `proactive` mode, add the personality snippet to that agent's system prompt instead. See [examples/system-prompts/](examples/system-prompts/) for templates.

#### Session-Level Identity (`MCP_API_KEYS`)

Map API keys to user IDs so the LLM doesn't need to pass `user_id` in every tool call:

```bash
MCP_API_KEYS='{"mnm-key-for-filip": "filip", "mnm-shared-service-key": "*"}'
```

- `"key": "username"` — authenticates AND binds `user_id=username` to the session
- `"key": "*"` — authenticates only (wildcard), `user_id` must come from identity headers or tool parameter

**Identity resolution priority (user_id):**
1. API key mapping (non-wildcard) — most secure, cannot be overridden
2. `X-User-Id` HTTP header — explicit identity header
3. `X-OpenWebUI-User-Email` HTTP header — automatic Open WebUI integration
4. Tool parameter — backward compatible fallback

**Agent ID** is set via the `X-Agent-Id` HTTP header per client connection:
- Open WebUI: `X-Agent-Id: open-webui`
- Claude Code: `X-Agent-Id: claude-code`

`MCP_API_KEY` (single key) is kept for backward compatibility — it authenticates but does not bind to a user. If both `MCP_API_KEYS` and `MCP_API_KEY` are set, `MCP_API_KEYS` is checked first, then `MCP_API_KEY` as fallback.

### Memory Behavior

| Variable | Default | Description |
|---|---|---|
| `HISTORY_DB_PATH` | `/data/history.db` | SQLite history database path |
| `MAX_MEMORY_LENGTH` | `1000` | Max characters for fast memory content |
| `MAX_ARTIFACT_SIZE` | `102400` | Max bytes per artifact (100KB) |
| `MAX_CORE_CONTEXT_LENGTH` | `4000` | Max characters for get_core_memories response |
| `DEFAULT_RECENT_HOURS` | `24` | Default hours for recent context in core memories |
| `AUTO_CLASSIFY` | `true` | Auto-classify memory metadata (type, categories, importance, pinned) via LLM when not provided |
| `CLASSIFY_CACHE_TTL` | `300` | TTL in seconds for the category cache used during auto-classification |
| `CORE_MEMORIES_CACHE_TTL` | `300` | TTL in seconds for the core memories cache (get_core_memories). Set to 0 to disable. Invalidated on memory mutations. |
| `TTL_FACT` | (none) | Default TTL in days for `fact` memories (empty = permanent) |
| `TTL_PREFERENCE` | (none) | Default TTL in days for `preference` memories (empty = permanent) |
| `TTL_EPISODIC` | `90` | Default TTL in days for `episodic` memories |
| `TTL_PROCEDURAL` | `60` | Default TTL in days for `procedural` memories |
| `TTL_CONTEXT` | `7` | Default TTL in days for `context` memories |
| `TRACK_MEMORY_ACCESS` | `true` | Update last_accessed_at and reset TTL on search/recall |

## Memory Model

### Two-Tier Architecture

**Fast Memory** (vector store): Concise facts and summaries, max 1000 characters. Semantically searchable via embeddings. Stored in Qdrant or Chroma via mem0.

**Slow Memory** (artifact store): Detailed content — research reports, analysis, logs, code, images. Stored in S3/MinIO or local filesystem. Attached to fast memories, retrieved on demand with pagination.

### Memory Types

| Type | Purpose | Default TTL |
|---|---|---|
| `preference` | Likes, dislikes, style choices | Permanent |
| `fact` | Biographical, factual information | Permanent |
| `episodic` | Events, interactions, conclusions | 90 days |
| `procedural` | Workflows, habits, "how to" | 60 days |
| `context` | Session/short-term context | 7 days |

Default TTLs are configurable via environment variables (see Memory Behavior config).

### Categories

Predefined categories ensure LLM discoverability:

| Category | Description |
|---|---|
| `personal` | Personal life, family, relationships |
| `preferences` | Likes, dislikes, style preferences |
| `health` | Physical, mental, medical |
| `work` | Job, career, professional |
| `technical` | Tools, languages, infrastructure |
| `finance` | Money, investments, billing |
| `home` | House, appliances, maintenance |
| `vehicles` | Cars, bikes, maintenance history |
| `travel` | Trips, places, itineraries |
| `entertainment` | Movies, music, books, games |
| `goals` | Objectives, plans, ambitions |
| `decisions` | Conclusions, choices, reasoning |
| `project` | Project-specific (use `project:<name>`) |

Dynamic subcategories via prefix: `project:myapp`, `project:domecek/k8s-manifests`.

### Importance Levels

| Level | Search Weight | Use For |
|---|---|---|
| `low` | 0.1 | Minor details, temporary notes |
| `normal` | 0.4 | Standard memories (default) |
| `high` | 0.7 | Important facts, key decisions |
| `critical` | 1.0 | Essential information, always-relevant |

Search results are reranked: `combined_score = similarity * 0.7 + importance_weight * 0.3`.

### Pinned Memories

Memories with `pinned: true` are loaded at every conversation start via `get_core_memories`. Use for:
- User identity facts ("Lives in Prague", "DevOps engineer")
- Core preferences ("Prefers direct communication")
- Agent identity ("Your name is Bob", "You speak casually")
- Agent knowledge ("You researched X and concluded Y")

### TTL (Time-To-Live)

Memories can have a TTL that causes them to decay (soft-expire) after a set number of days. Each memory type has a configurable default TTL (see Memory Types table above). You can override the default by passing `ttl_days` to `add_memory`.

**Lifecycle:**
1. Memory is created with `expires_at` calculated from `ttl_days`
2. When `expires_at` passes, the memory enters **decayed** state (soft-deleted)
3. Decayed memories are excluded from search and list by default
4. Use `include_decayed=true` to browse historical/expired memories
5. Decayed memories can be restored via `update_memory` (set new `ttl_days`)

**Reinforcement:** When a memory is accessed via `search_memories`, its TTL is automatically reset — `expires_at` is recalculated from now + original `ttl_days`. This means frequently-used memories stay alive. Controlled by `TRACK_MEMORY_ACCESS` config.

**Pinned exemption:** Pinned memories (`pinned: true`) are exempt from TTL — they never decay, even if `expires_at` is set.

**Existing memories:** No migration needed. Memories without TTL fields are treated as permanent.

### Role

The `role` parameter on `add_memory` controls who the memory is about:

| Role | Description | Example |
|---|---|---|
| `user` (default) | Facts about the user | "User lives in Prague", "User prefers dark mode" |
| `assistant` | Facts about the agent itself | "Your name is Bob", "You speak casually" |

When `role="assistant"`, the server passes content to mem0 as an assistant-role message, triggering mem0's agent-specific fact extraction prompt. When `role="user"` (default), user fact extraction is used. The `role` is stored in metadata and can be used as a filter in `search_memories` and `list_memories`.

`get_core_memories` uses `role` to organize agent-scoped memories into sections:
- **Agent Identity**: pinned, `role=assistant`, fact/preference type
- **Agent Knowledge**: pinned, `role=assistant`, other types
- **Agent Instructions**: pinned, `role=user`, agent-scoped (user preferences specific to this agent)

### Scoping

- `user_id` (required): Every memory belongs to a user. Shared across all agents. Can be set at session level via API key mapping (`MCP_API_KEYS`) or `X-User-Id` header, eliminating the need to pass it per tool call.
- `agent_id` (optional): Set for agent-scoped memories. Two use cases:
  - **Agent identity** (`role="assistant"`): The agent's name, personality, capabilities, knowledge.
  - **Agent-scoped user preferences** (`role="user"`): User preferences that apply only to this agent (e.g., "User wants me to create commit messages").
  
  Different agents see different agent memories but share user memories. Can be set at session level via `X-Agent-Id` header.

#### Sub-agents

Sub-agents allow creating multiple independent agent identities under a single session. Use a colon-separated prefix matching the session's `X-Agent-Id`:

```
X-Agent-Id: openwebui          ← session agent
agent_id: openwebui:bob        ← sub-agent "bob" (allowed)
agent_id: openwebui:alice      ← sub-agent "alice" (allowed)
agent_id: cursor:foo           ← different parent (blocked)
```

Sub-agents are fully independent — they have their own memories and do NOT inherit from the parent agent. The session agent can access and manage all its sub-agents' memories.

## MCP Tools

### Memory Operations

| Tool | Description |
|---|---|
| `add_memory` | Store a memory with optional metadata, `infer` flag, and `role` |
| `add_memories` | Batch-add multiple memories in a single call |
| `search_memories` | Semantic search with type/category/role filters, importance reranking |
| `get_core_memories` | Load pinned + recent context at conversation start |
| `list_memories` | List all/filtered memories |
| `update_memory` | Update content or metadata of existing memory |
| `delete_memory` | Delete a memory and its artifacts |
| `delete_all_memories` | Delete all memories in scope |
| `list_categories` | List categories with counts for discoverability |

### Artifact Operations

| Tool | Description |
|---|---|
| `save_artifact` | Attach detailed content to a memory |
| `get_artifact` | Retrieve artifact content with pagination |
| `list_artifacts` | List artifacts on a memory |
| `delete_artifact` | Remove an artifact |

## Client Configuration

### Open WebUI (v0.6.31+)

1. Go to **Admin Settings > External Tools > Add Server**
2. Type: **MCP (Streamable HTTP)**
3. URL: `http://mnemory:8050/mcp` (same namespace) or `https://mem.example.com/mcp` (via ingress)
4. Auth: **Bearer**, Key: `your-api-key`
5. Custom headers: `X-Agent-Id: open-webui`
6. Enable tools on models: **Workspace > Models > Advanced Params > Function Calling: Native**

**Multi-user setup** (recommended): Enable `ENABLE_FORWARD_USER_INFO_HEADERS=true` in Open WebUI so it forwards `X-OpenWebUI-User-Email` per user. Use a wildcard API key in mnemory — user identity is resolved automatically from the email header:

```bash
# Open WebUI environment
ENABLE_FORWARD_USER_INFO_HEADERS=true

# Mnemory environment
MCP_API_KEYS='{"shared-openwebui-key": "*"}'
```

**Single-user setup**: Use a non-wildcard API key that binds directly to a user:

```bash
MCP_API_KEYS='{"your-api-key": "filip"}'
```

### Claude Code / Opencode

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "https://mem.example.com/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "claude-code"
      }
    }
  }
}
```

With `MCP_API_KEYS` configured, the LLM doesn't need to pass `user_id` or `agent_id` — both are resolved from the API key mapping and `X-Agent-Id` header.

### Cursor / VS Code

Add to MCP settings with the Streamable HTTP URL. Set `Authorization` and `X-Agent-Id` headers.

## System Prompts & Examples

mnemory ships behavioral instructions via the MCP protocol — most clients inject these into the LLM's system prompt automatically. With `INSTRUCTION_MODE=proactive` (the default), memory works out of the box with zero system prompt configuration.

For advanced setups, example system prompts and guides are available in [`examples/system-prompts/`](examples/system-prompts/):

| Template | Description |
|---|---|
| [Quick Start](examples/system-prompts/quickstart.md) | Get running in 5 minutes — step-by-step guide |
| [Open WebUI — Basic](examples/system-prompts/openwebui-basic.md) | Minimal setup, memory works automatically |
| [Open WebUI — Personality](examples/system-prompts/openwebui-personality.md) | Agent with evolving identity and "soul" (sub-agent pattern) |
| [Claude Code / Opencode](examples/system-prompts/claude-code.md) | Coding assistant with project context memory |

### Per-Agent Personality (without server-wide `personality` mode)

To give a specific agent personality behavior while keeping the server in `proactive` mode, add this snippet to that agent's system prompt:

```
## Memory-Driven Identity

You can develop and maintain your own identity through memories. Your core
memories define who you are — load them at conversation start with
get_core_memories.

If you do not have identity memories yet, you start as a blank slate.
Develop your personality through interactions with the user.

### Storing identity memories
Store identity-defining content with role="assistant" and your agent_id:
- Your name, personality traits, communication style
- Behavioral rules and principles you follow
- Knowledge and conclusions from your research
- How you should behave toward this specific user

Pin important identity memories so they load at every conversation start.

### Role decision rule
- Memory describes YOU (identity, personality, knowledge) → role="assistant"
- Memory describes THE USER (facts, preferences, context) → role="user"
- Content has both → split into separate memories with correct roles

### Building knowledge
Use artifacts to build your knowledge base — save detailed research,
analysis notes, and reference material as artifacts attached to summary
memories. Your memories and artifacts form your evolving knowledge and
experience.

Regularly reflect on interactions and update your self-understanding.
Your identity should feel consistent but can evolve naturally over time.
```

For sub-agents (e.g., `openwebui:yoda`), you must also hardcode the `agent_id` in the system prompt — see the [personality template](examples/system-prompts/openwebui-personality.md) for details.

## How It Works

### Storing a Memory

1. You call `add_memory(content="I drive a Škoda Octavia 2019", user_id="filip", categories=["vehicles"])`
2. mem0's LLM extracts facts: "User drives a Škoda Octavia 2019"
3. mem0 checks for duplicates/contradictions in existing memories
4. If new: creates embedding, stores in Qdrant with metadata
5. If update: merges with existing memory (e.g., "User now drives a Tesla Model 3")

With `infer=false`, steps 2-3 are skipped — the content is embedded and stored directly. This is much faster (single embedding call vs. 2 LLM calls + embedding + vector search).

### Searching

1. You call `search_memories(query="what car do I have", user_id="filip")`
2. Query is embedded, vector similarity search in Qdrant
3. Results filtered by category/type if specified
4. Reranked by combined similarity + importance score
5. Returns top N results with artifact indicators

### Core Memories

1. At conversation start, LLM calls `get_core_memories(user_id="filip", agent_id="bob")`
2. Fetches all pinned memories for agent "bob", organized by `role`:
   - **Agent Identity** (`role=assistant`, fact/preference): name, personality
   - **Agent Knowledge** (`role=assistant`, other types): researched conclusions
   - **Agent Instructions** (`role=user`): user preferences specific to this agent
3. Fetches all pinned user memories (facts, preferences) — shared across agents
4. Fetches recent context memories from last 24h
5. Returns structured text injected into conversation context

### Artifacts

1. Agent does deep research, stores summary: `add_memory(content="Researched washing machines. Samsung WW90T best for price/quality.")`
2. Attaches full report: `save_artifact(memory_id="...", content="<5000 word report>", filename="research.md")`
3. Later, search finds the summary with `has_artifacts: true`
4. Agent fetches details: `get_artifact(memory_id="...", artifact_id="...")`

## Development

```bash
# Install with dev dependencies
pip install -e ".[all,dev]"

# Run locally
export LLM_API_KEY=sk-your-key
export VECTOR_BACKEND=chroma
export ARTIFACT_BACKEND=filesystem
mnemory

# Run tests
pytest

# Lint
ruff check mnemory/
```

## License

MIT
