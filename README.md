<p align="center">
  <img src="files/banner.png" alt="mnemory banner" />
</p>

# mnemory

Give your AI agents persistent memory. mnemory is a self-hosted [MCP](https://modelcontextprotocol.io/) server that adds personalization and long-term memory to any AI assistant — Claude Code, Open WebUI, Opencode, Cursor, or any MCP-compatible client.

**Plug and play.** Connect mnemory and your agent immediately starts remembering user preferences, facts, decisions, and context across conversations. No system prompt changes needed.

**Self-hosted and secure.** Your data stays on your infrastructure. No cloud dependencies, no third-party access to your memories.

**Intelligent.** Uses a unified LLM pipeline for fact extraction, deduplication, and contradiction resolution in a single call. Memories are semantically searchable, automatically categorized, and expire naturally when no longer relevant.

## Table of Contents

- [Quick Start](#quick-start)
- [Client Configuration](#client-configuration)
- [System Prompts & Examples](#system-prompts--examples)
- [Configuration](#configuration)
- [Memory Model](#memory-model)
- [MCP Tools](#mcp-tools)
- [REST API](#rest-api)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Plugins](#plugins)
- [Benchmark](#benchmark)
- [Development](#development)

## Quick Start

mnemory needs an OpenAI-compatible API key for LLM and embeddings. It picks up `OPENAI_API_KEY` from your environment automatically.

### Using uvx (recommended)

```bash
uvx mnemory
```

That's it. mnemory starts on `http://localhost:8050/mcp`, stores data in `~/.mnemory/`.

Now connect your client — for **Claude Code** or **Opencode**, add to your MCP config:

```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "X-Agent-Id": "claude-code"
      }
    }
  }
}
```

Start a new conversation. Memory works automatically.

### Using Docker

```bash
export OPENAI_API_KEY=sk-your-key
docker-compose up -d
```

### Using pip

```bash
pip install mnemory
mnemory
```

### Production (Qdrant + S3)

For production, use remote Qdrant for vectors and S3/MinIO for artifacts:

```bash
docker run -d \
  -p 8050:8050 \
  -e OPENAI_API_KEY=sk-your-key \
  -e LLM_BASE_URL=https://your-litellm-proxy/v1 \
  -e LLM_MODEL=gpt-5-mini \
  -e QDRANT_HOST=qdrant.example.com \
  -e ARTIFACT_BACKEND=s3 \
  -e S3_ENDPOINT=http://minio.example.com:9000 \
  -e S3_ACCESS_KEY=admin \
  -e S3_SECRET_KEY=secret \
  -e MCP_API_KEYS='{"your-api-key": "your-username"}' \
  -v mnemory-data:/data \
  genunix/mnemory:latest
```

## Client Configuration

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

### Open WebUI (v0.6.31+)

1. Go to **Admin Settings > External Tools > Add Server**
2. Type: **MCP (Streamable HTTP)**
3. URL: `http://mnemory:8050/mcp` (same namespace) or `https://mem.example.com/mcp` (via ingress)
4. Auth: **Bearer**, Key: `your-api-key`
5. Custom headers: `X-Agent-Id: open-webui`
6. Enable tools on models: **Workspace > Models > Advanced Params > Function Calling: Native**

**Note on MCP instructions**: Open WebUI does not inject MCP server instructions into the LLM's system prompt — it only exposes tool descriptions. For optimal proactive memory behavior, add this to your model's system prompt:

> Always call initialize_memory at the start of each conversation.

Or use the full system prompt templates in [examples/system-prompts/](examples/system-prompts/).

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

To give a specific agent personality behavior while keeping the server in `proactive` mode, add the personality snippet to that agent's system prompt. See [examples/system-prompts/openwebui-personality.md](examples/system-prompts/openwebui-personality.md) for the full template, or use this snippet:

```
## Memory-Driven Identity

Your personality and knowledge are stored in memory. At the start of every
conversation, call get_core_memories to load your identity and context.
If you have no identity memories yet, you start as a blank slate — develop
your personality through interactions.

### Storing identity memories

Store identity-defining content with role="assistant" and your agent_id:
- Your personality traits and communication style
- Behavioral rules and principles you follow
- Knowledge and conclusions from your research
- How you should behave toward this specific user

Pin important identity memories so they load at every conversation start.

### Role decision rule

- Memory describes YOU (identity, personality, knowledge) → role="assistant", agent_id=your_agent_id
- Memory describes THE USER (facts, preferences, context) → role="user"
- Memory describes user preference specific to THIS agent → role="user", agent_id=your_agent_id
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

## Configuration

All configuration is via environment variables. Defaults are optimized for local development — just set `OPENAI_API_KEY` and run.

Data is stored in `~/.mnemory/` by default. Override with `DATA_DIR` env var. In Docker, `DATA_DIR` is set to `/data` for volume mounting.

### LLM & Embeddings

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | | OpenAI API key (also used as fallback for `LLM_API_KEY`) |
| `OPENAI_API_BASE` | | OpenAI-compatible API base URL (also used as fallback for `LLM_BASE_URL`) |
| `LLM_API_KEY` | (falls back to `OPENAI_API_KEY`) | API key for LLM provider |
| `LLM_BASE_URL` | (falls back to `OPENAI_API_BASE`, then `https://api.openai.com/v1`) | OpenAI-compatible API base URL |
| `LLM_MODEL` | `gpt-5-mini` | LLM model for fact extraction and deduplication |
| `LLM_REASONING_EFFORT` | | Reasoning effort for LLM (none/minimal/low/medium/high). Models that don't support it auto-skip. |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBED_BASE_URL` | (falls back to `LLM_BASE_URL`) | Separate base URL for embeddings |
| `EMBED_DIMS` | `1536` | Embedding dimensions |

### Data Storage

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `~/.mnemory` | Base data directory (all paths below default to subdirectories of this) |
| `QDRANT_HOST` | | Qdrant host (empty = local embedded mode, set for remote Qdrant) |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_PATH` | `{DATA_DIR}/qdrant` | Local embedded Qdrant storage path (used when `QDRANT_HOST` is not set) |
| `QDRANT_API_KEY` | | Qdrant API key (optional) |
| `QDRANT_COLLECTION` | `mnemory` | Qdrant collection name |
| `ARTIFACT_BACKEND` | `filesystem` | Artifact backend: `filesystem` (local) or `s3` (production) |
| `ARTIFACT_PATH` | `{DATA_DIR}/artifacts` | Local filesystem path for artifacts |
| `S3_ENDPOINT` | `http://localhost:9000` | S3/MinIO endpoint URL |
| `S3_ACCESS_KEY` | (required for s3) | S3 access key |
| `S3_SECRET_KEY` | (required for s3) | S3 secret key |
| `S3_BUCKET` | `mnemory` | S3 bucket name |
| `S3_REGION` | | S3 region (optional) |

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

**Timezone** is set via the `X-Timezone` HTTP header per client connection. This overrides the `DEFAULT_TIMEZONE` env var for the session, affecting how naive `event_date` values are interpreted. Use an IANA timezone name (e.g., `Europe/Prague`, `America/New_York`).

`MCP_API_KEY` (single key) is kept for backward compatibility — it authenticates but does not bind to a user. If both `MCP_API_KEYS` and `MCP_API_KEY` are set, `MCP_API_KEYS` is checked first, then `MCP_API_KEY` as fallback.

### Memory Behavior

| Variable | Default | Description |
|---|---|---|
| `MAX_MEMORY_LENGTH` | `1000` | Max characters for fast memory content |
| `MAX_ARTIFACT_SIZE` | `102400` | Max bytes per artifact (100KB) |
| `MAX_CORE_CONTEXT_LENGTH` | `4000` | Max characters for get_core_memories response |
| `DEFAULT_RECENT_DAYS` | `7` | Default days for recent context in core memories |
| `RECENT_LIMIT_USER` | `25` | Max recent user memories to include |
| `RECENT_LIMIT_AGENT` | `25` | Max recent agent memories to include |
| `AUTO_CLASSIFY` | `true` | Auto-classify memory metadata (type, categories, importance, pinned) via LLM when not provided |
| `CLASSIFY_CACHE_TTL` | `300` | TTL in seconds for the category cache used during auto-classification |
| `CORE_MEMORIES_CACHE_TTL` | `300` | TTL in seconds for the core memories cache (get_core_memories). Set to 0 to disable. Invalidated on memory mutations. |
| `TTL_FACT` | (none) | Default TTL in days for `fact` memories (empty = permanent) |
| `TTL_PREFERENCE` | (none) | Default TTL in days for `preference` memories (empty = permanent) |
| `TTL_EPISODIC` | `90` | Default TTL in days for `episodic` memories |
| `TTL_PROCEDURAL` | `60` | Default TTL in days for `procedural` memories |
| `TTL_CONTEXT` | `7` | Default TTL in days for `context` memories |
| `TRACK_MEMORY_ACCESS` | `true` | Update last_accessed_at and reset TTL on search/recall |
| `SEARCH_SCORE_THRESHOLD` | `0.30` | Minimum score for search results (0.0-1.0). Results below this are filtered out |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.4` | Minimum similarity for deduplication matching during memory ingestion |
| `SEARCH_SIMILARITY_WEIGHT` | `0.9` | Weight for cosine similarity in search ranking (remainder goes to importance). Default 0.9 = 90% similarity, 10% importance |
| `SEARCH_KEYWORD_WEIGHT` | `0.2` | Weight for keyword overlap boost in search results. Set to 0.0 to disable |
| `DEFAULT_TIMEZONE` | | Default IANA timezone for naive `event_date` values (e.g., `Europe/Prague`). Empty = server local timezone. Can be overridden per session via `X-Timezone` header |
| `FIND_MEMORIES_QUERIES` | `5` | Maximum number of search queries the LLM generates for `find_memories` (may return fewer or zero) |
| `MAX_INPUT_LENGTH` | `400000` | Max chars for input to `add_memory(infer=True)` and `remember()`. ~100k tokens |
| `MEMORY_SESSION_TTL` | `3600` | Default session idle TTL in seconds (1 hour) |
| `MEMORY_SESSION_SWEEP_INTERVAL` | `300` | Interval in seconds between session cleanup sweeps (5 minutes) |
| `RECALL_MAX_RESULTS` | `10` | Max search results returned by recall endpoint |
| `REMEMBER_RATE_LIMIT` | `10` | Max remember requests per minute per user. 0 = no limit |

## Memory Model

### Two-Tier Architecture

**Fast Memory** (vector store): Concise facts and summaries, max 1000 characters. Semantically searchable via embeddings. Stored in Qdrant (local embedded or remote).

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

Search results are reranked: `combined_score = similarity * 0.9 + importance_weight * 0.1` (configurable via `SEARCH_SIMILARITY_WEIGHT`). A keyword boost is then blended in: `final = (1 - keyword_weight) * combined_score + keyword_weight * keyword_overlap` (configurable via `SEARCH_KEYWORD_WEIGHT`, default 0.2). Results below `SEARCH_SCORE_THRESHOLD` (default 0.30) are filtered out.

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

When `role="assistant"`, the server uses an agent-specific fact extraction prompt that focuses on the assistant's identity, personality, and capabilities. When `role="user"` (default), user fact extraction is used. The `role` is stored in metadata and can be used as a filter in `search_memories` and `list_memories`.

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

### Session Initialization

| Tool | Description |
|---|---|
| `initialize_memory` | **Start here.** Returns behavioral instructions + core memories. Use for clients that don't inject MCP server instructions (e.g., Open WebUI). |

### Memory Operations

| Tool | Description |
|---|---|
| `add_memory` | Store a memory with optional metadata, `infer` flag, `role`, and `event_date` |
| `add_memories` | Batch-add multiple memories in a single call |
| `search_memories` | Semantic search with type/category/role filters, importance reranking |
| `find_memories` | AI-powered search: generates multiple queries, searches, and reranks by relevance to your question. Temporal-aware — resolves "last week", "in 2023", etc. |
| `get_core_memories` | Load pinned + recent context at conversation start. Use for clients that inject MCP server instructions (e.g., Claude Code). |
| `get_recent_memories` | Get recent activity from the last N days with scope filter (user/agent/all) |
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

## REST API

mnemory exposes a full REST API alongside the MCP server. The FastAPI sub-app is mounted at `/api/` with auto-generated OpenAPI spec at `/api/openapi.json` and Swagger UI at `/api/docs`.

Both MCP and REST share the same `MemoryService` backend and authentication middleware.

### Memory CRUD

| Endpoint | Method | Description |
|---|---|---|
| `/api/memories` | POST | Add a memory |
| `/api/memories/batch` | POST | Batch add memories |
| `/api/memories/search` | POST | Semantic search |
| `/api/memories/find` | POST | AI-powered multi-query search |
| `/api/memories/core` | GET | Core memories (pinned + recent) |
| `/api/memories/recent` | GET | Recent memories |
| `/api/memories` | GET | List memories |
| `/api/memories/{id}` | PUT | Update memory |
| `/api/memories/{id}` | DELETE | Delete memory |
| `/api/memories/{id}/artifacts` | POST | Save artifact |
| `/api/memories/{id}/artifacts` | GET | List artifacts |
| `/api/memories/{id}/artifacts/{aid}` | GET | Get artifact |
| `/api/memories/{id}/artifacts/{aid}` | DELETE | Delete artifact |
| `/api/categories` | GET | List categories |

### Intelligence Layer

Two high-level endpoints designed for plugin-driven automatic memory management:

**POST /api/recall** — Combined initialize + search. Call on each user message.

```json
{
  "session_id": null,
  "query": "Should I buy a dog?",
  "include_instructions": true,
  "managed": true,
  "score_threshold": 0.5
}
```

- First call (no `session_id`): creates session, returns instructions + core memories + `find_memories` results
- Subsequent calls: returns only NEW relevant memories via fast `search_memories` (no LLM), filtering out already-returned IDs
- `score_threshold`: optional per-request minimum score (0.0-1.0) for search results, applied on top of server's `SEARCH_SCORE_THRESHOLD`. Prevents context bloat from weak matches on follow-up messages.
- Graceful degradation: `find_memories` fails → `search_memories` → core memories only → empty response

**POST /api/remember** — Fire-and-forget memory storage. Call after each exchange.

```json
{
  "session_id": "sess_abc123",
  "messages": [
    {"role": "user", "content": "I just moved to Berlin"},
    {"role": "assistant", "content": "That's exciting!"}
  ]
}
```

- Returns `{"accepted": true}` immediately, processes in background
- Reuses the same extraction pipeline as `add_memory(infer=True)` — no extra LLM calls
- Stored memory IDs are added to the session to prevent echo on next recall
- Rate limited per user (configurable via `REMEMBER_RATE_LIMIT`)

### Memory Sessions

The recall/remember endpoints use server-side sessions (`MemorySession`) to track which memories the client already has. This prevents context bloat by only returning new memories on subsequent calls.

- Sessions are created on first recall and expire after idle timeout (default 1 hour)
- Losing a session is harmless — next recall creates a new one
- Periodic background sweep cleans up expired sessions

## How It Works

### Storing a Memory

1. You call `add_memory(content="I drive a Skoda Octavia 2019", categories=["vehicles"])`
2. Content is embedded and similar existing memories are retrieved
3. A single LLM call extracts facts, classifies them, and deduplicates against existing memories
4. If new: creates embedding, stores in Qdrant with metadata
5. If update: merges with existing memory (e.g., "User now drives a Tesla Model 3")

With `infer=false`, the LLM call is skipped — the content is embedded and stored directly. This is much faster (single embedding call vs. LLM + embedding).

**Temporal anchoring:** Pass `event_date` to anchor relative time references. For example, if the content says "I bought it yesterday" and `event_date="2023-05-08"`, the extraction resolves "yesterday" to May 7, 2023 — not today's date. The `event_date` is also stored as metadata for temporal search queries.

### Searching

**search_memories** — fast single-query search:

1. You call `search_memories(query="what car do I have")`
2. Query is embedded, vector similarity search in Qdrant
3. Results filtered by category/type if specified
4. Reranked by combined similarity + importance score, then keyword-boosted
5. Results below score threshold (default 0.30) are filtered out
6. Returns top N results with artifact indicators

**find_memories** — AI-powered multi-query search:

1. You call `find_memories(question="What do I think about dogs? Should I buy one?")`
2. LLM generates multiple search queries covering different angles and associations (e.g., dogs, pets, partner, house, garden, lifestyle). Temporal-aware — knows today's date to resolve "last week", "recently", etc.
3. Each query runs through `search_memories` independently
4. Results are merged and deduplicated across all queries
5. LLM reranks all results by relevance to the original question, using `event_date` metadata for temporal scoring
6. Returns top N results with scores on the same 0.0-1.0 scale

### Core Memories

1. At conversation start, LLM calls `get_core_memories()`
2. Fetches all pinned memories, organized by `role`:
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

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client Layer                                │
│                                                                  │
│  Open WebUI Filter    Open WebUI OpenAPI Tool    OpenCode Plugin │
│  (auto recall/remember)  (LLM-driven ops)       (auto recall)   │
│  Claude Code / Opencode / Cursor (MCP)                          │
└──────────┬──────────────────┬──────────────────────┬────────────┘
           │ REST              │ REST (OpenAPI)        │ MCP
           ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    mnemory                                       │
│                                                                  │
│  FastAPI (/api/)                  MCP (/mcp)                    │
│  ├─ POST /api/recall              16 MCP tools                  │
│  ├─ POST /api/remember                                          │
│  ├─ POST /api/memories            OpenAPI spec at /api/docs     │
│  ├─ POST /api/memories/search                                   │
│  ├─ POST /api/memories/find                                     │
│  ├─ GET  /api/memories/core                                     │
│  ├─ ... (full CRUD)                                             │
│  └─ GET  /api/categories                                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Shared: MemoryService, VectorStore, LLMClient           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────┐  ┌───────────────────────┐                 │
│  │  Fast Memory     │  │  Slow Memory          │                 │
│  │  (Qdrant)        │  │  (S3/MinIO or FS)     │                 │
│  │  Searchable      │  │  Detailed artifacts   │                 │
│  │  facts/summaries │  │  retrieved on demand   │                 │
│  └────────┬─────────┘  └───────────┬───────────┘                 │
└───────────┼────────────────────────┼────────────────────────────┘
            │                        │
            ▼                        ▼
     ┌────────────┐          ┌──────────────┐
     │   Qdrant   │          │  S3 / MinIO  │
     │            │          │  or local FS │
     └────────────┘          └──────────────┘
            │
            ▼
     ┌────────────┐
     │  LiteLLM   │
     │ or OpenAI  │
     │(LLM+embed) │
     └────────────┘
```

## Plugins

Native plugins that automatically recall memories on each user message and store memories after each exchange — no LLM tool-calling required.

### Open WebUI Filter

A filter function that calls `/api/recall` on inlet (before LLM) and `/api/remember` on outlet (after LLM). Injects memories into the system prompt automatically.

See [`examples/openwebui/`](examples/openwebui/) for the filter and setup instructions.

### OpenCode Plugin

A TypeScript plugin that hooks into session lifecycle for automatic recall/remember.

See [`examples/opencode/`](examples/opencode/) for the plugin and setup instructions.

### Hybrid Approach

Plugins handle automatic recall/remember. The LLM still has access to tools (via OpenAPI or MCP) for explicit operations like searching mid-conversation or deleting a memory the user asks to forget.

## Benchmark

mnemory is evaluated on the [LoCoMo](https://github.com/snap-research/locomo) (Long Conversation Memory) benchmark — 10 multi-session dialogues with 1540 QA questions across 4 categories. All systems use gpt-4o-mini for answering and judging, following the [Memobase evaluation convention](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md).

| System | single_hop | multi_hop | temporal | open_domain | Overall |
|---|---|---|---|---|---|
| **mnemory** | **63.1** | **53.1** | **74.8** | **78.2** | **73.2** |
| Memobase | 70.9 | 52.1 | 85.0 | 77.2 | 75.8 |
| Mem0-Graph | 65.7 | 47.2 | 58.1 | 75.7 | 68.4 |
| Mem0 | 67.1 | 51.2 | 55.5 | 72.9 | 66.9 |
| Zep | 61.7 | 41.4 | 49.3 | 76.6 | 66.0 |
| LangMem | 62.2 | 47.9 | 23.4 | 71.1 | 58.1 |

Configuration: `gpt-5-mini` for extraction/embeddings, `text-embedding-3-small` for vectors, `search_memories` with limit=30. See [`benchmarks/`](benchmarks/) for reproduction instructions.

## Development

```bash
# Install with dev dependencies
pip install -e ".[all,dev]"

# Run locally (uses OPENAI_API_KEY, data in ~/.mnemory/)
mnemory

# Run tests
pytest

# Lint
ruff check mnemory/
```

## License

Apache 2.0
