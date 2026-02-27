<p align="center">
  <img src="files/banner.png" alt="mnemory banner" />
</p>

# mnemory

Give your AI agents persistent memory. mnemory is a self-hosted [MCP](https://modelcontextprotocol.io/) server that adds personalization and long-term memory to any AI assistant — Claude Code, ChatGPT, Open WebUI, Cursor, or any MCP-compatible client.

**Plug and play.** Connect mnemory and your agent immediately starts remembering user preferences, facts, decisions, and context across conversations. No system prompt changes needed.

**Self-hosted and secure.** Your data stays on your infrastructure. No cloud dependencies, no third-party access to your memories.

**Intelligent.** Uses a unified LLM pipeline for fact extraction, deduplication, and contradiction resolution in a single call. Memories are semantically searchable, automatically categorized, and expire naturally when no longer relevant.

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Clients](#supported-clients)
- [Configuration](#configuration)
- [Memory Model](#memory-model)
- [MCP Tools](#mcp-tools)
- [REST API](#rest-api)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Management UI](#management-ui)
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

Now connect your client — for **Claude Code**, add to your MCP config:

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

## Supported Clients

mnemory works with any MCP-compatible client. Some clients also have dedicated plugins for automatic recall/remember.

| Client | MCP | Plugin | Setup Guide |
|---|---|---|---|
| Claude Code | Yes | Yes ([hooks](integrations/claude-code/)) | [docs/clients/claude-code.md](docs/clients/claude-code.md) |
| ChatGPT | Yes (MCP connector) | -- | [docs/clients/chatgpt.md](docs/clients/chatgpt.md) |
| Claude Desktop | Yes | -- | [docs/clients/claude-desktop.md](docs/clients/claude-desktop.md) |
| Open WebUI | Yes | Yes ([filter](integrations/openwebui/)) | [docs/clients/open-webui.md](docs/clients/open-webui.md) |
| OpenCode | Yes | Yes ([plugin](integrations/opencode/)) | [docs/clients/opencode.md](docs/clients/opencode.md) |
| Cursor | Yes | -- | [docs/clients/cursor.md](docs/clients/cursor.md) |
| Windsurf | Yes | -- | [docs/clients/windsurf.md](docs/clients/windsurf.md) |
| Cline | Yes | -- | [docs/clients/cline.md](docs/clients/cline.md) |
| Continue.dev | Yes | -- | [docs/clients/continue.md](docs/clients/continue.md) |
| Codex CLI | Yes | -- | [docs/clients/codex.md](docs/clients/codex.md) |

**MCP** = works via Model Context Protocol (LLM-driven tool calls). **Plugin** = dedicated integration with automatic recall/remember (no LLM tool-calling needed).

See [docs/](docs/) for all setup guides, system prompt templates, and the [quick start guide](docs/quickstart.md).

### Per-Agent Personality (without server-wide `personality` mode)

To give a specific agent personality behavior while keeping the server in `proactive` mode, add the personality snippet to that agent's system prompt. See [docs/system-prompts/openwebui-personality.md](docs/system-prompts/openwebui-personality.md) for the full template, or use this snippet:

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

For sub-agents (e.g., `openwebui:yoda`), you must also hardcode the `agent_id` in the system prompt — see the [personality template](docs/system-prompts/openwebui-personality.md) for details.

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
| `ENABLE_METRICS` | `true` | Enable the `/metrics` Prometheus endpoint |
| `METRICS_CACHE_TTL` | `60` | Cache TTL in seconds for Qdrant gauge aggregation on `/metrics` |
| `MGMT_PORT` | | Management port for `/health` and `/metrics` (see below) |
| `MGMT_HOST` | (falls back to `MCP_HOST`) | Bind host for the management port |
| `LOG_LEVEL` | `INFO` | Logging level |

#### Instruction Modes (`INSTRUCTION_MODE`)

Controls how aggressively the LLM uses memory tools. These instructions are sent via the MCP protocol and injected into the LLM's system prompt by supporting clients.

| Mode | Description |
|---|---|
| `passive` | Soft guidance — use memory when asked or clearly relevant. Minimal behavioral directives. |
| `proactive` | **Default.** Always search before answering, proactively store new information, treat memory as primary context. Plug-and-play magic. |
| `personality` | Proactive + identity development. The agent can develop and maintain its own personality, knowledge, and "soul" through `role=assistant` memories. |

For most setups, `proactive` (default) is the right choice — it makes memory work automatically without any system prompt configuration. Use `personality` when all connected agents should develop their own identity. Use `passive` for manual control.

To activate personality behavior for a **specific agent** while keeping the server in `proactive` mode, add the personality snippet to that agent's system prompt instead. See [docs/system-prompts/](docs/system-prompts/) for templates.

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

#### Management Port (`MGMT_PORT`)

By default, `/health` and `/metrics` are served on the main port (`MCP_PORT`) and go through standard API key authentication. Set `MGMT_PORT` to run these endpoints on a separate port **without authentication** — useful for Kubernetes probes and Prometheus scraping:

```bash
MGMT_PORT=9090          # /health and /metrics on port 9090, no auth
MGMT_HOST=127.0.0.1     # Optional: bind management to localhost only
```

When `MGMT_PORT` is set and differs from `MCP_PORT`:
- `/health` and `/metrics` are served on `MGMT_PORT` without auth
- The main port (`MCP_PORT`) does NOT serve `/health` or `/metrics`
- All MCP and REST API endpoints remain on the main port with auth

When `MGMT_PORT` is not set (default):
- `/health` and `/metrics` are on the main port with standard auth
- Kubernetes probes and Prometheus must send a valid API key

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
| `CORE_TOP_MEMORIES` | `10` | Max non-pinned memories to include in core sections by importance. Set to 0 to disable (only pinned memories). |
| `CORE_MIN_IMPORTANCE` | `normal` | Minimum importance for non-pinned memories in core sections. Options: low, normal, high, critical |
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
 | `FSCK_CACHE_TTL` | `86400` | How long memory check results are cached in seconds (24 hours) |
 | `FSCK_LLM_CONCURRENCY` | `4` | Max concurrent LLM calls during memory check. Set to 1 for sequential |
 | `FSCK_LLM_MODEL` | | Override LLM model for memory check (empty = use main `LLM_MODEL`) |
 | `FSCK_REASONING_EFFORT` | `medium` | Reasoning effort for memory check LLM calls (empty = use main `LLM_REASONING_EFFORT`). Defaults to `medium` for better accuracy |
 | `FSCK_AUTO_INTERVAL` | `0` | Interval in hours between automatic background memory checks. `0` = disabled |
 | `FSCK_AUTO_MIN_CONFIDENCE` | `0.95` | Minimum confidence score (0.0–1.0) for a fix to be auto-applied |
 | `FSCK_AUTO_MIN_SEVERITY` | `medium` | Minimum severity for a fix to be auto-applied. Options: `low`, `medium`, `high` |

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

Search results are reranked: `combined_score = similarity * 0.9 + importance_weight * 0.1` (configurable via `SEARCH_SIMILARITY_WEIGHT`). An additive keyword boost is then applied: `final = min(1.0, combined_score + keyword_weight * keyword_overlap)` (configurable via `SEARCH_KEYWORD_WEIGHT`, default 0.2). Keyword matching only boosts scores — memories without keyword overlap keep their original score. Results below `SEARCH_SCORE_THRESHOLD` (default 0.30) are filtered out.

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
| `search_memories` | Semantic search with type/category/role/date filters, importance reranking |
| `find_memories` | AI-powered search: generates multiple queries, searches, and reranks by relevance to your question. Temporal-aware — resolves "last week", "in 2023", etc. |
| `get_core_memories` | Load pinned + recent context at conversation start. Use for clients that inject MCP server instructions (e.g., Claude Code). |
| `get_recent_memories` | Get recent activity from the last N days with scope filter (user/agent/all) |
| `list_memories` | List all/filtered memories |
| `update_memory` | Update content or metadata of existing memory (including `event_date`) |
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

### Memory Check (fsck)

Built-in memory consistency checker. Runs a three-phase pipeline to detect quality issues and suggest fixes:

| Endpoint | Method | Description |
|---|---|---|
| `/api/fsck` | POST | Start a memory check (runs in background) |
| `/api/fsck/{id}` | GET | Poll check status, progress, and results |
| `/api/fsck/{id}/apply` | POST | Apply selected fixes from a completed check |

**Three-phase pipeline:**

1. **Security scan** (instant) — regex-based detection of prompt injection patterns
2. **Duplicate detection** — vector similarity clustering + LLM evaluation of each cluster for duplicates and contradictions
3. **Quality check** — LLM batch evaluation for spelling/grammar errors, meaningless memories, memories that should be split, and metadata misclassification

Phases 2 and 3 run LLM calls in parallel (`FSCK_LLM_CONCURRENCY`, default 4 workers) for ~4x speedup on large memory sets. Results are cached with configurable TTL (`FSCK_CACHE_TTL`, default 24 hours) so you can review issues in the UI and apply fixes without re-running the check.

**Issue types:** `duplicate`, `quality`, `split`, `contradiction`, `reclassify`, `security`

**Workflow:** Start check -> poll until completed -> review issues in UI -> select and apply fixes. Each fix is a set of actions (update content, update metadata, delete memory, or add new memory).

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

### Periodic Maintenance

mnemory includes a built-in background maintenance service that automatically runs memory consistency checks and applies fixes on a configurable schedule.

Enable it by setting `FSCK_AUTO_INTERVAL` to a non-zero number of hours:

```bash
FSCK_AUTO_INTERVAL=24          # Run every 24 hours
FSCK_AUTO_MIN_CONFIDENCE=0.95  # Only apply fixes with ≥95% confidence
FSCK_AUTO_MIN_SEVERITY=medium  # Only apply medium or high severity fixes
```

**How it works:**

1. The maintenance loop sleeps for the configured interval, then wakes up
2. All users in the collection are enumerated via a Qdrant scroll
3. For each user, a full fsck check is run (security scan → duplicate detection → quality check)
4. Issues that meet both the confidence and severity thresholds are automatically applied
5. Results are recorded in Prometheus metrics (`mnemory_autofsck_*` counters + last-run timestamp)
6. Per-user errors are isolated — one failing user does not stop the run

The loop is **sleep-first**: it waits the full interval before the first run, so startup is not impacted. Auto-applied fixes are visible in the management UI (Check tab status banner) and in the Grafana dashboard (Auto-fsck row).

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
3. Results filtered by category/type/date range if specified
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
2. Fetches all pinned memories in a single query, organized by `role`:
   - **Agent Identity** (`role=assistant`, fact/preference): name, personality
   - **Agent Knowledge** (`role=assistant`, other types): researched conclusions
   - **Agent Instructions** (`role=user`): user preferences specific to this agent
3. Fetches all pinned user memories (facts, preferences) — shared across agents
4. Fetches top-N non-pinned memories by importance (`CORE_TOP_MEMORIES`, default 10) at or above `CORE_MIN_IMPORTANCE` (default `normal`). These are added after pinned memories in each section.
5. All memories within each section are sorted by importance (critical > high > normal > low), then by recency
6. Fetches recent context memories from last N days (configurable)
7. If output exceeds `MAX_CORE_CONTEXT_LENGTH`, recent context entries are trimmed one by one (most recent kept). Hard truncation only as last resort.
8. Returns structured text injected into conversation context

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
│  Open WebUI Filter    Open WebUI OpenAPI Tool    Claude Code Plugin │
│  (auto recall/remember)  (LLM-driven ops)       (auto recall)   │
│  Claude Code / OpenCode / Cursor (MCP)                          │
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

## Management UI

mnemory includes a built-in management UI at `/ui` on the main server port. No external dependencies, no build step required — it ships pre-built with the package.

<p align="center">
  <img src="files/screenshots/ui-dashboard.jpg" alt="Dashboard — memory totals, breakdowns, and operation counts" width="800" />
  <br><em>Dashboard — memory totals, breakdowns by type/category/role, and operation counts</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-search.jpg" alt="Search — semantic and AI-powered search with filters" width="800" />
  <br><em>Search — semantic and AI-powered find with type, role, agent, and category filters</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-memories.jpg" alt="Browse — memory list with full CRUD and artifact management" width="800" />
  <br><em>Browse — full CRUD with server-side sorting, inline editing, and artifact management</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-graph.jpg" alt="Graph — D3.js force-directed memory relationship visualization" width="800" />
  <br><em>Graph — D3.js force-directed visualization of memory relationships</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-check-progress.jpg" alt="Check — three-phase memory consistency scan with live progress" width="800" />
  <br><em>Check — three-phase memory consistency scan with live progress tracking</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-check-results.jpg" alt="Check — issue summary cards with severity and confidence filters" width="800" />
  <br><em>Check — issue summary with severity/confidence filters, grouped by type</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-check-detail.jpg" alt="Check — issue detail with affected memories and proposed actions" width="800" />
  <br><em>Check — issue detail with affected memories, metadata, and proposed fix actions</em>
</p>

### Features

- **Dashboard** — Memory totals, breakdowns by type/category/role (Chart.js donut charts), operation counts, per-user filtering, server version display, auto-refresh with manual refresh button
- **Search** — Semantic search (`search_memories`) and AI-powered find (`find_memories`) with filters: memory type, role, agent ID, categories multi-select, "has artifacts" toggle, include decayed, result limit. Sort results by score, newest, or oldest. Two-zone card layout with type/importance/category badges on the left, agent/artifacts/assistant/pinned indicators on the right
- **Browse** — List all memories with server-side sorting (newest/oldest via Qdrant `order_by`). Full filter bar: memory type, role, agent ID, categories multi-select, "has artifacts" toggle, include decayed. Add Memory button with modal form. Inline expand with edit modal (content, type, categories, importance, pinned, TTL), delete with confirmation, and artifact management (upload, view with pagination, delete)
- **Graph** — D3.js force-directed visualization of memory relationships (shared categories = edges, node size = importance, color = memory type). Type filter checkboxes, node limit slider, click-to-select detail panel
- **Check** — Memory consistency checker (fsck). Run a three-phase scan (security, duplicates, quality), review issues grouped by type with full reasoning and affected memory cards, select and apply fixes. Parallel LLM evaluation with live progress tracking

### Access

The UI is served at `http://localhost:8050/ui/`. Static files are exempt from API key authentication — API calls from the browser use the API key entered on the login screen.

**Login:** Enter your `MCP_API_KEY` or any key from `MCP_API_KEYS`. Wildcard keys (`*`) enable multi-user switching via a dropdown.

### Tech Stack

Alpine.js + Tailwind CSS + Chart.js + D3.js — all vendored as static files. Zero external requests at runtime, zero npm/node dependencies.

### Development

To modify the UI, edit files in `mnemory/ui/static/` (JS, HTML) or `mnemory/ui/src/input.css` (Tailwind). Rebuild CSS after changes:

```bash
# One-time: download Tailwind CLI (https://github.com/tailwindlabs/tailwindcss/releases)
# Then:
make ui-build    # Build minified CSS
make ui-watch    # Watch mode for development
```

## Monitoring

mnemory exposes a `/metrics` endpoint in [Prometheus text exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/), enabled by default (`ENABLE_METRICS=true`).

### Endpoint Access

By default, `/metrics` and `/health` are served on the main port with standard API key authentication. For production, set `MGMT_PORT` to serve them on a separate port without auth (see [Management Port](#management-port-mgmt_port)).

### Available Metrics

#### Counters (in-memory, reset on restart)

| Metric | Labels | Description |
|---|---|---|
| `mnemory_operations_total` | `operation`, `user_id`, `agent_id` | Total MCP/REST operations. Operations: `add_memory`, `add_memories`, `search_memories`, `find_memories`, `update_memory`, `delete_memory`, `delete_all`, `get_core_memories`, `get_recent_memories`, `list_memories`, `save_artifact`, `get_artifact`, `list_artifacts`, `delete_artifact`, `list_categories`, `recall`, `remember`, `initialize_memory`, `fsck_check`, `fsck_apply` |

#### Gauges (from Qdrant, cached)

Refreshed on each scrape with a configurable cache TTL (`METRICS_CACHE_TTL`, default 60s). Aggregated by scrolling all Qdrant points.

| Metric | Labels | Description |
|---|---|---|
| `mnemory_memories_total` | `user_id`, `agent_id`, `memory_type`, `role` | Total memories by all key dimensions |
| `mnemory_memories_decayed_total` | `user_id`, `agent_id` | Decayed (expired) memories |
| `mnemory_memories_pinned_total` | `user_id`, `agent_id` | Pinned memories |
| `mnemory_memories_by_category_total` | `user_id`, `category` | Memories per category |
| `mnemory_memories_with_artifacts_total` | `user_id`, `agent_id` | Memories with artifacts attached |
| `mnemory_active_sessions` | — | Active memory sessions (recall/remember) |
| `mnemory_info` | `version`, `vector_backend`, `artifact_backend` | Server metadata (always 1) |

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: mnemory
    scrape_interval: 60s
    static_configs:
      - targets: ['mnemory:9090']  # MGMT_PORT
```

### Grafana Dashboard

A pre-built Grafana dashboard is available in [`integrations/grafana/`](integrations/grafana/). Import `dashboard.json` into Grafana for an overview of memories, operations, and breakdowns by type/category/role with user and agent filtering.

## Plugins

Native plugins that automatically recall memories on each user message and store memories after each exchange — no LLM tool-calling required.

### Claude Code Plugin

A hooks-based integration that calls `/api/recall` on each user prompt and `/api/remember` after each response. Injects memories as additional context.

See [`integrations/claude-code/`](integrations/claude-code/) for the plugin and setup instructions.

### Open WebUI Filter

A filter function that calls `/api/recall` on inlet (before LLM) and `/api/remember` on outlet (after LLM). Injects memories into the system prompt automatically.

See [`integrations/openwebui/`](integrations/openwebui/) for the filter and setup instructions.

### OpenCode Plugin

A TypeScript plugin that hooks into session lifecycle for automatic recall/remember.

See [`integrations/opencode/`](integrations/opencode/) for the plugin and setup instructions.

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

# Run tests (unit only, fast, no API key needed)
pytest

# Run e2e tests (requires LLM_API_KEY or OPENAI_API_KEY, ~5 min)
pytest -m e2e -v

# Run all tests
pytest -m '' -v

# Lint
ruff check mnemory/
```

## License

Apache 2.0
