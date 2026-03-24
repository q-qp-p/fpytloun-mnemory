# REST API

mnemory exposes a full REST API alongside the MCP server. The FastAPI sub-app is mounted at `/api/` with auto-generated OpenAPI spec at `/api/openapi.json` and Swagger UI at `/api/docs`.

Both MCP and REST share the same `MemoryService` backend and authentication middleware.

## Memory CRUD

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
| `/api/memories/{id}/artifacts/{aid}` | GET | Get artifact (JSON with base64 for binary) |
| `/api/memories/{id}/artifacts/{aid}/download-token` | POST | Generate a signed download token |
| `/api/memories/{id}/artifacts/{aid}/raw` | GET | Download raw artifact bytes (token or API key auth) |
| `/api/memories/{id}/artifacts/{aid}` | DELETE | Delete artifact |
| `/api/categories` | GET | List categories |

## Session Summaries

Persistent session summaries from the remember endpoint, used by the consolidation service.

| Endpoint | Method | Description |
|---|---|---|
| `/api/sessions` | GET | List session summaries (optional `consolidation_state` filter) |
| `/api/sessions/{id}` | GET | Get a single session summary |
| `/api/sessions/{id}` | DELETE | Delete a session summary (optional `delete_memories=true` to also delete linked raw memories) |
| `/api/sessions/{id}/consolidate` | POST | Trigger consolidation for a specific session (returns result) |

## Memory Check (fsck)

Built-in memory consistency checker. Runs a three-phase pipeline to detect quality issues and suggest fixes:

| Endpoint | Method | Description |
|---|---|---|
| `/api/fsck` | POST | Start a memory check (runs in background) |
| `/api/fsck/{id}` | GET | Poll check status, progress, and results |
| `/api/fsck/{id}/apply` | POST | Apply selected fixes from a completed check |

### Three-Phase Pipeline

1. **Security scan** (instant) — regex-based detection of prompt injection patterns
2. **Duplicate detection** — vector similarity clustering + LLM evaluation of each cluster for duplicates and contradictions
3. **Quality check** — LLM batch evaluation for spelling/grammar errors, meaningless memories, memories that should be split, and metadata misclassification

Phases 2 and 3 run LLM calls in parallel (`FSCK_LLM_CONCURRENCY`, default 4 workers) for ~4x speedup on large memory sets. Results are cached with configurable TTL (`FSCK_CACHE_TTL`, default 24 hours) so you can review issues in the UI and apply fixes without re-running the check.

**Issue types:** `duplicate`, `quality`, `split`, `contradiction`, `reclassify`, `security`

**Workflow:** Start check -> poll until completed -> review issues in UI -> select and apply fixes. Each fix is a set of actions (update content, update metadata, delete memory, or add new memory).

## Intelligence Layer

Two high-level endpoints designed for plugin-driven automatic memory management:

### POST /api/recall

Combined initialize + search. Call on each user message.

```json
{
  "session_id": null,
  "query": "Should I buy a dog?",
  "include_instructions": true,
  "managed": true,
  "score_threshold": 0.5,
  "labels": {"project": "myapp"}
}
```

- First call (no `session_id`): creates session, returns instructions + core memories + `find_memories` results
- Subsequent calls: returns only NEW relevant memories via fast `search_memories` (no LLM), filtering out already-returned IDs
- `score_threshold`: optional per-request minimum score (0.0-1.0) for search results, applied on top of server's `SEARCH_SCORE_THRESHOLD`. Prevents context bloat from weak matches on follow-up messages.
- `labels`: optional label filter — only return memories matching these key-value pairs. See [Memory Model — Labels](memory-model.md#labels).
- Graceful degradation: `find_memories` fails -> `search_memories` -> core memories only -> empty response

### POST /api/remember

Fire-and-forget memory storage. Call after each exchange.

```json
{
  "session_id": "sess_abc123",
  "messages": [
    {"role": "user", "content": "I just moved to Berlin"},
    {"role": "assistant", "content": "That's exciting!"}
  ],
  "role": null,
  "labels": {"conversation_id": "conv-123"}
}
```

The `role` parameter controls the extraction point of view:

| Value | Behavior |
|---|---|
| `null` (default) | **Auto mode** — extracts facts from ALL participants. Each fact is attributed to the correct role (`user` or `assistant`). Assistant facts are silently dropped if no `agent_id` is set in the session. |
| `"user"` | Extracts only user facts, suppresses assistant content. |
| `"assistant"` | Extracts only assistant facts (identity, recommendations, conclusions). Requires `agent_id` via `X-Agent-Id` header. |

- Returns `{"accepted": true}` immediately, processes in background
- Reuses the same extraction pipeline as `add_memory(infer=True)` — no extra LLM calls
- `labels`: optional key-value metadata attached to all extracted memories. Bypasses LLM — stored as-is. See [Memory Model — Labels](memory-model.md#labels).
- Stored memory IDs are added to the session to prevent echo on next recall
- Rate limited per user (configurable via `REMEMBER_RATE_LIMIT`)

## Memory Sessions

The recall/remember endpoints use server-side sessions (`MemorySession`) to track which memories the client already has. This prevents context bloat by only returning new memories on subsequent calls.

- Sessions are created on first recall and expire after idle timeout (default 24 hours, configurable via `MEMORY_SESSION_TTL`)
- Both recall and remember calls reset the idle timer (auto-prolong), so active conversations never expire
- Sessions are persisted to a pluggable backend (SQLite by default, Redis for clustered deployments) and survive server restarts
- Losing a session is harmless — next recall creates a new one
- Periodic background sweep cleans up expired sessions from both the in-memory cache and the backend

### Session Persistence

Sessions use a write-through cache: an in-memory dict for fast reads, with all mutations written to the backend for durability. On cache miss (e.g., after restart), sessions are loaded lazily from the backend.

| Backend | Use case | Config |
|---|---|---|
| `sqlite` (default) | Single-node, local development, `uvx mnemory` | `SESSION_PATH` (default `~/.mnemory/sessions.db`) |
| `redis` | Clustered deployments, multiple replicas | `REDIS_URL` (e.g., `redis://host:6379/0`) |
| `memory` | Tests, no persistence needed | No additional config |

Auto-detection: if `REDIS_URL` is set and `SESSION_BACKEND` is not explicitly set, defaults to `redis`. See [Configuration](configuration.md) for all session env vars.

## Periodic Maintenance

mnemory includes a built-in background maintenance service that automatically runs memory consistency checks and applies fixes on a configurable schedule.

Enable it by setting `FSCK_AUTO_INTERVAL` to a non-zero number of hours:

```bash
FSCK_AUTO_INTERVAL=24          # Run every 24 hours
FSCK_AUTO_MIN_CONFIDENCE=0.95  # Only apply fixes with >=95% confidence
FSCK_AUTO_MIN_SEVERITY=medium  # Only apply medium or high severity fixes
```

**How it works:**

1. The maintenance loop sleeps for the configured interval, then wakes up
2. All users in the collection are enumerated via a Qdrant scroll
3. For each user, a full fsck check is run (security scan -> duplicate detection -> quality check)
4. Issues that meet both the confidence and severity thresholds are automatically applied
5. Results are recorded in Prometheus metrics (`mnemory_autofsck_*` counters + last-run timestamp)
6. Per-user errors are isolated — one failing user does not stop the run

The loop is **sleep-first**: it waits the full interval before the first run, so startup is not impacted. Auto-applied fixes are visible in the [management UI](management-ui.md) (Check tab status banner) and in the Grafana dashboard (Auto-fsck row).
