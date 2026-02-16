# TODO: Memory System Enhancements

This document describes two major features to be implemented in mnemory:
1. **TTL Support** — Dynamic memory expiration with decay and reinforcement
2. **Recall/Remember API** — High-level orchestration endpoints for intelligent memory retrieval and storage

---

## Background & Motivation

### The Problem

Currently, mnemory relies on LLMs calling MCP tools to manage memory. This has limitations:

1. **LLMs forget to search** — When a user asks "Should I buy a dog?", the LLM answers immediately without thinking to search for relevant context (past pets, living situation, preferences).

2. **No automatic memory capture** — LLMs must explicitly decide to call `add_memory`. They often miss important facts or store noise.

3. **All memories are permanent** — No concept of short-term vs long-term memory. Session context lives forever alongside core facts.

4. **Integration complexity** — Each client (Open WebUI, Claude Code, n8n) must implement their own orchestration logic.

### The Solution

Build an **intelligence layer** on top of mnemory that:

1. **Recall** (`/api/recall`) — Given any context, automatically retrieve relevant memories using query expansion, multi-query search, and smart filtering. Called at session start.

2. **Remember** (`/api/remember`) — Given any content, evaluate what's worth storing, assign appropriate TTL, and handle reinforcement of existing memories. Called async after each exchange.

3. **TTL with Decay** — Memories can expire over time but are reinforced when accessed or when similar content is remembered.

4. **MCP tools remain** — For mid-conversation use, LLMs continue to use MCP tools (`search_memories`, `add_memory`, etc.) for explicit memory operations.

### Intended Usage Pattern

```
1. SESSION START
   └─► Client calls POST /api/recall
       └─► Returns relevant memories as text
       └─► Client injects into system prompt

2. DURING CONVERSATION
   └─► LLM uses MCP tools as needed (search_memories, add_memory, etc.)
       └─► Explicit, LLM-driven memory operations

3. AFTER EACH EXCHANGE
   └─► Client calls POST /api/remember (fire-and-forget)
       └─► mnemory evaluates and stores worthy memories async
```

### Design Principles

- **mnemory owns the intelligence** — Clients are thin; all smart logic lives in mnemory
- **Versatile input** — Accept free text, conversation arrays, or any payload
- **One behavior, done well** — No modes or complexity; each endpoint does the best job it can
- **Human-like memory** — Inspired by cognitive science (cue-dependent retrieval, associative memory, decay with reinforcement)
- **Graceful degradation** — If task model fails, fall back to simpler approaches; never return errors for transient failures
- **Leverage mem0** — Use mem0 for what it's good at (fact extraction, deduplication, vector search); build orchestration on top

### What mem0 Provides (We Use)

| Feature | Description |
|---------|-------------|
| Fact extraction | LLM extracts structured facts from conversations |
| Deduplication | Detects and merges duplicate memories |
| Contradiction resolution | Updates conflicting facts (old → new) |
| Embedding + vector storage | Stores embeddings in Qdrant/Chroma |
| Semantic search | Vector similarity search |
| Metadata storage | Custom fields stored as flat Qdrant payload |

### What We Build on Top

| Feature | Description |
|---------|-------------|
| Query expansion | Generate multiple related queries from context via task model |
| Multi-query search | Run parallel searches, combine and deduplicate results |
| Category-aware retrieval | Use our category taxonomy to guide search |
| Core memories | Pinned + recent memories concept |
| Importance reranking | Boost by importance level |
| Memory evaluation | Decide if content is worth storing via task model |
| TTL / expiration | Time-based memory decay with reinforcement |
| Recall/Remember API | High-level orchestration endpoints |

---

## Task 1: TTL Support

### Overview

Implement dynamic TTL (Time-To-Live) for memories with decay and reinforcement mechanisms. This is a standalone feature that improves memory lifecycle management independent of the recall/remember API.

### Memory Lifecycle

```
Memory created (with or without TTL)
    │
    ▼
ACTIVE state
    │
    ├─── Has TTL? TTL counting down from expires_at
    │    │
    │    ├─── Accessed in recall/search → TTL reset (reinforcement)
    │    │
    │    ├─── Similar content remembered via /api/remember → TTL reset
    │    │
    │    └─── expires_at reached → DECAYED state
    │
    ├─── No TTL (permanent) → stays ACTIVE forever
    │
    └─── User deletes → DELETED state (hard delete)
```

### Memory States

| State | Description | In Recall? | In Search? | Restorable? |
|-------|-------------|------------|------------|-------------|
| **Active** | TTL not expired, or permanent (no TTL) | Yes | Yes | N/A |
| **Decayed** | TTL expired, soft-deleted | No (by default) | With `include_decayed` flag | Yes |
| **Deleted** | Hard-deleted by user | No | No | No |

### Metadata Fields

New fields added to memory metadata (stored as flat Qdrant payload fields alongside existing fields):

```python
{
    # Existing fields (unchanged)
    "memory_type": "fact",
    "categories": ["personal"],
    "importance": "normal",
    "pinned": False,
    "created_at_utc": "2024-01-15T10:30:00Z",

    # NEW: TTL fields
    "ttl_days": 30,                                # Original TTL setting (null = permanent)
    "expires_at": "2024-02-14T10:30:00Z",          # Calculated expiration (null = never). Full ISO 8601.
    "decayed_at": null,                             # When memory entered decayed state (null = active)
    "last_accessed_at": "2024-01-20T14:00:00Z",    # Last time returned in recall/search
    "access_count": 5,                              # Number of times accessed (for analytics)
}
```

### Decay Detection: Passive/Lazy

The codebase has no background task infrastructure. Decay is detected **at query time**:

1. When searching/listing memories, filter by `expires_at` using Qdrant's `DatetimeRange` filter (same pattern already used in `get_recent_memories`).
2. Memories with `expires_at < now` are excluded from results by default.
3. When an expired memory is encountered (e.g., via `include_decayed=true`), lazily set `decayed_at` if not already set.
4. No background jobs, no periodic scans.

For Chroma backend (which lacks date range filtering), fall back to post-filtering in Python after fetching results.

### Pinned Memories Are Exempt from TTL

If `pinned=true`, the memory is **exempt from TTL**:
- `expires_at` is ignored even if set
- The memory is always treated as active
- Rationale: Pinned memories are "always shown" in core memories — it would be contradictory for them to decay

### TTL Assignment Rules

Default TTL based on memory type (configurable via env):

| Memory Type | Default TTL | Rationale |
|-------------|-------------|-----------|
| `fact` | permanent | Core facts about user don't expire |
| `preference` | permanent | Preferences are long-term |
| `episodic` | 90 days | Events fade but stay longer |
| `procedural` | 60 days | Workflows may change |
| `context` | 7 days | Session context is short-term |

These defaults apply when `ttl_days` is not explicitly provided. The task model in `/api/remember` can override defaults with a specific TTL.

### Reinforcement Mechanism

Memory TTL is reset (extended) when:

1. **Accessed** — Memory is returned in `/api/recall` or `search_memories`. The `expires_at` is recalculated from now + original `ttl_days`. The `last_accessed_at` and `access_count` are updated.

2. **Reinforced** — Similar content is processed by `/api/remember` (see Task 2). Before storing a new memory, `/api/remember` searches for similar existing memories. If found, the existing memory is updated/restored instead of creating a duplicate.

**Important**: Reinforcement logic lives **only in `/api/remember`**, not in the base `add_memory`. The MCP tool `add_memory` stays fast and simple — it does not perform similarity searches before storing. This keeps the existing MCP tool behavior unchanged and avoids adding ~200ms latency to every `add_memory` call.

### Decay Behavior

- Decayed memories are **excluded from recall and search by default**
- Decayed memories **can be searched** with `include_decayed=true` parameter
- Decayed memories are **kept forever** (no hard delete) — useful for historical context ("what was I working on last year?")
- Decayed memories can be **manually restored** via `update_memory` (set new `ttl_days`, clear `decayed_at`)
- Decayed memories are **automatically restored** when similar content is remembered via `/api/remember`

### Existing Memories (Migration)

When TTL is deployed, existing memories have no TTL fields. **No migration is needed**:
- Missing `ttl_days` / `expires_at` → treated as permanent (no expiration)
- Missing `decayed_at` → treated as active
- Missing `last_accessed_at` / `access_count` → initialized on first access
- New fields are added lazily via `set_payload` when memories are updated

### mem0 Metadata Overwrite Risk

AGENTS.md notes: "mem0's public `update()` method drops custom metadata." When mem0 processes deduplication/contradiction resolution during `add()` with `infer=true`, it may overwrite TTL fields on existing memories via its internal update mechanism.

**Mitigation**: After any `add_memory` call that triggers mem0 inference, verify TTL fields are preserved on affected memories. If mem0 overwrites them, re-apply via Qdrant `set_payload`. This needs investigation during implementation — the risk may be limited to memories that mem0 decides to merge/update.

### Configuration

```bash
# Default TTL by memory type (days, null = permanent)
TTL_FACT=null
TTL_PREFERENCE=null
TTL_EPISODIC=90
TTL_PROCEDURAL=60
TTL_CONTEXT=7

# Access tracking
TRACK_MEMORY_ACCESS=true  # Update last_accessed_at on recall/search
```

### Implementation Steps

#### 1.1 TTL Module

- [ ] Create `mnemory/ttl.py` with TTL management functions:
  - `calculate_expiration(ttl_days: int | None, from_time: datetime | None = None) -> str | None`
  - `get_default_ttl(memory_type: str, config: MemoryConfig) -> int | None`
  - `is_expired(memory: dict) -> bool` — checks `expires_at < now`, respects `pinned` exemption
  - `is_decayed(memory: dict) -> bool` — checks `decayed_at` is set
  - `should_exclude(memory: dict, include_decayed: bool = False) -> bool` — combined check
  - `build_expiry_metadata(ttl_days: int | None, memory_type: str, config: MemoryConfig) -> dict` — returns TTL fields for new memory

#### 1.2 Schema Changes

- [ ] Update `MemoryService.add_memory` to:
  - Accept optional `ttl_days` parameter
  - Calculate `expires_at` from `ttl_days` (or default based on `memory_type`)
  - Include TTL fields in metadata dict passed to `VectorStore.add`
- [ ] Initialize `last_accessed_at` and `access_count` to `None` / `0`

#### 1.3 Filter Decayed Memories

- [ ] Update `VectorStore.search` to filter expired memories:
  - Qdrant backend: Add `DatetimeRange` filter on `expires_at` (exclude where `expires_at < now` AND `pinned != true`)
  - Chroma backend: Post-filter in Python
- [ ] Add `include_decayed: bool = False` parameter to:
  - `MemoryService.search_memories`
  - `MemoryService.search_memories_dual_scope`
  - `MemoryService.list_memories`
- [ ] Update `MemoryService.get_core_memories` to exclude decayed memories
- [ ] Lazily set `decayed_at` when expired memories are encountered via `include_decayed=true`

#### 1.4 Access Tracking

- [ ] Update `MemoryService.search_memories` to track access on returned memories:
  - Set `last_accessed_at` to current UTC timestamp
  - Increment `access_count`
  - Reset `expires_at` if memory has TTL (reinforcement)
- [ ] Run access tracking **asynchronously** to avoid adding latency:
  - Use Qdrant batch `set_payload` (single call for all returned memory IDs)
  - Fire-and-forget from a thread (don't block the response)
- [ ] Make tracking configurable via `TRACK_MEMORY_ACCESS` env var

#### 1.5 MCP Tool Updates

- [ ] Update `add_memory` tool to accept optional `ttl_days` parameter
- [ ] Update `add_memories` batch tool to accept `ttl_days` per item
- [ ] Update `search_memories` tool to accept `include_decayed` parameter
- [ ] Update `list_memories` tool to:
  - Accept `include_decayed` parameter
  - Show decay state in results (`is_decayed`, `expires_at`, `decayed_at`)
- [ ] Update `update_memory` tool to accept `ttl_days` (allows manual TTL change/restore)
- [ ] Update tool docstrings and `instructions.py`

#### 1.6 Configuration

- [ ] Add TTL config fields to `MemoryConfig` in `mnemory/config.py`:
  - `ttl_fact`, `ttl_preference`, `ttl_episodic`, `ttl_procedural`, `ttl_context`
  - `track_memory_access`
- [ ] Add environment variable parsing

#### 1.7 Tests

- [ ] Test TTL calculation and expiration detection
- [ ] Test pinned memory exemption from TTL
- [ ] Test decay state transitions (active → decayed)
- [ ] Test filtering of decayed memories in search
- [ ] Test filtering of decayed memories in core memories
- [ ] Test access tracking and TTL reset on access
- [ ] Test existing memories without TTL fields (treated as permanent)
- [ ] Test `add_memory` with explicit `ttl_days`
- [ ] Test `add_memory` with default TTL from memory type
- [ ] Test `include_decayed` parameter

#### 1.8 Documentation

- [ ] Update README with TTL feature description
- [ ] Update README configuration table
- [ ] Update AGENTS.md with TTL metadata fields
- [ ] Update `instructions.py` with TTL guidance for LLMs

---

## Task 2: Recall/Remember API

### Overview

Two new REST endpoints that provide high-level memory orchestration:

- **`POST /api/recall`** — Retrieve relevant memories for any context. Called at session start.
- **`POST /api/remember`** — Evaluate and store worthy memories from any content. Called async after each exchange.

These endpoints complement (not replace) the existing MCP tools. The intended split:
- **Recall/Remember**: Automatic, client-driven, runs at session boundaries
- **MCP tools**: Explicit, LLM-driven, runs during conversation when the LLM decides

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Intelligence Layer (NEW)                          │
│                                                                      │
│  POST /api/recall                 POST /api/remember                 │
│    │                                │                                │
│    ├─ Core memories (cached)        ├─ Evaluation (task model)       │
│    ├─ Query expansion (task model)  ├─ TTL assignment                │
│    ├─ Multi-query search (parallel) ├─ Reinforcement check           │
│    ├─ Deduplication + scoring       └─ Async storage via add_memory  │
│    └─ Formatting (text or JSON)                                      │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Uses existing services
┌──────────────────────────▼──────────────────────────────────────────┐
│                    Existing mnemory Layer                            │
│                                                                      │
│  MemoryService: add, search, update, delete, get_core_memories       │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                    mem0 (Storage Engine)                             │
│                                                                      │
│  Fact extraction, deduplication, embeddings, vector search           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Technical Constraints

The existing codebase is **entirely synchronous** — mem0, qdrant_client, openai, VectorStore, MemoryService are all blocking. The new endpoints need async capabilities for:

- **Task model calls** — Should not block the event loop
- **Parallel searches** — Multiple searches should run concurrently
- **Fire-and-forget** — Remember should return immediately

**Approach**:
- Use `openai.AsyncOpenAI` for task model calls (new pattern, only for recall/remember)
- Use `asyncio.get_event_loop().run_in_executor()` with `ThreadPoolExecutor` to run sync `VectorStore.search` calls in parallel
- Use Starlette `BackgroundTasks` for remember's fire-and-forget processing
- Keep existing MCP tools synchronous (no refactor needed)

---

### `POST /api/recall`

#### Purpose

Given any context (text, conversation, query), retrieve relevant memories using intelligent query expansion and multi-query search. Designed to be called once at session start to prime the LLM with relevant context.

#### Why Query Expansion?

Human memory works by **association**, not keyword search. When someone asks "Should I buy a dog?", relevant memories include:

- Direct: "User had a dog named Max who died in 2020"
- Associated: "User lives in a house with a garden"
- Contextual: "User mentioned dogs require too much time"
- Preference: "User's wife loves animals"

A single semantic search for "buy a dog" might miss the living situation and time concerns. A task model generates multiple related queries to capture the full associative context — similar to how Open WebUI uses a task model for RAG query generation.

#### Request

```json
{
  "context": "Should I buy a dog?",
  "messages": [
    {"role": "user", "content": "Should I buy a dog?"}
  ],
  "user_id": "filip",
  "agent_id": "open-webui"
}
```

Input flexibility:
- `context` (string): Any free text — a question, topic, document, anything
- `messages` (array): OpenAI-style conversation messages array
- If both provided: Combined (context + last user message from messages)
- If neither provided: Return only core memories (no search)

#### Messages Normalization

When `messages` is provided, extract text for processing:
- **For recall**: Use the last user message as primary context. If the conversation has multiple exchanges, include the last 2-3 user messages for broader context.
- **For remember**: Include the full last exchange (user + assistant). The task model needs both sides to evaluate what was learned.
- If `context` (free text) is also provided, prepend it.

#### Response Format

Negotiated via `Accept` header:

**`Accept: text/plain`** (default) — Ready for system prompt injection:
```
## About You
- Lives in Prague, Czech Republic
- DevOps engineer
- Prefers direct, concise communication

## Relevant Context
- Had a dog named Max who passed away in 2020
- Lives in a house with a small garden
- Mentioned dogs require too much time commitment
- Wife loves animals
```

**`Accept: application/json`** — For programmatic consumers:
```json
{
  "core_memories": [
    {"id": "mem-1", "content": "Lives in Prague, Czech Republic", "type": "fact", "importance": "high"}
  ],
  "search_results": [
    {"id": "mem-5", "content": "Had a dog named Max who passed away in 2020", "type": "episodic", "score": 0.89}
  ],
  "stats": {
    "core_count": 3,
    "search_count": 4,
    "queries": ["dog pets animals", "living situation home garden", "time commitment lifestyle"],
    "latency_ms": 450
  }
}
```

#### Query Expansion (Task Model)

Prompt:

```
You are a memory retrieval assistant. Given a user's message, generate search
queries to find relevant personal memories about them.

Think like a human remembering: what associations would this topic trigger?

User's message: "{context}"

Available memory categories: {categories}

Generate search queries that cover:
1. DIRECT concepts mentioned in the message
2. ASSOCIATED concepts that relate to the topic
3. CONTEXTUAL concepts about the user's situation that might be relevant

Output JSON:
{
  "queries": [
    {"text": "keywords for search", "weight": 1.0},
    {"text": "associated concepts", "weight": 0.8}
  ],
  "categories": ["relevant", "categories"],
  "skip_search": false
}

Rules:
- Max 4 queries, 3-6 words each
- Higher weight = more important query
- Set skip_search=true ONLY for pure greetings ("hello", "hi") or completely
  impersonal factual questions ("what is 2+2") that cannot benefit from
  personal memory
- When in doubt, search — it's better to search and find nothing than to skip
  and miss relevant context
```

#### Processing Flow

```
1. Parse input
   └─ Normalize context/messages to text

2. Fetch core memories (parallel with step 3)
   └─ Call existing get_core_memories (cached via CORE_MEMORIES_CACHE_TTL)

3. Query expansion (parallel with step 2)
   └─ Call task model with AsyncOpenAI
   └─ Parse JSON response
   └─ On failure: fall back to using raw context as single query

4. Multi-query search
   └─ For each query: run VectorStore.search in thread pool (parallel)
   └─ Apply category filters from task model output
   └─ Collect all results

5. Combine results
   └─ Deduplicate by memory ID (keep highest score)
   └─ Boost memories appearing in multiple queries
   └─ Apply importance reranking (existing logic)
   └─ Deduplicate against core memories (remove search results already in core)
   └─ Limit to RECALL_MAX_RESULTS

6. Format response
   └─ Based on Accept header: text/plain or application/json
   └─ Combine core memories + search results

7. Track access (fire-and-forget)
   └─ Update last_accessed_at on returned memories
   └─ Reset TTL on accessed memories (reinforcement)
```

#### Failure Handling

The recall endpoint should **never fail** for transient errors. Fallback chain:

1. Task model fails → Fall back to raw context as single search query (no expansion)
2. Search fails → Return only core memories
3. Core memories fail → Return empty text/JSON with error flag
4. Always return HTTP 200 with whatever we could retrieve

Only return HTTP 4xx/5xx for:
- 401: Authentication failure
- 400: Missing both `context` and `messages`
- 500: Catastrophic internal error (should be rare)

---

### `POST /api/remember`

#### Purpose

Given any content (text, conversation), evaluate what's worth remembering and store it with appropriate TTL. Fire-and-forget — returns immediately, processes asynchronously.

#### Why Evaluation?

Not everything should be stored:
- "Hello!" — Not worth remembering
- "What's 2+2?" — Generic question, no personal info
- "I just moved to Berlin" — Worth remembering (personal fact, permanent)
- "I'm working on project X this week" — Worth remembering (short-term context, TTL 30d)
- "The weather is nice today" — Not worth remembering

The task model evaluates content and extracts structured memories with appropriate metadata and TTL.

#### Request

```json
{
  "content": "I just moved to Berlin and I'm looking for a new apartment near Kreuzberg",
  "messages": [
    {"role": "user", "content": "I just moved to Berlin!"},
    {"role": "assistant", "content": "That's exciting! How do you like it so far?"},
    {"role": "user", "content": "It's great, looking for an apartment near Kreuzberg"}
  ],
  "user_id": "filip",
  "agent_id": "open-webui"
}
```

Input flexibility:
- `content` (string): Any free text
- `messages` (array): OpenAI-style conversation messages
- If both provided: Combined for evaluation
- At least one must be provided

#### Response

Immediate (before async processing):

```json
{
  "accepted": true
}
```

Fire-and-forget. The caller does not wait for storage to complete.

#### Memory Evaluation (Task Model)

Prompt:

```
You are a memory evaluation assistant. Given content from a conversation,
decide what should be remembered about the user.

Content:
{content}

Recently stored memories (to avoid duplicates):
{recent_memories_summary}

Available categories: {categories}

Memory types and typical TTL:
- fact: Personal facts (location, job, family) — typically permanent
- preference: Likes, dislikes, style choices — typically permanent
- episodic: Events, interactions, conclusions — medium-term (30-90 days)
- procedural: Workflows, habits, how-to — medium-term (30-60 days)
- context: Session/task context — short-term (7 days)

Evaluate what's worth remembering. Consider:
- Is this new information or already known?
- Is this personal/specific or generic?
- How long is this likely to be relevant?

Output JSON:
{
  "memories": [
    {
      "content": "concise fact to remember",
      "type": "fact|preference|episodic|procedural|context",
      "categories": ["category1"],
      "importance": "low|normal|high|critical",
      "ttl_days": null|7|30|60|90
    }
  ]
}

Rules:
- Be selective — quality over quantity, don't store noise
- ttl_days: null = permanent. Use null for core facts and preferences.
- Don't set permanent TTL for clearly temporary information (current projects,
  tasks, short-term plans)
- Don't set short TTL for core facts that should persist (location, job, family)
- If nothing worth remembering: {"memories": []}
```

#### Processing Flow (Async)

```
1. Parse input
   └─ Normalize content/messages to text

2. Return immediately
   └─ {"accepted": true}

3. Background processing (Starlette BackgroundTasks):

   a. Evaluate (task model)
      └─ Fetch recent memories summary for dedup context (last 10 memories)
      └─ Call task model with AsyncOpenAI
      └─ Parse JSON response
      └─ On failure: log warning, skip (don't store anything on eval failure)

   b. For each extracted memory:

      i. Reinforcement check
         └─ Search for similar existing memory (threshold: REINFORCEMENT_SIMILARITY_THRESHOLD)
         └─ If similar ACTIVE memory found:
            └─ Update content if meaningfully different
            └─ Reset TTL
            └─ Skip creating new memory
         └─ If similar DECAYED memory found:
            └─ Restore from decay (clear decayed_at)
            └─ Update content
            └─ Reset TTL
            └─ Skip creating new memory

      ii. Store new memory
          └─ Call existing MemoryService.add_memory
          └─ Pass type, categories, importance, ttl_days from evaluation
          └─ Use infer=false (task model already extracted the fact)

   c. Log results
      └─ Log what was stored/updated/skipped for debugging
```

#### Failure Handling

Since remember is fire-and-forget, failures are logged but never returned to the caller:

1. Task model fails → Log warning, skip entirely (don't store noise)
2. Reinforcement search fails → Log warning, proceed to store as new memory
3. Storage fails → Log error with memory content for debugging
4. Never crash the background task — catch all exceptions

---

### Authentication

Same as existing MCP endpoints — the `APIKeyMiddleware` already wraps all Starlette routes:

- Bearer token (`Authorization: Bearer <key>`) or `X-API-Key` header
- `MCP_API_KEYS` mapping for user_id resolution (non-wildcard key → bound user_id)
- `X-User-Id` header as fallback
- `X-Agent-Id` header for agent_id
- `user_id` / `agent_id` in request body as final fallback

Priority: API key mapping > headers > request body (same as MCP tools).

### Configuration

```bash
# Task model for recall/remember (fast, cheap model recommended)
# Falls back to LLM_MODEL / LLM_BASE_URL / LLM_API_KEY if not set
TASK_MODEL=gpt-4o-mini
TASK_MODEL_BASE_URL=https://api.openai.com/v1
TASK_MODEL_API_KEY=

# Recall settings
RECALL_MAX_QUERIES=4          # Max queries from expansion
RECALL_MAX_RESULTS=10         # Max search results to return
RECALL_CACHE_TTL=60           # Cache identical recall requests (seconds)

# Remember settings
REMEMBER_RATE_LIMIT=10        # Max requests per minute per user (0 = no limit)

# Reinforcement (used by /api/remember)
REINFORCEMENT_SIMILARITY_THRESHOLD=0.85  # Similarity threshold for finding existing memories
```

### Implementation Steps

#### 2.1 Async Infrastructure

- [ ] Add `openai` async client support:
  - Create `AsyncOpenAI` client factory (similar to existing `_get_openai_client` in classify.py)
  - Configure with `TASK_MODEL` / `TASK_MODEL_BASE_URL` / `TASK_MODEL_API_KEY`
- [ ] Add `ThreadPoolExecutor` for running sync VectorStore operations in parallel
- [ ] Verify Starlette `BackgroundTasks` works with our middleware (contextvars propagation)

#### 2.2 File Structure

Create new files:
- [ ] `mnemory/api.py` — Starlette REST routes for `/api/recall` and `/api/remember`
- [ ] `mnemory/recall.py` — Recall orchestration logic
- [ ] `mnemory/remember.py` — Remember orchestration logic

#### 2.3 Recall Implementation

- [ ] Create `RecallService` class in `mnemory/recall.py`:
  - `async def recall(context, messages, user_id, agent_id) -> dict`
  - `def _normalize_input(context, messages) -> str`
  - `async def _expand_query(text, categories) -> dict`
  - `async def _multi_query_search(queries, user_id, agent_id, categories) -> list`
  - `def _combine_results(core_memories, search_results, max_results) -> list`
  - `def _deduplicate(core_ids, search_results) -> list`
  - `def _format_text(core_memories, search_results) -> str`
  - `def _format_json(core_memories, search_results, stats) -> dict`

- [ ] Implement query expansion:
  - Task model prompt (see above)
  - Parse JSON response with validation
  - Handle `skip_search=true` (return core memories only)
  - On failure: log warning, fall back to raw context as single query

- [ ] Implement multi-query search:
  - Run searches in parallel via `run_in_executor` + `asyncio.gather`
  - Pass category filters from task model output
  - Deduplicate by memory ID across queries
  - Score: `original_score * query_weight`, boost for multi-query hits
  - Apply existing importance reranking

- [ ] Implement result formatting:
  - Deduplicate search results against core memories
  - Format based on `Accept` header
  - Include stats in JSON format

#### 2.4 Remember Implementation

- [ ] Create `RememberService` class in `mnemory/remember.py`:
  - `async def remember(content, messages, user_id, agent_id) -> None` (runs in background)
  - `def _normalize_input(content, messages) -> str`
  - `async def _evaluate(text, recent_memories) -> list[dict]`
  - `def _check_reinforcement(content, user_id, threshold) -> dict | None`
  - `def _store_or_reinforce(memory, user_id, agent_id) -> None`

- [ ] Implement evaluation:
  - Fetch recent memories summary (last 10, for dedup context in prompt)
  - Task model prompt (see above)
  - Parse JSON response with validation
  - On failure: log warning, skip entirely

- [ ] Implement reinforcement:
  - Search for similar existing memory using `VectorStore.search`
  - Compare similarity against `REINFORCEMENT_SIMILARITY_THRESHOLD`
  - If found active: update content + reset TTL
  - If found decayed: restore + update + reset TTL
  - If not found: create new via `add_memory(infer=false)`

- [ ] Implement rate limiting:
  - In-memory counter per user_id (simple, sufficient for now)
  - Return 429 if over `REMEMBER_RATE_LIMIT` per minute
  - Rate limit checked before accepting (before returning 200)

#### 2.5 API Routes

- [ ] Create Starlette routes in `mnemory/api.py`:
  - `POST /api/recall` — async handler, calls RecallService
  - `POST /api/remember` — async handler, adds to BackgroundTasks, returns immediately
- [ ] Add request validation (at least one of context/messages required)
- [ ] Add `Accept` header parsing for recall response format
- [ ] Add error handling (try/except returning JSON error responses)
- [ ] Mount routes in `mnemory/server.py` `create_app()`:
  ```python
  routes=[
      Route("/health", health_check),
      Route("/api/recall", api_recall, methods=["POST"]),
      Route("/api/remember", api_remember, methods=["POST"]),
      Mount("/", app=mcp.streamable_http_app()),
  ]
  ```

#### 2.6 Configuration

- [ ] Add `TaskModelConfig` dataclass to `mnemory/config.py`:
  - `model`, `base_url`, `api_key` (with fallbacks to LLM config)
- [ ] Add recall/remember config fields to `MemoryConfig`:
  - `recall_max_queries`, `recall_max_results`, `recall_cache_ttl`
  - `remember_rate_limit`
  - `reinforcement_similarity_threshold`
- [ ] Add environment variable parsing

#### 2.7 Tests

- [ ] Test recall with text context
- [ ] Test recall with messages array
- [ ] Test recall with both context and messages
- [ ] Test recall with neither (core memories only)
- [ ] Test recall query expansion (mock task model)
- [ ] Test recall query expansion failure (fallback to raw query)
- [ ] Test recall multi-query search and deduplication
- [ ] Test recall core + search deduplication
- [ ] Test recall text format response
- [ ] Test recall JSON format response
- [ ] Test remember with text content
- [ ] Test remember with messages array
- [ ] Test remember evaluation (mock task model)
- [ ] Test remember evaluation failure (nothing stored)
- [ ] Test remember reinforcement (similar active memory found)
- [ ] Test remember reinforcement (similar decayed memory restored)
- [ ] Test remember new memory creation
- [ ] Test remember rate limiting
- [ ] Test authentication on both endpoints

#### 2.8 Documentation

- [ ] Update README with new endpoints
- [ ] Document configuration options
- [ ] Add usage examples (curl, Python)
- [ ] Update AGENTS.md with new files and architecture

---

## Future Work (Not in Scope)

### Open WebUI Filter Function

After recall/remember API is stable, create a thin Open WebUI filter function:

```python
class Filter:
    class Valves(BaseModel):
        mnemory_url: str = "http://localhost:8050"
        mnemory_api_key: str = ""

    async def inlet(self, body, __user__, __event_emitter__):
        # Call POST /api/recall with last user message
        # Inject returned memories into system prompt
        pass

    async def outlet(self, body, __user__, __event_emitter__):
        # Call POST /api/remember with last exchange (fire-and-forget)
        pass
```

### Multi-Hop Retrieval (v2)

Found memories trigger additional searches:
- Found: "User had a dog named Max"
- Triggers: Search for "Max" to find more context about Max

### Graph Memory (v2)

Build relationship graph between memories:
- "User" → "lives in" → "Prague"
- "User" → "had pet" → "Max" → "was a" → "dog"

### Memory Importance Decay (v2)

Importance decreases over time unless reinforced:
- Critical → High → Normal → Low → Decayed

---

## Dependencies

### Task 1 (TTL) Dependencies
- None (builds on existing mnemory)

### Task 2 (Recall/Remember) Dependencies
- Task 1 (TTL) — For TTL assignment in remember and TTL reset on access in recall
- `openai` package with async support (already a dependency)

---

## Testing Strategy

### Unit Tests
- TTL calculation, expiration detection, pinned exemption
- Decay state management
- Query expansion prompt building and response parsing
- Messages normalization
- Result combination and deduplication logic
- Rate limiting

### Integration Tests
- Full recall flow with mocked task model and mocked VectorStore
- Full remember flow with mocked task model and mocked MemoryService
- Reinforcement with mocked vector search
- Access tracking with mocked VectorStore
- Authentication and user_id resolution on REST endpoints

### End-to-End Tests (require API key)
- Recall with real task model
- Remember with real task model
- Full cycle: remember → recall → verify memory appears
