# Changelog

## [Unreleased]

## [1.8.0] — 2026-03-14

### Client-Provided Labels

- **Labels metadata on memories**: Memories now support optional key-value labels (e.g., `project`, `topic`, `conversation_id`) that bypass LLM extraction and are stored exactly as provided. Labels are inherited by all facts extracted during `infer=True`. Pass `labels` to `add_memory`, `add_memories`, `remember`, and `update_memory`. Filter by labels in `search_memories`, `find_memories`, `ask_memories`, and `list_memories` with AND logic (all must match). List values use any-of matching within a single key ([`18d18b1`](https://github.com/fpytloun/mnemory/commit/18d18b1), [`fa7e566`](https://github.com/fpytloun/mnemory/commit/fa7e566))
- **Labels in REST API**: All REST endpoints (add, search, find, ask, list, update, remember, recall) accept and return labels. Pydantic schemas updated ([`18d18b1`](https://github.com/fpytloun/mnemory/commit/18d18b1), [`fa7e566`](https://github.com/fpytloun/mnemory/commit/fa7e566))
- **Labels in UI**: Management UI supports labels input (JSON), display as badges, client-side filtering, and edit modal support ([`18d18b1`](https://github.com/fpytloun/mnemory/commit/18d18b1))
- **Labels validation**: Keys must be alphanumeric + underscore, starting with letter or underscore. Values can be str, int, float, bool, or list[str]. Max 20 labels per memory (configurable). Reserved keys (memory_type, user_id, etc.) are rejected ([`fa7e566`](https://github.com/fpytloun/mnemory/commit/fa7e566))
- **Labels config**: New env vars `LABELS_MAX_FIELDS` (default 20), `LABELS_MAX_KEY_LENGTH` (default 64), `LABELS_MAX_VALUE_LENGTH` (default 256), `LABELS_INDEXES` (optional Qdrant payload indexes for label keys) ([`fa7e566`](https://github.com/fpytloun/mnemory/commit/fa7e566))

### Prometheus Timing Instrumentation

- **LLM, embedding, and Qdrant latency histograms**: 3 new Prometheus histograms (`mnemory_llm_duration_seconds`, `mnemory_embedding_duration_seconds`, `mnemory_qdrant_duration_seconds`) with operation labels. All 13 LLM call sites, embedding operations, sparse embedding, and all Qdrant vector operations are instrumented ([`937d112`](https://github.com/fpytloun/mnemory/commit/937d112))
- **find_memories pipeline timing**: Per-step timing breakdown (query_gen, embed, search, rerank) logged at INFO level ([`937d112`](https://github.com/fpytloun/mnemory/commit/937d112))
- **Grafana dashboard**: 6 new latency panels (LLM p95 by operation, call rate, embedding p95, Qdrant p95, LLM p50 vs p95 by model) ([`937d112`](https://github.com/fpytloun/mnemory/commit/937d112))

### Token Usage Optimization

- **Compact MCP tool docstrings**: Reduced tool description overhead from ~5,500 to ~1,500 tokens per request by moving detailed parameter guidance from docstrings to server instructions. Estimated savings: ~4,000 prompt tokens per LLM request ([`79818c6`](https://github.com/fpytloun/mnemory/commit/79818c6))
- **`include_instructions` flag on `initialize_memory`**: Set to `false` when the client already has instructions via MCP instructions field, avoiding duplicate instructions in context ([`79818c6`](https://github.com/fpytloun/mnemory/commit/79818c6))
- **OpenWebUI filter: strip redundant MCP tools**: Filter now strips `initialize_memory`, `get_core_memories`, `get_recent_memories` from tool_ids (saving ~800 tokens) and injects memory context before the last user message for better prompt caching ([`41f7e8f`](https://github.com/fpytloun/mnemory/commit/41f7e8f))

### Bug Fixes

- **Sub-agent memories now visible in MCP search tools**: The `search_memories`, `find_memories`, `ask_memories`, and `list_memories` MCP tools silently ignored the `agent_id` tool parameter when `X-Agent-Id` header was set, making sub-agent memories (e.g., `openwebui:miroslav`) invisible in search results. Now properly resolves the `agent_id` parameter, honoring sub-agent scope while blocking cross-agent access ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))

### Performance

- **Eliminated redundant embedding API call in dual-scope search**: `search_memories_dual_scope()` now pre-computes the dense embedding once and reuses it across both sub-searches, halving embedding latency for dual-scope searches ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))

### Observability

- **Granular search mode tracking**: Vector store tracks `hybrid`, `dense-formula`, and `dense-simple` modes, logged at INFO level with duration and result count ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))
- **Search summary logging**: Both search methods log mode, threshold, result count, and filtered count at INFO level ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))
- **Warnings for dense-only fallback**: WARNING-level logs when sparse vector is unavailable or hybrid search fails ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))

### Documentation

- **Corrected RRF score range**: Qdrant's default RRF k=1 produces scores ~0.1-1.0, not ~0.01-0.03 as previously documented for k=60 ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))

## [1.7.0] — 2026-03-05

### Secure Artifact Download Tokens

- **HMAC-signed download tokens**: Replaced long-lived API keys in URLs with short-lived, stateless HMAC-signed tokens for artifact raw access. Tokens are scoped to a specific artifact, expire after a configurable TTL (default 1 hour, max 24 hours), and are verified without server-side storage. Browser-embedded `<img src="...?token=...">` and download links now use these tokens instead of API keys ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **New MCP tool `get_artifact_url`**: Generates a signed download URL for direct browser access to artifacts. Returns URL, expiry, content type, filename, and size. Use when artifacts are binary or larger than 1 MB ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **New REST endpoint `POST /api/memories/{id}/artifacts/{aid}/download-token`**: Authenticated endpoint that generates a download token with optional custom TTL. Returns the token and a ready-to-use URL ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **1 MB binary inline cap on `get_artifact`**: Binary artifacts larger than 1 MB are no longer returned inline via the MCP `get_artifact` tool. Instead, a guidance message directs the client to use `get_artifact_url`. Text artifacts are unaffected ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **Removed `key` query parameter**: The `key` query parameter for passing API keys in URLs has been removed from the auth middleware. All browser-embedded artifact access must use download tokens ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **New config vars**: `SERVER_BASE_URL` (for full URL generation), `DOWNLOAD_TOKEN_TTL` (default 3600s), `DOWNLOAD_TOKEN_MAX_TTL` (default 86400s) ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))

### Binary Artifact Support

- **Fixed binary artifact retrieval**: Binary artifacts (images, PDFs, etc.) now return full content as a single base64 blob instead of being truncated by text pagination. Added `load_raw()` method for direct byte access and `GET /api/memories/{id}/artifacts/{aid}/raw` endpoint for browser rendering ([`ff89fc7`](https://github.com/fpytloun/mnemory/commit/ff89fc7))
- **Increased max artifact size to 10 MB**: `MAX_ARTIFACT_SIZE` default raised from 100 KB to 10 MB (10485760 bytes) ([`ff89fc7`](https://github.com/fpytloun/mnemory/commit/ff89fc7))
- **UI binary rendering**: Management UI now detects binary artifacts — images render inline via `<img>`, other binary types show download links. Text artifacts retain paginated display ([`ff89fc7`](https://github.com/fpytloun/mnemory/commit/ff89fc7))

### New MCP Tool: ask_memories

- **`ask_memories` tool**: Ask a natural language question and get a human-readable prose answer synthesized from stored memories. Uses `find_memories` internally for retrieval, then passes results to an LLM for answer generation. Set `include_memories=true` to also receive the supporting memories. Most expensive operation (3 LLM calls) — use when you need a synthesized answer rather than raw results ([`db2fd19`](https://github.com/fpytloun/mnemory/commit/db2fd19))

### Performance

- **Parallelized find/ask search queries**: `find_memories` and `ask_memories` now run generated search queries concurrently using `ThreadPoolExecutor` instead of sequentially, reducing latency by ~400–1200 ms depending on query count ([`407926d`](https://github.com/fpytloun/mnemory/commit/407926d))
- **Configurable find/ask LLM model**: New `FIND_LLM_MODEL` env var allows using a faster/cheaper model for query generation and reranking in the find/ask pipeline ([`407926d`](https://github.com/fpytloun/mnemory/commit/407926d))

### Bug Fixes

- **Core memories no longer hard-truncated**: Replaced hard character truncation with per-section entry limits (`CORE_MAX_PER_SECTION`). Main sections are never truncated — only the recent context section is gracefully trimmed per-entry. Improved agent memory guidance in core memories output ([`856a564`](https://github.com/fpytloun/mnemory/commit/856a564))
- **Code analysis no longer misclassified as permanent facts**: Extraction prompts updated to prevent transient technical observations and code analysis conclusions from being stored as permanent `fact` type memories ([`1c51431`](https://github.com/fpytloun/mnemory/commit/1c51431))
- **Session-specific assistant memories filtered from extraction**: Extraction prompts now filter out session-specific assistant memories (e.g., "I concluded X in this session") that should not persist across conversations ([`7ec1b3d`](https://github.com/fpytloun/mnemory/commit/7ec1b3d))

## [1.6.1] — 2026-03-04

### Bug Fixes

- **Episodic memories now always have `event_date`**: Episodic memories (decisions, intents, goals, interactions) created via the remember endpoint or add_memory were stored without `event_date` because extraction prompts instructed the LLM to set null for facts without explicit temporal anchors. Updated all 5 extraction prompt sections and ~18 examples to clarify that episodic events should default to today's date. Added a code-level fallback in `_execute_action()` that auto-sets `event_date` to today (UTC) for episodic memories when neither the LLM nor the caller provides one ([`f2d038f`](https://github.com/fpytloun/mnemory/commit/f2d038f))

## [1.6.0] — 2026-03-04

### Hybrid Search

- **BM25 sparse vectors + RRF fusion**: Search now uses Qdrant's native hybrid search — dense vectors (semantic similarity) and BM25 sparse vectors (keyword matching) fused with Reciprocal Rank Fusion (RRF). Replaces the old Python-side keyword boost post-processing. Results are more accurate for queries containing specific terms, names, or identifiers ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **fastembed is now a required dependency**: `fastembed>=0.4.0` moved from optional to main dependencies. The BM25 model (~50MB) is pre-downloaded in the Docker image at build time ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **Importance reranking**: Post-RRF importance boost applied in Python (weighted by `IMPORTANCE_WEIGHTS`), replacing the previous `FormulaQuery` approach which is unreliable inside Qdrant Prefetch ([qdrant/qdrant#6836](https://github.com/qdrant/qdrant/issues/6836)) ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **New config vars**: `SEARCH_SPARSE_MODEL` (default `Qdrant/bm25`), `SEARCH_SCORE_THRESHOLD_HYBRID` (default `0.0` — RRF scores are ~0.016–0.033, not 0–1 like cosine) ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **Deprecated**: `SEARCH_KEYWORD_WEIGHT` is no longer used (keyword matching is now handled by BM25 sparse vectors natively in Qdrant)
- **Dense-only fallback**: If hybrid search fails at query time (e.g., Qdrant version mismatch), the system gracefully falls back to dense-only search ([`3199a47`](https://github.com/fpytloun/mnemory/commit/3199a47))
- **Python 3.14 not supported**: fastembed's BM25 model depends on `py_rust_stemmers`, a Rust extension that segfaults on Python 3.14 due to a PyO3/C API incompatibility ([qdrant/fastembed#576](https://github.com/qdrant/fastembed/issues/576)). Use Python 3.11–3.13 (3.13 recommended). The Docker image uses `python:3.13-slim` ([`9f2b920`](https://github.com/fpytloun/mnemory/commit/9f2b920))

### Data Migrations

- **Automatic migration framework**: New `MigrationRunner` in `mnemory/migration.py` — idempotent, resumable with batch checkpointing, state tracked in `_mnemory_meta` Qdrant collection. Health endpoint returns 503 during migration. Existing instances get sparse vectors added via `001_add_sparse_vectors` on first startup. **Note: migration may take several minutes for large instances** (processes all existing memories in batches of 100) ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **Collection recreation fallback**: Migration handles Qdrant rejecting sparse vector addition by recreating the collection with correct config ([`4b4e7a7`](https://github.com/fpytloun/mnemory/commit/4b4e7a7))

### Session Persistence

- **Persistent session storage**: Sessions now survive server restarts via pluggable backends — SQLite (default for local/single-node), Redis (for clustered deployments), or in-memory (tests). Uses a write-through cache pattern: in-memory dict for fast reads, backend for durability, lazy loading on cache miss ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))
- **24-hour session TTL**: Default `MEMORY_SESSION_TTL` increased from 1 hour to 24 hours. Both recall and remember calls reset the idle timer (auto-prolong), so active conversations never expire ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))
- **Remember auto-prolong**: The remember endpoint now touches the session to reset its idle timeout, preventing session expiry during active conversations that only use remember (not recall) ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))
- **Redis optional dependency**: Added `redis>=5.0.0` as optional dependency, installable via `pip install mnemory[redis]` ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))
- **New env vars**: `SESSION_BACKEND` (sqlite/redis/memory), `SESSION_PATH` (SQLite DB path), `REDIS_URL` (Redis connection). Auto-detects Redis when `REDIS_URL` is set ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))

### Remember Pipeline

- **Two-stage extract+dedup pipeline**: Remember pipeline rewritten to separate fact extraction from deduplication, with session context tracking for conversation continuity across multiple remember calls ([`b089a48`](https://github.com/fpytloun/mnemory/commit/b089a48))
- **Three-way role semantics with auto mode**: Remember pipeline now supports `role=None` (auto mode) that extracts facts from all conversation participants, attributing each to the correct role. Assistant facts require `agent_id` ([`521b8ba`](https://github.com/fpytloun/mnemory/commit/521b8ba))

### Role Handling

- **Role extraction fix for `role=assistant`**: Agent memories with `infer=True` now correctly use the agent extraction prompt ([`ff25699`](https://github.com/fpytloun/mnemory/commit/ff25699))
- **Role metadata written on UPDATE actions**: Dedup UPDATE actions now correctly persist the role field in metadata ([`942568a`](https://github.com/fpytloun/mnemory/commit/942568a))

### Core Memories

- **Dedup core memories against search results**: `get_core_memories` results are deduplicated against search results, and recent memories are filtered by importance ([`969b786`](https://github.com/fpytloun/mnemory/commit/969b786))

### Fsck Improvements

- **Detect and fix wrong role metadata**: Quality phase now detects memories with incorrect `role` field and proposes fixes ([`23bc415`](https://github.com/fpytloun/mnemory/commit/23bc415))

### Prompt & Extraction Improvements

- **Softer dedup language**: Reduced over-suppression of new memories by softening dedup prompt guidance ([`ea3407e`](https://github.com/fpytloun/mnemory/commit/ea3407e))
- **Restored inline examples**: Re-added extraction examples and rules that were dropped during the remember pipeline rewrite ([`bbf5f8d`](https://github.com/fpytloun/mnemory/commit/bbf5f8d))
- **Improved extraction reliability**: Better remember extraction reliability and classification ([`b378182`](https://github.com/fpytloun/mnemory/commit/b378182))

## [1.5.0] — 2026-02-28

### Batch Operations

- **Batch delete API and MCP tool**: New `delete_memories` MCP tool and `DELETE /api/memories/batch` endpoint for deleting multiple memories in a single call ([`94698a9`](https://github.com/fpytloun/mnemory/commit/94698a9))
- **Bulk selection in UI**: Browse tab now supports select-all, shift-click range selection, and batch delete for managing large memory sets ([`c9637c9`](https://github.com/fpytloun/mnemory/commit/c9637c9))

### Memory Intelligence

- **Context hint in remember pipeline**: Plugins can pass a context hint (e.g., working directory, current topic) to the remember endpoint, improving extraction relevance for project-scoped work ([`7a00f72`](https://github.com/fpytloun/mnemory/commit/7a00f72))
- **Dual-scope deduplication**: Agent-scoped memories are now deduplicated against both agent-specific and shared user memories, preventing cross-scope duplicates ([`9220a10`](https://github.com/fpytloun/mnemory/commit/9220a10))
- **Improved memory_type classification**: Better prompt guidance for classifying goals, intents, and decisions — goals/intents are now correctly classified as `episodic` rather than `fact`, with expanded e2e test coverage ([`d6c3b8d`](https://github.com/fpytloun/mnemory/commit/d6c3b8d))
- **Clarified extraction guidance**: Fact-extraction prompts refined to better distinguish stable facts from transient observations ([`be30809`](https://github.com/fpytloun/mnemory/commit/be30809))

### Fsck / Health Checks

- **Artifact-aware fsck**: Split actions are skipped and delete actions trigger warnings for memories with attached artifacts, preventing accidental data loss ([`4562a36`](https://github.com/fpytloun/mnemory/commit/4562a36))
- **Auto-fsck UI**: Check tab now shows scheduling status banner with interval, next-run countdown, run-now button, and last auto-fsck results display ([`590374b`](https://github.com/fpytloun/mnemory/commit/590374b))
- **Auto-fsck race condition fix**: Resolved issue where UI showed auto-applied issues as unapplied due to a timing race between apply and UI refresh ([`2e6e488`](https://github.com/fpytloun/mnemory/commit/2e6e488))

### Management UI

- **Refresh button**: Manual refresh button added to the dashboard header ([`12d3dde`](https://github.com/fpytloun/mnemory/commit/12d3dde))

### Integrations

- **MNEMORY_INCLUDE_ASSISTANT option**: Claude Code and OpenCode plugins can now include assistant messages in remember flows, enabling the server to extract agent identity and capability facts ([`1df5e63`](https://github.com/fpytloun/mnemory/commit/1df5e63))
- **OpenCode child session handling**: OpenCode plugin skips memory recall/remember for child/subtask sessions to avoid noise from automated sub-agent work ([`6439dd4`](https://github.com/fpytloun/mnemory/commit/6439dd4))

### Documentation

- **Docs reorganization**: README split into dedicated documentation pages under `docs/` — architecture, configuration, deployment, development, management UI, MCP tools, memory model, monitoring, and REST API ([`9475191`](https://github.com/fpytloun/mnemory/commit/9475191))

## [1.4.0] — 2026-02-27

### Periodic Maintenance (Auto-fsck)

- **Background maintenance service**: New `MaintenanceService` (`mnemory/maintenance.py`) runs automatic memory consistency checks on a configurable schedule. Sleep-first loop (waits the full interval before the first run), per-user error isolation, and `asyncio.to_thread()` for sync LLM work ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **Auto-apply with thresholds**: Fixes that meet configurable confidence (`FSCK_AUTO_MIN_CONFIDENCE`, default 0.95) and severity (`FSCK_AUTO_MIN_SEVERITY`, default `medium`) thresholds are applied automatically. Issues below either threshold are skipped ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **New config vars**: `FSCK_AUTO_INTERVAL` (hours, default 0 = disabled), `FSCK_AUTO_MIN_CONFIDENCE` (0.0–1.0), `FSCK_AUTO_MIN_SEVERITY` (low/medium/high) ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **User enumeration**: New `VectorStore.list_user_ids()` scrolls all Qdrant points to enumerate distinct users for the maintenance loop ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

### Metrics

- **Auto-fsck Prometheus metrics**: Four new counters (`mnemory_autofsck_runs_total`, `mnemory_autofsck_issues_found_total`, `mnemory_autofsck_fixes_applied_total`, `mnemory_autofsck_fixes_failed_total`) and a gauge (`mnemory_autofsck_last_run_timestamp`), all labeled by `user_id` ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **Stats API extension**: `GET /api/stats` now includes an `autofsck` section with enabled flag, config, per-user run history, and aggregated totals ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

### Management UI

- **Auto-fsck status banner**: Check tab shows a teal status banner when auto-fsck is enabled, displaying interval, confidence, severity, last run age, and fixes applied ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **Last Auto-Check stat card**: Dashboard shows a 7th stat card with the age of the last auto-fsck run (conditional on auto-fsck being enabled) ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

### Grafana Dashboard

- **Auto-fsck row**: New "Auto-fsck" row with four panels — Last Auto-Check age (color thresholds: green <1h, yellow <24h, red >24h), Issues Found, Fixes Applied, and an Activity timeseries showing fixes applied and issues found per hour ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

### Fsck Improvements

- **Apply idempotency**: `apply_check` tracks applied issue IDs and skips re-applying on repeated calls ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **Background cleanup task**: `FsckStore` now has `start_cleanup_task()` / `stop_cleanup_task()` for periodic sweep of expired checks, wired into server lifespan ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **Apply validations**: Category and memory type values in update actions are validated and stripped if invalid, preventing bad metadata from being written ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **Ownership checks**: Apply actions verify memory ownership before mutating, skipping memories belonging to other users ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **Scroll pagination**: `scroll_with_vectors` now paginates correctly for large collections ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **FSCK_LLM_MODEL / FSCK_REASONING_EFFORT**: Override the LLM model and reasoning effort specifically for fsck checks, independent of the main `LLM_MODEL` ([`df4be81`](https://github.com/fpytloun/mnemory/commit/df4be81))
- **LLM security re-evaluation**: Regex-flagged memories are re-evaluated by the LLM before being reported as security issues, reducing false positives ([`75febcf`](https://github.com/fpytloun/mnemory/commit/75febcf))
- **Confidence field**: All fsck issues now carry a `confidence` score (0.0–1.0) from the LLM, surfaced in the UI and used for auto-apply threshold filtering ([`75febcf`](https://github.com/fpytloun/mnemory/commit/75febcf))

### Core Memories

- **Top-N non-pinned memories**: `get_core_memories` now includes the top-N most important non-pinned memories per section, configurable via `CORE_TOP_MEMORIES` (default 10) and `CORE_MIN_IMPORTANCE` (default `normal`) ([`40c085e`](https://github.com/fpytloun/mnemory/commit/40c085e))

### Documentation

- **Check UI screenshots**: Three new screenshots for the Check tab (progress, results, detail) added to README ([`4b87ca9`](https://github.com/fpytloun/mnemory/commit/4b87ca9))
- **Periodic Maintenance section**: README documents the auto-fsck feature with configuration example and how-it-works description ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

## [1.3.0] — 2026-02-25

### Management UI

- **Built-in web UI**: Full management UI at `/ui` — dashboard, search, browse/CRUD, and graph visualization. Alpine.js + Tailwind CSS + Chart.js + D3.js, all vendored as static files with zero external dependencies ([`14d1546`](https://github.com/fpytloun/mnemory/commit/14d1546))
- **Dashboard**: Memory totals, breakdowns by type/category/role (Chart.js donut charts), operation counts, per-user filtering, server version display, auto-refresh with manual refresh button ([`66dbf1e`](https://github.com/fpytloun/mnemory/commit/66dbf1e))
- **Search tab**: Semantic search (`search_memories`) and AI-powered find (`find_memories`) with filters — memory type, role, agent ID, categories multi-select, "has artifacts" toggle, include decayed, result limit. Sort by score, newest, or oldest ([`66dbf1e`](https://github.com/fpytloun/mnemory/commit/66dbf1e), [`33d2a7f`](https://github.com/fpytloun/mnemory/commit/33d2a7f), [`7cb089f`](https://github.com/fpytloun/mnemory/commit/7cb089f))
- **Browse tab**: Full CRUD with server-side sorting (newest/oldest via Qdrant `order_by`), Add Memory modal, inline edit modal (content, type, categories, importance, pinned, TTL), delete with confirmation, and artifact management (upload, view with pagination, delete) ([`bbeb69c`](https://github.com/fpytloun/mnemory/commit/bbeb69c), [`33d2a7f`](https://github.com/fpytloun/mnemory/commit/33d2a7f))
- **Graph tab**: D3.js force-directed visualization of memory relationships (shared categories = edges, node size = importance, color = memory type). Type filter checkboxes, node limit slider, click-to-select detail panel ([`14d1546`](https://github.com/fpytloun/mnemory/commit/14d1546))
- **Card layout**: Two-zone badge row — left: type + importance + `#category` tags; right: agent badge + artifacts indicator + assistant chip + pinned star ([`7cb089f`](https://github.com/fpytloun/mnemory/commit/7cb089f))
- **Agent filtering**: Agent ID dropdown filter in both Search and Browse tabs, with agent badges (robot icon + name) visible on cards without expanding ([`7cb089f`](https://github.com/fpytloun/mnemory/commit/7cb089f))
- **Session persistence**: API key and selected user stored in localStorage (persists across tab closes) ([`8171f02`](https://github.com/fpytloun/mnemory/commit/8171f02))

### Backend Improvements

- **Server-side sorting**: `GET /api/memories` accepts `sort` query param (newest/oldest). Uses Qdrant `order_by` with automatic datetime payload index creation on startup ([`33d2a7f`](https://github.com/fpytloun/mnemory/commit/33d2a7f), [`ff83be4`](https://github.com/fpytloun/mnemory/commit/ff83be4))
- **Index management**: `_ensure_indexes()` extracted as separate method, runs on every startup (not just collection creation). Existing collections now get new indexes automatically ([`ff83be4`](https://github.com/fpytloun/mnemory/commit/ff83be4))
- **Ordering fallback**: `get_all()` gracefully falls back to Python-side sorting if Qdrant `order_by` fails (e.g., missing index on older collections) ([`ff83be4`](https://github.com/fpytloun/mnemory/commit/ff83be4))
- **agent_id metadata fix**: `format_memory_item()` now injects top-level `agent_id` back into metadata dict, fixing agent badges not appearing in API responses ([`7cb089f`](https://github.com/fpytloun/mnemory/commit/7cb089f))
- **OpenAPI sanitization**: Schema cleanup for OpenAI function-calling compatibility ([`7ce33a9`](https://github.com/fpytloun/mnemory/commit/7ce33a9))
- **Non-conversational message filtering**: `remember()` ignores tool calls, system messages, and other non-conversational content ([`9f08f38`](https://github.com/fpytloun/mnemory/commit/9f08f38))
- **Runtime version loading**: Package version loaded via `importlib.metadata` at runtime with dev fallback ([`2640aad`](https://github.com/fpytloun/mnemory/commit/2640aad))

### Extraction & Search

- **English extraction**: Extraction prompts now require English output regardless of input language ([`a567984`](https://github.com/fpytloun/mnemory/commit/a567984))
- **Third-party guidance**: Extraction prompts include guidance for handling third-party information mentioned in conversations ([`a567984`](https://github.com/fpytloun/mnemory/commit/a567984))

### Testing

- **End-to-end test suite**: Comprehensive e2e tests exercising the full pipeline (extraction → storage → search) with real LLM calls and embedded Qdrant. Excluded from default `pytest` run via `-m 'not e2e'` ([`4b7353e`](https://github.com/fpytloun/mnemory/commit/4b7353e))

### Documentation

- **Client setup guides**: Dedicated docs for Claude Code, ChatGPT, Claude Desktop, Open WebUI, OpenCode, Cursor, Windsurf, Cline, Continue.dev, and Codex CLI ([`e283c48`](https://github.com/fpytloun/mnemory/commit/e283c48))
- **UI screenshots**: Dashboard, search, browse, and graph screenshots added to README ([`47defec`](https://github.com/fpytloun/mnemory/commit/47defec))

## [1.2.0] — 2026-02-24

### Monitoring

- **Prometheus metrics endpoint**: New `/metrics` endpoint with operation counters (`mnemory_operations_total`) and Qdrant-backed gauges for memory counts by type, category, role, decay, and pinned status. Configurable cache TTL and optional `prometheus_client` dependency ([`3951649`](https://github.com/fpytloun/mnemory/commit/3951649))
- **Management port**: New `MGMT_PORT` config to serve `/health` and `/metrics` on a separate port without authentication — designed for Kubernetes probes and Prometheus scraping ([`3951649`](https://github.com/fpytloun/mnemory/commit/3951649))
- **Grafana dashboard**: Pre-built dashboard with memory overview, operation rates, and breakdowns by type/category/role with user and agent filtering ([`f48c909`](https://github.com/fpytloun/mnemory/commit/f48c909))

### Security

- **Prompt injection safeguards**: All 6 LLM prompt builders hardened with Unicode boundary tags (`⟨user_input⟩`, `⟨existing_memories⟩`) and anti-injection preamble. Content within tags is escaped to prevent breakout ([`95f61bb`](https://github.com/fpytloun/mnemory/commit/95f61bb))
- **Injection detection logging**: Common prompt injection patterns (role overrides, instruction manipulation, boundary tag abuse) are detected and logged at WARNING level without blocking storage ([`95f61bb`](https://github.com/fpytloun/mnemory/commit/95f61bb))
- **Category name validation**: `project:<name>` categories validated against a safe character regex to prevent prompt format injection ([`95f61bb`](https://github.com/fpytloun/mnemory/commit/95f61bb))
- **Role validation**: REST `remember` endpoint and Pydantic schemas enforce an allowlist of valid message roles ([`95f61bb`](https://github.com/fpytloun/mnemory/commit/95f61bb))
- **Per-memory boundary tags**: Core memories output wraps each memory in `⟨memory_item⟩` tags to prevent cross-memory poisoning and section header forgery ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))
- **Agent identity gating**: `role="assistant"` memories now require `infer=True` — direct injection of arbitrary agent personality via `infer=False` is blocked with `ValueError` ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))
- **Input length validation**: `infer=False` path now has the same 400K character cap as `infer=True`. Pydantic `max_length` constraints added to 10 request fields across all schemas ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))
- **Output escaping**: Markdown headers in memory text are escaped in `format_memory_item()` and `_format_memories()` to prevent section forgery in search results and core memories ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))
- **Error sanitization**: `initialize_memory` error messages truncated to 200 chars with newlines stripped to prevent information leakage ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))

## [1.1.0] — 2026-02-24

### REST API & Intelligence Layer

- **REST API foundation**: Full memory CRUD + artifact operations as FastAPI sub-app mounted at `/api/`, with auto-generated OpenAPI spec at `/api/docs` ([`ceb07f1`](https://github.com/fpytloun/mnemory/commit/ceb07f1))
- **Recall endpoint** (`POST /api/recall`): Combined initialize + search in one call with server-side session tracking to avoid context bloat. First call returns instructions + core memories + AI-powered search; subsequent calls return only new memories ([`1747af6`](https://github.com/fpytloun/mnemory/commit/1747af6))
- **Remember endpoint** (`POST /api/remember`): Fire-and-forget memory storage from conversation turns with rate limiting and session deduplication ([`1747af6`](https://github.com/fpytloun/mnemory/commit/1747af6))
- **Configurable search modes**: Per-request `search_mode` (find vs search) and `score_threshold` on recall endpoint ([`2c240de`](https://github.com/fpytloun/mnemory/commit/2c240de), [`1f1e259`](https://github.com/fpytloun/mnemory/commit/1f1e259), [`13b3dde`](https://github.com/fpytloun/mnemory/commit/13b3dde))
- **Cross-agent access control**: Dual-scope search and agent memory isolation in REST endpoints ([`7997e29`](https://github.com/fpytloun/mnemory/commit/7997e29))
- **Input validation and session bounds**: Request validation, response models, and `_MAX_KNOWN_IDS` cap on session tracking ([`bfd90a2`](https://github.com/fpytloun/mnemory/commit/bfd90a2))

### Plugins

- **Open WebUI filter**: Automatic recall on inlet, remember on outlet, with configurable status messages and `show_status` user valve ([`75bb955`](https://github.com/fpytloun/mnemory/commit/75bb955), [`6a43998`](https://github.com/fpytloun/mnemory/commit/6a43998), [`444b2d8`](https://github.com/fpytloun/mnemory/commit/444b2d8))
- **OpenCode plugin**: TypeScript plugin with system prompt injection via `experimental.chat.system.transform`, pre-fetched recall with async caching, fire-and-forget remember, and compaction persistence ([`f658c48`](https://github.com/fpytloun/mnemory/commit/f658c48), [`1592071`](https://github.com/fpytloun/mnemory/commit/1592071))
- **Managed instructions**: New `build_managed_instructions()` for plugin-driven setups — tells the LLM what's automatic and what's available for explicit use ([`1747af6`](https://github.com/fpytloun/mnemory/commit/1747af6))

### Memory Intelligence

- **Context-aware query generation**: `find_memories` accepts an optional `context` hint (e.g., working directory) and injects the user's `project:*` categories into the query generation prompt for more targeted search ([`8247906`](https://github.com/fpytloun/mnemory/commit/8247906))
- **Temporal reasoning**: `event_date` parameter on `add_memory` anchors relative time references ("yesterday" resolves correctly). `find_memories` generates date-aware queries using session timezone ([`d537d62`](https://github.com/fpytloun/mnemory/commit/d537d62), [`9ecdc64`](https://github.com/fpytloun/mnemory/commit/9ecdc64))
- **Improved extraction quality**: Better extraction rules, conversation-aware prompts that filter assistant noise, and LLM-backed metadata validation with retry ([`737f7b2`](https://github.com/fpytloun/mnemory/commit/737f7b2), [`a3f0e48`](https://github.com/fpytloun/mnemory/commit/a3f0e48), [`f9c1df5`](https://github.com/fpytloun/mnemory/commit/f9c1df5))

### Artifacts

- **Auto-artifact creation**: LLM-guided for `infer=True` (extraction prompt decides via `store_artifact` flag), size-based for `infer=False` (content exceeding `MAX_MEMORY_LENGTH` auto-saved). New `remember()` method for conversation processing ([`08a998c`](https://github.com/fpytloun/mnemory/commit/08a998c))
- **M:N artifact linking**: Artifacts decoupled from individual memories — one artifact can be referenced by multiple memories. Storage path changed from `{user_id}/{memory_id}/{artifact_id}/` to `{user_id}/{artifact_id}/` ([`cee1f6a`](https://github.com/fpytloun/mnemory/commit/cee1f6a))
- **Artifact garbage collection**: Deleting a memory checks if other memories still reference the same artifact; orphaned artifacts are cleaned up ([`cee1f6a`](https://github.com/fpytloun/mnemory/commit/cee1f6a))

### LLM & Provider Compatibility

- **Provider adaptation**: Automatic handling of parameter incompatibilities across OpenAI-compatible providers (reasoning_effort, json_schema support) ([`7c585b7`](https://github.com/fpytloun/mnemory/commit/7c585b7))
- **Robust error handling**: Retry logic and graceful handling of `BadRequestError` for chat completions ([`f2c11fa`](https://github.com/fpytloun/mnemory/commit/f2c11fa))

### Benchmarks

- **LoCoMo benchmark suite**: Evaluation on 1540 QA questions across 4 categories (single_hop, multi_hop, temporal, open_domain). Overall score: 73.2 with `gpt-5-mini` ([`26798ff`](https://github.com/fpytloun/mnemory/commit/26798ff), [`7c976ca`](https://github.com/fpytloun/mnemory/commit/7c976ca))

### Bug Fixes

- Serialize local embedded Qdrant write operations to prevent concurrency issues ([`3efc793`](https://github.com/fpytloun/mnemory/commit/3efc793))
- Handle session compaction with fresh recall and re-injection ([`bdef0e0`](https://github.com/fpytloun/mnemory/commit/bdef0e0))
- Configurable `request_timeout` in Open WebUI filter instead of hardcoded 10s ([`444b2d8`](https://github.com/fpytloun/mnemory/commit/444b2d8))

### Breaking Changes

- **Artifact storage path changed**: `{user_id}/{memory_id}/{artifact_id}/{filename}` → `{user_id}/{artifact_id}/{filename}`. Existing artifacts at the old path become inaccessible. Re-create affected artifacts or manually move them.

## [1.0.0] — 2025-02-02

Initial release. MCP server with persistent two-tier memory (Qdrant + S3/filesystem), unified LLM extraction/classification/deduplication pipeline, semantic search with importance reranking, TTL-based memory decay, role-based scoping, and configurable behavioral instructions.
