# Changelog

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
