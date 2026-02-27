# MCP Tools

mnemory exposes 16 tools via the [Model Context Protocol](https://modelcontextprotocol.io/). These are available to any MCP-compatible client.

## Session Initialization

| Tool | Description |
|---|---|
| `initialize_memory` | **Start here.** Returns behavioral instructions + core memories. Use for clients that don't inject MCP server instructions (e.g., Open WebUI). |

## Memory Operations

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

## Artifact Operations

| Tool | Description |
|---|---|
| `save_artifact` | Attach detailed content to a memory |
| `get_artifact` | Retrieve artifact content with pagination |
| `list_artifacts` | List artifacts on a memory |
| `delete_artifact` | Remove an artifact |

## Tool Usage Patterns

### Storing Memories

- **`infer=true`** (default): The server runs a single LLM call that extracts facts, classifies metadata (type, categories, importance), and deduplicates against existing memories. Use when passing raw conversation text or unstructured content.
- **`infer=false`**: Skips the LLM call — content is embedded and stored directly. Much faster (single embedding call vs. LLM + embedding). Use when your content is already a clean, concise fact.
- **`add_memories`** (batch): Processes multiple memories in a single call, avoiding per-item round-trip latency.

### Searching

Two search tools are available:

- **`search_memories`**: Fast single-query vector search. Use for simple lookups and routine memory recall. Preferred for most cases.
- **`find_memories`**: AI-powered multi-query search. Takes a natural language question, generates multiple targeted searches following associations (e.g., "dogs" -> pets, partner, house, lifestyle), and uses AI to rerank results by relevance. Use for complex, multi-faceted questions where a single search query wouldn't capture all relevant context. Slower (2 extra LLM calls) but higher quality for complex queries.

### Artifacts

For detailed content too long for fast memory (research reports, analysis, code, data), save it as an artifact attached to a memory. The memory holds the searchable summary; the artifact holds the full details. Search results show which memories have artifacts — fetch them with `get_artifact` when you need the details.

### Role Parameter

The `role` parameter tells the server who the memory is about:

- **`role="user"`** (default): Facts about the user — preferences, personal info, context, decisions.
- **`role="assistant"`**: Facts about the agent — identity, personality, capabilities, knowledge. Requires `agent_id` to be set.

When searching or listing, `role` filters results by who the memory is about. Search automatically returns both agent-specific and shared user memories.
