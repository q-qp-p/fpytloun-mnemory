# @fpytloun/opencode-mnemory

OpenCode plugin for [mnemory](https://github.com/fpytloun/mnemory) — persistent AI memory with automatic recall, automatic capture, and 16 explicit memory tools.

**No MCP server configuration needed.** All memory tools are built into the plugin.

## How It Works

| Phase | Hook | Action |
|---|---|---|
| **Session start** | `session.created` | Pre-fetches core memories and instructions from mnemory (non-blocking) |
| **Each user message** | `chat.message` | Starts a semantic search with the user's query (non-blocking) |
| **Before each LLM call** | `experimental.chat.system.transform` | Injects instructions + core memories + search results into system prompt |
| **After each exchange** | `session.idle` | Extracts the last user+assistant exchange and sends to mnemory for memory extraction (fire-and-forget) |
| **On compaction** | `experimental.session.compacting` | Preserves core memories across context window compaction |
| **After compaction** | `session.compacted` | Resets state and re-fetches memories |
| **Session cleanup** | `session.deleted` | Cleans up session state |

The LLM also has access to 16 memory tools for explicit operations (search, add, update, delete, artifacts).

## Installation

### From npm (recommended)

Add to your `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@fpytloun/opencode-mnemory"]
}
```

Set environment variables:

```bash
export MNEMORY_URL=http://localhost:8050
export MNEMORY_API_KEY=your-api-key  # if auth is enabled
```

### From local files (for development)

```bash
# Global
cp integrations/opencode/*.ts ~/.config/opencode/plugins/

# Or project-level
cp integrations/opencode/*.ts .opencode/plugins/
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MNEMORY_URL` | `http://localhost:8050` | Mnemory server URL |
| `MNEMORY_API_KEY` | (empty) | Bearer token for authentication |
| `MNEMORY_AGENT_ID` | `opencode` | Agent ID sent to mnemory |
| `MNEMORY_USER_ID` | (empty) | User ID (optional if API key maps to user) |
| `MNEMORY_SCORE_THRESHOLD` | `0.5` | Minimum relevance score for recalled memories (0.0-1.0) |
| `MNEMORY_INCLUDE_ASSISTANT` | `false` | Include assistant messages in remember calls |
| `MNEMORY_SEARCH_MODE` | `search` | Default search mode for subsequent turns: `find` (AI-powered) or `search` (fast vector) |
| `MNEMORY_FIND_FIRST` | `true` | Use AI-powered search on the first turn of each session |
| `MNEMORY_MANAGED` | `true` | Include mnemory behavioral instructions in the system prompt |
| `MNEMORY_TIMEOUT` | `30000` | HTTP request timeout in milliseconds |

## Tools

The plugin registers 16 tools that the LLM can call for explicit memory operations:

| Tool | Description |
|---|---|
| `memory_search` | Semantic search across memories |
| `memory_find` | AI-powered multi-query search with LLM reranking |
| `memory_ask` | Ask a question and get a synthesized answer from memories |
| `memory_add` | Store a new memory (auto-extracts facts, deduplicates) |
| `memory_add_batch` | Store multiple memories in one call |
| `memory_update` | Update existing memory content or metadata |
| `memory_delete` | Delete a memory by ID |
| `memory_delete_batch` | Delete multiple memories |
| `memory_list` | List memories with optional filters |
| `memory_categories` | List available predefined categories |
| `memory_recent` | Get recent memories from last N days |
| `memory_save_artifact` | Attach artifact (report, code, data) to a memory |
| `memory_get_artifact` | Retrieve artifact content |
| `memory_get_artifact_url` | Generate signed download URL for large/binary artifacts |
| `memory_list_artifacts` | List artifacts attached to a memory |
| `memory_delete_artifact` | Delete an artifact |

## Architecture

```
index.ts        Plugin entry point — wires hooks + tools
hooks.ts        Lifecycle hooks — auto-recall, auto-remember, compaction
tools.ts        16 custom tool definitions
client.ts       HTTP client for mnemory REST API
helpers.ts      Config, session store, escaping, text extraction
```

### Two-Phase Recall

1. **Init recall** (`session.created`): Pre-fetches instructions + core memories (no query). Cached for session lifetime.
2. **Per-turn search** (`chat.message` → `system.transform`): On each user message, starts a search with the user's query. Results are awaited and injected before the LLM call.

First turn uses `find` mode (AI-powered multi-query search, higher quality). Subsequent turns use `search` mode (fast vector search, no LLM overhead). Configurable via `MNEMORY_SEARCH_MODE` and `MNEMORY_FIND_FIRST`.

### Graceful Degradation

- If the mnemory server is offline, the plugin logs a warning and the LLM works normally without memory context.
- All API calls have timeouts and never throw — errors are logged via OpenCode's structured logging.
- Per-turn search has an 8-second timeout in `system.transform` to avoid blocking the LLM call.

## Troubleshooting

**Memories not appearing?**
- Check that `MNEMORY_URL` is correct and the server is running
- Look for `mnemory:` messages in OpenCode logs
- Verify the API key is valid (if auth is enabled)

**Search results not relevant?**
- Try lowering `MNEMORY_SCORE_THRESHOLD` (e.g., `0.3`)
- Use `MNEMORY_FIND_FIRST=true` for AI-powered search on the first turn

**Too much latency on first turn?**
- Set `MNEMORY_FIND_FIRST=false` to use fast vector search on all turns
- The init recall runs in the background and shouldn't add latency

## Development

```bash
# Run tests
cd integrations/opencode
bun test
```

## License

Apache 2.0
