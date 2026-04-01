# Hermes Agent Plugin -- Mnemory

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that provides long-term semantic memory backed by a [mnemory](https://github.com/fpytloun/mnemory) server.

## Features

- **Auto-Recall**: Automatically fetches relevant memories and injects them into the system prompt before each agent turn
- **Auto-Capture**: Extracts and stores memories from conversations after each exchange (fire-and-forget)
- **16 Explicit Tools**: `memory_search`, `memory_find`, `memory_ask`, `memory_add`, `memory_add_batch`, `memory_update`, `memory_delete`, `memory_delete_batch`, `memory_list`, `memory_categories`, `memory_recent`, `memory_save_artifact`, `memory_get_artifact`, `memory_get_artifact_url`, `memory_list_artifacts`, `memory_delete_artifact`
- **Compaction-safe**: Detects context compression and re-injects memories automatically
- **Graceful degradation**: If mnemory is offline, the agent continues working normally

## Prerequisites

A running [mnemory](https://github.com/fpytloun/mnemory) server accessible via HTTP.

## Install

### pip (recommended)

```bash
pip install hermes-mnemory
```

The plugin is auto-discovered via the `hermes_agent.plugins` entry point.

### Directory install

Copy this directory to your Hermes plugins folder:

```bash
cp -r integrations/hermes ~/.hermes/plugins/mnemory
```

## Configure

Add to your `~/.hermes/.env`:

```bash
MNEMORY_URL=http://localhost:8050
MNEMORY_API_KEY=your-api-key  # optional if auth is disabled
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MNEMORY_URL` | *(required)* | Mnemory server URL |
| `MNEMORY_API_KEY` | `""` | Bearer token for authentication |
| `MNEMORY_USER_ID` | `""` | User ID (required when API key is wildcard or auth disabled) |
| `MNEMORY_AGENT_PREFIX` | `hermes` | Value sent as `X-Agent-Id` header |
| `MNEMORY_AUTO_RECALL` | `true` | Auto-inject memories into context |
| `MNEMORY_AUTO_CAPTURE` | `true` | Auto-extract memories from conversations |
| `MNEMORY_RECALL_FIND_FIRST` | `true` | Use AI-powered search on first turn (higher quality, slower) |
| `MNEMORY_RECALL_SEARCH_MODE` | `search` | Search mode for subsequent turns: `find` or `search` |
| `MNEMORY_SCORE_THRESHOLD` | `0.5` | Min relevance score for recalled memories (0.0-1.0) |
| `MNEMORY_INCLUDE_ASSISTANT` | `true` | Send assistant messages for extraction |
| `MNEMORY_MANAGED` | `true` | Include behavioural instructions in system prompt |
| `MNEMORY_TIMEOUT` | `60` | HTTP request timeout in seconds |

## How It Works

1. **`on_session_start`**: Pre-fetches instructions and core memories from `/api/recall` in a background thread (non-blocking)
2. **`pre_llm_call`**: Two-phase recall per turn:
   - Awaits the init recall (instructions + core memories) if still pending
   - Sends the current user message as a search query to `/api/recall` for topical memories
   - First turn uses AI-powered `find` mode (configurable via `MNEMORY_RECALL_FIND_FIRST`), subsequent turns use fast `search` mode (configurable via `MNEMORY_RECALL_SEARCH_MODE`)
   - Search results are **replaced** each turn (not accumulated) -- the server deduplicates via session tracking
   - Returns context via `{"context": "..."}` for injection into the system prompt
3. **`post_llm_call`**: Extracts the last user+assistant exchange and sends to `/api/remember` for background extraction
4. **`on_session_end`**: Cleans up session state

### Compaction handling

The plugin detects context compression by tracking conversation history length. When the history shrinks between turns (indicating Hermes compressed the context), the plugin resets its mnemory session and re-fetches core memories, ensuring memories survive compaction.

## Built-in Memory Coexistence

This plugin does not disable Hermes's built-in memory system (`MEMORY.md` / `USER.md`). They can coexist -- mnemory provides richer semantic search and cross-session memory, while the built-in system provides a simple scratchpad.

To disable the built-in memory when using mnemory, set in `~/.hermes/config.yaml`:

```yaml
memory:
  memory_enabled: false
  user_profile_enabled: false
```

## License

MIT
