# OpenCode Plugin — Automatic Memory

A plugin for [OpenCode](https://opencode.ai) that automatically recalls memories at session start, stores new memories after each exchange, and preserves memory context across session compaction.

## How It Works

1. **`session.created`**: Pre-fetches memories from `/api/recall` (non-blocking).
2. **`experimental.chat.system.transform`**: Injects recalled memories and behavioral instructions into the system prompt before each LLM call. On the first call, awaits the pre-fetch (~1-2s). On subsequent calls, uses the cached result (instant). Also handles resumed sessions where `session.created` didn't fire.
3. **`session.idle`**: After each LLM exchange, extracts the last user + assistant messages and calls `/api/remember` to store new memories (fire-and-forget).
4. **`experimental.session.compacting`**: Re-injects core memories into the compaction context so they survive session compaction.

The plugin injects both memory content and behavioral instructions (what to do automatically vs. what to leave for explicit user requests) directly into the system prompt. No separate rules file is needed.

The LLM also has access to mnemory MCP tools for explicit operations (search, update, delete). The plugin handles the automatic parts.

## Setup

### 1. Environment Variables

The plugin makes direct HTTP calls to the mnemory REST API. These environment variables **must be set** for the plugin to work — without them, API calls fail silently:

```bash
export MNEMORY_URL=http://localhost:8050
export MNEMORY_API_KEY=your-api-key        # required if mnemory uses MCP_API_KEYS
export MNEMORY_AGENT_ID=opencode           # optional, defaults to "opencode"
export MNEMORY_USER_ID=your-username       # optional if using API key mapping
export MNEMORY_SCORE_THRESHOLD=0.5         # optional, min relevance score (0.0-1.0)
```

| Variable | Default | Description |
|---|---|---|
| `MNEMORY_URL` | `http://localhost:8050` | mnemory server URL |
| `MNEMORY_API_KEY` | (empty) | API key for authentication. **Required** if mnemory has `MCP_API_KEYS` set. |
| `MNEMORY_AGENT_ID` | `opencode` | Agent ID sent to mnemory |
| `MNEMORY_USER_ID` | (empty) | User ID (optional if API key maps to a user) |
| `MNEMORY_SCORE_THRESHOLD` | `0.5` | Minimum relevance score for recalled memories. Higher = fewer but more relevant. Prevents context bloat from weak matches. |

### 2. Install the Plugin

Copy `mnemory.ts` to your OpenCode plugins directory:

```bash
# Global (recommended — memory works across all projects)
mkdir -p ~/.config/opencode/plugins
cp mnemory.ts ~/.config/opencode/plugins/

# Or project-level
mkdir -p .opencode/plugins
cp mnemory.ts .opencode/plugins/
```

Local plugins are loaded automatically — no config entry needed.

### 3. Configure OpenCode

Add the MCP server to your `~/.config/opencode/opencode.json` (global) or `opencode.json` (project):

```json
{
  "mcp": {
    "mnemory": {
      "type": "remote",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "opencode"
      }
    }
  }
}
```

That's it. The plugin injects behavioral instructions into the system prompt automatically — no rules file or `"instructions"` config needed.

See `opencode.json` in this directory for a complete example.

## Hybrid Approach

The plugin and MCP server work together:

| Component | Handles |
|---|---|
| **Plugin** (automatic) | Recall at session start, remember after each exchange, system prompt injection, compaction persistence |
| **MCP tools** (LLM-driven) | Explicit search, add, update, delete when the user asks |

The plugin injects managed instructions into the system prompt that tell the LLM which operations are automatic and which it can use explicitly. This prevents the LLM from calling `initialize_memory` or `add_memory` when the plugin already handles those.

## Compaction

OpenCode compacts long sessions to stay within context limits. The plugin handles this in two ways:

1. **System prompt injection**: Since memories are injected via `experimental.chat.system.transform`, they are automatically present on every LLM call — including after compaction. No special handling needed.
2. **Compaction context**: The `experimental.session.compacting` hook also pushes core memories into the compaction context, ensuring the compaction summary preserves memory-related information.

On `session.compacted`, the plugin resets its cache and pre-fetches fresh memories for the next LLM call.

## Troubleshooting

- **No memories appearing**: Check that `MNEMORY_URL` and `MNEMORY_API_KEY` are set. The plugin fails silently without them. Look for "Plugin initialized" in OpenCode logs (`--print-logs`).
- **Memories lost after compaction**: Ensure the plugin is loaded (check logs). The system prompt injection and compaction hook should preserve memories automatically.
- **LLM still calls initialize_memory**: The plugin injects managed instructions that override MCP tool descriptions. If the LLM still calls init tools, check that the plugin is loaded and the mnemory server is reachable (instructions come from the `/api/recall` response).
- **Duplicate memory storage**: The extraction pipeline deduplicates against existing memories. If you see duplicates, check that the mnemory server is reachable for the remember calls.
