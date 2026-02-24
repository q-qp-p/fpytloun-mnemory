# OpenCode Plugin — Automatic Memory

A plugin for [OpenCode](https://opencode.ai) that automatically recalls memories at session start, stores new memories after each exchange, and preserves memory context across session compaction.

## How It Works

1. **`session.created`**: Calls `/api/recall` to load core memories and relevant context. Injects them into the session via `session.prompt({ noReply: true })`.
2. **`session.idle`**: After each LLM exchange, extracts the last user + assistant messages and calls `/api/remember` to store new memories (fire-and-forget).
3. **`experimental.session.compacting`**: Re-injects core memories into the compaction context so they survive session compaction.

The LLM also has access to mnemory MCP tools for explicit operations (search, update, delete). The plugin handles the automatic parts.

## Setup

### 1. Environment Variables

```bash
export MNEMORY_URL=http://localhost:8050
export MNEMORY_API_KEY=your-api-key
export MNEMORY_AGENT_ID=opencode       # optional, defaults to "opencode"
export MNEMORY_USER_ID=your-username    # optional if using API key mapping
export MNEMORY_SCORE_THRESHOLD=0.5      # optional, min relevance score (0.0-1.0)
```

| Variable | Default | Description |
|---|---|---|
| `MNEMORY_URL` | `http://localhost:8050` | mnemory server URL |
| `MNEMORY_API_KEY` | (empty) | API key for authentication |
| `MNEMORY_AGENT_ID` | `opencode` | Agent ID sent to mnemory |
| `MNEMORY_USER_ID` | (empty) | User ID (optional if API key maps to a user) |
| `MNEMORY_SCORE_THRESHOLD` | `0.5` | Minimum relevance score for recalled memories. Higher = fewer but more relevant. Prevents context bloat from weak matches. |

### 2. Install the Plugin

Copy `mnemory.ts` to your OpenCode plugins directory:

```bash
# Project-level
mkdir -p .opencode/plugins
cp mnemory.ts .opencode/plugins/

# Or global
mkdir -p ~/.config/opencode/plugins
cp mnemory.ts ~/.config/opencode/plugins/
```

### 3. Add the Rules File

The rules file tells the LLM not to duplicate the plugin's automatic behavior:

```bash
mkdir -p .opencode/rules
cp memory.md .opencode/rules/
```

### 4. Configure OpenCode

Add to your `opencode.json`:

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
  },
  "rules": [".opencode/rules/memory.md"],
  "plugin": [".opencode/plugins/mnemory.ts"]
}
```

See `opencode.json` in this directory for a complete example.

## Hybrid Approach

The plugin and MCP server work together:

| Component | Handles |
|---|---|
| **Plugin** (automatic) | Recall at session start, remember after each exchange, compaction persistence |
| **MCP tools** (LLM-driven) | Explicit search, add, update, delete when the user asks |

The rules file (`memory.md`) tells the LLM which operations are automatic and which it can use explicitly. This prevents the LLM from calling `initialize_memory` or `add_memory` when the plugin already handles those.

## Compaction

OpenCode compacts long sessions to stay within context limits. Without special handling, memories injected at session start would be lost after compaction.

The plugin registers an `experimental.session.compacting` hook that re-injects core memories into the compaction context. This ensures the compaction summary preserves memory context, so the LLM retains user knowledge across compaction boundaries.

## Troubleshooting

- **No memories appearing**: Check that `MNEMORY_URL` is reachable. Look for "Plugin initialized" in OpenCode logs.
- **Memories lost after compaction**: Ensure the plugin is loaded (check logs). The compaction hook should re-inject core memories automatically.
- **LLM still calls initialize_memory**: Ensure `memory.md` is in your rules directory and listed in `opencode.json`. The rules file overrides MCP tool descriptions.
- **Duplicate memory storage**: The extraction pipeline deduplicates against existing memories. If you see duplicates, check that the mnemory server is reachable for the remember calls.
