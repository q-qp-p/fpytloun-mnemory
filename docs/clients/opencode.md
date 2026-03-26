# OpenCode

OpenCode has full MCP support and a plugin system that enables automatic memory recall and storage via lifecycle hooks, plus 16 built-in memory tools.

## Integration Options

| Method | What it does | Setup effort |
|---|---|---|
| **Plugin** (recommended) | Automatic recall/remember + 16 built-in tools. No MCP needed. | Install npm package |
| **MCP only** | LLM-driven memory via MCP tool calls | Add MCP config |

## Plugin Setup (Recommended)

The plugin handles everything automatically — memory recall, per-turn search, and storage. It also registers 16 memory tools for explicit operations (search, add, update, delete, artifacts).

### Install from npm

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

### Install from local files

```bash
# Global (recommended)
cp integrations/opencode/*.ts ~/.config/opencode/plugins/

# Or project-level
cp integrations/opencode/*.ts .opencode/plugins/
```

### How the Plugin Works

| Phase | Hook | Action |
|---|---|---|
| Session start | `session.created` | Pre-fetches core memories and instructions (non-blocking) |
| Each user message | `chat.message` | Starts semantic search with user's query (non-blocking) |
| Before each LLM call | `experimental.chat.system.transform` | Injects instructions + core memories + search results into system prompt |
| After each exchange | `session.idle` | Sends last exchange to mnemory for memory extraction (fire-and-forget) |
| On compaction | `experimental.session.compacting` | Preserves core memories across context window compaction |
| After compaction | `session.compacted` | Resets state and re-fetches memories |

### Configuration

All configuration is via environment variables. See [`integrations/opencode/README.md`](../../integrations/opencode/README.md) for the full list.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MNEMORY_URL` | `http://localhost:8050` | Mnemory server URL |
| `MNEMORY_API_KEY` | (empty) | Bearer token for authentication |
| `MNEMORY_AGENT_ID` | `opencode` | Agent ID sent to mnemory |
| `MNEMORY_MANAGED` | `true` | Include behavioral instructions in system prompt |
| `MNEMORY_FIND_FIRST` | `true` | Use AI-powered search on first turn |

## MCP Setup (Alternative)

If you prefer MCP-only (without the plugin), add to your `opencode.json`:

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

Note: With MCP-only, memory recall and storage are LLM-driven (the LLM must call tools explicitly). The plugin approach is recommended for automatic recall/remember.

## Tips

- The plugin registers 16 memory tools — no separate MCP config needed
- Memory persists across session compaction
- First turn uses AI-powered search for higher quality results; subsequent turns use fast vector search
- If the mnemory server is offline, the plugin degrades gracefully — the LLM works normally without memory context
