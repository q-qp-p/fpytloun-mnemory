# OpenCode

OpenCode has full MCP support and a plugin system that enables automatic memory recall and storage via lifecycle hooks.

## Integration Options

| Method | What it does | Setup effort |
|---|---|---|
| **MCP only** | LLM-driven memory via tool calls | Add MCP config |
| **MCP + Plugin** (recommended) | Automatic recall/remember via hooks + LLM tools for explicit ops | Copy plugin + MCP config |

## MCP Setup

Add to your `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

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

## Plugin Setup (Recommended)

The plugin hooks into OpenCode's session lifecycle for automatic recall/remember. See [`integrations/opencode/`](../../integrations/opencode/) for the plugin code and detailed setup instructions.

### Quick Install

1. Set environment variables:
   ```bash
   export MNEMORY_URL=http://localhost:8050
   export MNEMORY_API_KEY=your-api-key
   ```

2. Copy the plugin:
   ```bash
   # Global (recommended)
   mkdir -p ~/.config/opencode/plugins
   cp integrations/opencode/mnemory.ts ~/.config/opencode/plugins/

   # Or project-level
   mkdir -p .opencode/plugins
   cp integrations/opencode/mnemory.ts .opencode/plugins/
   ```

3. Add the MCP config above for explicit tool access.

### How the Plugin Works

| Hook | Action |
|---|---|
| `session.created` | Pre-fetches memories from `/api/recall` (non-blocking) |
| `experimental.chat.system.transform` | Injects memories into system prompt before each LLM call |
| `session.idle` | Calls `/api/remember` with the last exchange (fire-and-forget) |
| `experimental.session.compacting` | Preserves memories across session compaction |

## Tips

- The plugin and MCP tools work together -- plugin handles automatic parts, MCP tools handle explicit search/update/delete
- Memory persists across session compaction
- See the [coding assistant system prompt guide](../system-prompts/claude-code.md) for tips on what gets remembered
