# Claude Code

Claude Code has full MCP support and a hooks system that enables automatic memory recall and storage without relying on the LLM to call tools.

## Integration Options

| Method | What it does | Setup effort |
|---|---|---|
| **MCP only** | LLM-driven memory via tool calls | Minimal -- just add MCP config |
| **MCP + Plugin** (recommended) | Automatic recall/remember via hooks + LLM tools for explicit ops | Copy plugin files + MCP config |

## MCP Setup

Add to your MCP configuration (`~/.claude/claude_code_config.json`):

**Local (no auth):**
```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "X-Agent-Id": "claude-code"
      }
    }
  }
}
```

**Production (with auth):**
```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "https://mem.example.com/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "claude-code"
      }
    }
  }
}
```

With `MCP_API_KEYS` configured on the server, `user_id` and `agent_id` are resolved automatically. No system prompt changes needed -- Claude Code injects MCP server instructions into the LLM's context.

## Plugin Setup (Recommended)

The plugin uses Claude Code's hooks system to automatically recall memories at session start and store new memories after each exchange. This is more reliable than relying on the LLM to call tools.

See [`integrations/claude-code/`](../../integrations/claude-code/) for the plugin code and detailed setup instructions.

### Quick Install

1. Set environment variables:
   ```bash
   export MNEMORY_URL=http://localhost:8050
   export MNEMORY_API_KEY=your-api-key        # if using MCP_API_KEYS
   ```

2. Install the plugin:
   ```bash
   # Copy to your Claude Code settings directory
   cp -r integrations/claude-code/.claude-plugin ~/.claude/
   cp integrations/claude-code/hooks/hooks.json ~/.claude/hooks.json
   cp -r integrations/claude-code/scripts ~/.claude/scripts
   chmod +x ~/.claude/scripts/*.sh
   ```

3. Add the MCP config above for explicit tool access (search, update, delete).

### How the Plugin Works

| Hook | Action |
|---|---|
| `SessionStart` | Calls `/api/recall` to fetch core memories + instructions |
| `UserPromptSubmit` | Injects recalled memories as additional context |
| `Stop` | Calls `/api/remember` with the last user + assistant exchange |

The plugin handles automatic recall/remember. The MCP tools remain available for explicit operations (searching mid-conversation, deleting a memory, etc.).

## How Memory Works

With `INSTRUCTION_MODE=proactive` (default), Claude Code will:

1. **Load context at session start** -- calls `get_core_memories` (MCP) or gets injected context (plugin)
2. **Search before answering** -- when you ask about past decisions, project context, or "why did we do X?"
3. **Store important context** -- architecture decisions, coding conventions, project structure

## Tips

- **Use `project:<name>` categories** -- tag project-specific memories with `project:myapp`, `project:backend`, etc.
- **Pin architecture decisions** -- important decisions should be pinned so they load at every session start
- **Artifacts for deep analysis** -- save full bug investigation reports or architecture research as artifacts
- **Context type for current work** -- "Currently working on X" uses `context` type with 7-day auto-expiry
- **Shared vs agent-scoped** -- technical preferences and project facts should be shared (no `agent_id`) so they're available across all your tools

See the [coding assistant system prompt guide](../system-prompts/claude-code.md) for more tips on what gets remembered.
