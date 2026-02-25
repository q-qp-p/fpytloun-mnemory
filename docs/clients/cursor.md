# Cursor

Cursor has native MCP support. However, it does **not** inject MCP server instructions into the LLM's context, so you need a one-line addition to your rules for proactive memory behavior.

## MCP Setup

Add to `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "cursor"
      }
    }
  }
}
```

Or configure via **Cursor Settings > MCP > Add Server**.

## System Prompt

Cursor does **not** inject MCP server instructions. Add this to your Cursor rules (`.cursor/rules/` or global rules):

```
Always call initialize_memory at the start of each conversation and follow received instructions for further memory interactions.
```

The `initialize_memory` tool returns behavioral instructions and core memories in one call, so the LLM knows how to use all memory tools effectively.

See [system prompt templates](../system-prompts/) for more detailed options.

## How Memory Works

Once the rules line is added, Cursor will:

1. **Load context at session start** -- calls `initialize_memory`
2. **Search before answering** -- when questions benefit from personal or project context
3. **Store new information** -- architecture decisions, coding conventions, project structure

## Tips

- **Use `project:<name>` categories** -- tag project-specific memories for easy filtering
- **Pin architecture decisions** -- they load at every session start
- **Shared memories** -- memories stored without `agent_id` are shared across all your tools (Claude Code, Cursor, etc.)
- **Cursor Rules** -- you can add more detailed memory-related instructions to `.cursor/rules/` for customized behavior
