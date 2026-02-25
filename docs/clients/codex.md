# Codex CLI

OpenAI's Codex CLI supports MCP servers natively. However, it likely does **not** inject MCP server instructions, so you need a one-line system prompt addition for proactive memory behavior.

## MCP Setup

Add to your Codex CLI MCP configuration:

```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "codex"
      }
    }
  }
}
```

## System Prompt

Codex CLI likely does **not** inject MCP server instructions. Add this to your system prompt or instructions:

```
Always call initialize_memory at the start of each conversation and follow received instructions for further memory interactions.
```

The `initialize_memory` tool returns behavioral instructions and core memories in one call, so the LLM knows how to use all memory tools effectively.

See [system prompt templates](../system-prompts/) for more detailed options.

## How Memory Works

Once the system prompt line is added, Codex will:

1. **Load context at session start** -- calls `initialize_memory`
2. **Search before answering** -- when questions benefit from personal or project context
3. **Store new information** -- architecture decisions, coding conventions, project structure

## Tips

- **Cross-tool memory** -- memories are shared across Codex, Claude Code, Cursor, and any other MCP client
- **Use `project:<name>` categories** -- tag project-specific memories for easy filtering
- **Pin architecture decisions** -- they load at every session start
