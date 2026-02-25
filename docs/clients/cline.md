# Cline

Cline (VS Code extension) has deep MCP support with a built-in MCP Marketplace. However, it does **not** inject MCP server instructions, so you need a one-line system prompt addition for proactive memory behavior.

## MCP Setup

### Via Cline Settings

1. Open Cline in VS Code
2. Go to **MCP Servers** settings
3. Add a new remote server:
   - URL: `http://localhost:8050/mcp`
   - Headers: `Authorization: Bearer your-api-key`, `X-Agent-Id: cline`

### Via Configuration File

Add to your Cline MCP configuration:

```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "cline"
      }
    }
  }
}
```

## System Prompt

Cline does **not** inject MCP server instructions. Add this to your Cline custom instructions:

```
Always call initialize_memory at the start of each conversation and follow received instructions for further memory interactions.
```

The `initialize_memory` tool returns behavioral instructions and core memories in one call, so the LLM knows how to use all memory tools effectively.

See [system prompt templates](../system-prompts/) for more detailed options.

## How Memory Works

Once the custom instructions are added, Cline will:

1. **Load context at session start** -- calls `initialize_memory`
2. **Search before answering** -- when questions benefit from personal or project context
3. **Store new information** -- architecture decisions, coding conventions, project structure

## Tips

- **Cross-tool memory** -- memories are shared across Cline, Claude Code, Cursor, and any other MCP client
- **Use `project:<name>` categories** -- tag project-specific memories for easy filtering
- **Model flexibility** -- Cline supports many LLM providers; mnemory works with all of them since it uses its own LLM for extraction
