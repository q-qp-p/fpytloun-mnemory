# Claude Desktop

Claude Desktop supports MCP servers natively. However, it does **not** inject MCP server instructions into the LLM's system prompt, so you need a one-line system prompt addition for proactive memory behavior.

## MCP Setup

### Streamable HTTP (Remote Server)

For a remote mnemory instance (available on Pro, Max, Team, Enterprise plans):

1. Open Claude Desktop settings
2. Navigate to the MCP servers section
3. Add a new Streamable HTTP server:

```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "https://mem.example.com/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "claude-desktop"
      }
    }
  }
}
```

### Local STDIO

For a local mnemory instance, configure in `claude_desktop_config.json`:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Since mnemory is an HTTP server (not stdio), you'll need a wrapper or use the Streamable HTTP transport directly if your plan supports it.

## System Prompt

Claude Desktop does **not** inject MCP server instructions into the LLM's context. Add this to your system prompt or project instructions:

```
Always call initialize_memory at the start of each conversation and follow received instructions for further memory interactions.
```

The `initialize_memory` tool returns behavioral instructions and core memories in one call, so the LLM knows how to use all memory tools effectively.

See [system prompt templates](../system-prompts/) for more detailed options.

## How Memory Works

Once the system prompt line is added, Claude will:

1. **Load context at conversation start** -- calls `initialize_memory` to load instructions + pinned facts + recent context
2. **Search before answering** -- when questions benefit from personal context
3. **Store new information** -- when you share preferences, facts, or decisions

## Limitations

- **No hooks or plugin system** -- unlike Claude Code, Claude Desktop has no lifecycle hooks. Memory relies on the LLM following the instructions returned by `initialize_memory`.
- **Claude's built-in memory** -- Claude Desktop has its own native memory feature. It works independently from mnemory. You can use both, but be aware of potential overlap.

## Tips

- **Pin important facts** -- pinned memories load at every conversation start via `initialize_memory`, making them reliably available
- **Start conversations with context** -- if Claude doesn't automatically load memories, you can say "check my memories" or "what do you remember about me?" to prompt it
