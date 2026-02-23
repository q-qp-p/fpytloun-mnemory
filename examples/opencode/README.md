# OpenCode Plugin — Automatic Memory

A TypeScript plugin skeleton for [OpenCode](https://opencode.ai) that hooks into session lifecycle for automatic memory recall and storage.

> **Note**: This is a skeleton implementation. OpenCode's plugin API is still evolving — the hooks for injecting context and extracting messages may need adjustment as the API stabilizes.

## How It Works

1. **session.created**: Calls `/api/recall` to initialize memory and load core context.
2. **message.part.updated**: After each assistant response, calls `/api/remember` to store new memories (fire-and-forget).

## Setup

### 1. Environment Variables

```bash
export MNEMORY_URL=http://localhost:8050
export MNEMORY_API_KEY=your-api-key
export MNEMORY_AGENT_ID=opencode  # optional, defaults to "opencode"
```

### 2. Install the Plugin

Copy `mnemory.ts` to your OpenCode plugins directory (e.g., `.opencode/plugins/`).

### 3. Configure OpenCode

Add to your `opencode.json`:

```json
{
  "plugins": [".opencode/plugins/mnemory.ts"]
}
```

### 4. MCP Server (for LLM-driven operations)

For explicit operations (search, update, delete), also configure the MCP server:

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

## Hybrid Approach

The plugin handles automatic recall/remember. The LLM still has access to MCP tools for explicit operations like searching mid-conversation or deleting a memory the user asks to forget.
