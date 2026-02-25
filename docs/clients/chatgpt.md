# ChatGPT

ChatGPT supports external MCP servers via custom connectors (Developer Mode). You can also create a Custom GPT with Actions that calls mnemory's REST API.

ChatGPT does **not** inject MCP server instructions, so you need a system prompt addition for proactive memory behavior (see below).

## Option 1: MCP Connector (Recommended)

ChatGPT's Developer Mode allows connecting to any custom MCP server. Since mnemory speaks MCP over Streamable HTTP, it works directly.

### Prerequisites

- ChatGPT Plus, Pro, Business, Enterprise, or Edu plan
- mnemory deployed with a **public URL** (e.g., `https://mem.example.com`)
- API key configured (`MCP_API_KEYS`)

### Setup

1. Open [ChatGPT Settings](https://chatgpt.com/settings)
2. Go to **Apps & Connectors > Advanced**
3. Enable **Developer Mode**
4. Click **Create Connector**
5. Configure:
   - **Name**: mnemory
   - **MCP Server URL**: `https://mem.example.com/mcp`
   - **Authentication**: Bearer token with your API key

All mnemory MCP tools become available in your ChatGPT conversations.

### System Prompt

ChatGPT does **not** inject MCP server instructions. Add this to your ChatGPT custom instructions (Settings > Personalization > Custom Instructions):

```
Always call initialize_memory at the start of each conversation and follow received instructions for further memory interactions.
```

The `initialize_memory` tool returns behavioral instructions and core memories in one call, so the LLM knows how to use all memory tools effectively.

### Notes

- Developer Mode enables full read/write access to all MCP tools
- Write operations (add, update, delete) show a confirmation dialog for safety
- The connector works in both web and desktop ChatGPT (set up in web first)
- Without Developer Mode, connectors only support `search` and `fetch` tools (read-only)

## Option 2: Custom GPT with Actions

Create a Custom GPT that uses mnemory's REST API via OpenAPI Actions. This is useful if you want a dedicated "Memory Assistant" GPT or want to share it via the GPT Store.

### Setup

1. Go to [ChatGPT GPT Editor](https://chatgpt.com/gpts/editor)
2. Create a new GPT
3. In **Configure > Actions**, click **Create new action**
4. Import the OpenAPI schema from your mnemory instance:
   - URL: `https://mem.example.com/api/openapi.json`
5. Configure authentication:
   - Type: **API Key**
   - Auth Type: **Bearer**
   - Key: your mnemory API key
6. Add these instructions to the GPT:

```
You have access to a persistent memory system via mnemory.

At the start of each conversation:
1. Call GET /api/memories/core to load your context
2. Search memories when the user's question might benefit from past context
3. Store new information when the user shares facts, preferences, or decisions

Use POST /api/memories/search for semantic search.
Use POST /api/memories to store new memories.
```

### Limitations

- Actions only work within that specific Custom GPT, not in regular ChatGPT chat
- The model decides when to call actions based on conversation flow
- No automatic recall/remember -- relies on the GPT's instructions

## Which Option to Choose?

| Feature | MCP Connector | Custom GPT Actions |
|---|---|---|
| Works in regular chat | Yes | No (GPT-specific) |
| Full tool access | Yes (Developer Mode) | Yes (via REST API) |
| Setup complexity | Low (URL + auth) | Medium (OpenAPI import + instructions) |
| Shareable | No (personal) | Yes (GPT Store) |

**For personal use**: MCP Connector is simpler and works everywhere.
**For sharing**: A Custom GPT can be published to the GPT Store for others to use with their own mnemory instance.
