# Open WebUI — Basic Memory-Enhanced Agent

A minimal setup where mnemory enhances any Open WebUI model with persistent memory. No custom system prompt needed — mnemory's server instructions handle memory behavior automatically.

## Setup

1. Add mnemory as an MCP server in Open WebUI:
   - **Admin Settings > External Tools > Add Server**
   - Type: **MCP (Streamable HTTP)**
   - URL: `http://mnemory:8050/mcp`
   - Auth: **Bearer**, Key: `your-api-key`
   - Custom headers: `X-Agent-Id: open-webui`

2. Enable function calling on your model:
   - **Workspace > Models > Advanced Params > Function Calling: Native**

3. For multi-user setups, enable user info forwarding:
   ```
   # Open WebUI environment
   ENABLE_FORWARD_USER_INFO_HEADERS=true

   # mnemory environment
   MCP_API_KEYS='{"shared-openwebui-key": "*"}'
   ```

## System Prompt

With `INSTRUCTION_MODE=proactive` (the default), the system prompt can be minimal. mnemory's server instructions tell the LLM to search before answering, store new information proactively, and call `get_core_memories` at conversation start.

```
You are a helpful assistant.
```

That's it. The memory behavior is automatic.

### Optional: Add personality flavor

```
You are a helpful assistant. You are friendly, concise, and practical.
You remember things about the user across conversations and use that
context to give better, more personalized answers.
```

## How It Works

1. **Conversation start**: The LLM calls `get_core_memories` and loads pinned user facts and recent context.
2. **During conversation**: The LLM searches memories before answering relevant questions and stores new information the user shares.
3. **Over time**: The agent builds up knowledge about the user — preferences, facts, projects, decisions — making every conversation more personalized.

## Notes

- The `X-Agent-Id: open-webui` header means agent-scoped memories (like agent identity) are tied to Open WebUI. Other clients (Claude Code, Cursor) have their own agent scope but share user memories.
- With `proactive` mode, the LLM is instructed to search and store without being asked. If you prefer manual control, set `INSTRUCTION_MODE=passive` on the mnemory server.
- All user memories are shared across agents by default. Only agent-scoped memories (identity, agent-specific preferences) are isolated.
