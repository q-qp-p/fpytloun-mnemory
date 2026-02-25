# Open WebUI

Open WebUI supports MCP servers and has a filter function system that enables automatic memory recall and storage.

## Integration Options

| Method | What it does | Setup effort |
|---|---|---|
| **MCP only** | LLM-driven memory via tool calls | Add MCP server + one line in system prompt |
| **MCP + Filter** (recommended) | Automatic recall/remember + LLM tools for explicit ops | Add MCP server + install filter |
| **Filter only** | Automatic recall/remember, no explicit tools | Install filter only |

## MCP Setup

1. Go to **Admin Settings > External Tools > Add Server**
2. Type: **MCP (Streamable HTTP)**
3. URL: `http://mnemory:8050/mcp` (same Docker network) or `https://mem.example.com/mcp` (via ingress)
4. Auth: **Bearer**, Key: `your-api-key`
5. Custom headers: `X-Agent-Id: open-webui`
6. Enable tools on models: **Workspace > Models > Advanced Params > Function Calling: Native**

**Important**: Open WebUI does not inject MCP server instructions into the LLM's system prompt. Add this to your model's system prompt:

```
Always call initialize_memory at the start of each conversation and follow received instructions for further memory interactions.
```

Or use the full system prompt templates in [docs/system-prompts/](../system-prompts/).

## Filter Setup (Recommended)

The filter automatically recalls memories before each LLM response and stores new memories after each exchange. No LLM tool-calling required.

See [`integrations/openwebui/`](../../integrations/openwebui/) for the filter code and detailed setup instructions.

### Quick Install

1. In Open WebUI, go to **Workspace > Functions > Add Function**
2. Paste the contents of [`integrations/openwebui/mnemory_filter.py`](../../integrations/openwebui/mnemory_filter.py)
3. Save and enable the function
4. Configure valves (gear icon): set `mnemory_url` and `api_key`

The filter handles recall on inlet (before LLM) and remember on outlet (after LLM). The MCP tools remain available for explicit operations.

## Multi-User Setup (Recommended)

Enable user identity forwarding so each user gets their own memories:

```bash
# Open WebUI environment
ENABLE_FORWARD_USER_INFO_HEADERS=true

# mnemory environment
MCP_API_KEYS='{"shared-openwebui-key": "*"}'
```

Set `api_key` in the filter valves (or MCP auth) to `shared-openwebui-key`.

## Single-User Setup

Use a non-wildcard API key that binds directly to a user:

```bash
MCP_API_KEYS='{"your-api-key": "your-username"}'
```

## System Prompt Templates

- [Basic](../system-prompts/openwebui-basic.md) -- minimal setup, memory works automatically
- [Personality](../system-prompts/openwebui-personality.md) -- agent with evolving identity (sub-agent pattern)

## Tips

- **Filter + MCP together** -- the filter handles automatic recall/remember; MCP tools let the LLM do explicit search, update, delete when the user asks
- **`recall_score_threshold`** -- set to 0.5 (default) to filter out weak matches. Lower for more context, higher for precision.
- **`recall_search_mode`** -- use `search` (default) for fast recall, `find` for AI-powered thorough search (slower but better for first messages)
