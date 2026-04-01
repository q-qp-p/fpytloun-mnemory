# Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research supports plugins with lifecycle hooks, making it ideal for automatic memory integration.

## Integration Options

| Method | What it does | Setup effort |
|---|---|---|
| **Plugin** (recommended) | Automatic recall/remember + 16 explicit memory tools | Install plugin + set env vars |
| **MCP only** | LLM-driven memory via tool calls | Add MCP server config |

## Plugin Setup (Recommended)

The plugin provides automatic recall before each turn, automatic memory extraction after each exchange, and 16 explicit memory tools for the LLM.

### Install

```bash
pip install hermes-mnemory
```

Or copy the plugin directory:

```bash
cp -r integrations/hermes ~/.hermes/plugins/mnemory
```

### Configure

Add to `~/.hermes/.env`:

```bash
MNEMORY_URL=http://localhost:8050
MNEMORY_API_KEY=your-api-key
```

See [`integrations/hermes/`](../../integrations/hermes/) for the full configuration reference.

### Verify

Start Hermes and check the plugin loaded:

```
/plugins
```

You should see:

```
Plugins (1):
  ✓ mnemory v0.1.0 (16 tools, 4 hooks)
```

## MCP Setup

Alternatively, add mnemory as an MCP server. The LLM decides when to use memory tools (no automatic recall/remember).

In `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  mnemory:
    url: "http://localhost:8050/mcp"
    headers:
      X-Agent-Id: hermes
      Authorization: "Bearer your-api-key"
```

**Note**: Add this to your system prompt or persona (`SOUL.md`) for best results:

```
Always call initialize_memory at the start of each conversation and follow received instructions for further memory interactions.
```

## Built-in Memory

Hermes has a built-in memory system using `MEMORY.md` and `USER.md` files. The mnemory plugin can coexist with it, or you can disable the built-in memory:

```yaml
# ~/.hermes/config.yaml
memory:
  memory_enabled: false
  user_profile_enabled: false
```
