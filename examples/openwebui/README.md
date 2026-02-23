# Open WebUI Filter — Automatic Memory

A filter function that automatically recalls memories before each LLM response and stores new memories after each exchange. No LLM tool-calling required.

## How It Works

1. **Inlet** (before LLM): Calls `/api/recall` with the user's message. Injects memories and instructions into the system prompt.
2. **Outlet** (after LLM): Calls `/api/remember` with the last user + assistant messages. Fire-and-forget — the LLM response is not delayed.

The filter tracks sessions per `chat_id` so subsequent turns only receive NEW relevant memories.

## Setup

### 1. Install the Filter

1. In Open WebUI, go to **Workspace > Functions > Add Function**
2. Paste the contents of `mnemory_filter.py`
3. Save and enable the function

### 2. Configure Valves

Click the gear icon on the filter to set:

| Valve | Default | Description |
|---|---|---|
| `mnemory_url` | `http://localhost:8050` | mnemory server URL |
| `api_key` | (empty) | API key for authentication |
| `agent_id` | `open-webui` | Agent ID sent to mnemory |
| `recall_mode` | `first_only` | When to recall: `first_only` (first message only) or `always` (every message) |
| `recall_search_mode` | `find` | Search mode for first recall: `find` (AI-powered, thorough) or `search` (fast, no LLM) |

### 3. Multi-User Setup (Recommended)

Enable user identity forwarding in Open WebUI so each user gets their own memories:

```bash
# Open WebUI environment
ENABLE_FORWARD_USER_INFO_HEADERS=true

# mnemory environment
MCP_API_KEYS='{"shared-openwebui-key": "*"}'
```

Set `api_key` in the filter valves to `shared-openwebui-key`.

### 4. Single-User Setup

Use a non-wildcard API key that binds to a specific user:

```bash
MCP_API_KEYS='{"your-api-key": "your-username"}'
```

### 5. Optional: Add OpenAPI Tool

For LLM-driven operations (explicit search, delete, etc.), add the OpenAPI spec as a tool:

1. Go to **Workspace > Tools > Add Tool**
2. Import from URL: `http://mnemory:8050/api/openapi.json`
3. Enable on your models

This gives the LLM access to search, update, and delete operations alongside the automatic filter.

## User Valves

Each user can configure:

| Valve | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable/disable memory for this user |
| `show_status` | `true` | Show "Recalling memories..." status in chat |

## Recall Modes

The `recall_mode` valve controls when the filter calls `/api/recall`:

| Mode | Behavior | Latency |
|---|---|---|
| `first_only` (default) | Recall on first message only. Core memories + relevant context loaded once. Subsequent messages add zero latency. | ~200-500ms on first message, 0ms after |
| `always` | Recall on every message. First message uses AI-powered search (thorough). Subsequent messages use fast vector search for new context. | ~200-500ms first, ~50-200ms after |

**Recommendation:** Use `first_only` (default) for most setups. The first message loads all core memories and relevant context. If the user switches topics mid-conversation, the LLM can use MCP/OpenAPI tools to search explicitly.

Use `always` if your conversations frequently shift to unrelated topics and you want automatic context updates on every turn.

### Search Modes

The `recall_search_mode` valve controls how the first recall searches for relevant memories:

| Mode | Behavior | Latency |
|---|---|---|
| `find` (default) | AI-powered: LLM generates up to 5 targeted queries, searches each, reranks by relevance. Can return 0 queries for irrelevant input (e.g., "ok", "format as table") — skipping search entirely. | ~200-500ms (or ~100ms when skipped) |
| `search` | Fast: single vector similarity search, no LLM calls. Good enough for most use cases. | ~50-200ms |

Subsequent calls (in `always` mode) always use fast `search` regardless of this setting.

## Troubleshooting

- **No memories appearing**: Check that `mnemory_url` is reachable from Open WebUI. If both run in Docker, use the container name (e.g., `http://mnemory:8050`).
- **All users share memories**: Enable `ENABLE_FORWARD_USER_INFO_HEADERS=true` in Open WebUI and use a wildcard API key in mnemory.
- **Filter not running**: Ensure the filter is enabled and assigned to the model. Check Open WebUI logs for errors.
