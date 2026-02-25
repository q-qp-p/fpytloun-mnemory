# Claude Code Integration -- Automatic Memory

A hooks-based integration for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) that automatically recalls memories at session start and stores new memories after each exchange.

## How It Works

1. **`UserPromptSubmit`**: Before Claude processes each user message, the hook calls `/api/recall` and injects memories as `additionalContext`. The first call fetches core memories + instructions + relevant search results. Subsequent calls fetch only NEW relevant memories.
2. **`Stop`**: After Claude finishes responding, the hook calls `/api/remember` with the user's message to store new memories (fire-and-forget).

The LLM also has access to mnemory MCP tools for explicit operations (search, update, delete). The hooks handle the automatic parts.

## Setup

### 1. Environment Variables

The hooks make direct HTTP calls to the mnemory REST API. Set these in your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
export MNEMORY_URL=http://localhost:8050
export MNEMORY_API_KEY=your-api-key        # required if mnemory uses MCP_API_KEYS
export MNEMORY_AGENT_ID=claude-code        # optional, defaults to "claude-code"
export MNEMORY_USER_ID=your-username       # optional if using API key mapping
export MNEMORY_SCORE_THRESHOLD=0.5         # optional, min relevance score (0.0-1.0)
```

| Variable | Default | Description |
|---|---|---|
| `MNEMORY_URL` | `http://localhost:8050` | mnemory server URL |
| `MNEMORY_API_KEY` | (empty) | API key for authentication. **Required** if mnemory has `MCP_API_KEYS` set. |
| `MNEMORY_AGENT_ID` | `claude-code` | Agent ID sent to mnemory |
| `MNEMORY_USER_ID` | (empty) | User ID (optional if API key maps to a user) |
| `MNEMORY_SCORE_THRESHOLD` | `0.5` | Minimum relevance score for recalled memories. Higher = fewer but more relevant. |

### 2. Install the Hooks

Copy the hooks configuration and scripts to your Claude Code settings:

```bash
# Copy hooks config
cp integrations/claude-code/hooks.json ~/.claude/settings.json
# Or merge with existing settings.json if you have other hooks

# Copy scripts
cp -r integrations/claude-code/scripts ~/.claude/scripts
chmod +x ~/.claude/scripts/*.sh
```

If you already have a `~/.claude/settings.json`, merge the `hooks` section from `hooks.json` into it.

### 3. Configure MCP Server

Add the MCP server for explicit tool access (search, update, delete):

Add to `~/.claude/claude_code_config.json`:

```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "claude-code"
      }
    }
  }
}
```

## Hybrid Approach

The hooks and MCP server work together:

| Component | Handles |
|---|---|
| **Hooks** (automatic) | Recall before each prompt, remember after each response |
| **MCP tools** (LLM-driven) | Explicit search, add, update, delete when the user asks |

The hooks inject managed instructions into the context that tell the LLM which operations are automatic and which it can use explicitly. This prevents the LLM from calling `initialize_memory` or `get_core_memories` when the hooks already handle those.

## Session Tracking

The hooks use mnemory's server-side sessions to track which memories the client already has:

- First `UserPromptSubmit`: creates a session, returns core memories + instructions + search results
- Subsequent prompts: returns only NEW relevant memories (avoids context bloat)
- Session ID is stored in a temp file (`/tmp/mnemory_session_*`) per Claude Code session

## Troubleshooting

- **No memories appearing**: Check that `MNEMORY_URL` and `MNEMORY_API_KEY` are set. Run `~/.claude/scripts/recall.sh` manually to test.
- **Scripts not executing**: Ensure they're executable (`chmod +x ~/.claude/scripts/*.sh`). Check Claude Code logs for hook errors.
- **LLM still calls initialize_memory**: The hooks inject managed instructions that override MCP tool descriptions. If the LLM still calls init tools, check that the hooks are configured correctly in `settings.json`.
- **Duplicate memory storage**: The extraction pipeline deduplicates against existing memories. If you see duplicates, check that the mnemory server is reachable for the remember calls.
