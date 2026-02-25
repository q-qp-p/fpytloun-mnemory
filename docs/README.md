# Documentation

Setup guides, system prompt templates, and integration instructions for mnemory.

## Quick Start

New to mnemory? Start here: **[Quick Start Guide](quickstart.md)** — get running in 5 minutes.

## Client Setup Guides

Step-by-step setup instructions for each supported client. Every guide covers MCP configuration, authentication, and tips.

| Client | MCP | Plugin | Guide |
|---|---|---|---|
| [Claude Code](clients/claude-code.md) | Yes | Yes (automatic recall/remember) | Hooks-based plugin for automatic memory |
| [Claude Desktop](clients/claude-desktop.md) | Yes | -- | MCP config, proactive instructions |
| [ChatGPT](clients/chatgpt.md) | Yes (MCP connector) | -- | Developer Mode connector + GPT Actions |
| [Open WebUI](clients/open-webui.md) | Yes | Yes (filter) | Filter for automatic recall/remember |
| [OpenCode](clients/opencode.md) | Yes | Yes (automatic recall/remember) | Plugin for automatic memory |
| [Cursor](clients/cursor.md) | Yes | -- | MCP config |
| [Windsurf](clients/windsurf.md) | Yes | -- | MCP config |
| [Cline](clients/cline.md) | Yes | -- | MCP config |
| [Continue.dev](clients/continue.md) | Yes | -- | MCP config |
| [Codex CLI](clients/codex.md) | Yes | -- | MCP config |

**MCP** = works via Model Context Protocol (all clients). **Plugin** = dedicated integration with automatic recall/remember (no LLM tool-calling needed).

## System Prompt Templates

Most MCP clients do **not** inject MCP server instructions into the LLM's context. For these clients, add this one-liner to your system prompt or rules:

```
Always call initialize_memory at the start of each conversation and follow received instructions for further memory interactions.
```

Only **Claude Code** is confirmed to inject MCP server instructions automatically. All other clients (Claude Desktop, Cursor, Windsurf, Cline, ChatGPT, etc.) need the system prompt line above. Each [client guide](clients/) has specific instructions.

For more detailed system prompt templates:

| Template | Description |
|---|---|
| [Open WebUI -- Basic](system-prompts/openwebui-basic.md) | Minimal setup, memory works automatically |
| [Open WebUI -- Personality](system-prompts/openwebui-personality.md) | Agent with evolving identity and "soul" (sub-agent pattern) |
| [Claude Code / OpenCode](system-prompts/claude-code.md) | Coding assistant with project context memory |

## Integrations (Plugins)

Runnable plugin code lives in the [`integrations/`](../integrations/) directory:

| Integration | Description |
|---|---|
| [Claude Code](../integrations/claude-code/) | Hooks-based plugin for automatic recall/remember |
| [OpenCode](../integrations/opencode/) | TypeScript plugin for automatic recall/remember |
| [Open WebUI](../integrations/openwebui/) | Filter function for automatic recall/remember |
| [Grafana](../integrations/grafana/) | Pre-built monitoring dashboard |

## Configuration Reference

See the main [README.md](../README.md) for the full configuration reference (environment variables, memory model, REST API, architecture).
