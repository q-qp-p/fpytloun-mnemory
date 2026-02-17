# Claude Code / Opencode — Coding Assistant with Memory

Use mnemory with Claude Code or Opencode to remember project context, architecture decisions, coding preferences, and technical knowledge across sessions.

## Setup

Add to your MCP configuration (Claude Code: `~/.claude/claude_code_config.json`, Opencode: `opencode.json`):

```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "https://mem.example.com/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "claude-code"
      }
    }
  }
}
```

With `MCP_API_KEYS` configured on the server, user_id and agent_id are resolved automatically. No system prompt changes needed.

## How It Works

Claude Code and Opencode natively support MCP server instructions. When connected to mnemory with `INSTRUCTION_MODE=proactive` (the default), the coding assistant will:

1. **Load context at session start**: Calls `get_core_memories` to load your technical preferences, project context, and recent activity.

2. **Search before answering**: When you ask about project architecture, past decisions, or "why did we do X?", the assistant searches memory for relevant context.

3. **Remember important context**: When you make architecture decisions, establish coding conventions, or share project context, it stores this automatically.

## What Gets Remembered

The assistant will naturally store things like:

| What | Type | Category | Example |
|---|---|---|---|
| Tech stack preferences | preference | technical | "Prefers TypeScript over JavaScript" |
| Coding conventions | preference | technical | "Uses single quotes, 2-space indent" |
| Architecture decisions | fact | project:myapp | "Chose PostgreSQL for the main DB" |
| Project structure | fact | project:myapp | "API lives in src/api/, uses Express" |
| Deployment setup | procedural | technical | "Deploy via GitHub Actions to AWS ECS" |
| Current sprint tasks | context | work | "Working on auth refactor this week" |
| Bug investigations | episodic | project:myapp | "Memory leak was caused by unclosed DB connections" |

## System Prompt (Optional)

Claude Code / Opencode don't require a custom system prompt — MCP server instructions handle everything. But if you want to add project-specific guidance:

```
When working in this project, use the project:myapp category for
project-specific memories. Key architecture decisions and coding
conventions should be stored as pinned facts.
```

## Tips

- **Use `project:<name>` categories**: Tag project-specific memories with `project:myapp`, `project:backend`, etc. This makes searching by project easy.
- **Pin architecture decisions**: Important decisions (database choice, framework, deployment strategy) should be pinned so they load at every session start.
- **Artifacts for deep analysis**: When investigating a complex bug or doing architecture research, save the full analysis as an artifact. The memory holds the conclusion; the artifact holds the reasoning.
- **Context type for current work**: "Currently working on X" is a `context` type memory with 7-day TTL. It'll naturally expire when you move on.
- **Shared vs agent-scoped**: Technical preferences and project facts should be shared (no agent_id) so they're available across all your coding tools. Only agent-specific behavior rules need agent_id.

## Multi-Project Setup

If you work on multiple projects, use category prefixes consistently:

```
project:frontend    — React app memories
project:backend     — API server memories
project:infra       — Infrastructure/DevOps memories
project:mobile      — Mobile app memories
```

The assistant can then search within a specific project:
```
search_memories(query="database schema", categories=["project:backend"])
```
