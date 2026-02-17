# Quick Start: Your First Memory-Powered Agent

Get a memory-enhanced AI assistant running in 5 minutes.

## Step 1: Start mnemory

```bash
git clone https://github.com/fpytloun/mnemory.git
cd mnemory
export OPENAI_API_KEY=sk-your-key
docker-compose up -d
```

mnemory is now running at `http://localhost:8050/mcp`.

**Alternative** (no Docker):
```bash
pip install mnemory[all]
export LLM_API_KEY=$OPENAI_API_KEY
export VECTOR_BACKEND=chroma
export ARTIFACT_BACKEND=filesystem
mnemory
```

## Step 2: Connect Your Client

### Open WebUI

1. **Admin Settings > External Tools > Add Server**
2. Type: **MCP (Streamable HTTP)**
3. URL: `http://mnemory:8050/mcp` (or `http://localhost:8050/mcp`)
4. Custom headers: `X-Agent-Id: open-webui`
5. Enable on your model: **Workspace > Models > Advanced Params > Function Calling: Native**

### Claude Code / Opencode

Add to your MCP config:
```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "X-Agent-Id": "claude-code"
      }
    }
  }
}
```

## Step 3: Start Chatting

That's it. With the default `INSTRUCTION_MODE=proactive`, your agent will automatically:

- **Load your context** at the start of each conversation
- **Search memories** before answering questions that benefit from personal context
- **Store new information** when you share personal facts, preferences, or decisions

Try saying something like:
- "My name is Alex and I'm a frontend developer in Berlin"
- "I prefer TypeScript over JavaScript and use React for most projects"
- "I'm currently working on a dashboard app called MetricsHub"

Then start a **new conversation** and ask:
- "What tech stack should I use for my next project?"
- "What am I working on right now?"

The agent will search its memories and give personalized answers.

## Step 4 (Optional): Add Authentication

For production or multi-user setups, add API key authentication:

```bash
# mnemory environment
MCP_API_KEYS='{"your-secret-key": "alex"}'
```

Then add the key to your client configuration:
- Open WebUI: Auth type **Bearer**, Key: `your-secret-key`
- Claude Code: Add `"Authorization": "Bearer your-secret-key"` to headers

## Step 5 (Optional): Create a Personality Agent

Want an agent with its own evolving personality? See [openwebui-personality.md](openwebui-personality.md) for the full guide. Short version:

1. Set `INSTRUCTION_MODE=personality` on the server (or add the personality snippet to the agent's system prompt)
2. Create a new model in Open WebUI with a system prompt that defines the agent's character
3. The agent develops and maintains its identity through memory

## What's Happening Under the Hood

```
Conversation start:
  Agent calls get_core_memories()
  → Loads pinned facts: "Alex is a frontend developer in Berlin"
  → Loads recent context: "Working on MetricsHub dashboard"

You ask: "What React library should I use for charts?"
  Agent calls search_memories("React charts dashboard")
  → Finds: "Prefers TypeScript", "Working on MetricsHub dashboard"
  → Gives personalized recommendation based on your stack and project

You say: "I decided to go with Recharts for the dashboard"
  Agent calls add_memory("Chose Recharts for MetricsHub dashboard charts")
  → Stored as: type=decision, category=project:metricshub
  → Available in all future conversations
```

## Next Steps

- [openwebui-basic.md](openwebui-basic.md) — Detailed Open WebUI setup
- [openwebui-personality.md](openwebui-personality.md) — Agents with evolving personality
- [claude-code.md](claude-code.md) — Coding assistants with memory
- [README.md](../../README.md) — Full configuration reference
