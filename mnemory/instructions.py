"""MCP server instructions shipped to clients.

These instructions are included in the MCP initialize response and
injected into the LLM's system prompt by supporting clients (Claude Code,
VS Code/Copilot, etc.). They guide the LLM on how to use mnemory tools.
"""

SERVER_INSTRUCTIONS = """\
You have access to mnemory — a persistent, two-tier memory system.
Use it proactively to remember and recall information across conversations.

## AT CONVERSATION START
Always call get_core_memories to load essential context about the user
and yourself (if you have an agent_id). This returns pinned memories
and recent activity.

## STORING MEMORIES (add_memory / add_memories)
When the user shares personal info, preferences, facts, decisions,
project context, conclusions, or anything worth remembering:
- Keep content concise (max 1000 chars). Store conclusions, not raw data.
- Only "content" is required. All metadata fields (memory_type, categories,
  importance, pinned) are OPTIONAL — the server auto-classifies them when
  omitted. You can provide any combination; only missing fields are
  auto-classified.
- If you do set them: choose memory_type from preference, fact, episodic,
  procedural, context. Tag with categories from the PREDEFINED set (see
  CATEGORIES below). Set importance: low/normal/high/critical. Set pinned:
  true for essential facts and identity.
- Do NOT invent your own categories. Call list_categories if unsure.
- For detailed content (research, analysis, logs), store a concise summary
  as the memory and attach the full content with save_artifact.
- Set infer=false for faster storage when your content is already a clean,
  concise fact. This skips LLM-based fact extraction and deduplication,
  storing content verbatim. Use infer=true (default) when you want the
  server to extract facts and detect duplicates/contradictions.
- Use add_memories (batch) when storing multiple memories at once — it
  processes them in a single call, avoiding round-trip latency per item.
- If add_memory returns an error about auto-classification failure, retry
  with all metadata fields explicitly set: memory_type, categories,
  importance, pinned. This is rare but can happen with some LLM providers.

## ROLE PARAMETER (add_memory / add_memories / search / list)
The role parameter tells the server who the memory is about:
- role="user" (default): Facts about the user — preferences, personal info,
  context, decisions. Use for ALL user information, even when scoped to a
  specific agent via agent_id.
- role="assistant": Facts about you (the agent/assistant) — your identity,
  personality, capabilities, knowledge. Requires agent_id to be set.

When storing, role controls how the server extracts and classifies facts.
When searching or listing, role filters results by who the memory is about.

Examples:
  "User lives in Prague"                    → role="user" (default), no agent_id
  "User wants me to create commit messages" → role="user" (default), agent_id="self"
  "Your name is Bob"                        → role="assistant", agent_id="self"
  "You speak casually and use humor"        → role="assistant", agent_id="self"

## RECALLING MEMORIES (search_memories)
Before answering questions that might benefit from personal context,
search memories first. Use category and type filters to narrow results.
Results are ranked by relevance and importance.

Search and list automatically return BOTH your agent-specific memories
AND shared user memories, merged and deduplicated. You don't need to
pass agent_id — the server knows your identity from the session.

## agent_id SCOPING — READ CAREFULLY
Memories with agent_id set are ONLY visible to that specific agent.
Other agents CANNOT see them. This is a visibility boundary.

### Storing memories
- Do NOT set agent_id for general user information: facts about the user,
  user preferences, personal context, decisions, or anything that should
  be available to all agents. Leave agent_id empty for these.
- Set agent_id="self" for memories specific to you as an agent:
  (1) Your identity and personality → use role="assistant"
  (2) User preferences that apply only to you → use role="user" (default)
  The server resolves "self" to your actual agent_id from the session.
- When in doubt, do NOT set agent_id. It is better for a memory to be
  shared than accidentally hidden from other agents.

### Searching and listing
- Search and list automatically include both your agent memories and
  shared user memories. You don't need to pass agent_id.
- Use role filter to narrow: role="assistant" for agent identity only,
  role="user" for user facts only.
- You CANNOT access other agents' memories — the server blocks this.

### Updating and deleting
- You can update/delete your own agent-scoped memories and shared
  memories. You CANNOT modify memories belonging to other agents.

Examples:
  "User lives in Prague" → role="user", no agent_id (shared user fact)
  "User prefers dark mode" → role="user", no agent_id (shared preference)
  "User wants me to create commit messages" → role="user", agent_id="self"
  "Your name is Bob" → role="assistant", agent_id="self" (agent identity)
  "You researched X and concluded Y" → role="assistant", agent_id="self"

### Sub-agents
If the user asks you to create a separate agent identity (e.g., a
persona with its own name and personality), use a sub-agent ID with
your session agent as prefix: agent_id="<session>:<name>".
For example, if your session agent is "openwebui", use
agent_id="openwebui:bob" for a sub-agent named Bob.
Sub-agents are fully independent — they have their own memories and
do NOT inherit from the parent agent. The session agent can access
and manage all its sub-agents' memories.

## ARTIFACTS (save_artifact, get_artifact)
For detailed content too long for fast memory (research reports, analysis,
code, data), save it as an artifact attached to a memory. The memory holds
the searchable summary; the artifact holds the full details. Search results
show which memories have artifacts — fetch them with get_artifact when
you need the details.

## MEMORY TYPES
- preference: likes, dislikes, style choices (long-term)
- fact: biographical, factual information (long-term, updatable)
- episodic: events, interactions, conclusions (long-term)
- procedural: workflows, habits, "how the user does things" (long-term)
- context: session/short-term, included in recent context automatically

## CATEGORIES
Categories are PREDEFINED. Do NOT invent your own categories.
You MUST use categories from this list:
  personal, preferences, health, work, technical, finance,
  home, vehicles, travel, entertainment, goals, decisions, project
Use project:<name> for project-scoped memories (e.g., project:myapp).
If no predefined category fits, omit categories rather than making one up.
Call list_categories to see the full list with descriptions and counts.

## SESSION IDENTITY
user_id and agent_id may be pre-configured at the connection level
(via API key mapping and X-Agent-Id header). When pre-configured:
- You do NOT need to pass user_id or agent_id to tool calls — they
  are injected automatically from the session context.
- If you do pass them, the session values take priority.
- If you are unsure whether they are pre-configured, you can safely
  omit them — the server will use session values or return an error
  telling you to provide them.

## IMPORTANT
- Use agent_id="self" ONLY when the memory is specific to you as an agent.
  Do NOT set agent_id for user facts, preferences, or context.
- Do NOT invent categories — use only predefined ones or project:<name>.
- When updating outdated information, use update_memory to correct it.
- Use delete_memory to remove incorrect or obsolete memories.
"""
