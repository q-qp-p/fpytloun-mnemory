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

## STORING MEMORIES (add_memory)
When the user shares personal info, preferences, facts, decisions,
project context, conclusions, or anything worth remembering:
- Keep content concise (max 1000 chars). Store conclusions, not raw data.
- Choose memory_type: preference, fact, episodic, procedural, or context.
- Tag with categories from the PREDEFINED set (see CATEGORIES below).
  Do NOT invent your own categories. Call list_categories if unsure.
- Set importance: low/normal/high/critical. Critical memories get boosted
  in search results.
- Set pinned: true for memories that should always load at conversation
  start (key facts, identity, core preferences).
- For detailed content (research, analysis, logs), store a concise summary
  as the memory and attach the full content with save_artifact.

## RECALLING MEMORIES (search_memories)
Before answering questions that might benefit from personal context,
search memories first. Use category and type filters to narrow results.
Results are ranked by relevance and importance.
When searching, omit agent_id to find shared user memories. Only set
agent_id if you specifically need your own agent-scoped memories.

## agent_id SCOPING — READ CAREFULLY
Memories with agent_id set are ONLY visible to that specific agent.
Other agents CANNOT see them. This is a visibility boundary.

- Do NOT set agent_id for general user information: facts about the user,
  user preferences, personal context, decisions, or anything that should
  be available to all agents. Leave agent_id empty for these.
- ONLY set agent_id for memories that are specific to you as an agent:
  your identity, your personality, your name, knowledge that only you
  should have.
- When in doubt, do NOT set agent_id. It is better for a memory to be
  shared than accidentally hidden from other agents.

Examples:
  "User lives in Prague" → do NOT set agent_id (shared user fact)
  "User prefers dark mode" → do NOT set agent_id (shared preference)
  "Your name is Bob" → set agent_id (agent identity)
  "You researched X and concluded Y" → set agent_id (agent knowledge)

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
- Do NOT set agent_id unless the memory is specific to you as an agent.
- Do NOT invent categories — use only predefined ones or project:<name>.
- When updating outdated information, use update_memory to correct it.
- Use delete_memory to remove incorrect or obsolete memories.
"""
