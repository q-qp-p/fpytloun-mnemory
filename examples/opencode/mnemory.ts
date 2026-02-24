/**
 * mnemory — Automatic memory recall and storage for OpenCode.
 *
 * This plugin hooks into OpenCode's session lifecycle to:
 * 1. Recall memories on session start (inject into context)
 * 2. Store memories after each exchange (fire-and-forget)
 * 3. Preserve memories across session compaction
 *
 * The LLM also has access to mnemory MCP tools for explicit operations
 * (search, update, delete). The plugin handles the automatic parts.
 *
 * Configuration via environment variables:
 *   MNEMORY_URL             - mnemory server URL (default: http://localhost:8050)
 *   MNEMORY_API_KEY         - API key for authentication
 *   MNEMORY_AGENT_ID        - Agent ID (default: opencode)
 *   MNEMORY_USER_ID         - User ID (optional, can be set via API key mapping)
 *   MNEMORY_SCORE_THRESHOLD - Min relevance score for recalled memories (default: 0.5)
 */

import type { Plugin } from "@opencode-ai/plugin"

interface SessionState {
  mnemorySessionId: string | null
  lastMessageCount: number
}

interface RecallResult {
  session_id?: string
  instructions?: string
  core_memories?: string
  search_results?: Array<{ memory?: string }>
}

export const MnemoryPlugin: Plugin = async ({ client }) => {
  // ── Configuration ────────────────────────────────────────────────
  const baseUrl = process.env.MNEMORY_URL || "http://localhost:8050"
  const apiKey = process.env.MNEMORY_API_KEY || ""
  const agentId = process.env.MNEMORY_AGENT_ID || "opencode"
  const userId = process.env.MNEMORY_USER_ID || ""
  const scoreThreshold = parseFloat(
    process.env.MNEMORY_SCORE_THRESHOLD || "0.5",
  )

  // ── State ────────────────────────────────────────────────────────
  // Track mnemory session per OpenCode session.
  // Bounded to prevent unbounded growth in long-running instances.
  const MAX_SESSIONS = 100
  const sessions = new Map<string, SessionState>()

  // ── API Client ───────────────────────────────────────────────────

  async function callApi(
    path: string,
    payload: object,
  ): Promise<any | null> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "X-Agent-Id": agentId,
    }
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`
    }
    if (userId) {
      headers["X-User-Id"] = userId
    }

    try {
      const resp = await fetch(`${baseUrl}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30_000),
      })
      if (resp.ok) return await resp.json()
    } catch {
      // Graceful degradation — memory is best-effort
    }
    return null
  }

  // ── Helpers ──────────────────────────────────────────────────────

  function trackSession(sessionId: string, state: SessionState): void {
    sessions.set(sessionId, state)
    // Evict oldest entries if over limit
    if (sessions.size > MAX_SESSIONS) {
      const excess = sessions.size - MAX_SESSIONS
      let count = 0
      for (const key of sessions.keys()) {
        if (count >= excess) break
        sessions.delete(key)
        count++
      }
    }
  }

  function buildInjectionText(result: RecallResult): string {
    const parts: string[] = []

    if (result.instructions) {
      parts.push(result.instructions)
    }
    if (result.core_memories) {
      parts.push(result.core_memories)
    }
    if (result.search_results?.length) {
      const memories = result.search_results
        .filter((m) => m.memory)
        .map((m) => `- ${m.memory}`)
        .join("\n")
      if (memories) {
        parts.push(`## Recalled Memories\n${memories}`)
      }
    }

    return parts.join("\n\n")
  }

  /**
   * Recall memories from mnemory and inject into an OpenCode session.
   * Used on session creation and after compaction.
   */
  async function recallAndInject(sessionId: string): Promise<void> {
    const result: RecallResult | null = await callApi("/api/recall", {
      include_instructions: true,
      managed: true,
      score_threshold: scoreThreshold,
    })

    const mnemorySessionId = result?.session_id ?? null

    // Get current message count for remember tracking
    let messageCount = 0
    try {
      const response = await client.session.messages({
        path: { id: sessionId },
      })
      const allMessages = response?.data ?? response ?? []
      if (Array.isArray(allMessages)) {
        messageCount = allMessages.length
      }
    } catch {
      // Non-critical — worst case we re-process some messages
    }

    trackSession(sessionId, { mnemorySessionId, lastMessageCount: messageCount })

    if (!result) return

    const injectionText = buildInjectionText(result)
    if (!injectionText) return

    // Inject memories as a context-only message (no AI reply)
    try {
      await client.session.prompt({
        path: { id: sessionId },
        body: {
          noReply: true,
          parts: [{ type: "text", text: injectionText }],
        },
      })
    } catch {
      // Session may have been deleted between event and injection
    }
  }

  /**
   * Extract the last user + assistant message pair from session messages.
   * Returns null if no new complete exchange is found.
   */
  function extractLastExchange(
    messages: Array<{ info: { role: string }; parts: Array<{ type: string; text?: string }> }>,
    afterIndex: number,
  ): { user: string; assistant: string; newCount: number } | null {
    // Find the last user and assistant messages after afterIndex
    const newMessages = messages.slice(afterIndex)
    if (newMessages.length < 2) return null

    let lastUser = ""
    let lastAssistant = ""

    for (const msg of newMessages) {
      const text = msg.parts
        ?.filter((p) => p.type === "text" && p.text)
        .map((p) => p.text)
        .join("\n")

      if (!text) continue

      if (msg.info.role === "user") {
        lastUser = text
      } else if (msg.info.role === "assistant") {
        lastAssistant = text
      }
    }

    if (!lastUser && !lastAssistant) return null

    return {
      user: lastUser,
      assistant: lastAssistant,
      newCount: messages.length,
    }
  }

  // ── Hooks ────────────────────────────────────────────────────────

  await client.app.log({
    body: {
      service: "mnemory",
      level: "info",
      message: "Plugin initialized",
      extra: { baseUrl, agentId },
    },
  })

  return {
    // Handle session lifecycle events
    event: async ({ event }: { event: { type: string; properties: any } }) => {
      // ── Session Created: Recall + Inject ───────────────────────
      if (event.type === "session.created") {
        const sessionId: string = event.properties?.info?.id
        if (!sessionId) return
        await recallAndInject(sessionId)
      }

      // ── Session Compacted: Fresh Recall + Inject ───────────────
      // After compaction, old messages (including injected memories)
      // are summarized and discarded. The mnemory session's known_ids
      // are stale. Create a fresh mnemory session and re-inject.
      if (event.type === "session.compacted") {
        const sessionId: string = event.properties?.sessionID
        if (!sessionId) return
        await recallAndInject(sessionId)
      }

      // ── Session Idle: Remember ─────────────────────────────────
      // Note: session.idle is deprecated but still emitted.
      // Future versions may need to use session.status instead.
      if (event.type === "session.idle") {
        const sessionId: string = event.properties?.sessionID
        if (!sessionId) return

        const state = sessions.get(sessionId)
        if (!state) return

        try {
          // Get all messages in the session
          const response = await client.session.messages({
            path: { id: sessionId },
          })
          const allMessages = response?.data ?? response ?? []
          if (!Array.isArray(allMessages)) return

          const exchange = extractLastExchange(
            allMessages,
            state.lastMessageCount,
          )
          if (!exchange) return

          // Update tracked count before the async call
          state.lastMessageCount = exchange.newCount

          // Build messages array for remember endpoint
          const messages: Array<{ role: string; content: string }> = []
          if (exchange.user) {
            messages.push({ role: "user", content: exchange.user })
          }
          if (exchange.assistant) {
            messages.push({ role: "assistant", content: exchange.assistant })
          }

          if (messages.length === 0) return

          // Fire-and-forget
          callApi("/api/remember", {
            session_id: state.mnemorySessionId,
            messages,
          }).catch(() => {})
        } catch {
          // Graceful degradation
        }
      }
    },

    // ── Compaction: Preserve Memories ─────────────────────────────
    // Re-inject core memories into the compaction context so the
    // compaction summary preserves them. Without this, memories
    // injected via noReply are lost after compaction.
    "experimental.session.compacting": async (
      _input: any,
      output: { context: string[]; prompt?: string },
    ) => {
      const result: RecallResult | null = await callApi("/api/recall", {
        include_instructions: true,
        managed: true,
        score_mode: "search",
        score_threshold: scoreThreshold,
      })

      if (!result) return

      const parts: string[] = []

      if (result.instructions) {
        parts.push(result.instructions)
      }
      if (result.core_memories) {
        parts.push(result.core_memories)
      }

      if (parts.length > 0) {
        output.context.push(parts.join("\n\n"))
      }
    },
  }
}
