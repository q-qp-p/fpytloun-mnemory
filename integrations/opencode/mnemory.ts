/**
 * mnemory — Automatic memory recall and storage for OpenCode.
 *
 * This plugin hooks into OpenCode's session lifecycle to:
 * 1. Recall memories and inject into the system prompt (no race conditions)
 * 2. Store memories after each exchange (fire-and-forget)
 * 3. Preserve memories across session compaction
 *
 * Memory injection uses `experimental.chat.system.transform` to add memories
 * to the system prompt before each LLM call. This avoids the race condition
 * of injecting noReply messages that can appear mid-conversation.
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
  recallPromise: Promise<RecallResult | null> | null
  recallResult: RecallResult | null
}

interface RecallResult {
  session_id?: string
  instructions?: string
  core_memories?: string
  search_results?: Array<{ memory?: string }>
}

export const MnemoryPlugin: Plugin = async ({ client, worktree, directory }) => {
  // ── Configuration ────────────────────────────────────────────────
  const baseUrl = process.env.MNEMORY_URL || "http://localhost:8050"
  const apiKey = process.env.MNEMORY_API_KEY || ""
  const agentId = process.env.MNEMORY_AGENT_ID || "opencode"
  const workingDirectory = worktree || directory || ""
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

  function getOrCreateState(sessionId: string): SessionState {
    let state = sessions.get(sessionId)
    if (!state) {
      state = {
        mnemorySessionId: null,
        lastMessageCount: 0,
        recallPromise: null,
        recallResult: null,
      }
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
    return state
  }

  /**
   * Start a recall API call and store the promise in session state.
   * Does NOT await — the promise is resolved later in system.transform.
   */
  function startRecall(state: SessionState): void {
    state.recallResult = null
    state.recallPromise = callApi("/api/recall", {
      include_instructions: true,
      managed: true,
      score_threshold: scoreThreshold,
      ...(workingDirectory && { context: `Working directory: ${workingDirectory}` }),
    }).then((result: RecallResult | null) => {
      state.mnemorySessionId = result?.session_id ?? null
      state.recallResult = result
      return result
    })
  }

  function buildSystemText(result: RecallResult): string {
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
   * Extract the last user + assistant message pair from session messages.
   * Returns null if no new complete exchange is found.
   *
   * Only the LAST non-synthetic text part of each message is used.
   * This captures the final conclusion/response and discards all intermediate
   * narration ("I'm reading file X...", "I can see that...") that the LLM
   * emits during agentic work — which would otherwise pollute memory with
   * internal implementation details rather than meaningful conclusions.
   *
   * Synthetic parts (injected by the system, not generated by the LLM) are
   * always excluded.
   */
  function extractLastExchange(
    messages: Array<{ info: { role: string }; parts: Array<{ type: string; text?: string; synthetic?: boolean }> }>,
    afterIndex: number,
  ): { user: string; assistant: string; newCount: number } | null {
    const newMessages = messages.slice(afterIndex)
    if (newMessages.length < 2) return null

    let lastUser = ""
    let lastAssistant = ""

    for (const msg of newMessages) {
      // Take only the last non-empty, non-synthetic text part.
      // Skipping synthetic parts avoids re-storing injected system content
      // (e.g., recalled memories injected into the system prompt).
      // Taking only the last part avoids storing intermediate narration —
      // only the final response/conclusion is sent to /api/remember.
      const textParts = msg.parts?.filter(
        (p) => p.type === "text" && p.text && !p.synthetic,
      ) ?? []
      const lastPart = textParts[textParts.length - 1]
      const text = lastPart?.text ?? ""

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
    // ── System Prompt Injection ─────────────────────────────────
    // Inject memories into the system prompt before each LLM call.
    // On the first call, awaits the recall promise (adds ~1-2s).
    // On subsequent calls, uses the cached result (instant).
    // Also handles resumed sessions where session.created didn't fire.
    "experimental.chat.system.transform": async (
      input: { sessionID?: string },
      output: { system: string[] },
    ) => {
      const sessionId = input.sessionID
      if (!sessionId) return

      const state = getOrCreateState(sessionId)

      // If no recall has been started (e.g., resumed session), start one
      if (!state.recallPromise && !state.recallResult) {
        startRecall(state)
      }

      // Await the recall promise if not yet resolved
      if (state.recallPromise && !state.recallResult) {
        try {
          await state.recallPromise
        } catch {
          // Graceful degradation
        }
        state.recallPromise = null
      }

      // Inject cached result into system prompt
      if (state.recallResult) {
        const text = buildSystemText(state.recallResult)
        if (text) {
          output.system.push(text)
        }
      }
    },

    // ── Session Lifecycle Events ────────────────────────────────
    event: async ({ event }: { event: { type: string; properties: any } }) => {
      // ── Session Created: Pre-fetch Recall ────────────────────
      // Kick off recall early so it's ready by the time the first
      // LLM call hits experimental.chat.system.transform.
      if (event.type === "session.created") {
        const sessionId: string = event.properties?.info?.id
        if (!sessionId) return

        const state = getOrCreateState(sessionId)
        startRecall(state)
      }

      // ── Session Compacted: Reset + Fresh Recall ──────────────
      // After compaction, old context is summarized. Reset the
      // cached recall result so the next system.transform call
      // fetches fresh memories.
      if (event.type === "session.compacted") {
        const sessionId: string = event.properties?.sessionID
        if (!sessionId) return

        const state = getOrCreateState(sessionId)
        startRecall(state)
      }

      // ── Session Idle: Remember ───────────────────────────────
      if (event.type === "session.idle") {
        const sessionId: string = event.properties?.sessionID
        if (!sessionId) return

        const state = sessions.get(sessionId)
        if (!state) return

        try {
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

    // ── Compaction: Preserve Memories ───────────────────────────
    // Inject core memories into the compaction context so the
    // compaction summary preserves them.
    "experimental.session.compacting": async (
      input: { sessionID: string },
      output: { context: string[]; prompt?: string },
    ) => {
      // Use cached result if available, otherwise fetch fresh
      const state = input.sessionID ? sessions.get(input.sessionID) : null
      const result: RecallResult | null = state?.recallResult ?? await callApi("/api/recall", {
        include_instructions: true,
        managed: true,
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
