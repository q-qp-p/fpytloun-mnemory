import type { PluginContext } from "@opencode/plugin";

export default function mnemoryPlugin(ctx: PluginContext) {
  const baseUrl = process.env.MNEMORY_URL || "http://localhost:8050";
  const apiKey = process.env.MNEMORY_API_KEY || "";
  const agentId = process.env.MNEMORY_AGENT_ID || "opencode";

  let sessionId: string | null = null;

  async function callApi(path: string, payload: object): Promise<any> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "X-Agent-Id": agentId,
    };
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }

    try {
      const resp = await fetch(`${baseUrl}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      if (resp.ok) return await resp.json();
    } catch {
      // Graceful degradation
    }
    return null;
  }

  return {
    hooks: {
      "session.created": async () => {
        // Initialize memory on session start
        const result = await callApi("/api/recall", {
          include_instructions: true,
          managed: true,
        });
        if (result?.session_id) {
          sessionId = result.session_id;
        }
        // TODO: inject result into session context via ctx API
        // when OpenCode exposes the necessary hooks
      },

      "message.part.updated": async (event: any) => {
        // After assistant response: store memories (fire-and-forget)
        if (event?.part?.role !== "assistant") return;

        callApi("/api/remember", {
          session_id: sessionId,
          messages: [
            // TODO: extract last user + assistant messages from event
            // when OpenCode exposes the necessary hooks
          ],
        }).catch(() => {});
      },
    },
  };
}
