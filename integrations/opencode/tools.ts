/**
 * Custom tool definitions for the mnemory OpenCode plugin.
 *
 * 16 tools covering the full mnemory REST API surface.
 * These replace the need for a separate MCP server configuration.
 */

import { tool } from "@opencode-ai/plugin";
import type { MnemoryClient } from "./client.js";
import type { Logger } from "./helpers.js";

type ToolContext = {
  client: MnemoryClient;
  logger: Logger;
};

export function createTools(ctx: ToolContext) {
  const { client, logger } = ctx;

  return {
    // ------------------------------------------------------------------
    // Search tools
    // ------------------------------------------------------------------

    memory_search: tool({
      description:
        "Search long-term memories by semantic similarity. Returns relevant memories ranked by relevance and importance.",
      args: {
        query: tool.schema.string().describe("What to search for (natural language)"),
        limit: tool.schema
          .number()
          .min(1)
          .max(50)
          .optional()
          .describe("Max results to return (default 10)"),
        memory_type: tool.schema
          .string()
          .optional()
          .describe("Filter by type: preference, fact, episodic, procedural, context"),
        categories: tool.schema
          .array(tool.schema.string())
          .optional()
          .describe("Filter by categories"),
        role: tool.schema
          .string()
          .optional()
          .describe("Filter by role: user or assistant"),
        include_decayed: tool.schema
          .boolean()
          .optional()
          .describe("Include expired/decayed memories (default false)"),
        date_start: tool.schema
          .string()
          .optional()
          .describe("Filter start date (YYYY-MM-DD)"),
        date_end: tool.schema
          .string()
          .optional()
          .describe("Filter end date (YYYY-MM-DD)"),
        labels: tool.schema
          .record(tool.schema.string(), tool.schema.unknown())
          .optional()
          .describe("Filter by label key-value pairs (AND logic)"),
      },
      async execute(args) {
        const result = await client.searchMemories({
          query: args.query,
          limit: args.limit,
          memoryType: args.memory_type,
          categories: args.categories,
          role: args.role,
          includeDecayed: args.include_decayed,
          dateStart: args.date_start,
          dateEnd: args.date_end,
          labels: args.labels,
        });
        if (!result) return "Memory search unavailable — mnemory server may be offline.";
        if (result.results.length === 0) return "No memories found matching your query.";
        const lines = result.results.map((m, i) => {
          const score = m.score != null ? ` (${Math.round(m.score * 100)}%)` : "";
          const type = m.memory_type ? ` [${m.memory_type}]` : "";
          return `${i + 1}. ${m.memory}${score}${type} (id: ${m.id})`;
        });
        return `Found ${result.results.length} memories:\n${lines.join("\n")}`;
      },
    }),

    memory_find: tool({
      description:
        "Find memories relevant to a complex question using AI-powered search. " +
        "Generates multiple targeted searches covering different angles and associations, " +
        "then reranks results by relevance. Slower than memory_search (2 extra LLM calls) " +
        "but higher quality for complex, multi-faceted questions.",
      args: {
        question: tool.schema.string().describe("The question in natural language"),
        limit: tool.schema
          .number()
          .min(1)
          .max(100)
          .optional()
          .describe("Max results (default 10)"),
        memory_type: tool.schema
          .string()
          .optional()
          .describe("Filter by type: preference, fact, episodic, procedural, context"),
        categories: tool.schema
          .array(tool.schema.string())
          .optional()
          .describe("Filter by categories"),
        role: tool.schema
          .string()
          .optional()
          .describe("Filter by role: user or assistant"),
        include_decayed: tool.schema
          .boolean()
          .optional()
          .describe("Include expired/decayed memories (default false)"),
        context: tool.schema
          .string()
          .optional()
          .describe("Optional context hint for query generation"),
        labels: tool.schema
          .record(tool.schema.string(), tool.schema.unknown())
          .optional()
          .describe("Filter by label key-value pairs (AND logic)"),
      },
      async execute(args) {
        const result = await client.findMemories({
          question: args.question,
          limit: args.limit,
          memoryType: args.memory_type,
          categories: args.categories,
          role: args.role,
          includeDecayed: args.include_decayed,
          context: args.context,
          labels: args.labels,
        });
        if (!result) return "Memory find unavailable — mnemory server may be offline.";
        if (result.results.length === 0) return "No memories found for your question.";
        const lines = result.results.map((m, i) => {
          const score = m.score != null ? ` (${Math.round(m.score * 100)}%)` : "";
          const type = m.memory_type ? ` [${m.memory_type}]` : "";
          return `${i + 1}. ${m.memory}${score}${type} (id: ${m.id})`;
        });
        return `Found ${result.results.length} memories:\n${lines.join("\n")}`;
      },
    }),

    memory_ask: tool({
      description:
        "Ask a question and get a human-readable answer based on stored memories. " +
        "Uses AI-powered search internally, then generates a natural language answer. " +
        "Most expensive operation (3 LLM calls). Use when you need a synthesized answer " +
        "rather than raw memory results.",
      args: {
        question: tool.schema.string().describe("The question in natural language"),
        limit: tool.schema
          .number()
          .min(1)
          .max(100)
          .optional()
          .describe("Max supporting memories (default 10)"),
        memory_type: tool.schema
          .string()
          .optional()
          .describe("Filter by type: preference, fact, episodic, procedural, context"),
        categories: tool.schema
          .array(tool.schema.string())
          .optional()
          .describe("Filter by categories"),
        role: tool.schema
          .string()
          .optional()
          .describe("Filter by role: user or assistant"),
        include_decayed: tool.schema
          .boolean()
          .optional()
          .describe("Include expired/decayed memories (default false)"),
        context: tool.schema
          .string()
          .optional()
          .describe("Optional context hint for query generation"),
        include_memories: tool.schema
          .boolean()
          .optional()
          .describe("Include supporting memories in response (default false)"),
        labels: tool.schema
          .record(tool.schema.string(), tool.schema.unknown())
          .optional()
          .describe("Filter by label key-value pairs (AND logic)"),
      },
      async execute(args) {
        const result = await client.askMemories({
          question: args.question,
          limit: args.limit,
          memoryType: args.memory_type,
          categories: args.categories,
          role: args.role,
          includeDecayed: args.include_decayed,
          context: args.context,
          includeMemories: args.include_memories,
          labels: args.labels,
        });
        if (!result) return "Memory ask unavailable — mnemory server may be offline.";
        const parts = [result.answer];
        if (result.results && result.results.length > 0) {
          const memLines = result.results.map(
            (m, i) => `${i + 1}. ${m.memory} (id: ${m.id})`,
          );
          parts.push(`\nSupporting memories:\n${memLines.join("\n")}`);
        }
        return parts.join("\n");
      },
    }),

    // ------------------------------------------------------------------
    // CRUD tools
    // ------------------------------------------------------------------

    memory_add: tool({
      description:
        "Store a new memory. Content is automatically analyzed for facts, classified, and deduplicated against existing memories.",
      args: {
        content: tool.schema
          .string()
          .describe("The memory content to store (max 1000 chars)"),
        memory_type: tool.schema
          .string()
          .optional()
          .describe("Memory type: preference, fact, episodic, procedural, context"),
        importance: tool.schema
          .string()
          .optional()
          .describe("Importance: low, normal, high, critical"),
        categories: tool.schema
          .array(tool.schema.string())
          .optional()
          .describe(
            "Tags: personal, preferences, health, work, technical, finance, home, vehicles, travel, entertainment, goals, decisions, project",
          ),
        pinned: tool.schema
          .boolean()
          .optional()
          .describe("Pin this memory (loaded at every conversation start)"),
        infer: tool.schema
          .boolean()
          .optional()
          .describe(
            "Extract facts and dedup (default true). Set false to store content verbatim.",
          ),
        role: tool.schema
          .string()
          .optional()
          .describe(
            "Who the memory is about: user (default) or assistant (requires agent_id)",
          ),
        ttl_days: tool.schema
          .number()
          .min(1)
          .optional()
          .describe("Time-to-live in days. Omit for type defaults."),
        event_date: tool.schema
          .string()
          .optional()
          .describe("ISO 8601 datetime of when the event occurred"),
        labels: tool.schema
          .record(tool.schema.string(), tool.schema.unknown())
          .optional()
          .describe("Key-value metadata labels (e.g. project, topic)"),
      },
      async execute(args) {
        const result = await client.addMemory({
          content: args.content,
          memoryType: args.memory_type,
          importance: args.importance,
          categories: args.categories,
          pinned: args.pinned,
          infer: args.infer,
          role: args.role,
          ttlDays: args.ttl_days,
          eventDate: args.event_date,
          labels: { source: "opencode", ...args.labels },
        });
        if (!result) return "Failed to store memory — mnemory server may be offline.";
        if (result.error) return `Failed to store memory: ${result.message ?? "unknown error"}`;
        const stored = result.results?.length ?? 0;
        if (stored === 0) {
          return "Memory was processed but no new facts were extracted (may be a duplicate).";
        }
        const summaries = result.results.map((r) => `- ${r.memory} (id: ${r.id})`);
        return `Stored ${stored} memory item(s):\n${summaries.join("\n")}`;
      },
    }),

    memory_add_batch: tool({
      description:
        "Store multiple memories in a single call. Each memory is processed independently — " +
        "failures on individual items do not block the rest.",
      args: {
        memories: tool.schema
          .array(
            tool.schema.object({
              content: tool.schema.string().describe("Memory content (max 1000 chars)"),
              memory_type: tool.schema
                .string()
                .optional()
                .describe("Type: preference, fact, episodic, procedural, context"),
              importance: tool.schema
                .string()
                .optional()
                .describe("Importance: low, normal, high, critical"),
              categories: tool.schema.array(tool.schema.string()).optional(),
              infer: tool.schema
                .boolean()
                .optional()
                .describe("Extract facts and dedup (default true). Set false for verbatim."),
              role: tool.schema
                .string()
                .optional()
                .describe("Who the memory is about: user (default) or assistant"),
              ttl_days: tool.schema
                .number()
                .min(1)
                .optional()
                .describe("Time-to-live in days"),
              labels: tool.schema
                .record(tool.schema.string(), tool.schema.unknown())
                .optional()
                .describe("Key-value metadata labels"),
            }),
          )
          .describe("List of memories to store"),
      },
      async execute(args) {
        if (args.memories.length > 5) {
          return `Batch size ${args.memories.length} exceeds limit of 5. Split into smaller batches.`;
        }
        const items = args.memories.map((m) => ({
          content: m.content,
          memoryType: m.memory_type,
          importance: m.importance,
          categories: m.categories,
          infer: m.infer,
          role: m.role,
          ttlDays: m.ttl_days,
          labels: { source: "opencode", ...m.labels },
        }));
        const result = await client.addMemoriesBatch(items);
        if (!result) return "Failed to store memories — mnemory server may be offline.";
        return `Batch add: ${result.succeeded}/${result.total} succeeded, ${result.failed} failed.`;
      },
    }),

    memory_update: tool({
      description: "Update an existing memory's content or metadata by its ID.",
      args: {
        memory_id: tool.schema.string().describe("ID of the memory to update"),
        content: tool.schema.string().optional().describe("New content text"),
        memory_type: tool.schema
          .string()
          .optional()
          .describe("New type: preference, fact, episodic, procedural, context"),
        importance: tool.schema
          .string()
          .optional()
          .describe("New importance: low, normal, high, critical"),
        categories: tool.schema
          .array(tool.schema.string())
          .optional()
          .describe("New categories (replaces existing)"),
        pinned: tool.schema.boolean().optional().describe("New pinned state"),
        ttl_days: tool.schema
          .number()
          .min(1)
          .optional()
          .describe("New TTL in days. Restores decayed memories."),
        event_date: tool.schema
          .string()
          .optional()
          .describe("New event date (ISO 8601). Null to clear."),
        labels: tool.schema
          .record(tool.schema.string(), tool.schema.unknown())
          .optional()
          .describe("New labels (merged with existing). Empty object to clear."),
      },
      async execute(args) {
        const ok = await client.updateMemory(args.memory_id, {
          content: args.content,
          memoryType: args.memory_type,
          importance: args.importance,
          categories: args.categories,
          pinned: args.pinned,
          ttlDays: args.ttl_days,
          eventDate: args.event_date,
          labels: args.labels,
        });
        return ok
          ? `Memory ${args.memory_id} updated successfully.`
          : `Failed to update memory ${args.memory_id} — mnemory server may be offline.`;
      },
    }),

    memory_delete: tool({
      description: "Delete a memory by its ID.",
      args: {
        memory_id: tool.schema.string().describe("ID of the memory to delete"),
      },
      async execute(args) {
        const ok = await client.deleteMemory(args.memory_id);
        return ok
          ? `Memory ${args.memory_id} deleted successfully.`
          : `Failed to delete memory ${args.memory_id} — mnemory server may be offline.`;
      },
    }),

    memory_delete_batch: tool({
      description: "Delete multiple memories in a single call.",
      args: {
        memory_ids: tool.schema
          .array(tool.schema.string())
          .describe("List of memory IDs to delete"),
      },
      async execute(args) {
        const result = await client.deleteMemoriesBatch(args.memory_ids);
        if (!result) return "Failed to delete memories — mnemory server may be offline.";
        return `Batch delete: ${result.succeeded}/${result.total} succeeded, ${result.failed} failed.`;
      },
    }),

    // ------------------------------------------------------------------
    // List / Browse tools
    // ------------------------------------------------------------------

    memory_list: tool({
      description: "List stored memories with optional filters.",
      args: {
        memory_type: tool.schema
          .string()
          .optional()
          .describe("Filter by type: preference, fact, episodic, procedural, context"),
        limit: tool.schema
          .number()
          .min(1)
          .max(100)
          .optional()
          .describe("Max results (default 20)"),
        categories: tool.schema
          .array(tool.schema.string())
          .optional()
          .describe("Filter by categories"),
        role: tool.schema
          .string()
          .optional()
          .describe("Filter by role: user or assistant"),
        include_decayed: tool.schema
          .boolean()
          .optional()
          .describe("Include expired/decayed memories (default false)"),
        labels: tool.schema
          .record(tool.schema.string(), tool.schema.unknown())
          .optional()
          .describe("Filter by label key-value pairs (AND logic)"),
      },
      async execute(args) {
        const result = await client.listMemories({
          memoryType: args.memory_type,
          limit: args.limit ?? 20,
          categories: args.categories,
          role: args.role,
          includeDecayed: args.include_decayed,
          labels: args.labels,
        });
        if (!result) return "Memory list unavailable — mnemory server may be offline.";
        if (result.results.length === 0) return "No memories found.";
        const lines = result.results.map((m, i) => {
          const type = m.memory_type ? ` [${m.memory_type}]` : "";
          const pinned = m.pinned ? " (pinned)" : "";
          return `${i + 1}. ${m.memory}${type}${pinned} (id: ${m.id})`;
        });
        return `Showing ${result.results.length} memories:\n${lines.join("\n")}`;
      },
    }),

    memory_categories: tool({
      description:
        "List all available memory categories with descriptions and counts. " +
        "Categories are predefined — do not invent new ones.",
      args: {},
      async execute() {
        const result = await client.listCategories();
        if (!result) return "Categories unavailable — mnemory server may be offline.";
        const lines = result.categories.map((c) => {
          const desc = c.description ? ` — ${c.description}` : "";
          return `- ${c.name} (${c.count})${desc}`;
        });
        return `Available categories:\n${lines.join("\n")}`;
      },
    }),

    memory_recent: tool({
      description:
        "Get recent memories from the last N days, ordered by most recent first.",
      args: {
        days: tool.schema
          .number()
          .min(1)
          .optional()
          .describe("How many days back to look (default 7)"),
        scope: tool.schema
          .string()
          .optional()
          .describe("Scope: all (default), user, or agent"),
        limit: tool.schema
          .number()
          .min(1)
          .max(100)
          .optional()
          .describe("Max results (default 25)"),
        include_decayed: tool.schema
          .boolean()
          .optional()
          .describe("Include expired memories (default false)"),
      },
      async execute(args) {
        const result = await client.getRecentMemories({
          days: args.days,
          scope: args.scope,
          limit: args.limit,
          includeDecayed: args.include_decayed,
        });
        if (!result) return "Recent memories unavailable — mnemory server may be offline.";
        return result.text || "No recent memories found.";
      },
    }),

    // ------------------------------------------------------------------
    // Artifact tools
    // ------------------------------------------------------------------

    memory_save_artifact: tool({
      description:
        "Attach an artifact to a memory (slow memory tier). Use for detailed content " +
        "too long for fast memory — research reports, analysis, logs, notes, code, data.",
      args: {
        memory_id: tool.schema.string().describe("ID of the parent memory"),
        content: tool.schema
          .string()
          .describe("Text content or base64-encoded binary content"),
        filename: tool.schema
          .string()
          .optional()
          .describe("Name for the artifact (default: note.md)"),
        content_type: tool.schema
          .string()
          .optional()
          .describe("MIME type (default: text/markdown)"),
      },
      async execute(args) {
        const result = await client.saveArtifact(args.memory_id, {
          content: args.content,
          filename: args.filename,
          contentType: args.content_type,
        });
        if (!result) return "Failed to save artifact — mnemory server may be offline.";
        return `Artifact saved: ${result.filename} (${result.size} bytes, id: ${result.id})`;
      },
    }),

    memory_get_artifact: tool({
      description:
        "Retrieve artifact content attached to a memory. Text artifacts support pagination.",
      args: {
        memory_id: tool.schema.string().describe("ID of the parent memory"),
        artifact_id: tool.schema
          .string()
          .describe("ID of the artifact to retrieve"),
        offset: tool.schema
          .number()
          .min(0)
          .optional()
          .describe("Character offset for text (default 0)"),
        limit: tool.schema
          .number()
          .min(1)
          .optional()
          .describe("Max characters for text (default 5000)"),
      },
      async execute(args) {
        const result = await client.getArtifact(
          args.memory_id,
          args.artifact_id,
          { offset: args.offset, limit: args.limit },
        );
        if (!result) return "Failed to get artifact — mnemory server may be offline.";
        const more = result.has_more
          ? `\n(truncated — ${result.total_size} total bytes)`
          : "";
        return `${result.content}${more}`;
      },
    }),

    memory_get_artifact_url: tool({
      description:
        "Generate a short-lived signed URL for direct artifact download. " +
        "Use instead of memory_get_artifact for binary artifacts (images, PDFs) " +
        "or large artifacts (>1MB).",
      args: {
        memory_id: tool.schema.string().describe("ID of the parent memory"),
        artifact_id: tool.schema.string().describe("ID of the artifact"),
        ttl: tool.schema
          .number()
          .min(60)
          .optional()
          .describe("URL lifetime in seconds (default ~3600, max ~86400)"),
      },
      async execute(args) {
        const result = await client.getArtifactUrl(
          args.memory_id,
          args.artifact_id,
          args.ttl,
        );
        if (!result) return "Failed to generate artifact URL — mnemory server may be offline.";
        return `Download URL: ${result.url}\nExpires in: ${result.expires_in}s\nFile: ${result.filename} (${result.content_type}, ${result.size} bytes)`;
      },
    }),

    memory_list_artifacts: tool({
      description: "List all artifacts attached to a memory.",
      args: {
        memory_id: tool.schema.string().describe("ID of the parent memory"),
      },
      async execute(args) {
        const result = await client.listArtifacts(args.memory_id);
        if (!result) return "Failed to list artifacts — mnemory server may be offline.";
        if (result.length === 0) return "No artifacts found for this memory.";
        const lines = result.map(
          (a, i) =>
            `${i + 1}. ${a.filename} (${a.content_type}, ${a.size} bytes, id: ${a.id})`,
        );
        return `${result.length} artifact(s):\n${lines.join("\n")}`;
      },
    }),

    memory_delete_artifact: tool({
      description: "Delete an artifact from a memory.",
      args: {
        memory_id: tool.schema.string().describe("ID of the parent memory"),
        artifact_id: tool.schema
          .string()
          .describe("ID of the artifact to delete"),
      },
      async execute(args) {
        const ok = await client.deleteArtifact(
          args.memory_id,
          args.artifact_id,
        );
        return ok
          ? `Artifact ${args.artifact_id} deleted successfully.`
          : `Failed to delete artifact ${args.artifact_id} — mnemory server may be offline.`;
      },
    }),
  };
}
