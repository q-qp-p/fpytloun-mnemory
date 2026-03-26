/**
 * OpenCode Mnemory Plugin
 *
 * Persistent AI memory backed by a mnemory server (https://github.com/fpytloun/mnemory).
 * Provides automatic recall (inject relevant memories before each LLM call),
 * automatic capture (extract and store memories after conversations), and
 * 16 explicit memory tools (search, add, update, delete, artifacts).
 *
 * Uses mnemory's REST API (/api/recall, /api/remember, /api/memories/*).
 * No MCP server configuration needed — all tools are built into the plugin.
 *
 * Configuration via environment variables:
 *   MNEMORY_URL              - Server URL (default: http://localhost:8050)
 *   MNEMORY_API_KEY          - Bearer token for authentication
 *   MNEMORY_AGENT_ID         - Agent ID (default: opencode)
 *   MNEMORY_USER_ID          - User ID (optional if API key maps to user)
 *   MNEMORY_SCORE_THRESHOLD  - Min relevance score 0-1 (default: 0.5)
 *   MNEMORY_INCLUDE_ASSISTANT - Include assistant messages in remember (default: false)
 *   MNEMORY_SEARCH_MODE      - Default search mode: find or search (default: search)
 *   MNEMORY_FIND_FIRST       - Use AI search on first turn (default: true)
 *   MNEMORY_MANAGED          - Include behavioral instructions (default: true)
 *   MNEMORY_TIMEOUT          - HTTP timeout in ms (default: 30000)
 */

import type { Plugin } from "@opencode-ai/plugin";
import { MnemoryClient } from "./client.js";
import { createHooks } from "./hooks.js";
import { createTools } from "./tools.js";
import {
  createLogger,
  createSessionStore,
  parseConfig,
} from "./helpers.js";

export const MnemoryPlugin: Plugin = async ({ client, directory, worktree }) => {
  const config = parseConfig();
  const logger = createLogger(client);
  const store = createSessionStore(logger);

  const mnemoryClient = new MnemoryClient(config, logger);

  // Create hooks and tools
  const hooks = createHooks({
    config,
    logger,
    store,
    directory,
    worktree,
    sdkClient: client,
  });

  // Wire the client into hooks (avoids circular dependency)
  hooks.setClient(mnemoryClient);

  const tools = createTools({ client: mnemoryClient, logger });

  // Non-blocking health check on startup
  void mnemoryClient.healthCheck().then((ok) => {
    if (ok) {
      logger.info(
        `Initialized (server: ${config.url}, agent: ${config.agentId}, managed: ${config.managed})`,
      );
    } else {
      logger.warn(
        `Server at ${config.url} is unreachable — memories will be unavailable until the server is online`,
      );
    }
  });

  // Return hooks + tools (omit setClient from the returned object)
  const { setClient: _, ...hookHandlers } = hooks;

  return {
    ...hookHandlers,
    tool: tools,
  };
};

export default MnemoryPlugin;
