/**
 * Unit tests for mnemory OpenCode plugin helpers.
 *
 * Run with: bun test helpers.test.ts
 */

import { describe, expect, test } from "bun:test";
import {
  buildSystemText,
  createLogger,
  createSessionStore,
  escapeForPrompt,
  extractLastExchange,
  extractTextFromContent,
  extractTextFromParts,
  parseConfig,
} from "./helpers.js";

// ============================================================================
// parseConfig
// ============================================================================

describe("parseConfig", () => {
  test("returns defaults when no env vars set", () => {
    const saved = { ...process.env };
    delete process.env.MNEMORY_URL;
    delete process.env.MNEMORY_API_KEY;
    delete process.env.MNEMORY_AGENT_ID;
    delete process.env.MNEMORY_USER_ID;
    delete process.env.MNEMORY_SCORE_THRESHOLD;
    delete process.env.MNEMORY_INCLUDE_ASSISTANT;
    delete process.env.MNEMORY_SEARCH_MODE;
    delete process.env.MNEMORY_FIND_FIRST;
    delete process.env.MNEMORY_MANAGED;
    delete process.env.MNEMORY_TIMEOUT;

    const cfg = parseConfig();
    expect(cfg.url).toBe("http://localhost:8050");
    expect(cfg.apiKey).toBe("");
    expect(cfg.agentId).toBe("opencode");
    expect(cfg.userId).toBe("");
    expect(cfg.scoreThreshold).toBe(0.5);
    expect(cfg.includeAssistant).toBe(false);
    expect(cfg.searchMode).toBe("search");
    expect(cfg.findFirst).toBe(true);
    expect(cfg.managed).toBe(true);
    expect(cfg.timeout).toBe(30000);

    Object.assign(process.env, saved);
  });

  test("reads env vars correctly", () => {
    const saved = { ...process.env };
    process.env.MNEMORY_URL = "http://example.com:9090/";
    process.env.MNEMORY_API_KEY = "test-key";
    process.env.MNEMORY_AGENT_ID = "my-agent";
    process.env.MNEMORY_USER_ID = "user-1";
    process.env.MNEMORY_SCORE_THRESHOLD = "0.7";
    process.env.MNEMORY_INCLUDE_ASSISTANT = "true";
    process.env.MNEMORY_SEARCH_MODE = "find";
    process.env.MNEMORY_FIND_FIRST = "false";
    process.env.MNEMORY_MANAGED = "false";
    process.env.MNEMORY_TIMEOUT = "60000";

    const cfg = parseConfig();
    expect(cfg.url).toBe("http://example.com:9090"); // trailing slash stripped
    expect(cfg.apiKey).toBe("test-key");
    expect(cfg.agentId).toBe("my-agent");
    expect(cfg.userId).toBe("user-1");
    expect(cfg.scoreThreshold).toBe(0.7);
    expect(cfg.includeAssistant).toBe(true);
    expect(cfg.searchMode).toBe("find");
    expect(cfg.findFirst).toBe(false);
    expect(cfg.managed).toBe(false);
    expect(cfg.timeout).toBe(60000);

    Object.assign(process.env, saved);
  });

  test("clamps score threshold to 0-1 range", () => {
    const saved = { ...process.env };
    process.env.MNEMORY_SCORE_THRESHOLD = "2.5";
    expect(parseConfig().scoreThreshold).toBe(1);

    process.env.MNEMORY_SCORE_THRESHOLD = "-0.5";
    expect(parseConfig().scoreThreshold).toBe(0);

    Object.assign(process.env, saved);
  });

  test("enforces minimum timeout of 1000ms", () => {
    const saved = { ...process.env };
    process.env.MNEMORY_TIMEOUT = "500";
    expect(parseConfig().timeout).toBe(1000);
    Object.assign(process.env, saved);
  });

  test("ignores invalid search mode", () => {
    const saved = { ...process.env };
    process.env.MNEMORY_SEARCH_MODE = "invalid";
    expect(parseConfig().searchMode).toBe("search");
    Object.assign(process.env, saved);
  });
});

// ============================================================================
// escapeForPrompt
// ============================================================================

describe("escapeForPrompt", () => {
  test("escapes HTML entities", () => {
    expect(escapeForPrompt('<script>alert("xss")</script>')).toBe(
      "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;",
    );
  });

  test("escapes ampersands", () => {
    expect(escapeForPrompt("foo & bar")).toBe("foo &amp; bar");
  });

  test("escapes single quotes", () => {
    expect(escapeForPrompt("it's")).toBe("it&#x27;s");
  });

  test("passes through plain text unchanged", () => {
    expect(escapeForPrompt("Hello world")).toBe("Hello world");
  });
});

// ============================================================================
// buildSystemText
// ============================================================================

describe("buildSystemText", () => {
  test("returns instructions and context separately", () => {
    const result = buildSystemText({
      session_id: "s1",
      instructions: "Be helpful",
      core_memories: "## User Facts\n- Name: Alice",
      search_results: [
        { id: "1", memory: "User likes dogs" },
        { id: "2", memory: "User lives in Prague" },
      ],
    });

    expect(result.instructions).toBe("Be helpful");
    expect(result.context).toContain("## User Facts");
    expect(result.context).toContain("Name: Alice");
    expect(result.context).toContain("## Recalled Memories");
    expect(result.context).toContain("User likes dogs");
    expect(result.context).toContain("User lives in Prague");
  });

  test("handles missing instructions", () => {
    const result = buildSystemText({
      session_id: "s1",
      search_results: [],
    });
    expect(result.instructions).toBeUndefined();
    expect(result.context).toBe("");
  });

  test("handles empty search results", () => {
    const result = buildSystemText({
      session_id: "s1",
      core_memories: "## Core\n- Fact 1",
      search_results: [],
    });
    expect(result.context).toContain("## Core");
    expect(result.context).not.toContain("## Recalled Memories");
  });

  test("escapes search result content", () => {
    const result = buildSystemText({
      session_id: "s1",
      search_results: [
        { id: "1", memory: '<script>alert("xss")</script>' },
      ],
    });
    expect(result.context).toContain("&lt;script&gt;");
    expect(result.context).not.toContain("<script>");
  });

  test("filters empty search results", () => {
    const result = buildSystemText({
      session_id: "s1",
      search_results: [
        { id: "1", memory: "" },
        { id: "2", memory: "   " },
        { id: "3", memory: "Valid memory" },
      ],
    });
    expect(result.context).toContain("Valid memory");
    expect(result.context).not.toContain("- \n");
  });
});

// ============================================================================
// extractTextFromParts
// ============================================================================

describe("extractTextFromParts", () => {
  test("extracts text from text parts", () => {
    const parts = [
      { type: "text", text: "Hello world" },
    ];
    expect(extractTextFromParts(parts)).toBe("Hello world");
  });

  test("takes last non-synthetic text part", () => {
    const parts = [
      { type: "text", text: "First message" },
      { type: "text", text: "Second message" },
    ];
    expect(extractTextFromParts(parts)).toBe("Second message");
  });

  test("skips synthetic parts", () => {
    const parts = [
      { type: "text", text: "Real message" },
      { type: "text", text: "Synthetic narration", synthetic: true },
    ];
    expect(extractTextFromParts(parts)).toBe("Real message");
  });

  test("skips non-text parts", () => {
    const parts = [
      { type: "tool", callID: "123" },
      { type: "text", text: "Hello" },
      { type: "reasoning", text: "Thinking..." },
    ];
    expect(extractTextFromParts(parts)).toBe("Hello");
  });

  test("returns null for empty parts", () => {
    expect(extractTextFromParts([])).toBeNull();
  });

  test("returns null for only synthetic parts", () => {
    const parts = [
      { type: "text", text: "Synthetic", synthetic: true },
    ];
    expect(extractTextFromParts(parts)).toBeNull();
  });

  test("trims whitespace", () => {
    const parts = [{ type: "text", text: "  Hello  " }];
    expect(extractTextFromParts(parts)).toBe("Hello");
  });

  test("skips empty text", () => {
    const parts = [
      { type: "text", text: "" },
      { type: "text", text: "   " },
    ];
    expect(extractTextFromParts(parts)).toBeNull();
  });
});

// ============================================================================
// extractTextFromContent
// ============================================================================

describe("extractTextFromContent", () => {
  test("handles string content", () => {
    expect(extractTextFromContent("Hello")).toBe("Hello");
  });

  test("handles array content", () => {
    expect(
      extractTextFromContent([{ type: "text", text: "Hello" }]),
    ).toBe("Hello");
  });

  test("returns null for empty string", () => {
    expect(extractTextFromContent("")).toBeNull();
  });

  test("returns null for non-string non-array", () => {
    expect(extractTextFromContent(42)).toBeNull();
    expect(extractTextFromContent(null)).toBeNull();
    expect(extractTextFromContent(undefined)).toBeNull();
  });
});

// ============================================================================
// extractLastExchange
// ============================================================================

describe("extractLastExchange", () => {
  test("extracts user message from flat format", () => {
    const messages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
    ];
    const result = extractLastExchange(messages, 0, false);
    expect(result).not.toBeNull();
    expect(result!.user).toBe("Hello");
    expect(result!.assistant).toBeUndefined();
    expect(result!.newCount).toBe(2);
  });

  test("includes assistant when requested", () => {
    const messages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
    ];
    const result = extractLastExchange(messages, 0, true);
    expect(result!.user).toBe("Hello");
    expect(result!.assistant).toBe("Hi there");
  });

  test("handles OpenCode SDK format (info + parts)", () => {
    const messages = [
      {
        info: { role: "user" },
        parts: [{ type: "text", text: "Hello from SDK" }],
      },
      {
        info: { role: "assistant" },
        parts: [{ type: "text", text: "Response" }],
      },
    ];
    const result = extractLastExchange(messages, 0, true);
    expect(result!.user).toBe("Hello from SDK");
    expect(result!.assistant).toBe("Response");
  });

  test("respects afterIndex", () => {
    const messages = [
      { role: "user", content: "Old message" },
      { role: "assistant", content: "Old response" },
      { role: "user", content: "New message" },
      { role: "assistant", content: "New response" },
    ];
    const result = extractLastExchange(messages, 2, true);
    expect(result!.user).toBe("New message");
    expect(result!.assistant).toBe("New response");
    expect(result!.newCount).toBe(4);
  });

  test("returns null when no user message found", () => {
    const messages = [
      { role: "assistant", content: "Just a response" },
    ];
    expect(extractLastExchange(messages, 0, false)).toBeNull();
  });

  test("returns null for empty slice", () => {
    expect(extractLastExchange([], 0, false)).toBeNull();
    expect(extractLastExchange([{ role: "user", content: "Hi" }], 5, false)).toBeNull();
  });

  test("takes last user message when multiple exist", () => {
    const messages = [
      { role: "user", content: "First" },
      { role: "user", content: "Second" },
    ];
    const result = extractLastExchange(messages, 0, false);
    expect(result!.user).toBe("Second");
  });
});

// ============================================================================
// createSessionStore
// ============================================================================

describe("createSessionStore", () => {
  const mockLogger = createLogger(null);

  test("creates and retrieves session state", () => {
    const store = createSessionStore(mockLogger);
    const state = store.getOrCreate("s1");
    expect(state.mnemorySessionId).toBeNull();
    expect(state.turnCount).toBe(0);
    expect(state.isChildSession).toBe(false);

    // Same session returns same state
    const state2 = store.getOrCreate("s1");
    expect(state2).toBe(state);
  });

  test("removes session state", () => {
    const store = createSessionStore(mockLogger);
    const state = store.getOrCreate("s1");
    state.turnCount = 5;
    store.remove("s1");

    // New state after removal
    const state2 = store.getOrCreate("s1");
    expect(state2.turnCount).toBe(0);
    expect(state2).not.toBe(state);
  });

  test("resets session state", () => {
    const store = createSessionStore(mockLogger);
    const state = store.getOrCreate("s1");
    state.turnCount = 5;
    state.mnemorySessionId = "ms-1";

    const newState = store.reset("s1");
    expect(newState.turnCount).toBe(0);
    expect(newState.mnemorySessionId).toBeNull();
    expect(newState).not.toBe(state);
  });

  test("aborts controller on remove", () => {
    const store = createSessionStore(mockLogger);
    const state = store.getOrCreate("s1");
    const signal = state.abortController.signal;
    expect(signal.aborted).toBe(false);

    store.remove("s1");
    expect(signal.aborted).toBe(true);
  });

  test("aborts controller on reset", () => {
    const store = createSessionStore(mockLogger);
    const state = store.getOrCreate("s1");
    const signal = state.abortController.signal;

    store.reset("s1");
    expect(signal.aborted).toBe(true);
  });
});
