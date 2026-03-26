/**
 * Mnemory REST API client.
 *
 * Thin HTTP wrapper around mnemory's REST endpoints.
 * All methods use graceful error handling — API failures are logged
 * but never thrown, so the assistant keeps working if mnemory is offline.
 */

import type { Logger, MnemoryConfig } from "./helpers.js";

// ============================================================================
// Types
// ============================================================================

export type RecallParams = {
  sessionId?: string;
  query?: string;
  includeInstructions?: boolean;
  managed?: boolean;
  instructionMode?: string;
  searchMode?: "find" | "search";
  scoreThreshold?: number;
  context?: string;
  labels?: Record<string, unknown>;
};

export type RecallResponse = {
  session_id: string;
  instructions?: string;
  core_memories?: string;
  search_results: Array<{
    id: string;
    memory: string;
    score?: number;
    metadata?: Record<string, unknown>;
    has_artifacts?: boolean;
  }>;
  stats?: {
    core_count?: number;
    search_count?: number;
    new_count?: number;
    known_skipped?: number;
    latency_ms?: number;
  };
};

export type RememberParams = {
  sessionId?: string;
  messages: Array<{ role: string; content: string }>;
  context?: string;
  labels?: Record<string, unknown>;
};

export type SearchMemoriesParams = {
  query: string;
  memoryType?: string;
  categories?: string[];
  role?: string;
  limit?: number;
  includeDecayed?: boolean;
  dateStart?: string;
  dateEnd?: string;
  labels?: Record<string, unknown>;
};

export type SearchMemoriesResponse = {
  results: Array<{
    id: string;
    memory: string;
    score?: number;
    memory_type?: string;
    categories?: string[];
    importance?: string;
    created_at?: string;
    has_artifacts?: boolean;
  }>;
};

export type FindMemoriesParams = {
  question: string;
  memoryType?: string;
  categories?: string[];
  role?: string;
  limit?: number;
  includeDecayed?: boolean;
  context?: string;
  labels?: Record<string, unknown>;
};

export type AskMemoriesParams = {
  question: string;
  memoryType?: string;
  categories?: string[];
  role?: string;
  limit?: number;
  includeDecayed?: boolean;
  context?: string;
  includeMemories?: boolean;
  labels?: Record<string, unknown>;
};

export type AskMemoriesResponse = {
  answer: string;
  results: Array<{
    id: string;
    memory: string;
    score?: number;
    memory_type?: string;
  }>;
  count: number;
  queries: string[];
};

export type AddMemoryParams = {
  content: string;
  memoryType?: string;
  categories?: string[];
  importance?: string;
  pinned?: boolean;
  infer?: boolean;
  role?: string;
  ttlDays?: number;
  eventDate?: string;
  labels?: Record<string, unknown>;
};

export type AddMemoryResponse = {
  results: Array<{
    id: string;
    memory: string;
    event?: string;
  }>;
  error?: boolean;
  message?: string;
};

export type AddMemoriesBatchItem = {
  content: string;
  memoryType?: string;
  categories?: string[];
  importance?: string;
  pinned?: boolean;
  infer?: boolean;
  role?: string;
  ttlDays?: number;
  labels?: Record<string, unknown>;
};

export type AddMemoriesBatchResponse = {
  results: Array<{
    id: string;
    memory: string;
    event?: string;
  }>;
  errors: Array<{
    index: number;
    error: boolean;
    message: string;
  }>;
  total: number;
  succeeded: number;
  failed: number;
};

export type UpdateMemoryParams = {
  content?: string;
  memoryType?: string;
  categories?: string[];
  importance?: string;
  pinned?: boolean;
  ttlDays?: number;
  eventDate?: string;
  labels?: Record<string, unknown>;
};

export type ListMemoriesParams = {
  memoryType?: string;
  categories?: string[];
  role?: string;
  limit?: number;
  includeDecayed?: boolean;
  labels?: Record<string, unknown>;
};

export type MemoryItem = {
  id: string;
  memory: string;
  memory_type?: string;
  categories?: string[];
  importance?: string;
  pinned?: boolean;
  created_at?: string;
  has_artifacts?: boolean;
};

export type ListMemoriesResponse = {
  results: MemoryItem[];
};

export type DeleteMemoriesBatchResponse = {
  results: Array<Record<string, unknown>>;
  errors: Array<{
    memory_id: string;
    error: boolean;
    message: string;
  }>;
  total: number;
  succeeded: number;
  failed: number;
};

export type GetRecentMemoriesParams = {
  days?: number;
  scope?: string;
  limit?: number;
  includeDecayed?: boolean;
};

export type SaveArtifactParams = {
  content: string;
  filename?: string;
  contentType?: string;
};

export type ArtifactMetadata = {
  id: string;
  filename: string;
  content_type: string;
  size: number;
  created_at: string;
};

export type GetArtifactParams = {
  offset?: number;
  limit?: number;
};

export type GetArtifactResponse = {
  content: string;
  total_size: number;
  has_more: boolean;
};

export type GetArtifactUrlResponse = {
  token: string;
  url: string;
  expires_in: number;
  content_type: string;
  filename: string;
  size: number;
};

export type CategoryItem = {
  name: string;
  description?: string;
  count: number;
};

export type ListCategoriesResponse = {
  categories: CategoryItem[];
};

// ============================================================================
// Client
// ============================================================================

export class MnemoryClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly agentId: string;
  private readonly userId: string;
  private readonly logger: Logger;
  private readonly timeout: number;

  constructor(config: MnemoryConfig, logger: Logger) {
    this.baseUrl = config.url;
    this.apiKey = config.apiKey;
    this.agentId = config.agentId;
    this.userId = config.userId;
    this.timeout = config.timeout;
    this.logger = logger;
  }

  // --------------------------------------------------------------------------
  // Internal helpers
  // --------------------------------------------------------------------------

  private headers(): Record<string, string> {
    const h: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      h["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.userId) {
      h["X-User-Id"] = this.userId;
    }
    if (this.agentId) {
      h["X-Agent-Id"] = this.agentId;
    }
    return h;
  }

  /**
   * Combine a caller-provided AbortSignal with a timeout signal.
   * The request aborts when either fires first.
   */
  private combinedSignal(
    timeoutMs: number,
    signal?: AbortSignal,
  ): AbortSignal {
    if (!signal) return AbortSignal.timeout(timeoutMs);
    // AbortSignal.any() combines multiple signals — aborts when any fires
    return AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]);
  }

  private async post<T>(
    path: string,
    body: unknown,
    timeoutMs?: number,
    signal?: AbortSignal,
  ): Promise<T | null> {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(body),
        signal: this.combinedSignal(timeoutMs ?? this.timeout, signal),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        this.logger.warn(`POST ${path} returned ${res.status}: ${text}`);
        return null;
      }
      return (await res.json()) as T;
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        this.logger.warn(`POST ${path} failed: ${String(err)}`);
      }
      return null;
    }
  }

  private async get<T>(
    path: string,
    signal?: AbortSignal,
  ): Promise<T | null> {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: "GET",
        headers: this.headers(),
        signal: this.combinedSignal(this.timeout, signal),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        this.logger.warn(`GET ${path} returned ${res.status}: ${text}`);
        return null;
      }
      return (await res.json()) as T;
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        this.logger.warn(`GET ${path} failed: ${String(err)}`);
      }
      return null;
    }
  }

  private async put(
    path: string,
    body: unknown,
  ): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: "PUT",
        headers: this.headers(),
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeout),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        this.logger.warn(`PUT ${path} returned ${res.status}: ${text}`);
        return false;
      }
      return true;
    } catch (err) {
      this.logger.warn(`PUT ${path} failed: ${String(err)}`);
      return false;
    }
  }

  private async del(path: string): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: "DELETE",
        headers: this.headers(),
        signal: AbortSignal.timeout(this.timeout),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        this.logger.warn(`DELETE ${path} returned ${res.status}: ${text}`);
        return false;
      }
      return true;
    } catch (err) {
      this.logger.warn(`DELETE ${path} failed: ${String(err)}`);
      return false;
    }
  }

  // --------------------------------------------------------------------------
  // Health
  // --------------------------------------------------------------------------

  /** Non-blocking health check. Returns true if server is reachable. */
  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  // --------------------------------------------------------------------------
  // Recall / Remember
  // --------------------------------------------------------------------------

  async recall(
    params: RecallParams,
    signal?: AbortSignal,
  ): Promise<RecallResponse | null> {
    return this.post<RecallResponse>(
      "/api/recall",
      {
        session_id: params.sessionId ?? undefined,
        query: params.query ?? undefined,
        include_instructions: params.includeInstructions ?? false,
        managed: params.managed ?? false,
        instruction_mode: params.instructionMode ?? undefined,
        search_mode: params.searchMode ?? undefined,
        score_threshold: params.scoreThreshold ?? 0.5,
        context: params.context ?? undefined,
        labels: params.labels ?? undefined,
      },
      undefined,
      signal,
    );
  }

  async remember(
    params: RememberParams,
    signal?: AbortSignal,
  ): Promise<void> {
    // Fire-and-forget: we don't need the response
    void this.post(
      "/api/remember",
      {
        session_id: params.sessionId ?? undefined,
        messages: params.messages,
        context: params.context ?? undefined,
        labels: params.labels ?? undefined,
      },
      undefined,
      signal,
    );
  }

  // --------------------------------------------------------------------------
  // Search
  // --------------------------------------------------------------------------

  async searchMemories(
    params: SearchMemoriesParams,
  ): Promise<SearchMemoriesResponse | null> {
    return this.post<SearchMemoriesResponse>("/api/memories/search", {
      query: params.query,
      memory_type: params.memoryType ?? undefined,
      categories: params.categories ?? undefined,
      role: params.role ?? undefined,
      limit: params.limit ?? 10,
      include_decayed: params.includeDecayed ?? false,
      date_start: params.dateStart ?? undefined,
      date_end: params.dateEnd ?? undefined,
      labels: params.labels ?? undefined,
    });
  }

  async findMemories(
    params: FindMemoriesParams,
  ): Promise<SearchMemoriesResponse | null> {
    return this.post<SearchMemoriesResponse>("/api/memories/find", {
      question: params.question,
      memory_type: params.memoryType ?? undefined,
      categories: params.categories ?? undefined,
      role: params.role ?? undefined,
      limit: params.limit ?? 10,
      include_decayed: params.includeDecayed ?? false,
      context: params.context ?? undefined,
      labels: params.labels ?? undefined,
    });
  }

  async askMemories(
    params: AskMemoriesParams,
  ): Promise<AskMemoriesResponse | null> {
    return this.post<AskMemoriesResponse>("/api/memories/ask", {
      question: params.question,
      memory_type: params.memoryType ?? undefined,
      categories: params.categories ?? undefined,
      role: params.role ?? undefined,
      limit: params.limit ?? 10,
      include_decayed: params.includeDecayed ?? false,
      context: params.context ?? undefined,
      include_memories: params.includeMemories ?? false,
      labels: params.labels ?? undefined,
    });
  }

  // --------------------------------------------------------------------------
  // CRUD
  // --------------------------------------------------------------------------

  async addMemory(
    params: AddMemoryParams,
  ): Promise<AddMemoryResponse | null> {
    return this.post<AddMemoryResponse>("/api/memories", {
      content: params.content,
      memory_type: params.memoryType ?? undefined,
      categories: params.categories ?? undefined,
      importance: params.importance ?? undefined,
      pinned: params.pinned ?? undefined,
      infer: params.infer ?? true,
      role: params.role ?? undefined,
      ttl_days: params.ttlDays ?? undefined,
      event_date: params.eventDate ?? undefined,
      labels: params.labels ?? undefined,
    });
  }

  async addMemoriesBatch(
    memories: AddMemoriesBatchItem[],
  ): Promise<AddMemoriesBatchResponse | null> {
    return this.post<AddMemoriesBatchResponse>(
      "/api/memories/batch",
      {
        memories: memories.map((m) => ({
          content: m.content,
          memory_type: m.memoryType ?? undefined,
          categories: m.categories ?? undefined,
          importance: m.importance ?? undefined,
          pinned: m.pinned ?? undefined,
          infer: m.infer ?? undefined,
          role: m.role ?? undefined,
          ttl_days: m.ttlDays ?? undefined,
          labels: m.labels ?? undefined,
        })),
      },
      this.timeout * 5,
    );
  }

  async updateMemory(
    id: string,
    params: UpdateMemoryParams,
  ): Promise<boolean> {
    return this.put(`/api/memories/${encodeURIComponent(id)}`, {
      content: params.content ?? undefined,
      memory_type: params.memoryType ?? undefined,
      categories: params.categories ?? undefined,
      importance: params.importance ?? undefined,
      pinned: params.pinned ?? undefined,
      ttl_days: params.ttlDays ?? undefined,
      event_date: params.eventDate ?? undefined,
      labels: params.labels ?? undefined,
    });
  }

  async deleteMemory(id: string): Promise<boolean> {
    return this.del(`/api/memories/${encodeURIComponent(id)}`);
  }

  async deleteMemoriesBatch(
    memoryIds: string[],
  ): Promise<DeleteMemoriesBatchResponse | null> {
    return this.post<DeleteMemoriesBatchResponse>(
      "/api/memories/batch/delete",
      { memory_ids: memoryIds },
    );
  }

  async listMemories(
    params?: ListMemoriesParams,
  ): Promise<ListMemoriesResponse | null> {
    const sp = new URLSearchParams();
    if (params?.memoryType) sp.set("memory_type", params.memoryType);
    if (params?.role) sp.set("role", params.role);
    if (params?.limit) sp.set("limit", String(params.limit));
    if (params?.includeDecayed) sp.set("include_decayed", "true");
    if (params?.categories) {
      for (const cat of params.categories) {
        sp.append("categories", cat);
      }
    }
    if (params?.labels) sp.set("labels", JSON.stringify(params.labels));
    const qs = sp.toString();
    return this.get<ListMemoriesResponse>(
      `/api/memories${qs ? `?${qs}` : ""}`,
    );
  }

  // --------------------------------------------------------------------------
  // Recent / Categories
  // --------------------------------------------------------------------------

  async getRecentMemories(
    params?: GetRecentMemoriesParams,
  ): Promise<{ text: string } | null> {
    const sp = new URLSearchParams();
    if (params?.days != null) sp.set("days", String(params.days));
    if (params?.scope) sp.set("scope", params.scope);
    if (params?.limit != null) sp.set("limit", String(params.limit));
    if (params?.includeDecayed) sp.set("include_decayed", "true");
    const qs = sp.toString();
    return this.get<{ text: string }>(
      `/api/memories/recent${qs ? `?${qs}` : ""}`,
    );
  }

  async listCategories(): Promise<ListCategoriesResponse | null> {
    return this.get<ListCategoriesResponse>("/api/categories");
  }

  // --------------------------------------------------------------------------
  // Artifacts
  // --------------------------------------------------------------------------

  async saveArtifact(
    memoryId: string,
    params: SaveArtifactParams,
  ): Promise<ArtifactMetadata | null> {
    return this.post<ArtifactMetadata>(
      `/api/memories/${encodeURIComponent(memoryId)}/artifacts`,
      {
        content: params.content,
        filename: params.filename ?? "note.md",
        content_type: params.contentType ?? "text/markdown",
      },
    );
  }

  async getArtifact(
    memoryId: string,
    artifactId: string,
    params?: GetArtifactParams,
  ): Promise<GetArtifactResponse | null> {
    const sp = new URLSearchParams();
    if (params?.offset != null) sp.set("offset", String(params.offset));
    if (params?.limit != null) sp.set("limit", String(params.limit));
    const qs = sp.toString();
    return this.get<GetArtifactResponse>(
      `/api/memories/${encodeURIComponent(memoryId)}/artifacts/${encodeURIComponent(artifactId)}${qs ? `?${qs}` : ""}`,
    );
  }

  async getArtifactUrl(
    memoryId: string,
    artifactId: string,
    ttl?: number,
  ): Promise<GetArtifactUrlResponse | null> {
    return this.post<GetArtifactUrlResponse>(
      `/api/memories/${encodeURIComponent(memoryId)}/artifacts/${encodeURIComponent(artifactId)}/download-token`,
      ttl != null ? { ttl } : {},
    );
  }

  async listArtifacts(
    memoryId: string,
  ): Promise<ArtifactMetadata[] | null> {
    return this.get<ArtifactMetadata[]>(
      `/api/memories/${encodeURIComponent(memoryId)}/artifacts`,
    );
  }

  async deleteArtifact(
    memoryId: string,
    artifactId: string,
  ): Promise<boolean> {
    return this.del(
      `/api/memories/${encodeURIComponent(memoryId)}/artifacts/${encodeURIComponent(artifactId)}`,
    );
  }
}
