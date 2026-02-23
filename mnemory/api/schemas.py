"""Pydantic request/response models for the REST API.

These models auto-generate the OpenAPI spec that clients (Open WebUI,
Cursor, etc.) consume as native tools.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Memory CRUD ───────────────────────────────────────────────────────


class AddMemoryRequest(BaseModel):
    """Request to add a single memory."""

    content: str = Field(
        ..., description="Memory content (max 1000 chars for infer=False)"
    )
    memory_type: str | None = Field(
        None, description="Memory type: preference, fact, episodic, procedural, context"
    )
    categories: list[str] | None = Field(
        None, description="Category tags from predefined set"
    )
    importance: str | None = Field(
        None, description="Importance: low, normal, high, critical"
    )
    pinned: bool | None = Field(
        None, description="Pin to core memories (loaded at conversation start)"
    )
    infer: bool = Field(
        True,
        description="Use LLM for fact extraction and dedup (True) or store as-is (False)",
    )
    role: str = Field(
        "user", description="Who this memory is about: 'user' or 'assistant'"
    )
    ttl_days: int | None = Field(
        None, description="Time-to-live in days (None = use type default)"
    )
    event_date: str | None = Field(
        None, description="ISO 8601 datetime for when the event occurred"
    )


class BatchMemoryItem(BaseModel):
    """Single item in a batch add request."""

    content: str
    memory_type: str | None = None
    categories: list[str] | None = None
    importance: str | None = None
    pinned: bool | None = None
    infer: bool = True
    role: str = "user"
    ttl_days: int | None = None
    event_date: str | None = None


class AddMemoriesBatchRequest(BaseModel):
    """Request to batch-add multiple memories."""

    memories: list[BatchMemoryItem] = Field(..., description="List of memories to add")


class MemoryActionResult(BaseModel):
    """Result of a single memory action (ADD, UPDATE, DELETE)."""

    id: str
    memory: str
    event: str  # ADD, UPDATE, DELETE


class ArtifactInfo(BaseModel):
    """Info about an auto-created artifact."""

    id: str
    filename: str
    size: int
    linked_memories: int


class AddMemoryResponse(BaseModel):
    """Response from adding memory/memories."""

    results: list[MemoryActionResult]
    artifact: ArtifactInfo | None = None


class SearchMemoriesRequest(BaseModel):
    """Request for semantic memory search."""

    query: str = Field(..., description="Search query (natural language)")
    memory_type: str | None = Field(None, description="Filter by memory type")
    categories: list[str] | None = Field(None, description="Filter by categories")
    role: str | None = Field(None, description="Filter by role: 'user' or 'assistant'")
    limit: int = Field(10, description="Max results to return")
    include_decayed: bool = Field(False, description="Include expired/decayed memories")


class FindMemoriesRequest(BaseModel):
    """Request for AI-powered multi-query search."""

    question: str = Field(..., description="Natural language question")
    memory_type: str | None = Field(None, description="Filter by memory type")
    categories: list[str] | None = Field(None, description="Filter by categories")
    role: str | None = Field(None, description="Filter by role")
    limit: int = Field(10, description="Max results to return")
    include_decayed: bool = Field(False, description="Include expired/decayed memories")


class MemoryItem(BaseModel):
    """A memory item in search/list results."""

    id: str
    memory: str
    score: float | None = None
    metadata: dict | None = None
    has_artifacts: bool = False


class SearchMemoriesResponse(BaseModel):
    """Response from memory search."""

    results: list[MemoryItem]


class UpdateMemoryRequest(BaseModel):
    """Request to update an existing memory."""

    content: str | None = Field(None, description="New content text")
    memory_type: str | None = Field(None, description="New memory type")
    categories: list[str] | None = Field(
        None, description="New categories (replaces existing)"
    )
    importance: str | None = Field(None, description="New importance level")
    pinned: bool | None = Field(None, description="New pinned state")
    ttl_days: int | None = Field(None, description="New TTL in days")


class CoreMemoriesResponse(BaseModel):
    """Response from get_core_memories."""

    text: str = Field(..., description="Formatted core memories text")


class RecentMemoriesRequest(BaseModel):
    """Query parameters for recent memories (used as query params)."""

    days: int = Field(7, description="How many days back to look")
    scope: str = Field("all", description="Scope: 'all', 'user', or 'agent'")
    limit: int = Field(25, description="Max results per scope")
    include_decayed: bool = Field(False, description="Include expired memories")


class ListMemoriesRequest(BaseModel):
    """Query parameters for listing memories."""

    memory_type: str | None = Field(None, description="Filter by type")
    categories: list[str] | None = Field(None, description="Filter by categories")
    role: str | None = Field(None, description="Filter by role")
    limit: int = Field(50, description="Max results")
    include_decayed: bool = Field(False, description="Include expired memories")


# ── Artifacts ─────────────────────────────────────────────────────────


class SaveArtifactRequest(BaseModel):
    """Request to save an artifact attached to a memory."""

    content: str = Field(..., description="Text or base64-encoded binary content")
    filename: str = Field("note.md", description="Artifact filename")
    content_type: str = Field("text/markdown", description="MIME type")


class ArtifactMetadataResponse(BaseModel):
    """Artifact metadata in responses."""

    id: str
    filename: str
    content_type: str
    size: int
    created_at: str


class GetArtifactResponse(BaseModel):
    """Response from getting artifact content."""

    content: str
    total_size: int
    has_more: bool


# ── Categories ────────────────────────────────────────────────────────


class CategoryItem(BaseModel):
    """A category with description and count."""

    name: str
    description: str
    count: int


class ListCategoriesResponse(BaseModel):
    """Response from listing categories."""

    categories: list[CategoryItem]


# ── Intelligence Layer ────────────────────────────────────────────────


class RecallRequest(BaseModel):
    """Request for the recall endpoint."""

    session_id: str | None = Field(
        None, description="Session ID from previous call. Null = first call."
    )
    query: str | None = Field(None, description="Free text search query")
    messages: list[dict] | None = Field(
        None,
        description="OpenAI-format messages. Last user message used as query if query not provided.",
    )
    include_instructions: bool = Field(
        False, description="Include behavioral instructions in response"
    )
    managed: bool = Field(
        False, description="Use managed-mode instructions (plugin-driven)"
    )
    instruction_mode: str | None = Field(
        None, description="Override instruction mode (passive/proactive/personality)"
    )
    recent_days: int = Field(7, description="Days of recent context for core memories")
    ttl: int | None = Field(
        None, description="Session TTL in seconds (only used on first call)"
    )


class RecallStats(BaseModel):
    """Statistics from a recall operation."""

    core_count: int = 0
    search_count: int = 0
    new_count: int = 0
    known_skipped: int = 0
    latency_ms: int = 0


class RecallResponse(BaseModel):
    """Response from the recall endpoint."""

    session_id: str
    instructions: str | None = None
    core_memories: str | None = None
    search_results: list[MemoryItem] = Field(default_factory=list)
    stats: RecallStats = Field(default_factory=RecallStats)


class RememberRequest(BaseModel):
    """Request for the remember endpoint."""

    session_id: str | None = Field(
        None, description="Session ID to update known memory IDs"
    )
    messages: list[dict] = Field(
        ..., description="OpenAI-format messages (typically last 2: user + assistant)"
    )


class RememberResponse(BaseModel):
    """Response from the remember endpoint (immediate)."""

    accepted: bool = True


# ── Common ────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: bool = True
    message: str
