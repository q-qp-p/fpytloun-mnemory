#!/usr/bin/env bash
# mnemory recall hook for Claude Code
#
# Called on UserPromptSubmit. Reads the hook input from stdin,
# calls /api/recall with the user's message, and outputs
# additionalContext with memories + instructions.
#
# Environment variables:
#   MNEMORY_URL             - mnemory server URL (default: http://localhost:8050)
#   MNEMORY_API_KEY         - API key for authentication
#   MNEMORY_AGENT_ID        - Agent ID (default: claude-code)
#   MNEMORY_USER_ID         - User ID (optional if using API key mapping)
#   MNEMORY_SCORE_THRESHOLD - Min relevance score (default: 0.5)

set -euo pipefail

MNEMORY_URL="${MNEMORY_URL:-http://localhost:8050}"
MNEMORY_API_KEY="${MNEMORY_API_KEY:-}"
MNEMORY_AGENT_ID="${MNEMORY_AGENT_ID:-claude-code}"
MNEMORY_USER_ID="${MNEMORY_USER_ID:-}"
MNEMORY_SCORE_THRESHOLD="${MNEMORY_SCORE_THRESHOLD:-0.5}"

# Read hook input from stdin
INPUT=$(cat)

# Extract the user's message from the hook input
# UserPromptSubmit provides: { "sessionId": "...", "message": "..." }
USER_MESSAGE=$(echo "$INPUT" | jq -r '.message // empty' 2>/dev/null || true)

# Session file for tracking mnemory session ID across hook calls
SESSION_ID_FROM_INPUT=$(echo "$INPUT" | jq -r '.sessionId // empty' 2>/dev/null || true)
SESSION_FILE="/tmp/mnemory_session_${SESSION_ID_FROM_INPUT:-default}"

# Load existing mnemory session ID if available
MNEMORY_SESSION_ID=""
if [ -f "$SESSION_FILE" ]; then
    MNEMORY_SESSION_ID=$(cat "$SESSION_FILE" 2>/dev/null || true)
fi

# Build request headers
HEADERS=(-H "Content-Type: application/json" -H "X-Agent-Id: $MNEMORY_AGENT_ID")
if [ -n "$MNEMORY_API_KEY" ]; then
    HEADERS+=(-H "Authorization: Bearer $MNEMORY_API_KEY")
fi
if [ -n "$MNEMORY_USER_ID" ]; then
    HEADERS+=(-H "X-User-Id: $MNEMORY_USER_ID")
fi

# Build request body
BODY=$(jq -n \
    --arg query "$USER_MESSAGE" \
    --arg session_id "$MNEMORY_SESSION_ID" \
    --argjson score_threshold "$MNEMORY_SCORE_THRESHOLD" \
    '{
        include_instructions: true,
        managed: true,
        score_threshold: $score_threshold
    }
    + (if $query != "" then {query: $query} else {} end)
    + (if $session_id != "" then {session_id: $session_id} else {} end)'
)

# Call /api/recall
RESPONSE=$(curl -s --max-time 25 \
    -X POST \
    "${HEADERS[@]}" \
    -d "$BODY" \
    "${MNEMORY_URL}/api/recall" 2>/dev/null || echo '{}')

# Save session ID for subsequent calls
NEW_SESSION_ID=$(echo "$RESPONSE" | jq -r '.session_id // empty' 2>/dev/null || true)
if [ -n "$NEW_SESSION_ID" ]; then
    echo "$NEW_SESSION_ID" > "$SESSION_FILE"
fi

# Build the additional context from the recall response
CONTEXT_PARTS=""

# Instructions (only on first call)
INSTRUCTIONS=$(echo "$RESPONSE" | jq -r '.instructions // empty' 2>/dev/null || true)
if [ -n "$INSTRUCTIONS" ]; then
    CONTEXT_PARTS="$INSTRUCTIONS"
fi

# Core memories (only on first call)
CORE_MEMORIES=$(echo "$RESPONSE" | jq -r '.core_memories // empty' 2>/dev/null || true)
if [ -n "$CORE_MEMORIES" ]; then
    if [ -n "$CONTEXT_PARTS" ]; then
        CONTEXT_PARTS="$CONTEXT_PARTS

$CORE_MEMORIES"
    else
        CONTEXT_PARTS="$CORE_MEMORIES"
    fi
fi

# Search results
SEARCH_RESULTS=$(echo "$RESPONSE" | jq -r '
    .search_results // [] |
    map(select(.memory != null and .memory != "")) |
    map("- " + .memory) |
    join("\n")' 2>/dev/null || true)

if [ -n "$SEARCH_RESULTS" ]; then
    RECALLED="## Recalled Memories
$SEARCH_RESULTS"
    if [ -n "$CONTEXT_PARTS" ]; then
        CONTEXT_PARTS="$CONTEXT_PARTS

$RECALLED"
    else
        CONTEXT_PARTS="$RECALLED"
    fi
fi

# Output additionalContext for Claude Code to inject
if [ -n "$CONTEXT_PARTS" ]; then
    jq -n --arg ctx "$CONTEXT_PARTS" '{additionalContext: $ctx}'
else
    echo '{}'
fi
