#!/usr/bin/env bash
# mnemory remember hook for Claude Code
#
# Called on Stop. Reads the hook input from stdin,
# extracts the conversation transcript, and calls /api/remember
# to store new memories. Fire-and-forget.
#
# Environment variables:
#   MNEMORY_URL                - mnemory server URL (default: http://localhost:8050)
#   MNEMORY_API_KEY            - API key for authentication
#   MNEMORY_AGENT_ID           - Agent ID (default: claude-code)
#   MNEMORY_USER_ID            - User ID (optional if using API key mapping)
#   MNEMORY_INCLUDE_ASSISTANT  - Include assistant messages in remember calls (default: false)

set -euo pipefail

MNEMORY_URL="${MNEMORY_URL:-http://localhost:8050}"
MNEMORY_API_KEY="${MNEMORY_API_KEY:-}"
MNEMORY_AGENT_ID="${MNEMORY_AGENT_ID:-claude-code}"
MNEMORY_USER_ID="${MNEMORY_USER_ID:-}"
MNEMORY_INCLUDE_ASSISTANT="${MNEMORY_INCLUDE_ASSISTANT:-false}"

# Read hook input from stdin
INPUT=$(cat)

# Extract the stop reason and transcript from the hook input
# Stop provides: { "sessionId": "...", "stopReason": "...", "transcript": [...] }
STOP_REASON=$(echo "$INPUT" | jq -r '.stopReason // empty' 2>/dev/null || true)

# Only remember on normal stops (not errors or interrupts)
if [ "$STOP_REASON" = "error" ] || [ "$STOP_REASON" = "interrupt" ]; then
    echo '{}'
    exit 0
fi

# Extract the last user + assistant messages from the transcript
# The transcript is an array of {role, content} objects
if [ "$MNEMORY_INCLUDE_ASSISTANT" = "true" ]; then
    MESSAGES=$(echo "$INPUT" | jq -c '
        .transcript // [] |
        # Get the last user message and the last assistant message
        (map(select(.role == "user")) | last // null) as $user |
        (map(select(.role == "assistant")) | last // null) as $assistant |
        [
            (if $user then {role: "user", content: ($user.content // "")} else null end),
            (if $assistant then {role: "assistant", content: ($assistant.content // "")} else null end)
        ] | map(select(. != null and .content != ""))
    ' 2>/dev/null || echo '[]')
else
    MESSAGES=$(echo "$INPUT" | jq -c '
        .transcript // [] |
        # Get only the last user message
        (map(select(.role == "user")) | last // null) as $user |
        [
            (if $user then {role: "user", content: ($user.content // "")} else null end)
        ] | map(select(. != null and .content != ""))
    ' 2>/dev/null || echo '[]')
fi

# Skip if no messages to remember
if [ "$MESSAGES" = "[]" ] || [ "$MESSAGES" = "null" ] || [ -z "$MESSAGES" ]; then
    echo '{}'
    exit 0
fi

# Load mnemory session ID
SESSION_ID_FROM_INPUT=$(echo "$INPUT" | jq -r '.sessionId // empty' 2>/dev/null || true)
SESSION_FILE="/tmp/mnemory_session_${SESSION_ID_FROM_INPUT:-default}"
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

# Build labels for provenance tracking
LABELS=$(jq -n --arg src "claude-code" --arg sid "$SESSION_ID_FROM_INPUT" \
    '{source: $src} + (if $sid != "" then {session_id: $sid} else {} end)')

# Build request body
BODY=$(jq -n \
    --argjson messages "$MESSAGES" \
    --arg session_id "$MNEMORY_SESSION_ID" \
    --argjson labels "$LABELS" \
    '{messages: $messages, labels: $labels}
    + (if $session_id != "" then {session_id: $session_id} else {} end)'
)

# Fire-and-forget: call /api/remember in background
curl -s --max-time 10 \
    -X POST \
    "${HEADERS[@]}" \
    -d "$BODY" \
    "${MNEMORY_URL}/api/remember" >/dev/null 2>&1 &

# Output empty (no modifications to Claude's behavior)
echo '{}'
