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

# Stop provides: { "session_id": "...", "transcript_path": "...", "stop_hook_active": bool }

# Don't re-run while a previous Stop hook is active
STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false' 2>/dev/null || echo "false")
if [ "$STOP_HOOK_ACTIVE" = "true" ]; then
    echo '{}'
    exit 0
fi

# Read the transcript from transcript_path (fallback: inline .transcript)
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null || true)
INLINE_TRANSCRIPT=$(echo "$INPUT" | jq -c '.transcript // empty' 2>/dev/null || true)

# Last non-empty text for a role. content is a string or an array of
# blocks; keep only text blocks (skip tool_result/thinking/tool_use).
extract_last() {
    local role="$1"
    if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
        jq -rs --arg role "$role" '
            [ .[]
              | select(.type == $role)
              | select((.isMeta // false) | not)
              | select((.isSidechain // false) | not)
              | .message.content
              | if type == "string" then .
                elif type == "array" then ([ .[] | select(.type == "text") | .text ] | join("\n"))
                else "" end
            ]
            | map(select(. != null and . != ""))
            | last // ""
        ' "$TRANSCRIPT_PATH" 2>/dev/null || echo ""
    elif [ -n "$INLINE_TRANSCRIPT" ]; then
        echo "$INLINE_TRANSCRIPT" | jq -r --arg role "$role" '
            [ .[] | select(.role == $role) | (.content // "") ]
            | map(select(. != "")) | last // ""
        ' 2>/dev/null || echo ""
    fi
}

USER_MSG=$(extract_last user)

# Skip if there's no user message to remember
if [ -z "$USER_MSG" ]; then
    echo '{}'
    exit 0
fi

if [ "$MNEMORY_INCLUDE_ASSISTANT" = "true" ]; then
    ASSISTANT_MSG=$(extract_last assistant)
    MESSAGES=$(jq -n --arg u "$USER_MSG" --arg a "$ASSISTANT_MSG" \
        '[{role: "user", content: $u}] + (if $a != "" then [{role: "assistant", content: $a}] else [] end)')
else
    MESSAGES=$(jq -n --arg u "$USER_MSG" '[{role: "user", content: $u}]')
fi

# Load mnemory session ID
SESSION_ID_FROM_INPUT=$(echo "$INPUT" | jq -r '.session_id // .sessionId // empty' 2>/dev/null || true)
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
