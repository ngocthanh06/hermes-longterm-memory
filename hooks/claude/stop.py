#!/usr/bin/env python3
"""Claude Code Stop hook: mirror the completed turn into episodic memory.

Registered in ~/.claude/settings.json by scripts/configure_claude.py. When
Claude finishes responding, extract the last user prompt + final assistant
reply from the session transcript and POST them to /memory/append —
the Claude Code counterpart of the Hermes post_llm_call hook.

The transcript JSONL format is internal to Claude Code and can change
between releases, so parsing is defensive: unknown lines are skipped, and
an empty extraction is silently dropped (the raw payload is still in
logs/claude-hook-debug.jsonl for diagnosis). The final assistant transcript
entry's own uuid is passed along as turn_id, a real idempotency key the
service uses instead of guessing retry-vs-new-occurrence from content and
session state alone. Best-effort throughout.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import post_json, read_payload, resolve_project  # noqa: E402


def _text_of(content) -> str:
    """Flatten a message content field: plain string, or list of blocks
    (only `text` blocks count — tool_use/tool_result/thinking are noise)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


def _is_tool_result(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def extract_last_turn(transcript_path: str) -> tuple:
    """(user_message, assistant_response, assistant_uuid) of the LAST
    completed turn.

    A turn = the last real user prompt (not a tool result, not meta, not a
    sidechain) and the final assistant text after it — later assistant
    entries overwrite earlier ones, so text emitted between tool calls
    doesn't shadow the closing reply. `assistant_uuid` is that final
    assistant entry's own transcript uuid — a stable per-turn idempotency
    key (see memory_store.add_message's turn_id) as long as this exact
    entry is what actually gets recorded; main() clears it when the
    fresher payload override applies instead (see there).
    """
    last_user, last_assistant, last_assistant_uuid = "", "", ""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("isSidechain") or entry.get("isMeta"):
                    continue
                message = entry.get("message") or {}
                content = message.get("content")
                if entry.get("type") == "user":
                    if _is_tool_result(content):
                        continue
                    text = _text_of(content)
                    if text.strip():
                        last_user, last_assistant, last_assistant_uuid = text, "", ""
                elif entry.get("type") == "assistant":
                    text = _text_of(content)
                    if text.strip():
                        last_assistant = text
                        last_assistant_uuid = str(entry.get("uuid") or "")
    except OSError:
        return "", "", ""
    return last_user, last_assistant, last_assistant_uuid


def main():
    payload = read_payload()
    transcript_path = payload.get("transcript_path") or ""
    session_id = payload.get("session_id") or ""
    if not transcript_path or not session_id:
        return

    user_message, assistant_response, turn_id = extract_last_turn(transcript_path)
    # The transcript may not have flushed the closing reply yet when Stop
    # fires (observed on 2.1.201: the assistant entry landed after the hook
    # ran) — the payload's own last_assistant_message is authoritative. When
    # that override actually changes the text, the transcript-derived
    # turn_id no longer identifies what's being sent (it's the OLD entry's
    # uuid, if any) — drop it and let add_message's heuristic fallback
    # handle idempotency for this narrower case instead of risking a
    # mismatched id.
    payload_reply = payload.get("last_assistant_message")
    if isinstance(payload_reply, str) and payload_reply.strip() and payload_reply != assistant_response:
        assistant_response = payload_reply
        turn_id = ""
    if not user_message and not assistant_response:
        return

    project_id, project_source = resolve_project(payload.get("cwd") or "")
    post_json("/memory/append", {
        "session_id": session_id,
        "user_message": user_message,
        "assistant_response": assistant_response,
        "project_id": project_id,
        "project_source": project_source,
        "source_agent": "claude-code",
        "turn_id": turn_id,
    })


if __name__ == "__main__":
    main()
