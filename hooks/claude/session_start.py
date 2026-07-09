#!/usr/bin/env python3
"""Claude Code SessionStart hook: catch-up consolidation sweep + health check.

Registered in ~/.claude/settings.json by scripts/configure_claude.py.
Whenever a session starts, poke the memory service to consolidate any
sessions still pending (missed SessionEnd — crash, force-quit, rate limit).
The service debounces, so opening several sessions in a row costs at most
one sweep per debounce window. Best-effort.

Normally prints nothing: SessionStart stdout would be injected as context,
and memory injection is UserPromptSubmit's job (query-relevant, bounded) —
not a session-wide dump. The one exception is the health warning below:
when the memory service is unreachable, every memory feature (recall, turn
writes, consolidation, docs ingest) fails silently for the whole session,
so a single loud line at session start is the difference between "the user
finds out now" and "the user finds out days later" (it has happened: the
service was once down for ~24h and the only trace was the ingest watcher's
log file).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MEMORY_BASE, get_json, post_json, read_payload  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    read_payload()  # drain + debug-log; content not needed
    if post_json("/memory/consolidate-pending", {}) is not None:
        return
    # The sweep failed — distinguish "service down" from a transient hiccup
    # before alarming anyone.
    if get_json("/health") is not None:
        return
    print(
        f"⚠ hermes-agent memory service is NOT reachable at {MEMORY_BASE} — "
        "long-term memory is OFFLINE for this session (no recall, no turn "
        "writes, no docs ingest; nothing said here will be remembered). "
        f"To restore it, run: cd '{REPO}' && docker compose up -d "
        "(diagnose with: docker compose logs llamaindex). "
        "Tell the user about this at the start of your first reply."
    )


if __name__ == "__main__":
    main()
