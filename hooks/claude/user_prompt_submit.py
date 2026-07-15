#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook: auto-inject relevant long-term memory.

Registered in ~/.claude/settings.json by scripts/configure_claude.py. Before
Claude processes the prompt, recall memory relevant to it; plain stdout on
exit 0 is added to the turn's context. The model never needs to remember to
call the recall tool — same contract as the Hermes pre_llm_call hook.

Latency budget: one embed + three vector searches (~100-300ms local), 3s
timeout. Any failure degrades to no injection — never blocks the prompt.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import env_get, env_int, post_json, read_payload, resolve_project  # noqa: E402

TIMEOUT = float(env_get("LONGBRAIN_MEMORY_RECALL_TIMEOUT", "3"))
MAX_CONTEXT_CHARS = env_int("LONGBRAIN_MEMORY_MAX_CONTEXT", 6000)
# Prompts shorter than this ("ok", "tiếp tục", "continue") carry no searchable
# meaning — recall would only match noise, and every injected block costs the
# user's subscription tokens. The turn is still WRITTEN to memory by the Stop
# hook; only the lookup is skipped.
MIN_PROMPT_CHARS = env_int("LONGBRAIN_RECALL_MIN_PROMPT_CHARS", 15)

# Mirrors app.memories.is_vietnamese — this wrapper line is the one piece of
# injected text the hook itself controls (context_block's own headers are
# matched server-side), so it should follow the query's language too instead
# of guaranteeing a dose of English on every single Vietnamese turn.
_VN_CHARS_RE = re.compile(
    r"[ăâàáảãạằắẳẵặầấẩẫậêèéẻẽẹềếểễệìíỉĩịôơòóỏõọồốổỗộờớởỡợ"
    r"ưùúủũụừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def main():
    payload = read_payload()
    query = (payload.get("user_input") or payload.get("prompt") or "").strip()
    if len(query) < MIN_PROMPT_CHARS:
        return

    result = post_json("/memory/recall", {
        "query": query[:2000],
        "session_id": payload.get("session_id") or "",
        "project": resolve_project(payload.get("cwd") or "")[0],
        "recent_turns": 0,  # Claude Code carries its own session history
    }, timeout=TIMEOUT)
    if not result:
        return

    context = (result.get("context_block") or "").strip()
    if context:
        # Keep injection bounded: it costs the user's subscription tokens
        # on every turn.
        prefix = "Bộ nhớ dài hạn (tự động gọi lại):" if _VN_CHARS_RE.search(query) \
            else "Long-term memory (auto-recalled):"
        print(prefix + "\n" + context[:MAX_CONTEXT_CHARS])


if __name__ == "__main__":
    main()
