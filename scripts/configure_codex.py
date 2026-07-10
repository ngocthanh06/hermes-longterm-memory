#!/usr/bin/env python3
"""Wire Codex to the memory stack — MCP-only tier.

Codex has no automatic lifecycle hooks today, so this adapter registers only
the MCP server: the model can call memory_recall / save_memories /
search_history on its own initiative, but turns are NOT recorded
automatically (see adapters/README.md "Support tiers").

Run via setup.sh (or directly). Idempotent — safe to re-run. The single step:
add `[mcp_servers.longbrain]` with the Streamable HTTP URL to
`$CODEX_HOME/config.toml` (default `~/.codex/config.toml`).

The file is edited text-level (the macOS system python has no tomllib):
an existing `[mcp_servers.longbrain]` section is updated in place and its
sub-tables (`[mcp_servers.longbrain.tools.*]`, user-set approval modes) are
left untouched; a missing section is appended at end of file, which is
always a fresh valid TOML table.

Exit code 0 = wired (or Codex not installed — nothing to do); 1 = problem.
"""

import os
import shutil
import sys
import time
from pathlib import Path

CODEX_HOME = Path(os.environ.get("CODEX_HOME", "")) if os.environ.get("CODEX_HOME") \
    else Path.home() / ".codex"
CONFIG = CODEX_HOME / "config.toml"
SECTION = "[mcp_servers.longbrain]"
MCP_URL = "http://localhost:8800/mcp"
URL_LINE = f'url = "{MCP_URL}"'

ok_all = True


def note(msg: str) -> None:
    print(f"  {msg}")


def fail(msg: str) -> None:
    global ok_all
    ok_all = False
    print(f"  ✗ {msg}")


def detected() -> bool:
    return shutil.which("codex") is not None or CODEX_HOME.is_dir()


def register_mcp() -> None:
    print(f"==> {CONFIG} (MCP server)")
    lines = CONFIG.read_text().splitlines() if CONFIG.exists() else []

    header_idx = next(
        (i for i, line in enumerate(lines) if line.strip() == SECTION), None
    )
    if header_idx is None:
        block = ["", "# Longbrain shared memory (added by longbrain setup)",
                 SECTION, URL_LINE]
        new_lines = lines + block
        note(f"registered MCP longbrain -> {MCP_URL}")
    else:
        # Body of the main section only — it ends at the next table header,
        # which may be one of our own sub-tables (tools.* approval modes);
        # those belong to the user and are preserved as-is.
        end = next(
            (i for i in range(header_idx + 1, len(lines))
             if lines[i].lstrip().startswith("[")),
            len(lines),
        )
        url_idx = next(
            (i for i in range(header_idx + 1, end)
             if lines[i].split("=")[0].strip() == "url"),
            None,
        )
        if url_idx is not None and lines[url_idx].split("=", 1)[1].strip().strip('"') == MCP_URL:
            note("MCP longbrain already registered")
            return
        new_lines = list(lines)
        if url_idx is not None:
            new_lines[url_idx] = URL_LINE
            note(f"updated MCP longbrain url -> {MCP_URL}")
        else:
            new_lines.insert(header_idx + 1, URL_LINE)
            note(f"set MCP longbrain url -> {MCP_URL}")

    if CONFIG.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(CONFIG, CONFIG.with_name(f"config.toml.bak.{stamp}"))
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text("\n".join(new_lines) + "\n")
    note("config.toml written (restart Codex sessions to pick it up)")


def main() -> int:
    if not detected():
        print("Codex not found (no `codex` on PATH, no ~/.codex) — nothing to do.")
        return 0
    register_mcp()
    if ok_all:
        print("✓ Codex wired (MCP-only: memory tools available; turns are not "
              "recorded automatically — see adapters/README.md)")
        print("  Verify inside Codex: the longbrain tools should appear; "
              "Streamable HTTP MCP needs a recent Codex version.")
    else:
        print("✗ finished with problems (see above)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
