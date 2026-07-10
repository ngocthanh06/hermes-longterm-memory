# Contributing to Longbrain

Thanks for your interest! Longbrain is a local-first, single-user memory
stack for AI agents — contributions that strengthen that focus are the most
likely to land.

## Getting started

```bash
git clone https://github.com/ngocthanh06/longbrain.git
cd longbrain
./setup.sh          # Docker required; builds and starts the stack
```

The service lives in `llamaindex-service/` (FastAPI), agent adapters in
`hooks/` (Hermes Desktop, Claude Code) and `adapters/` (SDK + examples).
Architecture deep-dive: [ARCHITECTURE.md](ARCHITECTURE.md).

## Running tests

```bash
cd llamaindex-service && python3 -m pytest tests/ -q
```

Recall quality is guarded by an eval set — run it against a live stack
before and after changes that touch recall, and compare with the committed
baseline:

```bash
python3 scripts/recall_eval.py
```

The project philosophy is **measure first, optimize later**: a change that
claims to improve recall must come with numbers from `recall_eval.py`.

## What we're looking for

- **New agent adapters** — the highest-impact contribution. Start from
  [adapters/README.md](adapters/README.md) and the `python_minimal` example;
  an adapter only needs to call `POST /memory/recall` before a turn and
  `POST /memory/append` after it.
- Bug fixes with a reproducing test.
- Recall-quality improvements backed by eval numbers.

## What we'll likely decline

- Multi-user / sync / cloud features — single-user local-first is a
  deliberate design decision, not a missing feature.
- New required dependencies on the host (the host side is stdlib-only by
  design; heavy lifting belongs in the Docker service).

## Conventions

- Hooks must stay **best-effort**: a memory-service hiccup must never break
  or slow the user's conversation. Degrade to a no-op, don't raise.
- Env vars use the `LONGBRAIN_*` prefix; the `HERMES_*` alias exists only
  for pre-rename installs — don't add new `HERMES_*` names.
- Keep PRs focused; one topic per PR.
