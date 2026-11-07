# Agent configuration migration — `.claude/` to `AGENTS.md` + `.agents/` (2026-11-07)

Why: the old configuration was Claude-only (no `AGENTS.md` for Command Code or Codex), loaded ~74 KB of rules and
templates in every session, ran lint hooks whose output Claude never saw, gave sub-agents every tool (unsupported
`allowed-tools` field), let an agent that deletes conversation history be delegated automatically, and contradicted
itself on tests and environment commands. The generic parts now live in the personal agent-kit
(github.com/Mathis-Royer/agent-kit, v0.1.3); this repository keeps only what is specific to it.

## Ledger

| Former file | Destination | Disposition |
|---|---|---|
| `.claude/README.md`, `.claude/PROTOCOL.md` | agent-kit `AGENTS.md`, `README.md` | Deleted here; template documentation (referred to a non-existent `.claude-template/`) |
| `rules/00-claude-config.md` | agent-kit | Deleted; meta-documentation loaded every session |
| `rules/00-role.md` | `.agents/project/instructions.md` (role and priorities) | Synthesized; CRSP context corrected to Tiingo |
| `rules/01-python-guidelines.md` | kit `stacks/python/instructions.md` → `.agents/stacks/python/` | Synthesized |
| `rules/02-git-commit.md` | kit `common/git.md` → `.agents/common/` | Synthesized |
| `rules/03-code-quality.md` | kit `stacks/python/instructions.md` + kit hook | Synthesized; `pip install` advice and invisible hooks removed |
| `rules/04-task-completion.md` | kit `common/instructions.md` | Synthesized |
| `rules/05-comments-readme.md` | kit `common/instructions.md`, `stacks/python/instructions.md` | Synthesized |
| `rules/06-documentation-protocol.md` | `.agents/project/{reference,history}/` convention | Replaced; single always-growing changelog dropped (git log + history folders) |
| `rules/07-environment-setup.md` | kit `stacks/python/environment.md` + project instructions | Corrected (`uv sync --all-extras`; there is no requirements.txt) |
| `rules/08-testing-philosophy.md` | kit `stacks/python/testing.md` → `.agents/stacks/python/` | Moved unchanged |
| `rules/project/project-overview.md` | `.agents/project/reference/project-overview.md` | Moved (git mv) and rewritten: Tiingo instead of CRSP, current status, diagnostics pipeline |
| `rules/project/structure.md` | `.agents/project/reference/structure.md` | Moved and regenerated from `git ls-files` + module docstrings |
| `rules/project/changelog.md` | `.agents/project/history/2026-02-to-11-changelog/changelog.md` | Archived unchanged (not normative) |
| `rules/project/datasets.md` | `.agents/project/reference/data.md` | Rewritten from code and data (the old file was an unfilled template) |
| `commands/lint.md` | — | Deleted (the kit hook lints every edit) |
| `commands/test.md`, `agents/test-runner.md` | `.agents/skills/run-targeted-tests/` | Merged into one portable skill; the `pytest tests/` contradiction removed; test mapping by import grep |
| `commands/update-docs.md` | — | Deleted (history convention) |
| `agents/clear-conversations.md` | kit `scripts/clear_claude_history.sh` | Removed from the project: a destructive helper must not be auto-delegable |
| `agents/TEMPLATE-agent.md`, `skills/TEMPLATE-skill/` | — | Deleted (templates were loaded as real items); kit `write-skill` replaces them |
| `skills/external-subagents/` (untracked) | kit `skills/external-subagents/` | Canonical copy in the kit (newer: session resume) |
| `settings.json` | `.claude/settings.json` | Rewritten: lint hooks removed (kit plugin returns findings to Claude), agent-kit enabled, overlapping or off-topic plugins disabled for this project (Superpowers, Cloudflare, code-modernization, frontend/UI design, TypeScript LSP) |
| `settings.local.json.example` | unchanged | Kept |
| memory "light runs only" | `.agents/project/instructions.md` (machine limits) | Promoted from Claude memory to a project rule visible to every tool |
| (new) | `AGENTS.md`, `CLAUDE.md`, `.agents/README.md`, `.agents/project/*.txt` | Created from the kit template |

Not migrated: `docs/` (specification and dated analyses). Pylint is no longer run on every edit (the kit runs pyright
and flake8 from `.venv/bin`); run it by hand when needed.

## Checks

Run on 2026-11-07 with agent-kit v0.1.4:

- `check_config.py .` → `agent config OK` (it reported 36 errors on the old layout: ignored `allowed-tools` in three
  agents, stale `requirements.txt` / full-suite commands, 41 KB of always-loaded rules for a 16 KB budget).
- Always-loaded context: AGENTS.md + common and project instructions + `.agents/README.md` ≈ 8 KB (was ≈ 74 KB).
- `sync_kit.py` → all six kit-owned copies up to date; `run-targeted-tests` reported as project-specific.
- Fresh Claude Code session (headless, in this repository): quotes the machine limits from the project instructions,
  lists `run-targeted-tests` (through the `.claude/skills` stub) and the six `agent-kit:` skills; no Superpowers
  injection. Before the stub existed, the project skill was invisible to Claude: the kit validator now flags it.
- Fresh Command Code session: quotes the same instructions (`@path` imports) and lists `run-targeted-tests` and the
  kit skills from `~/.agents/skills`.
- Kit hooks (fresh Claude session, scratch directory): writing `.env` is refused by the PreToolUse guard; an unused
  import in a new `.py` file comes back to the model as ruff findings (`additionalContext`).

Left for the user (agents may not edit Claude permission files): `.claude/settings.json` still lists
`Write(**/.env)` and `Write(**/.env.*)` deny rules, which Claude Code ignores (only `Edit(...)` rules apply to file
tools); the `Edit(...)` rules and the kit guard already cover them.
