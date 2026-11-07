# Agent entry point

This file is read by every agent tool (Claude Code imports it through CLAUDE.md; Command Code, Codex and others read
it directly). It is the only entry point: do not add rules elsewhere.

The two instruction files below are imported (`@path`, supported by Claude Code and Command Code). Tools without
imports must read them first, in this order:

1. Working rules shared by all projects (kit-owned copy; change them in agent-kit, not here):
   @.agents/common/instructions.md
2. This project's rules, which refine or override the common ones:
   @.agents/project/instructions.md

Everything else is loaded only when relevant: see [.agents/README.md](.agents/README.md) (references, stack rules,
skills, history).
