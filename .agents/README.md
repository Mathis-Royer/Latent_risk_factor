# Agent documentation index

Single maintained source of agent rules for this repository. Load references only when the task needs them.

- Stack rules: [Python](stacks/python/instructions.md), [testing](stacks/python/testing.md),
  [environment](stacks/python/environment.md)
- Common policies: [git](common/git.md), [tools and plugins](common/tooling.md)
- Project reference: [index](project/reference/README.md) — [overview](project/reference/project-overview.md),
  [structure](project/reference/structure.md), [data](project/reference/data.md)
- Project skills: [run-targeted-tests](skills/run-targeted-tests/SKILL.md). Generic skills (external sub-agents,
  paired audits, parallel sessions, project setup, kit sync, skill writing) come from agent-kit, installed per user.
- Durable results: [history](project/history/README.md) — history is never proof of current code.

`common/` and `stacks/` are kit-owned copies (header `agent-kit: <source> v<version>`): change them in agent-kit and
resync, never here. After changing this directory, run `python3 <agent-kit>/scripts/check_config.py .`.
