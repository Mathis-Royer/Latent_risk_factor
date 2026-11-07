<!-- agent-kit: common/instructions.md v0.1.3 -->
# Working instructions (agent-kit common)

## Scope
- Treat "can you", "I want" and "help me" as requests to do the work, carried through verification.
- Ask only when missing information changes the outcome or an action exceeds the authorization given.
- User instructions override skills and plugins. Project instructions refine or override these common rules.
- Stay in scope. Preserve unrelated edits and local data; never stage, commit, stash or reset work you did not make.

## Understand before acting
- Read the code and the current state first. Documents, plans and past audits are history, not proof of current
  behavior: check every claim against the code and the data before relying on it.
- Reproduce a reported problem before fixing it, then trace it to its root cause. A stale cache or build is a
  hypothesis to prove, not an explanation.
- Prefer one structural correction at the shared boundary over a stack of local patches, and look for sibling
  occurrences of the same defect.
- Reuse existing code and platform features. Write the least code that solves the problem; no speculative
  abstraction. Name every shortcut or deferred item explicitly.
- When a practice, tool or API may have changed, read current official documentation and cite it.

## Verification and honesty
- Choose checks proportionate to the risk. Report what was tested, what was not, and what is blocked.
- Never claim a run, test or result that did not happen. Report failures with their output.
- Keep apart what documents claim, what the code does and what results show.

## Machine resources
- Respect the machine limits stated in the project instructions. Run at most one heavy job (training, full test
  suite, large data processing) at a time across all sessions: wrap it with the kit's `heavy_lock.py`.
- Use the smallest configuration that answers the question. Check free memory and disk before a heavy batch.
- Give every background process a bounded lifetime, stop what you start, delete your own temporary artifacts.

## Delegation and parallel work
- Delegate only bounded, independent work. Sub-agents read and report; one writer per working copy.
- Two sessions on one repository: follow the `parallel-sessions` skill (one task, one branch, one worktree).
- Agents of another tool or model: follow the `external-subagents` skill.

## Where results go
- Durable results (audit, validated plan, measurement, decision): `.agents/project/history/<YYYY-MM-DD>-<topic>/`.
- Maintained knowledge: `.agents/project/reference/`. Drafts: the tool's scratchpad, never the repository.
- Secrets never go into files, commits or tool output.

## Communication
- Answer in the user's language, lead with the outcome, use plain words.
- Repository files are written in English unless the project instructions say otherwise.
