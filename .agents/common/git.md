<!-- agent-kit: common/git.md v0.1.3 -->
# Git conventions (agent-kit common, loaded on demand)

- Commit or push only when the user asks, or when the project instructions grant it for the task.
- Commit messages explain the objective, not the mechanics: a short title, then one bullet per goal
  ("Keep the test suite side-effect free"), not per edited line.
- Before writing the message, look at what changed since the last push (`git diff origin/<branch>`) and describe
  the effect on the main entry points (scripts, notebooks, CLI), not on helpers.
- Never commit ignored files, secrets, generated artifacts or large data.
- Never force-push, `reset --hard`, rewrite shared history or delete branches you did not create.
- Work on a task branch when another session may be active on the same repository (`parallel-sessions` skill).
- Add the attribution lines your tool requires at the end of commit messages and pull request descriptions.
