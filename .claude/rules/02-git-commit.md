# Git Commit Guidelines

> **PORTABILITY NOTICE**: This file is designed to be **reusable across any Python project**.

## Commit Message Rules

- Write commit messages in English that explain the **objective/goal** of the changes, not the technical details.
- Use **one bullet point per objective** (use `-` as bullet marker).
- Focus on **why** the change was made, not **what** was changed technically.
- Do not commit files listed in `.gitignore`.
- Use VS Code extensions for linting and formatting (e.g., Pylint, Black).

## Before Writing a Commit Message

1. **Read `.claude/rules/project/changelog.md`** to understand the context and objectives of recent work.

2. **Compare with last pushed version**: Use `git diff origin/<branch> -- <file>` to see what changed since the last push. This helps identify what's new vs what was already committed.

3. **Identify the main file(s)**: Determine which file(s) represent the main entry points or pipelines (e.g., `main.py`, a notebook, CLI scripts). Understand utility file changes from the perspective of their impact on these main files. Ask yourself: "What does this change enable or fix in the main workflow?"

4. **Write the message** from the user's perspective: What problem was solved? What feature was added? What behavior changed?

## Example: Good Commit Message

```
Improve model training reproducibility

- Ensure datasets are loaded identically across sessions
- Fix random sampling to use fixed seed
```

## Example: Bad Commit Message

```
Update utils.py

- Added hashlib import at line 20
- Changed Random() to Random(42) in 3 functions
- Modified save_data() to handle edge case
```

## Useful Git Commands

```bash
# View changes before commit
git diff origin/main -- .

# View staged changes
git diff --cached

# View recent commits
git log --oneline -10

# View all changes since branching from main
git diff main...HEAD
```
