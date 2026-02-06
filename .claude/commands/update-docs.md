---
name: update-docs
description: Updates project documentation after modifications
allowed-tools: [Read, Write, Edit, Glob, Grep, Bash]
---

# Update Documentation Command

Updates project documentation in `.claude/rules/project/` after significant modifications.

## When Invoked with `/update-docs`

### 1. Analyze Recent Modifications

```bash
git status
git diff --stat
```

### 2. Read Current Documentation State

- `.claude/rules/project/changelog.md` - Current state and history
- `.claude/rules/project/structure.md` - Project tree
- `.claude/rules/project/datasets.md` - Dataset documentation (if exists)
- `.claude/rules/project/project-overview.md` - Pipelines and objectives

### 3. Update Appropriate Files

| Modification | File to Update |
|--------------|----------------|
| New file/folder | `structure.md` |
| New pipeline | `project-overview.md` |
| New dataset specificity | `datasets.md` |
| Any significant modification | `changelog.md` |

### 4. Changelog Format

Add entry at position 1, shift others down:

```markdown
| 1 | YYYY-MM-DD | **Title**: Brief description |
```

If > 10 entries, remove the oldest (#10).

### 5. Update "Current State" If Necessary

In `changelog.md`, section "Current State" - summarizes the current project state.

## Rules

- Do not create new documentation files
- Respect the existing format of each file
- Use `{TODO}` markers for missing information
- Paths must be absolute for remote directories
