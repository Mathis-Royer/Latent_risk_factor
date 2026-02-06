# Project Structure

> **Note:** This file documents the project tree and remote directories. Built incrementally. Sections marked with {TODO} need to be completed.

## Project Tree

```
{TODO: project-name}/
├── .claude/                          # Claude Code configuration
│   ├── settings.json                 # Hooks (pyright, pylint, flake8)
│   ├── rules/                        # Auto-loaded rules
│   │   ├── 00-claude-config.md       # Claude configuration
│   │   ├── 01-python-guidelines.md   # Python coding rules
│   │   ├── 02-git-commit.md          # Git commit rules
│   │   ├── 03-code-quality.md        # Quality verification
│   │   ├── 04-task-completion.md     # Task completion rules
│   │   ├── 05-comments-readme.md     # Comments and README rules
│   │   ├── 06-documentation-protocol.md  # Documentation protocol
│   │   ├── 07-environment-setup.md   # Environment setup
│   │   └── project/                  # Project-specific documentation
│   │       ├── project-overview.md   # Project goals by pipeline
│   │       ├── structure.md          # This file
│   │       ├── changelog.md          # Recent changes
│   │       └── {TODO}.md             # Domain-specific docs
│   ├── commands/                     # User slash commands
│   │   └── {TODO}.md
│   └── agents/                       # Specialized sub-agents
│       └── {TODO}.md
│
├── src/                              # {TODO: Main source code}
│   ├── {TODO}/                       # {TODO: Module description}
│   │   └── {TODO}.py                 # {TODO: File description}
│   └── {TODO}.py                     # {TODO: File description}
│
├── tests/                            # {TODO: Test files}
│   └── {TODO}
│
├── docs/                             # {TODO: Additional documentation}
│   └── {TODO}
│
├── README.md                         # Project overview
└── requirements.txt                  # Python dependencies
```

---

## Remote Directories

These are external data directories used by the project. They are not part of the git repository.

{TODO: Document external data directories if any. Remove this section if not applicable.}

### {TODO: Directory Name}

- **Absolute Path:** {TODO: `/path/to/directory`}
- **Description:** {TODO: What this directory contains}
- **Structure:**
```
{TODO: directory-name}/
├── {TODO}/
│   └── {TODO}
└── {TODO}
```
- **Used by:** {TODO: `src/file.py` - what it does with this data}

---

### {TODO: Second Directory Name}

- **Absolute Path:** {TODO}
- **Description:** {TODO}
- **Structure:**
```
{TODO}
```
- **Used by:** {TODO}

---

## Update History

| Date | Section | Change |
|------|---------|--------|
| {TODO: YYYY-MM-DD} | Initial | Created minimal structure |
