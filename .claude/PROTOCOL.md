# Claude Code Configuration Protocol

A comprehensive guide for building and maintaining `.claude/` directories in Python projects.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Configuration Hierarchy](#configuration-hierarchy)
3. [Directory Structure](#directory-structure)
4. [Rules System](#rules-system)
5. [Commands System](#commands-system)
6. [Skills System](#skills-system)
7. [Agents System](#agents-system)
8. [Hooks System](#hooks-system)
9. [Documentation Protocol](#documentation-protocol)
10. [Workflow Integration](#workflow-integration)
11. [Maintenance Guidelines](#maintenance-guidelines)

---

## Core Principles

### 1. Layered Configuration

Configuration flows from global to local, with later layers overriding earlier ones:

```
~/.claude/CLAUDE.md          # Global (all projects)
    ↓
.claude/settings.json        # Project (team-shared)
    ↓
.claude/settings.local.json  # Personal (gitignored)
```

### 2. Separation of Concerns

| Component | Purpose | Trigger | Example |
|-----------|---------|---------|---------|
| **Rules** | Provide context/guidelines | Auto-loaded | Coding standards |
| **Commands** | Execute workflows | User types `/name` | `/test`, `/deploy` |
| **Skills** | Domain expertise | Auto-detected | Testing patterns |
| **Agents** | Specialized sub-tasks | Task tool | Data processing |

### 3. Incremental Documentation

- **Build as you work** - Don't try to document everything upfront
- **{TODO} markers** - Mark incomplete sections for later
- **Update on change** - Keep docs current with code

### 4. Context Efficiency

- **Path filters** - Only load relevant rules for current work
- **Focused rules** - Under 300 lines per file
- **Minimal context** - Remove obsolete information

### 5. Team vs Personal

| Committed | Gitignored |
|-----------|------------|
| `settings.json` | `settings.local.json` |
| `rules/*.md` | Personal preferences |
| `commands/*.md` | Local experiments |

---

## Configuration Hierarchy

### Level 1: Global Configuration

Location: `~/.claude/`

For rules that apply to ALL your projects:

```markdown
# ~/.claude/CLAUDE.md

## Universal Rules
- Write all code in English
- Use type hints for all functions
- Prefer explicit over implicit
```

### Level 2: Project Configuration

Location: `.claude/`

For team-shared project standards:

```
.claude/
├── settings.json      # Hooks, permissions
└── rules/             # Project rules
```

### Level 3: Local Configuration

Location: `.claude/settings.local.json`

For personal preferences (gitignored):

```json
{
  "hooks": {
    "PostToolUse": []
  }
}
```

---

## Directory Structure

### Minimum Viable Setup

```
.claude/
├── settings.json
└── rules/
    └── project.md
```

### Recommended Full Setup

```
.claude/
├── settings.json              # Hooks, permissions
├── settings.local.json        # Personal (gitignored)
│
├── rules/                     # Auto-loaded instructions
│   ├── 00-claude-config.md    # Claude Code explanation
│   ├── 01-python-guidelines.md # Python standards
│   ├── 02-git-commit.md       # Git conventions
│   ├── 03-code-quality.md     # Linting, verification
│   ├── 04-task-completion.md  # Task checklist
│   ├── 05-comments-readme.md  # Documentation style
│   ├── 06-documentation-protocol.md # Doc maintenance
│   ├── 07-environment-setup.md # Setup instructions
│   └── project/               # Project-specific
│       ├── project-overview.md
│       ├── structure.md
│       ├── changelog.md
│       └── {domain}.md        # datasets.md, api.md, etc.
│
├── commands/                  # User slash commands
│   └── *.md
│
├── skills/                    # Domain expertise
│   └── {skill-name}/
│       └── SKILL.md
│
└── agents/                    # Specialized agents
    └── *.md
```

---

## Rules System

### Numbering Convention

```
00-*.md    # Core configuration
01-*.md    # Language-specific
02-*.md    # Workflow (git, CI/CD)
03-*.md    # Quality (linting, testing)
04-09-*.md # Other generic rules
project/   # Project-specific documentation
```

### Rule File Structure

```markdown
# Rule Title

> **PORTABILITY NOTICE**: This file is designed to be reusable.

## Section 1

Content...

## Section 2

Content...
```

### Conditional Rules (Path Filtering)

```markdown
---
paths:
  - "src/training/**"
  - "src/models/**"
---

# Training-Specific Rules

These rules only apply to training code.
```

**Path Pattern Examples:**
- `src/**/*.py` - All Python files in src/
- `tests/**` - All files in tests/
- `*.md` - Markdown files in root only
- `**/*.test.ts` - All test files

### Size Guidelines

| File Type | Max Lines | Split Strategy |
|-----------|-----------|----------------|
| Generic rules | 200 | By topic |
| Project rules | 300 | By domain |
| Datasets docs | 500+ | Per-dataset files |

---

## Commands System

### Command Structure

```markdown
---
name: command-name
description: Brief description with keywords
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep]
---

# Command Title

## When Invoked with `/command-name`

### Step 1: Action

Instructions...

### Step 2: Action

Instructions...

## Rules

- Rule 1
- Rule 2
```

### Frontmatter Options

| Option | Description | Example |
|--------|-------------|---------|
| `name` | Command name (required) | `test` |
| `description` | Help text with keywords | `Run tests on modified files` |
| `allowed-tools` | Permitted tools | `[Read, Bash]` |
| `context: fork` | Isolated context | For complex workflows |

### Common Commands

| Command | Purpose |
|---------|---------|
| `/test` | Run tests |
| `/lint` | Run linters |
| `/build` | Build project |
| `/deploy` | Deploy to environment |
| `/update-docs` | Update documentation |
| `/commit` | Guided commit workflow |

---

## Skills System

### Skill Structure

```markdown
---
name: skill-name
description: Keywords-rich description for auto-detection
allowed-tools: [Read, Write, Edit]
---

# Skill Title

## When to Use

Trigger conditions...

## Domain Knowledge

Patterns, best practices...

## Examples

Code examples...
```

### When to Create Skills

| Scenario | Create Skill? |
|----------|---------------|
| Complex domain patterns | Yes |
| Project-specific conventions | Yes |
| One-off instructions | No (use rules) |
| User-triggered workflows | No (use commands) |

---

## Agents System

### Agent Structure

```markdown
---
name: agent-name
description: What this agent specializes in
allowed-tools: [Read, Bash, Grep, Glob, Edit, Write]
---

# Agent Name

You are specialized in {domain}.

## Responsibilities

- Primary task
- Secondary task

## Domain Knowledge

Tables, references...

## Before Any Modification

Pre-check steps...

## Critical Points

Important constraints...
```

### When to Create Agents

| Scenario | Create Agent? |
|----------|---------------|
| Deep domain expertise needed | Yes |
| Repetitive specialized tasks | Yes |
| Focused context beneficial | Yes |
| Simple one-off tasks | No |

---

## Hooks System

### Hook Events

| Event | When | Can Block? |
|-------|------|------------|
| `SessionStart` | At session begin | No |
| `PreToolUse` | Before tool runs | Yes (exit 2) |
| `PostToolUse` | After tool runs | No |
| `UserPromptSubmit` | User sends message | No |
| `Stop` | Claude stops | No |

### settings.json Structure

```json
{
  "permissions": {
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push --force*)",
      "Edit(**/.env)",
      "Write(**/.env)"
    ]
  },
  "hooks": {
    "SessionStart": [
      {
        "hooks": [{
          "type": "command",
          "command": "echo '=== Git Status ===' && git status -s | head -10 || true"
        }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "your-command-here"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "blocking-check-command"
          }
        ]
      }
    ]
  }
}
```

### Common Hook Patterns

**Auto-lint Python files:**
```json
{
  "matcher": "Write|Edit",
  "hooks": [{
    "type": "command",
    "command": "file=$(cat | python3 -c \"import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))\"); [[ $file == *.py ]] && pyright $file || true"
  }]
}
```

**Block edits on main branch:**
```json
{
  "matcher": "Edit|Write",
  "hooks": [{
    "type": "command",
    "command": "[[ $(git branch --show-current) == 'main' ]] && exit 2 || true"
  }]
}
```

**Auto-format on save:**
```json
{
  "matcher": "Write",
  "hooks": [{
    "type": "command",
    "command": "file=$(cat | jq -r '.tool_input.file_path'); [[ $file == *.py ]] && black $file || true"
  }]
}
```

### Exit Codes

| Code | Meaning | Use Case |
|------|---------|----------|
| 0 | Success | Continue normally |
| 2 | Block action | PreToolUse only - prevent tool |
| Other | Non-blocking | Show feedback, continue |

### Permission Rules

The `permissions` section in settings.json controls what tools can do:

```json
{
  "permissions": {
    "allow": [
      "Read(~/**)",
      "Glob(~/**)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push --force*)",
      "Bash(git reset --hard*)",
      "Edit(**/.env)",
      "Edit(**/.env.*)",
      "Edit(~/.ssh/**)",
      "Write(**/.env)"
    ]
  }
}
```

**Pattern syntax:**
- `*` matches any characters except `/`
- `**` matches any characters including `/`
- Tool name required: `Bash(pattern)`, `Edit(pattern)`, etc.

**Common deny rules:**
| Pattern | Protects Against |
|---------|------------------|
| `Bash(rm -rf *)` | Accidental deletion |
| `Bash(git push --force*)` | Force push to remote |
| `Bash(git reset --hard*)` | Destructive resets |
| `Edit(**/.env)` | Secret file edits |
| `Edit(~/.ssh/**)` | SSH key modifications |

---

## Documentation Protocol

### changelog.md

**Format:**
```markdown
| # | Date | Modification |
|---|------|--------------|
| 1 | YYYY-MM-DD | **Title**: Description |
```

**Protocol:**
1. Add new entry at position 1
2. Shift existing entries down
3. Remove #10+ if > 10 entries
4. Update "Current State" section

### project-overview.md

**Focus:** WHAT and WHY, not HOW

**Required sections:**
- Overview (1-2 paragraphs)
- Pipelines (Goal, Main files, Input, Output)
- Supported Languages/Platforms
- Target Use Case
- Update History

### structure.md

**Required sections:**
- Project Tree (with descriptions)
- Remote Directories (absolute paths)
- Update History

### datasets.md (Optional)

**Required sections:**
- Normalized Column Names
- Per-dataset sections (path, structure, specificities)
- Encountered Specificities
- Update History

---

## Workflow Integration

### New Project Setup

```bash
# 1. Create structure
mkdir -p .claude/rules/project .claude/commands

# 2. Copy template files
cp -r template/.claude/* .claude/

# 3. Initialize
# - Fill project-overview.md
# - Update structure.md
# - Set initial changelog.md entry

# 4. Test hooks
# - Edit a Python file
# - Verify lint output appears
```

### Daily Workflow

1. **Start session** - Claude reads rules automatically
2. **Work on tasks** - Use `/commands` as needed
3. **After changes** - Run `/update-docs` or update manually
4. **Before commit** - Verify changelog updated

### Maintenance Workflow

| Frequency | Action |
|-----------|--------|
| After each change | Update changelog.md |
| Weekly | Review {TODO} markers |
| Monthly | Audit rules relevance |
| On major changes | Update project-overview.md |

---

## Maintenance Guidelines

### Keeping Rules Current

- **Remove obsolete rules** - Delete what's no longer relevant
- **Update examples** - Keep code examples working
- **Verify commands** - Test documented commands periodically
- **Sync with codebase** - Rules should reflect current practices

### Handling {TODO} Markers

```markdown
{TODO}                    # Needs to be filled
{TODO: specific question} # Needs specific information
{TODO: YYYY-MM-DD}       # Added on date, track progress
```

### Version Control

**Commit:**
- `settings.json`
- `rules/*.md`
- `commands/*.md`
- `agents/*.md`
- `skills/*/SKILL.md`

**Gitignore:**
- `settings.local.json`
- Personal experiments

### AGENTS.md Synchronization

If maintaining AGENTS.md for other tools:

1. Add redirect note at top:
   ```markdown
   > **NOTE**: Rules are in `.claude/rules/` and auto-loaded.
   > This file is kept for compatibility.
   ```

2. Mirror key rules from `.claude/rules/`

3. Update both when rules change

---

## Quick Reference

### File Purposes

| File | Purpose | Auto-loaded? |
|------|---------|--------------|
| `settings.json` | Hooks, permissions | Yes |
| `rules/*.md` | Guidelines, context | Yes |
| `commands/*.md` | Slash commands | On `/invoke` |
| `skills/*/SKILL.md` | Domain expertise | Auto-detected |
| `agents/*.md` | Sub-agents | Via Task tool |

### Key Commands

```bash
# Initialize new project
cp -r .claude-template /project/.claude

# Test hooks
# Edit any .py file and check output

# List available commands
# Type / in Claude Code
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Rules not loading | Check .md extension, path |
| Hooks not firing | Verify settings.json syntax |
| Commands missing | Check frontmatter name field |
| Agent not found | Verify file in agents/ dir |

---

## Sources

- [Claude Code Settings](https://code.claude.com/docs/en/settings)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Using CLAUDE.md Files](https://claude.com/blog/using-claude-md-files)
- [Claude Code Showcase](https://github.com/ChrisWiles/claude-code-showcase)
- [ClaudeLog Configuration](https://claudelog.com/configuration/)
