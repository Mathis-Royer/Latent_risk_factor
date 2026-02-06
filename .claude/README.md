# Claude Code Configuration Template

A reusable template for configuring Claude Code (`.claude/` directory) in any Python project.

## Quick Start

```bash
# 1. Copy template to your project
cp -r .claude-template /path/to/your/project/.claude

# 2. Remove template files
rm /path/to/your/project/.claude/README.md
rm /path/to/your/project/.claude/agents/TEMPLATE-agent.md
rm -rf /path/to/your/project/.claude/skills/TEMPLATE-skill

# 3. Add to .gitignore
echo "settings.local.json" >> /path/to/your/project/.claude/.gitignore

# 4. Replace {TODO} markers with project-specific information
```

---

## Directory Structure

```
.claude/
├── settings.json              # Hooks, permissions (committed to git)
├── settings.local.json        # Personal overrides (gitignored)
│
├── rules/                     # Auto-loaded knowledge (always active)
│   ├── 00-claude-config.md    # Claude Code structure explanation
│   ├── 01-python-guidelines.md # Python coding standards
│   ├── 02-git-commit.md       # Git commit conventions
│   ├── 03-code-quality.md     # Linting and verification
│   ├── 04-task-completion.md  # Task completion checklist
│   ├── 05-comments-readme.md  # Comment and README guidelines
│   ├── 06-documentation-protocol.md # Documentation protocol
│   ├── 07-environment-setup.md # uv/pip setup
│   └── project/               # Project-specific documentation
│       ├── project-overview.md # Goals and pipelines
│       ├── structure.md       # Directory tree
│       ├── changelog.md       # Recent changes
│       └── datasets.md        # Dataset documentation (optional)
│
├── commands/                  # User-triggered slash commands
│   ├── update-docs.md         # /update-docs
│   ├── test.md                # /test
│   └── lint.md                # /lint
│
├── skills/                    # Domain expertise (auto-detected)
│   └── TEMPLATE-skill/
│       └── SKILL.md
│
└── agents/                    # Specialized sub-agents (Task tool)
    └── TEMPLATE-agent.md
```

---

## Configuration Components

### 1. Rules (Auto-Loaded)

Rules in `.claude/rules/` are **automatically loaded** as project memory. Claude reads them at the start of every conversation.

**Numbering Convention:**
| Number | Purpose | Content |
|--------|---------|---------|
| 00 | Config | Claude Code structure explanation |
| 01 | Language | Python coding standards |
| 02 | Workflow | Git commit conventions |
| 03 | Quality | Linting and verification |
| 04 | Completion | Task verification checklist |
| 05 | Style | Comments and README guidelines |
| 06 | Docs | Documentation protocol |
| 07 | Setup | Environment setup |
| project/ | Specific | Project-specific documentation |

**Conditional Loading (Path Filtering):**

You can scope rules to specific paths using YAML frontmatter:

```markdown
---
paths:
  - "src/training/**"
---

# Training Rules
Only loaded when working in src/training/
```

### 2. Commands (User-Triggered)

Commands in `.claude/commands/` are available via the `/` menu.

| Command | Purpose |
|---------|---------|
| `/update-docs` | Update project documentation after changes |
| `/test` | Run tests on modified files |
| `/lint` | Run linters on modified files |

**Creating Custom Commands:**

```markdown
---
name: my-command
description: Description shown in /help
allowed-tools: [Read, Write, Edit, Bash]
---

# My Command

Instructions for what Claude should do when `/my-command` is invoked.
```

### 3. Skills (Auto-Detected)

Skills in `.claude/skills/` provide domain expertise. Claude automatically uses them based on the task context.

**When to create a skill:**
- Complex domain patterns (testing frameworks, API conventions)
- Project-specific knowledge that applies across multiple files
- Reusable patterns that need consistent application

### 4. Agents (Task Tool)

Agents in `.claude/agents/` are specialized sub-agents invoked via the Task tool for complex, focused tasks.

**When to create an agent:**
- Tasks requiring deep domain expertise
- Repetitive tasks benefiting from consistent instructions
- Tasks that need focused context (data loading, testing)

### 5. Hooks (Automation)

Hooks in `settings.json` automate actions on file changes.

**Current hooks:**
| Hook | Trigger | Action |
|------|---------|--------|
| PostToolUse | Write\|Edit on .py | Run pyright, pylint, flake8 |

**Hook Events:**
- `PreToolUse` - Before tool execution (can block)
- `PostToolUse` - After tool execution
- `UserPromptSubmit` - When user sends message
- `Stop` - When Claude stops

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | Block action (PreToolUse only) |
| Other | Non-blocking feedback |

---

## Customization Guide

### Step 1: Project Overview

Edit `rules/project/project-overview.md`:
1. Replace `{TODO}` markers with actual project information
2. Document each pipeline with Goal, Main files, Input, Output
3. Add supported languages/platforms
4. Describe target use cases

### Step 2: Project Structure

Edit `rules/project/structure.md`:
1. Update the directory tree to match your project
2. Document remote/external directories if any
3. Add file descriptions

### Step 3: Initialize Changelog

Edit `rules/project/changelog.md`:
1. Add initial entry with date
2. Describe current state
3. Keep updated after each significant change

### Step 4: Dataset Documentation (Optional)

If your project uses datasets, edit `rules/project/datasets.md`:
1. Define normalized column schema
2. Document each dataset with paths, structure, specificities
3. Remove this file if not applicable

### Step 5: Customize Hooks

Edit `settings.json` to:
1. Add/remove linters
2. Adjust line limits
3. Add pre-commit hooks

### Step 6: Create Project-Specific Commands

Copy `commands/test.md` pattern to create:
- `/deploy` - Deployment workflow
- `/build` - Build process
- `/release` - Release workflow

### Step 7: Create Domain Agents (Optional)

Use `agents/TEMPLATE-agent.md` to create:
- Data processing agents
- Testing agents
- Domain-specific experts

---

## Best Practices

### Rules

- **Keep files under 300 lines** - Split large files
- **Use {TODO} markers** for incomplete sections
- **Path filters** reduce context for irrelevant rules
- **Numbered prefixes** ensure consistent loading order

### Commands

- **Clear trigger keywords** in descriptions
- **Step-by-step instructions** for complex workflows
- **Explicit tool permissions** in frontmatter

### Documentation

- **WHAT and WHY, not HOW** - Focus on goals
- **Incremental updates** - Build as you learn
- **Changelog discipline** - Update after each change

### Hooks

- **Fail gracefully** - Use `|| true` for non-critical checks
- **Limit output** - Use `| head -N` to prevent spam
- **Test hooks** - Verify they work before committing

---

## Synchronization with AGENTS.md

If you maintain an `AGENTS.md` file for other tools:

1. **Keep AGENTS.md as a mirror** of `.claude/rules/` content
2. **Add redirect note** at top of AGENTS.md:
   ```markdown
   > **NOTE**: Rules are now stored in `.claude/rules/` and auto-loaded.
   > This file is kept for compatibility with other tools/agents.
   ```
3. **Update both** when rules change

---

## Troubleshooting

### Rules not loading

- Check file extension is `.md`
- Verify file is in `.claude/rules/` or subdirectory
- Check YAML frontmatter syntax if using path filters

### Hooks not triggering

- Verify `settings.json` syntax (valid JSON)
- Check matcher regex matches tool name
- Test command manually in terminal

### Commands not appearing

- Verify file is in `.claude/commands/`
- Check YAML frontmatter has `name` field
- Restart Claude Code session

---

## Sources

- [Claude Code Settings](https://code.claude.com/docs/en/settings)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Using CLAUDE.md Files](https://claude.com/blog/using-claude-md-files)
- [Claude Code Showcase](https://github.com/ChrisWiles/claude-code-showcase)
