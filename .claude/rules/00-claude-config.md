# Claude Code Configuration (.claude/)

> **PORTABILITY NOTICE**: This file is designed to be **reusable across any Python project**.

> **CRITICAL RULE - ENGLISH ONLY**: ALL files in ALL projects MUST be written in English. This includes code, comments, docstrings, documentation, configuration files, commit messages, and any text content. No exceptions.

The `.claude/` directory contains configuration and extensions for Claude Code.

## Directory Structure

```
.claude/
├── settings.json          # Project settings (hooks, permissions) - committed
├── settings.local.json    # Local overrides - gitignored
├── rules/                 # Project rules (auto-loaded)
│   ├── 00-role.md        # Claude's expert persona (loaded first)
│   ├── 00-claude-config.md  # This file - configuration reference
│   ├── 01-*.md           # Coding guidelines
│   ├── project/          # Project-specific documentation
│   │   └── project-overview.md  # Source for 00-role.md
│   └── *.md              # Other rules
├── commands/              # User-triggered slash commands
│   └── command-name.md   # /command-name
├── skills/                # Custom skills (domain knowledge)
│   └── skill-name/
│       └── SKILL.md      # Skill definition
└── agents/                # Custom sub-agents
    └── agent-name.md     # Agent definition
```

## Rules Directory (.claude/rules/)

Markdown files automatically loaded as project memory. All `.md` files in this directory (and subdirectories) are included.

### Basic Example

`.claude/rules/project-rules.md`:
```markdown
# Project Rules

- Use type hints for all functions
- Always use fixed seeds for reproducibility
- Follow the existing code style
```

### Conditional Rule with Frontmatter

Rules can be scoped to specific file paths using YAML frontmatter:

```markdown
---
paths:
  - "src/training/**"
---

# Training Rules

For training scripts:
- Log hyperparameters
- Save checkpoints regularly
```

**Path patterns**:
- `src/**/*.py` - All Python files in src/
- `tests/**` - All files in tests/
- `*.md` - All markdown files in root

## Skills vs Commands vs Agents

| Type | Trigger | Purpose | Location |
|------|---------|---------|----------|
| **Rules** | Auto-loaded | Provide context and guidelines | `.claude/rules/` |
| **Commands** | User types `/name` | Execute workflow | `.claude/commands/` |
| **Skills** | Auto-detected | Provide domain expertise | `.claude/skills/` |
| **Agents** | Task tool | Delegate specialized work | `.claude/agents/` |

## Role Prompting (.claude/rules/00-role.md)

The `00-role.md` file defines Claude's expert persona for the project. This file is loaded first (due to `00-` prefix) and establishes context before other rules.

### Why Role Prompting?

- **Enhanced accuracy**: Domain-specific expertise improves responses on complex tasks
- **Tailored tone**: Adjusts communication style to project needs
- **Improved focus**: Claude stays within the bounds of project requirements
- **Dynamic adaptation**: For each prompt, adopt the perspective of the best expert to answer

### Protocol: Creating 00-role.md

**Source:** `.claude/rules/project/project-overview.md`

**Steps:**

1. **Read project-overview.md** to extract:
   - Project domain (e.g., ASR, web development, data science)
   - Primary goal and target users
   - Key technologies and models used
   - Supported languages/platforms

2. **Write 00-role.md** with this structure:

```markdown
# Role Definition

You are a {seniority} {role} specializing in {domain} for {target application}.

## Dynamic Expertise

For each user request, identify and adopt the perspective of the best expert to answer:
- Ask yourself: "Who would be the ideal expert to handle this specific request?"
- Adapt your expertise while staying within the project's domain context
- Examples: debugging → senior debugger, architecture → system architect, optimization → performance engineer

## Expertise

- {Technology/framework 1}
- {Technology/framework 2}
- {Domain-specific skill}

## Context

{Brief description of project goal and target users}

## Priorities

- {Key priority derived from project goals}
- {Quality standard important for the domain}
- {Technical constraint or best practice}
```

3. **Extraction rules:**
   - `{role}`: Infer from project type (ML project → ML engineer, web app → software engineer)
   - `{domain}`: From "Overview" section
   - `{target application}`: From "Target Users" or "Target Use Case"
   - `{Expertise}`: From technologies mentioned (Model, pipelines, languages)
   - `{Priorities}`: From project goals and constraints

### Example Transformation

**From project-overview.md:**
```
Project domain: {TODO}
Model/technologies: {TODO}
Target Users: {TODO}
```

**To 00-role.md:**
```markdown
# Role Definition

You are a {TODO: seniority} {TODO: role} specializing in {TODO: domain} for {TODO: target application}.

## Dynamic Expertise

For each user request, identify and adopt the perspective of the best expert to answer:
- Ask yourself: "Who would be the ideal expert to handle this specific request?"
- Adapt your expertise while staying within the project's domain context
- Examples: {TODO: domain-specific examples}

## Expertise

- {TODO: Technology 1}
- {TODO: Technology 2}
- {TODO: Domain skill}

## Context

{TODO: Brief description of project goal and target users}

## Priorities

- {TODO: Key priority 1}
- {TODO: Key priority 2}
- {TODO: Key priority 3}
```

---

## Priority: Hooks > Manual Commands

**IMPORTANT**: If the project has hooks configured in `settings.json` to automatically run linters after file modifications, check the hook output for immediate feedback. Only use manual commands (pyright, pylint, flake8) as fallback or for targeted verifications.

## Template Synchronization

If this project has a `.claude-template/` directory (reusable template for other projects):

**When updating `.claude/` with generalizable changes:**

1. **Identify if change is project-specific or generic**
   - Project-specific: paths, datasets, pipelines → Only update `.claude/`
   - Generic: coding standards, protocols, hooks → Update BOTH `.claude/` AND `.claude-template/`

2. **Update both directories** for generic changes:
   ```bash
   # After modifying a generic rule in .claude/
   cp .claude/rules/01-python-guidelines.md .claude-template/rules/
   ```

3. **Keep template {TODO} markers** - When copying to template, replace project-specific content with `{TODO}` markers

4. **Update template documentation** - If adding new patterns, update `.claude-template/README.md` and/or `.claude-template/PROTOCOL.md`

**Decision Matrix:**

| Change Type | Update .claude/ | Update .claude-template/ |
|-------------|-----------------|--------------------------|
| New Python guideline | Yes | Yes |
| New dataset documentation | Yes | No (project-specific) |
| New hook pattern | Yes | Yes |
| Project pipeline docs | Yes | No |
| New command pattern | Yes | Yes (with {TODO}) |
| New agent for specific domain | Yes | No (or add as TEMPLATE) |
| **00-role.md content** | Yes | No (generated from project-overview.md) |
| **00-role.md protocol** | Yes | Yes (in 00-claude-config.md) |
