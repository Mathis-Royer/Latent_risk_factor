# Documentation Protocol

> **PORTABILITY NOTICE**: This file is designed to be **reusable across any Python project**. Replace `.claude/rules/project/` with your documentation directory path.

## Project Documentation Location

Project-specific documentation is stored in `.claude/rules/project/`:

```
.claude/rules/project/
├── project-overview.md   # Project goals by pipeline (WHAT, not HOW)
├── structure.md          # Project tree + remote directories
├── datasets.md           # Dataset documentation (if applicable)
├── changelog.md          # Recent changes and current state
└── architecture.md       # Application architecture (if single application)
```

## When to Read

**At the start of each conversation**, read the relevant files:
- Always read `project-overview.md` to understand project goals
- Read `changelog.md` to see recent changes and current state
- Read `structure.md` when navigating unfamiliar parts of the codebase
- Read `datasets.md` when working with data loading or processing
- Read `architecture.md` (if exists) when modifying application flow

---

## File Specifications

### project-overview.md

**Purpose:** Describe project goals and pipelines - the "WHAT" and "WHY", not the "HOW".

**Required sections:**
- **Overview**: 1-2 paragraphs describing project purpose and target users
- **Pipelines**: One subsection per pipeline with:
  - `**Goal:**` - What this pipeline achieves (business/research objective)
  - `**Main files:**` - Entry points and key files
  - `**Input:**` - What data/models it consumes
  - `**Output:**` - What it produces
- **Update History**: Table tracking changes
- **Supported Languages/Platforms**: If applicable
- **Target Use Case**: End-user scenarios

**What belongs:**
- High-level objectives and business goals
- Pipeline purposes and relationships
- User-facing features and capabilities

**What does NOT belong:**
- Implementation details (algorithms, parameters)
- Code snippets or function signatures
- Technical debt or TODOs (use code comments)

---

### structure.md

**Purpose:** Document project tree and external directories.

**Required sections:**
- **Project Tree**: Directory structure with inline descriptions
- **Remote Directories**: External data/resources not in repo
  - Absolute path
  - Description
  - Internal structure
  - Which code uses it

**Format:**
```
project_root/
├── folder/                    # Brief description
│   ├── file.py                # What this file does
│   └── {TODO}                 # Incomplete sections
```

**When to update:**
- New folder or significant file added
- New remote directory discovered
- File responsibilities change significantly

---

### datasets.md

**Purpose:** Document datasets in detail for data-intensive projects.

**Required sections:**
- **Normalized Column Names**: Standard schema across all loaders
- **Per-dataset sections** with:
  - Loader file path
  - Absolute and relative paths
  - Description and data types
  - Folder structure
  - Original column/field names
  - **Specificities Encountered**: Gotchas, edge cases, preprocessing quirks
- **Encountered Specificities**: Cross-dataset issues discovered during work
- **Update History**

**What belongs in "Specificities Encountered":**
- Data quality issues (missing values, encoding problems)
- Preprocessing requirements (normalization, filtering)
- Edge cases that caused bugs
- Performance considerations (large files, slow loading)
- Schema changes between versions

**When to update:**
- New dataset added
- New specificity discovered (bug caused by data quirk)
- Column names or paths change

---

### changelog.md

**Purpose:** Track recent significant changes and current project state.

**Required sections:**
- **Recent Changes**: Table with ~10 entries (date, description)
- **Current State**: Brief summary of project status

**Format:**
```markdown
| # | Date | Modification |
|---|------|--------------|
| 1 | YYYY-MM-DD | **Title**: Brief description |
```

**Update protocol:**
- After each significant modification, add entry at position 1
- If similar entry exists, update it instead of adding
- If > 10 entries, remove the oldest
- Update "Current State" when project status changes

---

### architecture.md (Optional)

**Include only if** the project is a single application (web app, CLI tool, service).

**Do NOT include if** the project is:
- A collection of scripts/pipelines
- A library or package
- Research code with multiple independent experiments

**Required sections (if included):**
- Component diagram or description
- Data flow between components
- Entry points and main loops
- External service integrations

---

## Incremental Update Protocol

These files are built **incrementally** as you understand the project.

### When to update

| Trigger | Action |
|---------|--------|
| Significant code modification | Add to `changelog.md` |
| Discovered new pipeline | Add to `project-overview.md` |
| Found new remote directory | Add to `structure.md` |
| Encountered data quirk/bug | Add to `datasets.md` Specificities |
| Completed {TODO} section | Fill in the details |
| Architecture understanding improved | Update relevant file |

### How to update

1. **Check existing content first** - avoid duplicates
2. **Add to existing sections** - don't create new top-level sections
3. **Use {TODO} markers** for incomplete information:
   - `{TODO}` - Needs to be filled
   - `{TODO: specific question}` - Needs specific information
4. **Update the Update History table** with date and change summary
5. **Keep format consistent** with existing entries

### Size Management

- No strict line limits for stable documentation files
- `changelog.md`: Keep ~10 recent entries (remove oldest when exceeding)
- Prioritize completeness over brevity
- If a section becomes too long, consider:
  - Creating sub-files (e.g., `datasets/commonvoice.md`)
  - Moving implementation details to code comments

---

## Verification Checklist

Before considering documentation complete:
- [ ] All {TODO} markers have been reviewed
- [ ] Paths are absolute and correct
- [ ] Update History reflects recent changes
- [ ] `changelog.md` updated if significant changes were made
