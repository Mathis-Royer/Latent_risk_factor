# Code Quality Verification

> **PORTABILITY NOTICE**: This file is designed to be **reusable across any Python project**.

## Command Maintenance Policy

**IMPORTANT**: Commands documented in these files must be functional. If a suggested command does not work:

1. **Identify the cause**: obsolete command, missing dependency, or incorrect syntax
2. **Fix the command** with the working version
3. **Add alternatives** if multiple commands can work depending on the context:
   ```bash
   # Option A: with uv (recommended)
   uv pip install -r requirements.txt

   # Option B: with standard pip
   pip install -r requirements.txt
   ```
4. **Document prerequisites** if the command requires specific configuration

This ensures that suggested commands always remain up-to-date and functional.

---

## Priority: Hooks > Manual Commands

**IMPORTANT**: If the project has Claude Code hooks configured (in `.claude/settings.json`) to automatically run linters after file modifications, check the hook output for immediate feedback. The manual commands below are provided as fallback.

## Tools Overview

| Tool | Purpose | Detects |
|------|---------|---------|
| **Pyright** | Type checker (CLI version of Pylance) | Type errors, missing imports, incorrect function signatures |
| **Pylint** | Linter | Code style issues, potential bugs, unused variables |
| **Flake8** | Linter | PEP8 violations, syntax errors, undefined names |

## Installation (if not already installed)

```bash
pip install pyright pylint flake8
```

## Verification Commands

Run these commands on modified files at the end of each task:

```bash
# Type checking (equivalent to Pylance/VSCode red underlines)
pyright src/your_modified_file.py

# Linting for code quality
pylint src/your_modified_file.py --disable=C0114,C0115,C0116  # Disable missing docstring warnings if needed

# Quick style check
flake8 src/your_modified_file.py --max-line-length=120
```

## Priority of Fixes

1. **Pyright errors** (highest priority): Type errors that will likely cause runtime failures
2. **Pylint errors (E)**: Probable bugs
3. **Pylint warnings (W)**: Potential issues
4. **Style issues**: Fix only if trivial, don't over-engineer

## When to Skip

- Minor documentation-only changes
- Configuration file edits
- When explicitly told by the user to skip verification

## Modification Verification Guidelines

**After each code modification**, verify changes according to the type of modification:

### Function Modifications

- **Find all callers**: `grep -rn "function_name(" src/`
- **Check signature compatibility**: changed parameters = all callers updated
- **Verify return type**: if changed, all usages must handle the new type

### Import/Module Modifications

- **Find all imports**:
  ```bash
  grep -rn "from old_module import" src/
  grep -rn "import old_module" src/
  ```
- **Test imports**: `python -c "from module import function"`

### Data Structure Modifications

- **Find all instantiations**: search for constructors
- **Check attribute access**: search for `.attribute_name`
- **Verify serialization**: if the class is saved/loaded

## Verification Checklist

Before considering a modification complete:
- [ ] Hook output shows no pyright errors
- [ ] Hook output shows no pylint E-level errors
- [ ] All callers of modified functions updated
- [ ] Imports correct and functional
- [ ] No hardcoded paths that break cross-platform compatibility
- [ ] `.claude/rules/project/changelog.md` updated if significant changes were made
