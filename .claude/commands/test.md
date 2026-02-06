---
name: test
description: Run tests on modified files
allowed-tools: [Bash, Read, Glob, Grep]
---

# Test Command

Runs tests on modified Python files to verify changes don't break existing functionality.

## When Invoked with `/test`

### 1. Identify Modified Files

```bash
# Get list of modified Python files
git diff --name-only | grep '\.py$'

# Or for unstaged changes
git diff --name-only HEAD | grep '\.py$'
```

### 2. Run Tests

**Option A: pytest (recommended)**
```bash
# Run tests related to modified files
pytest tests/ -v --tb=short

# Run specific test file
pytest tests/test_specific.py -v
```

**Option B: unittest**
```bash
python -m unittest discover tests/
```

### 3. Report Results

- List passed/failed tests
- Show error messages for failures
- Suggest fixes if obvious

## Configuration

{TODO: Customize test commands for your project}

```bash
# Project-specific test command
{TODO: pytest tests/ -v}
```

## Rules

- Always run tests before committing
- Fix failing tests before proceeding
- Add new tests for new functionality
