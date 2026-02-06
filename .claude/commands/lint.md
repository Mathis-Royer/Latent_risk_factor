---
name: lint
description: Run linters on modified files
allowed-tools: [Bash, Read, Glob]
---

# Lint Command

Runs code quality tools on modified Python files.

## When Invoked with `/lint`

### 1. Identify Modified Files

```bash
# Get list of modified Python files
git diff --name-only | grep '\.py$'
```

### 2. Run Linters

For each modified Python file:

```bash
# Type checking
echo "=== Pyright ===" && pyright $file 2>&1 | head -30

# Code quality
echo "=== Pylint ===" && pylint $file --disable=C0114,C0115,C0116 --max-line-length=120 2>&1 | head -20

# Style check
echo "=== Flake8 ===" && flake8 $file --max-line-length=120 2>&1 | head -15
```

### 3. Report Results

Prioritize fixes:
1. **Pyright errors** (highest priority): Type errors
2. **Pylint errors (E)**: Probable bugs
3. **Pylint warnings (W)**: Potential issues
4. **Style issues**: Fix only if trivial

## Quick Commands

```bash
# Lint all Python files in src/
pyright src/
pylint src/ --disable=C0114,C0115,C0116

# Lint specific file
pyright src/module.py && pylint src/module.py
```
