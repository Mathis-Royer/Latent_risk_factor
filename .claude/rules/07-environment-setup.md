# Environment Setup with uv

> **PORTABILITY NOTICE**: This file is designed to be **reusable across any Python project**.

Use `uv` to create and manage a fast, reproducible Python environment.

## 1. Install uv

You can install uv globally via pipx (recommended) or pip:

```bash
# Option A: via pipx (recommended)
pipx install uv

# Option B: via pip
python -m pip install --upgrade uv
```

## 2. Create a Virtual Environment

Specify the Python version required by the project (check README or pyproject.toml).

```bash
# From the repo root
uv venv --python 3.11  # Adjust version as needed

# Activate the environment
# Linux/macOS:
source .venv/bin/activate

# Windows:
.\.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

## 4. Updating and Maintenance

```bash
# Update packages quickly
uv pip install -U -r requirements.txt

# Freeze the current environment (optional snapshot)
uv pip freeze > requirements.lock.txt
```

## Alternative: Standard pip

If uv is not available:

```bash
# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```
