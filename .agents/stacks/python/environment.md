<!-- agent-kit: stacks/python/environment.md v0.1.3 -->
# Python environment (agent-kit)

The standard is `uv` with a `pyproject.toml` and a committed `uv.lock`.

```bash
# Create or refresh the environment exactly from the lock file (all optional groups)
uv sync --all-extras

# Run a tool inside the environment without activating it
uv run pytest tests/unit/test_x.py -q

# Add or upgrade a dependency (updates pyproject.toml and uv.lock together)
uv add <package>
uv lock --upgrade-package <package>
```

Fallback without uv (only when uv is unavailable):
```bash
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

Rules:
- Every imported package is declared in `pyproject.toml`; `uv lock` must succeed (no unsatisfiable extras).
- Never install packages globally or edit the lock file by hand.
- Tool hooks look for binaries in `.venv/bin/` first, so keep linters (pyright, ruff or flake8, pytest) in the dev
  dependencies.
- Parallel worktrees each get their own `.venv` (`uv sync` reuses the shared uv cache, so it is fast).
