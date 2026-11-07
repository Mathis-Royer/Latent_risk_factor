<!-- agent-kit: stacks/python/instructions.md v0.1.3 -->
# Python stack instructions (agent-kit)

## Code
- Type hints on every function signature; a docstring on every public function:
  ```python
  def function_name(param1: Type1, param2: Type2) -> ReturnType:
      """
      Short description.

      :param param1 (Type1): Description
      :param param2 (Type2): Description

      :return name (type): Description
      """
  ```
- PEP 8 naming: `snake_case` functions and variables, `UPPER_CASE` constants, `PascalCase` classes; 4 spaces.
- Pass values as parameters instead of hard-coding them or relying on globals.
- Before writing a new function, look for an existing one that does almost the same thing and extend it.
- When a function moves or its signature changes, update every import and call site (`grep -rn "name("`).
- Comments explain why, never the conversation that produced the code. No emojis in code or logs.

## Quality checks
- The kit hook (`scripts/agent_hooks.py`) reports pyright / ruff / flake8 findings on every edited `.py` file directly to
  the agent. Fix pyright errors first, then lint errors (E), then warnings; style only when trivial.
- Manual fallback on modified files: `pyright <file>`, `ruff check <file>` or `flake8 <file> --max-line-length=120`.
- A documented command that fails must be fixed in the document, with alternatives when the context varies.

## Notebooks
- To test a feature from a notebook, add a dedicated cell, iterate until it passes, then delete the cell.
- Notebook parameters override library defaults: say which source you rely on when they disagree.

## Dependencies and environment
See [environment.md](environment.md). Testing rules: [testing.md](testing.md).
