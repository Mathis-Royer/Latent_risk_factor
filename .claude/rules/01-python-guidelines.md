# Python Guidelines

> **PORTABILITY NOTICE**: This file is designed to be **reusable across any Python project**.

> **CRITICAL RULE - ENGLISH ONLY**: ALL files MUST be written in English. This includes code, comments, docstrings, documentation, configuration files, commit messages, and any text content. No exceptions.

## General Principles

- Write all code and comments in English.
- Use clear, descriptive names for variables, functions, and classes.
- Keep functions short and focused on a single task.

## Type Hints and Docstrings

- Each function must declare input and output types using type hints.
- Each function should have a docstring in English, following this format:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Short function description.

    :param param1 (Type1): Description of param1
    :param param2 (Type2): Description of param2

    :return return1 (type): Description of return1
    :return ... (type): Description of other returns
    """
```

## Code Style

- Use 4 spaces for indentation.
- Use snake_case for variable and function names (e.g., `my_variable`, `process_data`).
- Use UPPER_CASE for constants (e.g., `MAX_SIZE`, `DEFAULT_PATH`).
- Use PascalCase for class names (e.g., `DataProcessor`, `AudioModel`).

## Best Practices

- Prefer passing values as function parameters rather than hardcoding them or using local/global variables.
- Avoid passing functions as parameters to other functions.
- Avoid global variables.
- Use list comprehensions and built-in functions when appropriate.

## When Modifying Code

- When modifying or moving a function, update all related imports and every call to that function across the entire codebase.
- Whenever possible, prefer updating an existing function that already performs something close to the desired behavior rather than creating a new one with almost identical logic.
- When updating a function, do not break existing usages: either preserve the necessary parameters/behavior, or adapt both the function and all its call sites.
