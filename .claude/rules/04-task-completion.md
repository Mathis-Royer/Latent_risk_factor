# Task Completion Verification

> **PORTABILITY NOTICE**: This file is designed to be **reusable across any Python project**.

## At the End of Each Task

**Always re-read the original user prompt** to verify all requests have been addressed:

1. **Re-read the prompt**: Go back to the user's original request and check each requirement.

2. **Verify completion**: For each point in the prompt, confirm it was implemented correctly.

3. **If incomplete**: Create a new todo list with the remaining tasks and continue working until all requirements are met.

4. **If complete**: Briefly summarize what was done to confirm alignment with the request.

This ensures no requirement is forgotten or partially implemented.

## Testing in Notebooks

When the user asks to test something and the main test file is a `.ipynb` notebook:

1. Create a new cell dedicated to testing the requested feature
2. Execute the test cell, iterating and fixing issues until it works as expected
3. Once the test passes, delete the test cell to keep the notebook clean
