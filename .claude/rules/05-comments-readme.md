# Comments and README Guidelines

> **PORTABILITY NOTICE**: This file is designed to be **reusable across any Python project**.

## Comments

- Write comments in English only.
- Use comments to explain **why**, not what (the code should be self-explanatory).
- **NEVER** add comments indicating a fix or correction from a prompt (e.g., `# Fixed issue from prompt`, `# Correction: ...`). Comments must serve the code's long-term readability, not document the conversation history.

### Valid Uses for Comments

- Structure and explain complex logic
- Document edge cases or non-obvious behavior
- Mark critical sections with warnings (e.g., `# CRITICAL: Do not modify without updating X`)
- Highlight potential errors or limitations

### Emoji Policy

**NEVER** use emojis in code, including in `print()` statements, unless:
- Explicitly requested by the user
- Used as legend markers in visualizations or reports

---

## README.md

Write all README content in English only.

### Structure (in order)

1. **Overview**: Brief summary of project subject and goals (1-2 paragraphs)
2. **Model Description** (if relevant): Name, architecture, capabilities, languages supported, external links
3. **Datasets and Evaluation** (if relevant): For each dataset include:
   - Purpose and data sources
   - Preprocessing steps and filtering rules
   - Evaluation metrics (define acronyms on first use)
   - Results with ranges and constraints
4. **Project Structure**: Directory tree with inline comments describing file roles
5. **Performance Summary** (if relevant): Table with key metrics per dataset/strategy
6. **Authors**: Roles and contributions with @ tags

### Content Guidelines

- Focus on **why** decisions were made, not implementation details
- Use present tense for current capabilities, past tense for completed work
- Keep sections concise with bullet lists for procedures
- Use backticks for filenames and code identifiers
- If missing information, add section title with `{TODO}` tag and request details
