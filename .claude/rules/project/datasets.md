# Datasets Documentation

> **Note:** This file documents datasets in detail. Built incrementally as agents work with data. Sections marked with {TODO} need to be completed.
> **If your project doesn't use datasets, you can delete this file.**

---

## Normalized Column Names

All dataset loaders in this project normalize data to these standard columns:

{TODO: Define the standard schema for your project}

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `{TODO}` | str | {TODO: Description} | Yes |
| `{TODO}` | str | {TODO: Description} | Yes |
| `{TODO}` | float | {TODO: Description} | Optional |

---

## Dataset: {TODO: Dataset Name}

**Loader:** `{TODO: src/path/to/loader.py}`
**Absolute Path:** `{TODO: /path/to/dataset}`
**Relative Path:** `{TODO: dataset/subfolder}` (fallback)

### Description

{TODO: Brief description of the dataset - what it contains, source, purpose}

### Data Types

- **{TODO: Type 1}:** {TODO: Description}
- **{TODO: Type 2}:** {TODO: Description}

### Folder Structure

```
{TODO: dataset_name}/
├── {TODO}/                   # {TODO: Description}
│   └── {TODO}
└── {TODO}
```

### File/Column Format (Original)

{TODO: Document the original format before normalization}

| Column/Field | Description |
|--------------|-------------|
| `{TODO}` | {TODO: Description} |
| `{TODO}` | {TODO: Description} |

### Specificities Encountered

{TODO: Add specificities as they are discovered during development}

- {TODO: Data quality issues, edge cases, preprocessing quirks}

---

## Dataset: {TODO: Second Dataset Name}

**Loader:** `{TODO}`
**Absolute Path:** `{TODO}`

### Description

{TODO}

### Specificities Encountered

{TODO}

---

## Encountered Specificities

> Add discoveries here as you work with the data.

### Text/Data Preprocessing

{TODO: Document preprocessing steps}

- {TODO: Normalization rules}
- {TODO: Encoding issues}

### File Properties

{TODO: Document file formats and properties}

- Format: {TODO}
- Encoding: {TODO}

---

## Update History

| Date | Section | Change |
|------|---------|--------|
| {TODO: YYYY-MM-DD} | Initial | Created minimal structure |
