# Code Hash Module Guide

Files:

- `extract_structural_features/code_hash.py`
- `extract_structural_features/generate_code_hash.py`

## Purpose

Provide one shared, extractor-compatible way to generate code hashes.

This matches the behavior used in feature extraction:

- Prefer `code`, then fall back to `cleaned_code`
- Hash algorithm: SHA1
- Encoding: UTF-8
- Empty/missing/non-string code -> empty hash (`""`)

## Module API (`code_hash.py`)

### `compute_code_sha1(code_obj: object) -> str`

- Input: any object (usually code text)
- Output:
  - SHA1 hex digest if input is a non-empty string
  - `""` otherwise

### `choose_code_column(columns: Iterable[str], candidates: Sequence[str] = ["code", "cleaned_code"]) -> str`

- Selects the first available column from candidates.
- Default priority: `code` then `cleaned_code`
- Raises `ValueError` if neither exists.

### `compute_code_sha1_from_row(row: Mapping[str, object], candidates: Sequence[str] = ["code", "cleaned_code"]) -> tuple[str, str]`

- Picks column using `choose_code_column`.
- Returns:
  - `code_sha1`
  - `code_col_used`

## Standalone Script (`generate_code_hash.py`)

This is for teammates who want to generate hashes without importing code.

### Basic usage

```bash
python /Users/kimsia/Projects/City-of-Agents/extract_structural_features/generate_code_hash.py --cleaned-code 'print(1)'
```

Output:

```json
{"code_sha1":"f36c28972be9cd625bfda7a61a114cb2ed6a0436","code_col_used":"cleaned_code"}
```

### Print only hash value

```bash
python /Users/kimsia/Projects/City-of-Agents/extract_structural_features/generate_code_hash.py --code 'print(1)' --only-hash
```

### Read code from stdin

```bash
echo 'print(1)' | python /Users/kimsia/Projects/City-of-Agents/extract_structural_features/generate_code_hash.py --stdin-column code
```

### Inputs the script accepts

- `--cleaned-code <text>`
- `--code <text>`
- `--json-row '{"cleaned_code":"...","code":"..."}'`
- `--stdin-column cleaned_code|code`
- `--prefer-column cleaned_code|code`
- `--only-hash`

## Python import example

```python
from extract_structural_features.code_hash import compute_code_sha1_from_row

row = {"cleaned_code": "print(1)", "code": "print('fallback')"}
code_sha1, code_col_used = compute_code_sha1_from_row(row)
print(code_sha1, code_col_used)
```

## Notes for consistency

- Use this module whenever you need a stable join key (`code_sha1`).
- Avoid re-implementing hash logic in multiple scripts.
- Keeping one source of truth prevents mismatch across team outputs.
