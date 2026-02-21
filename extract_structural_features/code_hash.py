"""Shared code-hash helpers for source columns.

The hash behavior intentionally matches the existing extraction logic:
- Use SHA1 on UTF-8 encoded text.
- Return an empty string for missing/non-string/empty values.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Sequence, Tuple

# After discussing with Shenghua 2026-02-21 Saturday ard 3pm, we prioritize code over cleaned_code
CODE_COLUMN_CANDIDATES = ["code", "cleaned_code"]


def choose_code_column(
    columns: Iterable[str],
    candidates: Sequence[str] = CODE_COLUMN_CANDIDATES,
) -> str:
    col_set = set(columns)
    for candidate in candidates:
        if candidate in col_set:
            return candidate
    raise ValueError(
        f"None of {list(candidates)} found in dataset columns: {sorted(col_set)}"
    )


def compute_code_sha1(code_obj: object) -> str:
    """
    Compute SHA1 hash of code object.

    If code_obj is "", None, or a non-string, it returns "".
    """
    code = code_obj if isinstance(code_obj, str) else ""
    return hashlib.sha1(code.encode("utf-8")).hexdigest() if code else ""


def compute_code_sha1_from_row(
    row: Mapping[str, object],
    candidates: Sequence[str] = CODE_COLUMN_CANDIDATES,
) -> Tuple[str, str]:
    code_col = choose_code_column(row.keys(), candidates)
    return compute_code_sha1(row.get(code_col)), code_col
