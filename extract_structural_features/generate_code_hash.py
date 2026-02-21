#!/usr/bin/env python3
"""Generate code SHA1 hashes with the same logic as feature extraction.

Behavior matches extraction:
- Prefer `code`, then fall back to `cleaned_code`.
- Compute SHA1 over UTF-8 text.
- Return an empty hash for missing/non-string/empty code.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Sequence

try:
    from .code_hash import CODE_COLUMN_CANDIDATES, compute_code_sha1_from_row
except ImportError:
    from code_hash import CODE_COLUMN_CANDIDATES, compute_code_sha1_from_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate code hash using extractor-compatible logic"
    )
    parser.add_argument(
        "--cleaned-code", default=None, help="Value for cleaned_code column"
    )
    parser.add_argument("--code", default=None, help="Value for code column")
    parser.add_argument(
        "--json-row",
        default=None,
        help='JSON object containing cleaned_code/code (e.g. {"cleaned_code":"print(1)","code":"..."}).',
    )
    parser.add_argument(
        "--stdin-column",
        choices=CODE_COLUMN_CANDIDATES,
        default=None,
        help="Read code text from stdin into this column name.",
    )
    parser.add_argument(
        "--prefer-column",
        choices=CODE_COLUMN_CANDIDATES,
        default="cleaned_code",
        help="Column priority when both are present (default: cleaned_code).",
    )
    parser.add_argument(
        "--only-hash",
        action="store_true",
        help="Print only the hash string instead of JSON output.",
    )
    return parser.parse_args()


def build_candidates(prefer_column: str) -> Sequence[str]:
    return [prefer_column] + [c for c in CODE_COLUMN_CANDIDATES if c != prefer_column]


def build_row(args: argparse.Namespace) -> Dict[str, object]:
    row: Dict[str, object] = {}

    if args.json_row:
        parsed = json.loads(args.json_row)
        if not isinstance(parsed, dict):
            raise ValueError("--json-row must be a JSON object")
        row.update(parsed)

    if args.cleaned_code is not None:
        row["cleaned_code"] = args.cleaned_code
    if args.code is not None:
        row["code"] = args.code

    if args.stdin_column is not None:
        row[args.stdin_column] = sys.stdin.read()

    return row


def main() -> None:
    args = parse_args()
    row = build_row(args)

    candidates = build_candidates(args.prefer_column)
    code_sha1, code_col_used = compute_code_sha1_from_row(row, candidates=candidates)

    if args.only_hash:
        print(code_sha1)
        return

    print(
        json.dumps(
            {"code_sha1": code_sha1, "code_col_used": code_col_used}, ensure_ascii=True
        )
    )


if __name__ == "__main__":
    main()
