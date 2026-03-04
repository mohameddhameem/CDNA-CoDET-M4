#!/usr/bin/env python3
"""Extract aggregated Python code features from CoDET-M4 for baseline modeling.

This script:
1. Loads the dataset from Hugging Face (`datasets` library)
2. Filters rows to `language == python` (and optional source filter)
3. Extracts aggregated lexical + AST features per sample
4. Writes split-level Parquet files for downstream XGBoost / RF training

Example:
    python analysis/extract_python_aggregated_features.py \
      --dataset DaniilOr/CoDET-M4 \
      --output-dir data/processed_homo/python_agg \
      --num-proc 8

Optional source filter (e.g., CodeForces only):
    python analysis/extract_python_aggregated_features.py \
      --source codeforces
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import tokenize
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

try:
    from .code_hash import choose_code_column, compute_code_sha1
except ImportError:
    from code_hash import choose_code_column, compute_code_sha1


METADATA_COLUMNS = [
    "model",
    "language",
    "source",
    "target",
    "split",
    "source_type",
    "model_display",
]

AST_NODE_TYPES = {
    "FunctionDef": ast.FunctionDef,
    "AsyncFunctionDef": ast.AsyncFunctionDef,
    "ClassDef": ast.ClassDef,
    "Return": ast.Return,
    "Assign": ast.Assign,
    "AnnAssign": ast.AnnAssign,
    "AugAssign": ast.AugAssign,
    "For": ast.For,
    "AsyncFor": ast.AsyncFor,
    "While": ast.While,
    "If": ast.If,
    "Try": ast.Try,
    "With": ast.With,
    "AsyncWith": ast.AsyncWith,
    "Call": ast.Call,
    "Import": ast.Import,
    "ImportFrom": ast.ImportFrom,
    "ListComp": ast.ListComp,
    "DictComp": ast.DictComp,
    "SetComp": ast.SetComp,
    "GeneratorExp": ast.GeneratorExp,
    "Lambda": ast.Lambda,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract aggregated Python features from CoDET-M4")
    parser.add_argument("--dataset", default="DaniilOr/CoDET-M4", help="Hugging Face dataset id")
    parser.add_argument("--output-dir", default="data/processed_homo/python_agg", help="Output directory")
    parser.add_argument("--language", default="python", help="Language filter (default: python)")
    parser.add_argument("--source", default=None, help="Optional source filter (e.g. codeforces)")
    parser.add_argument("--num-proc", type=int, default=1, help="Parallel workers for datasets.map/filter")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap per split after filtering (for quick debug)",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated split names to process (default: all available)",
    )
    parser.add_argument(
        "--exclude-hf-features",
        action="store_true",
        help="Do not include existing nested CoDET-M4 'features' values",
    )
    return parser.parse_args()


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    try:
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def infer_hf_feature_keys(dataset: Dataset, probe_rows: int = 1024) -> List[str]:
    if "features" not in dataset.column_names:
        return []

    limit = min(len(dataset), probe_rows)
    keys = set()
    for i in range(limit):
        row = dataset[i]
        nested = row.get("features")
        if isinstance(nested, dict):
            keys.update(str(k) for k in nested.keys())

    return sorted(keys)


def compute_lexical_features(code: str) -> Dict[str, float]:
    lines = code.splitlines()
    loc = len(lines)
    non_empty_loc = sum(1 for line in lines if line.strip())

    char_count = len(code)
    whitespace_chars = sum(1 for ch in code if ch.isspace())
    whitespace_ratio = whitespace_chars / char_count if char_count else 0.0

    if loc:
        avg_line_length = sum(len(line) for line in lines) / loc
        max_line_length = max(len(line) for line in lines)
    else:
        avg_line_length = 0.0
        max_line_length = 0.0

    comment_line_numbers = set()
    token_count = 0
    significant_token_count = 0

    try:
        token_stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in token_stream:
            token_count += 1
            tok_type = tok.type
            if tok_type == tokenize.COMMENT:
                comment_line_numbers.add(tok.start[0])
            if tok_type not in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            }:
                significant_token_count += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Keep lexical metrics robust even for malformed snippets.
        token_count = 0
        significant_token_count = 0
        comment_line_numbers = set()

    comment_loc = len(comment_line_numbers)

    return {
        "char_count": float(char_count),
        "loc": float(loc),
        "non_empty_loc": float(non_empty_loc),
        "comment_loc": float(comment_loc),
        "comment_density": float(comment_loc / loc) if loc else 0.0,
        "avg_line_length": float(avg_line_length),
        "max_line_length": float(max_line_length),
        "token_count": float(token_count),
        "significant_token_count": float(significant_token_count),
        "whitespace_ratio": float(whitespace_ratio),
    }


def compute_ast_features(code: str) -> Dict[str, float]:
    default_counts = {f"ast_count_{name}": 0.0 for name in AST_NODE_TYPES}
    default_result = {
        "ast_parse_success": 0.0,
        "ast_node_count": 0.0,
        "ast_max_depth": 0.0,
        "function_count": 0.0,
        "class_count": 0.0,
        "loop_count": 0.0,
        "if_count": 0.0,
        "try_count": 0.0,
        "import_count": 0.0,
        "call_count": 0.0,
        "assign_count": 0.0,
        "return_count": 0.0,
        "comprehension_count": 0.0,
        "cyclomatic_complexity": 0.0,
        **default_counts,
    }

    try:
        # Some snippets trigger SyntaxWarning (e.g., malformed numeric literals).
        # We treat those the same way as other parse issues and continue.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return default_result

    nodes = list(ast.walk(tree))
    node_count = len(nodes)

    count_by_name = {name: 0 for name in AST_NODE_TYPES}
    for node in nodes:
        for name, node_type in AST_NODE_TYPES.items():
            if isinstance(node, node_type):
                count_by_name[name] += 1

    def max_depth(node: ast.AST, depth: int = 0) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return depth
        return max(max_depth(child, depth + 1) for child in children)

    ast_max_depth = max_depth(tree)

    loop_count = count_by_name["For"] + count_by_name["AsyncFor"] + count_by_name["While"]
    if_count = count_by_name["If"]
    try_count = count_by_name["Try"]
    import_count = count_by_name["Import"] + count_by_name["ImportFrom"]
    call_count = count_by_name["Call"]
    assign_count = count_by_name["Assign"] + count_by_name["AnnAssign"] + count_by_name["AugAssign"]
    return_count = count_by_name["Return"]
    comprehension_count = (
        count_by_name["ListComp"]
        + count_by_name["DictComp"]
        + count_by_name["SetComp"]
        + count_by_name["GeneratorExp"]
    )

    bool_op_count = sum(1 for node in nodes if isinstance(node, ast.BoolOp))
    except_handler_count = sum(1 for node in nodes if isinstance(node, ast.ExceptHandler))
    decision_points = if_count + loop_count + try_count + except_handler_count + comprehension_count + bool_op_count
    cyclomatic_complexity = 1 + decision_points

    result = {
        "ast_parse_success": 1.0,
        "ast_node_count": float(node_count),
        "ast_max_depth": float(ast_max_depth),
        "function_count": float(count_by_name["FunctionDef"] + count_by_name["AsyncFunctionDef"]),
        "class_count": float(count_by_name["ClassDef"]),
        "loop_count": float(loop_count),
        "if_count": float(if_count),
        "try_count": float(try_count),
        "import_count": float(import_count),
        "call_count": float(call_count),
        "assign_count": float(assign_count),
        "return_count": float(return_count),
        "comprehension_count": float(comprehension_count),
        "cyclomatic_complexity": float(cyclomatic_complexity),
    }
    for name, count in count_by_name.items():
        result[f"ast_count_{name}"] = float(count)

    return result


def extract_row_features(
    example: Dict[str, object],
    idx: int,
    *,
    code_col: str,
    split_name: str,
    hf_feature_keys: List[str],
    include_hf_features: bool,
) -> Dict[str, object]:
    code_obj = example.get(code_col)
    code = code_obj if isinstance(code_obj, str) else ""

    code_hash = compute_code_sha1(code_obj)
    result: Dict[str, object] = {
        "dataset_split": split_name,
        "split_index": int(idx),
        "sample_row_id": f"{split_name}:{idx}",
        "code_sha1": code_hash,
        "code_col_used": code_col,
    }

    for col in METADATA_COLUMNS:
        if col in example:
            result[col] = example.get(col)

    result.update(compute_lexical_features(code))
    result.update(compute_ast_features(code))

    if include_hf_features and hf_feature_keys:
        nested = example.get("features")
        nested_dict = nested if isinstance(nested, dict) else {}
        for key in hf_feature_keys:
            result[f"hf_{key}"] = safe_float(nested_dict.get(key))

    return result


def parse_splits_arg(arg_value: Optional[str], available_splits: List[str]) -> List[str]:
    if not arg_value:
        return available_splits
    requested = [part.strip() for part in arg_value.split(",") if part.strip()]
    invalid = [s for s in requested if s not in available_splits]
    if invalid:
        raise ValueError(f"Unknown split(s): {invalid}. Available: {available_splits}")
    return requested


def process_split(
    split_name: str,
    split_ds: Dataset,
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    if "language" not in split_ds.column_names:
        raise ValueError(f"Split '{split_name}' has no 'language' column")

    code_col = choose_code_column(split_ds.column_names)

    def keep_row(example: Dict[str, object]) -> bool:
        if example.get("language") != args.language:
            return False
        if args.source is not None and example.get("source") != args.source:
            return False
        if "model" in split_ds.column_names and example.get("model") in (None, ""):
            return False
        return True

    filtered = split_ds.filter(
        keep_row,
        num_proc=args.num_proc,
        desc=f"Filtering {split_name} ({args.language}{' + source=' + args.source if args.source else ''})",
    )

    if args.max_samples_per_split is not None:
        limited_n = min(args.max_samples_per_split, len(filtered))
        filtered = filtered.select(range(limited_n))

    hf_feature_keys = []
    if not args.exclude_hf_features:
        hf_feature_keys = infer_hf_feature_keys(filtered)

    featured = filtered.map(
        extract_row_features,
        with_indices=True,
        fn_kwargs={
            "code_col": code_col,
            "split_name": split_name,
            "hf_feature_keys": hf_feature_keys,
            "include_hf_features": not args.exclude_hf_features,
        },
        remove_columns=filtered.column_names,
        num_proc=args.num_proc,
        desc=f"Extracting features for {split_name}",
    )

    out_path = output_dir / f"{split_name}.parquet"
    featured.to_parquet(str(out_path))

    parse_success = int(sum(featured["ast_parse_success"])) if len(featured) else 0
    if "model" in featured.column_names:
        raw_models = set(featured["model"])
        # Exclude null labels to keep sorting and label mapping stable.
        models = sorted(str(m) for m in raw_models if m is not None)
    else:
        models = []

    return {
        "split": split_name,
        "rows": len(featured),
        "ast_parse_success_rows": parse_success,
        "ast_parse_success_rate": (parse_success / len(featured)) if len(featured) else 0.0,
        "models": models,
        "output": str(out_path),
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_obj = load_dataset(args.dataset)
    if not isinstance(dataset_obj, DatasetDict):
        raise RuntimeError(f"Expected DatasetDict, got {type(dataset_obj)}")

    available_splits = list(dataset_obj.keys())
    splits = parse_splits_arg(args.splits, available_splits)

    print(f"Dataset: {args.dataset}")
    print(f"Splits: {splits}")
    print(f"Language filter: {args.language}")
    if args.source:
        print(f"Source filter: {args.source}")
    print(f"Output dir: {output_dir.resolve()}")

    summaries: List[Dict[str, object]] = []
    all_models = set()

    for split_name in splits:
        summary = process_split(split_name, dataset_obj[split_name], args=args, output_dir=output_dir)
        summaries.append(summary)
        all_models.update(summary["models"])

    label_mapping = {model: idx for idx, model in enumerate(sorted(all_models))}
    mapping_path = output_dir / "label_mapping.json"
    mapping_path.write_text(json.dumps(label_mapping, indent=2, sort_keys=True), encoding="utf-8")

    summary_path = output_dir / "extraction_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\nExtraction complete.")
    for s in summaries:
        print(
            f"- {s['split']}: rows={s['rows']}, "
            f"ast_parse_success={s['ast_parse_success_rows']} ({s['ast_parse_success_rate']:.2%}), "
            f"file={s['output']}"
        )
    print(f"Label mapping: {mapping_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
