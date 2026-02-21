# Python Aggregated Feature Extraction Guide

Script: `/Users/kimsia/Projects/smu-mitb-2024-2027/smu-mitb-cs706-software-mining/project/avi-idea/analysis/extract_python_aggregated_features.py`

## What this script does

- Loads a dataset from Hugging Face (`datasets`)
- Filters to Python (`language == "python"` by default)
- Extracts per-sample aggregated lexical + AST features
- Writes split-level parquet files for baseline ML (XGBoost/RF)

## Current assumption (important)

The script is currently CoDET-M4-oriented. It expects dataset rows to include fields similar to:

- `language`
- `model` (label)
- `code` or `cleaned_code`
- optional metadata such as `source`, `split`, etc.

If your input does not follow this structure, you should normalize your data first.

Note: CoDET-M4 may not expose stable IDs like `sample_id` / `problem_id` in the base splits.
This extractor now emits generated IDs:

- `sample_row_id` = `<split>:<index>`
- `code_sha1` = SHA1 hash of code text

Use these for joins across separately extracted artifacts.

## Output columns and definitions

The extractor writes one row per code sample. Output columns are:

Note: the `quick_viz/*.csv` files are convenience exports and may include only a subset of these columns.

### ID / lineage columns

| Column | Definition |
| --- | --- |
| `dataset_split` | Split being processed by the script (`train`, `validation`, `test`, etc.). |
| `split_index` | Row index within that split after filtering (0-based from `datasets.map(..., with_indices=True)`). |
| `sample_row_id` | Generated row id: `<dataset_split>:<split_index>`. |
| `code_sha1` | SHA1 of the exact text used for feature extraction. |
| `code_col_used` | Which source code field was used: `cleaned_code` (preferred) or `code` (fallback). |

### Label / metadata columns (passed through when present)

| Column | Definition |
| --- | --- |
| `model` | Class label (e.g., `human`, `gpt`, `codellama`, etc.). |
| `language` | Programming language (filtered to `python` in your current run). |
| `source` | Dataset source/domain (e.g., `codeforces`). |
| `target` | Existing numeric target from source dataset if available. |
| `split` | Original split field from source row (if present). |
| `source_type` | Source subtype from dataset metadata (if present). |
| `model_display` | Human-readable model name if present. |

### Lexical / size features

| Column | Definition |
| --- | --- |
| `char_count` | Number of characters in code text. |
| `loc` | Number of lines (`splitlines()`). |
| `non_empty_loc` | Number of non-empty lines. |
| `comment_loc` | Number of distinct line numbers containing comment tokens. |
| `comment_density` | `comment_loc / loc`. |
| `avg_line_length` | Average characters per line. |
| `max_line_length` | Maximum line length (characters). |
| `token_count` | Total Python tokenizer tokens seen. |
| `significant_token_count` | Token count excluding structural tokens (`NL`, `NEWLINE`, `INDENT`, `DEDENT`, `ENDMARKER`). |
| `whitespace_ratio` | Whitespace characters divided by total characters. |

### AST / structural features

| Column | Definition |
| --- | --- |
| `ast_parse_success` | `1.0` if `ast.parse` succeeds, else `0.0`. |
| `ast_node_count` | Total AST nodes (`len(ast.walk(tree))`). |
| `ast_max_depth` | Maximum root-to-leaf AST depth. |
| `function_count` | Count of `FunctionDef` + `AsyncFunctionDef`. |
| `class_count` | Count of `ClassDef`. |
| `loop_count` | Count of `For` + `AsyncFor` + `While`. |
| `if_count` | Count of `If`. |
| `try_count` | Count of `Try`. |
| `import_count` | Count of `Import` + `ImportFrom`. |
| `call_count` | Count of `Call`. |
| `assign_count` | Count of `Assign` + `AnnAssign` + `AugAssign`. |
| `return_count` | Count of `Return`. |
| `comprehension_count` | Count of list/dict/set comprehensions + generator expressions. |
| `cyclomatic_complexity` | Approximation: `1 + decision_points`, where decision points include if/loops/try/except/comprehensions/bool ops. |

### AST node-type count columns

For each node type below, the extractor emits `ast_count_<NodeType>`:

- `FunctionDef`, `AsyncFunctionDef`, `ClassDef`
- `Return`, `Assign`, `AnnAssign`, `AugAssign`
- `For`, `AsyncFor`, `While`, `If`, `Try`
- `With`, `AsyncWith`, `Call`
- `Import`, `ImportFrom`
- `ListComp`, `DictComp`, `SetComp`, `GeneratorExp`, `Lambda`

### Optional Hugging Face nested feature columns

If source rows contain nested `features` and `--exclude-hf-features` is not used, extra columns are emitted as:

- `hf_<original_feature_key>`

Example: `hf_avgFunctionLength`, `hf_maintainabilityIndex`, etc.

## Reusing this on other datasets (JSONL)

Yes, you can reuse the same extraction logic for other team datasets in JSONL, as long as you convert them to a common schema first.

### Recommended minimal schema

For each JSONL row, standardize to:

- `sample_id` (string, unique) if available; otherwise generate one
- `code` (string)
- `language` (string, e.g., `python`)
- `label` (string, model/human class)
- `split` (optional: `train` / `validation` / `test`)
- `source` (optional: `codeforces`, etc.)

### Suggested workflow

1. Normalize each contributor's dataset (JSONL) into the common schema.
2. Run this feature extractor on normalized code rows (Python only).
3. Keep CPG outputs separate (from Joern pipeline).
4. Merge flat-feature table and CPG table later using a stable key:
   - preferred: shared `sample_id`
   - fallback: `code_sha1` (+ `language` if needed)

## Why this helps your team now

- People can extract different representations in parallel (flat AST features vs CPG).
- You avoid blocking on a single merged raw dataset.
- You can still train baselines immediately on flat features, then join with CPG later for RQ3 experiments.

## Example run (CoDET-M4 via Hugging Face)

```bash
python /Users/kimsia/Projects/smu-mitb-2024-2027/smu-mitb-cs706-software-mining/project/avi-idea/analysis/extract_python_aggregated_features.py \
  --dataset DaniilOr/CoDET-M4 \
  --language python \
  --output-dir /Users/kimsia/Projects/smu-mitb-2024-2027/smu-mitb-cs706-software-mining/project/avi-idea/data/processed_homo/python_agg \
  --num-proc 8
```

Optional CodeForces filter:

```bash
--source codeforces
```

## Future improvement (optional)

If needed, extend the script with:

- `--input-jsonl`
- `--code-col`
- `--language-col`
- `--label-col`
- `--split-col`

This makes one extractor reusable across all team JSONL variants without separate normalization scripts.
