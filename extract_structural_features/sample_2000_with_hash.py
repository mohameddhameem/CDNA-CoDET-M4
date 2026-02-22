#!/usr/bin/env python3
"""Sample 2400 rows from CoDET-M4 dataset with SHA-1 code hashes.

Strategy:
  - 2400 rows total, evenly split across 6 models (400 each):
    human, codellama, gpt, llama3.1, nxcode, qwen1.5
  - Each model follows an 8:1:1 train/test/val split:
    320 train + 40 test + 40 val = 400
  - SHA-1 hash is computed on the `code` column (preferred over
    `cleaned_code`) using the shared code_hash module using kim sia's logic.
  - Empty hash ("") is expected for rows with missing/empty code,
    as defined in code_hash.py.

Output columns (per split file):
  - code_sha1 : SHA-1 hex digest of `code` ("" if code is empty/missing)
  - index     : 0-based row number in the *original* parquet file
  - model     : ground-truth label

Outputs saved to dataset/empirical study/:
  - train.json, train.parquet   (1920 rows = 320 per model x 6)
  - test.json,  test.parquet    ( 240 rows =  40 per model x 6)
  - val.json,   val.parquet     ( 240 rows =  40 per model x 6)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Kim Sia's hash module
from code_hash import choose_code_column, compute_code_sha1


# Config
SEED = 42
DATASET_URL = "hf://datasets/DaniilOr/CoDET-M4/dataset_without_comments.parquet"

MODELS = ["human", "codellama", "gpt", "llama3.1", "nxcode", "qwen1.5"]
TOTAL_PER_MODEL = 400
# 8:1:1 split
TRAIN_PER_MODEL = 320
TEST_PER_MODEL = 40
VAL_PER_MODEL = 40

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "dataset" / "empirical_study"


def load_dataset(url: str) -> pd.DataFrame:
    """Load parquet from HuggingFace hub URL."""
    print(f"Loading dataset from {url} ...")
    df = pd.read_parquet(url)
    print(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


def stratified_sample(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample rows per model per split according to the 8:1:1 quota."""
    quota = {
        "train": TRAIN_PER_MODEL,
        "test": TEST_PER_MODEL,
        "val": VAL_PER_MODEL,
    }

    frames: list[pd.DataFrame] = []
    for model in MODELS:
        for split, n_needed in quota.items():
            pool = df[(df["model"] == model) & (df["split"] == split)]
            # We have verfied that the dataset contains enough rows for each (model, split) combination.
            sampled = pool.sample(n=n_needed, random_state=int(rng.integers(2**31)))
            frames.append(sampled)
            print(f"{model} / {split}: sampled {n_needed} from {len(pool)}")

    result = pd.concat(frames, ignore_index=False)
    print(f"Total sampled: {len(result)}")
    return result


def add_hashes(df: pd.DataFrame) -> pd.DataFrame:
    """Add code_sha1 column, hashing the `code` column (preferred).

    Uses compute_code_sha1 from code_hash module directly.
    Empty hash ("") is expected for rows with missing/empty/non-string code.
    """
    code_col = choose_code_column(df.columns)
    print(f"Hashing column: {code_col}")
    df = df.copy()
    df["code_sha1"] = df[code_col].apply(compute_code_sha1)
    n_empty = (df["code_sha1"] == "").sum()
    if n_empty:
        print(f"WARNNING: {n_empty} row(s) have empty hash (missing/empty code — expected)")
    return df


def build_output(sampled: pd.DataFrame) -> pd.DataFrame:
    """Build the final output DataFrame with code_sha1, index, model."""
    out = pd.DataFrame(
        {
            "code_sha1": sampled["code_sha1"],
            "index": sampled.index,  # 0-based original row number
            "model": sampled["model"],
        }
    )
    out = out.sort_values(["model", "index"]).reset_index(drop=True)
    return out


def save_outputs(sampled: pd.DataFrame, output_dir: Path) -> None:
    """Save train/test/val as separate JSON + Parquet file pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, group in sampled.groupby("split"):
        out = build_output(group)

        json_path = output_dir / f"{split_name}.json"
        parquet_path = output_dir / f"{split_name}.parquet"

        # JSON — list of records
        records = out.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, (np.integer,)):
                    rec[k] = int(v)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Saved {json_path.name}: {len(records)} records")

        # Parquet
        out.to_parquet(parquet_path, index=False)
        print(f"Saved {parquet_path.name}: {len(out)} rows")


def main() -> None:
    rng = np.random.default_rng(SEED)

    # 1. Load
    df = load_dataset(DATASET_URL)

    # 2. Sampling
    print(f"Sampling {TOTAL_PER_MODEL * len(MODELS)} rows ({TOTAL_PER_MODEL} per model, 8:1:1 split): ")
    sampled = stratified_sample(df, rng)

    # 3. Compute hashes
    print("Computing SHA-1 hashes")
    sampled = add_hashes(sampled)

    # 4. Build output
    out = build_output(sampled)

    # 5. Summary
    print()
    print("=== Sample summary ===")
    print(out.groupby("model").size().to_string())
    print(f"Total: {len(out)}")
    print(f"Unique hashes: {out['code_sha1'].nunique()}")
    print("Per split:")
    print(sampled.groupby("split").size().to_string())

    # 6. Save
    print()
    print(f"Saving outputs to {OUTPUT_DIR} ...")
    save_outputs(sampled, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
