"""
Universal Parser: Extracts Token Count and Max AST Depth (Logic Density)
from code files for methodology metrics.
"""

import csv
import os
from pathlib import Path


def get_metrics(filepath: str, language: str) -> tuple[int, int]:
    """
    Extract Token_Count and Max_Depth for a code file.

    Token Count: Split file content by whitespace, return length.
    Max AST Depth:
      - Python: Based on indentation (leading spaces / 4).
      - Java: Based on brace nesting ({ increment, } decrement).

    Returns:
        (token_count, max_depth)
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Token count: split by whitespace
    tokens = content.split()
    token_count = len(tokens)

    # Max AST depth (logic density)
    if language == "python":
        max_depth = 0
        for line in content.splitlines():
            stripped = line.lstrip(" \t")
            if not stripped:
                continue
            leading = len(line) - len(stripped)
            # Assume 4 spaces per indent; tabs count as 1 level
            spaces = leading
            level = spaces // 4 if spaces else 0
            if level > max_depth:
                max_depth = level
    elif language == "java":
        depth = 0
        max_depth = 0
        for char in content:
            if char == "{":
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            elif char == "}":
                depth -= 1
    else:
        max_depth = 0

    return token_count, max_depth


def get_language(filepath: str) -> str | None:
    """Return 'python' or 'java' from extension, else None."""
    ext = Path(filepath).suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".java":
        return "java"
    return None


def extract_model_from_filename(filename: str) -> str:
    """Extract model name from filename (e.g. gpt4_task1.py -> gpt4)."""
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def main():
    dataset_dir = Path(__file__).resolve().parent / "dataset"
    output_path = Path(__file__).resolve().parent / "real_experiment_data.csv"

    if not dataset_dir.is_dir():
        print(f"Dataset folder not found: {dataset_dir}")
        print("Creating empty dataset folder. Add .py/.java files and run again.")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return

    rows = []
    for filepath in dataset_dir.rglob("*"):
        if not filepath.is_file():
            continue
        language = get_language(str(filepath))
        if language is None:
            continue
        model = extract_model_from_filename(filepath.name)
        token_count, max_depth = get_metrics(str(filepath), language)
        rows.append({
            "Filename": filepath.name,
            "Model": model,
            "Language": language,
            "Token_Count": token_count,
            "Max_Depth": max_depth,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Filename", "Model", "Language", "Token_Count", "Max_Depth"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} files -> {output_path}")


if __name__ == "__main__":
    main()
