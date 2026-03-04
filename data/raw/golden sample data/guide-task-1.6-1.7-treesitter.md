# Guide: Tasks 1.6 & 1.7 -- Tree-sitter on the Golden Sample

> **For:** YQ (or whoever is handling Tree-sitter)
> **Depends on:** Task 1.4 (golden sample -- done)
> **Goal:** Install Tree-sitter for Python/Java/C++, then parse the golden sample code files to produce ASTs

---

## What files you need

You only need the **`code/`** directory. It contains 55 folders, one per CodeForces problem.

```text
data/golden-sample/code/
├── 1141_F1/
│   ├── human_python.py
│   ├── human_java.java
│   ├── human_cpp.cpp
│   ├── gpt4o_python.py
│   ├── codellama_python.py
│   └── llama31_python.py
├── 1303_C/
│   └── ...
└── ... (55 directories total)
```

### File naming convention

Each file is named `{model}_{language}.{ext}`:

| Model tag | What it is |
| --- | --- |
| `human` | Human solution from CodeForces |
| `gpt4o` | GPT-4o generated solution |
| `codellama` | CodeLlama 7B generated solution |
| `llama31` | Llama 3.1 8B generated solution |

### What languages are present

| Language | Extension | Approx. file count | Notes |
| --- | --- | --- | --- |
| Python | `.py` | ~220 | All 4 models have Python |
| Java | `.java` | ~56 | Almost entirely human (1 CodeLlama) |
| C++ | `.cpp` | ~55 | Human only |

**Important:** The LLMs (gpt4o, codellama, llama31) almost exclusively have Python solutions. Java and C++ files are almost all human-written. This is a property of the dataset, not a bug -- the raw LLM data from the CoDet-M4 author only had Python for the CodeForces subset.

This means you should expect Tree-sitter to process:

- **Python grammars** for all 4 models
- **Java and C++ grammars** primarily for human code

### Which files to start with for validation

For a quick smoke test, pick one problem directory (e.g., `1381_B/`) and try parsing each file type:

1. `1381_B/human_python.py` -- Python
2. `1381_B/human_java.java` -- Java
3. `1381_B/human_cpp.cpp` -- C++
4. `1381_B/gpt4o_python.py` -- LLM Python (to confirm it parses the same as human Python)

If all 4 parse successfully, you can batch-process the rest.

---

## What "success" looks like

After Tree-sitter is installed and validated, you should be able to:

1. Point Tree-sitter at any `.py`, `.java`, or `.cpp` file in `code/` and get back an AST (Abstract Syntax Tree)
2. From that AST, extract metrics like **max depth** and **node count** -- these feed directly into the City of Agents visualization (AST depth = building height)

You do NOT need to extract metrics yourself at this stage. Just confirm that:

- Tree-sitter parses all 55 directories without errors
- If some files fail to parse, note which ones and why (some CodeForces solutions use non-standard syntax or competitive programming tricks)

---

## Files you do NOT need

- `golden_sample_metadata.csv` -- this is for viz/feature analysis, not parsing
- `golden_sample_code.jsonl` -- this is for the GNN pipeline, not Tree-sitter
- `golden_sample_problems.csv` -- useful reference for what each problem is, but not needed for parsing
- `matching_report.json` -- internal matching data, ignore

---

## Quick reference: loading file paths programmatically

```python
from pathlib import Path

code_dir = Path("data/golden-sample/code")

# All Python files
py_files = sorted(code_dir.rglob("*.py"))

# All Java files
java_files = sorted(code_dir.rglob("*.java"))

# All C++ files
cpp_files = sorted(code_dir.rglob("*.cpp"))

# Parse the model and language from a filename
# e.g., "gpt4o_python.py" -> model="gpt4o", language="python"
for f in py_files:
    parts = f.stem.split("_", 1)  # ["gpt4o", "python"]
    model, lang = parts[0], parts[1]
    problem_id = f.parent.name      # "1381_B"
    print(f"{problem_id} | {model} | {lang} | {f}")
```
