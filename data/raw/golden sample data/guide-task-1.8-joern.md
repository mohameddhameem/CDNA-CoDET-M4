# Guide: Task 1.8 -- Joern CPG on the Golden Sample

> **For:** YQ (or whoever is handling Joern)
> **Depends on:** Task 1.5 (golden sample -- done)
> **Goal:** Install Joern and test CPG (Code Property Graph) generation on golden sample code files

---

## What files you need

Same as Tree-sitter: the **`code/`** directory with 55 problem folders.

```text
data/golden-sample/code/
├── 1381_B/
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

### Language coverage recap

| Language | Extension | Approx. file count | Notes |
| --- | --- | --- | --- |
| Python | `.py` | ~220 | All 4 models |
| Java | `.java` | ~56 | Almost entirely human |
| C++ | `.cpp` | ~55 | Human only |

The LLMs (gpt4o, codellama, llama31) almost exclusively have Python solutions. Java and C++ are almost all human-written.

---

## Joern language support -- important note

Joern's language support varies. As of early 2026:

- **C/C++**: Best supported (Joern was originally built for C/C++)
- **Java**: Well supported
- **Python**: Supported but may have limitations compared to C/C++

This is actually good news for us: the golden sample has C++ and Java files (human code) which are Joern's strong suit. Start validation with C++ files first.

### Suggested validation order

1. **C++ first** (best Joern support): `1381_B/human_cpp.cpp`
2. **Java second**: `1381_B/human_java.java`
3. **Python third**: `1381_B/human_python.py` then `1381_B/gpt4o_python.py`

If C++ and Java work but Python has issues, that's still useful -- document what works and what doesn't.

---

## What "success" looks like

After Joern is installed and validated, you should be able to:

1. Feed a code file to Joern and get back a CPG (Code Property Graph)
2. The CPG combines AST + Control Flow Graph + Data Flow Graph into a single structure
3. From the CPG, we can eventually extract richer metrics than Tree-sitter alone (e.g., data flow patterns, control flow complexity)

At this stage, just confirm:

- Joern can ingest files from the `code/` directory
- It produces CPG output for each language you test
- Note any files or languages that fail, and why

---

## How the code files are organized

Each file is named `{model}_{language}.{ext}`:

| Model tag | What it is |
| --- | --- |
| `human` | Human solution from CodeForces |
| `gpt4o` | GPT-4o generated solution |
| `codellama` | CodeLlama 7B generated solution |
| `llama31` | Llama 3.1 8B generated solution |

The folder name encodes the problem: `1381_B` means CodeForces contest 1381, problem B. You can look up any problem at `https://codeforces.com/problemset/problem/1381/B`.

---

## Files you do NOT need

- `golden_sample_metadata.csv` -- for viz/feature analysis
- `golden_sample_code.jsonl` -- for GNN pipeline
- `golden_sample_problems.csv` -- useful reference but not needed for Joern testing
- `matching_report.json` -- internal matching data, ignore

---

## Quick reference: listing files for batch processing

```python
from pathlib import Path

code_dir = Path("data/golden-sample/code")

# Group files by language for Joern processing
for ext, lang in [("*.cpp", "C++"), ("*.java", "Java"), ("*.py", "Python")]:
    files = sorted(code_dir.rglob(ext))
    print(f"\n{lang}: {len(files)} files")
    for f in files[:3]:  # preview first 3
        print(f"  {f.relative_to(code_dir)}")
```
