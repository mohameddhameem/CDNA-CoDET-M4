# Guide: Task 1.11 -- Visualization Mockups Using the Golden Sample

> **For:** Avi (or whoever is doing initial visualization mockups)
> **Depends on:** Task 1.5 (golden sample -- done)
> **Goal:** Create initial visualization mockups (Plotly/Streamlit) using golden sample metadata, as a stepping stone toward the full "City of Agents" visualization

---

## Overview: What data is available

The golden sample has **55 CodeForces problems** solved by **4 models** (human, GPT-4o, CodeLlama, Llama3.1) across up to **3 languages** (Python, Java, C++). The total is **2,623 code samples** with pre-computed metadata.

You have two CSV files that are ready to use for mockups -- no code parsing needed.

### Focus on Python for model comparison

All 4 models have Python solutions for all 55 problems, giving you **55 x 4 = 220 head-to-head data points**. This is the cleanest comparison in the golden sample and directly supports the proposal's RQ1 ("Do different LLMs produce distinct city skylines?").

Java and C++ in the golden sample are almost entirely human-only. This is because our matching strategy (function name matching) works well for Python code but not for Java/C++ where competitive programmers and LLMs tend to use generic names like `main()` or `solve()`. The LLM Java/C++ data does exist in the full CoDet-M4 dataset but couldn't be matched to specific problems.

**Bottom line for mockups: filter to `language == 'python'` for all cross-model comparisons.** The Java/C++ data is useful for comparing human coding style across languages, but not for comparing models against each other.

All 4 models (including human) have Python solutions for **54 of the 55 problems**. Only problem 1572/C "Paint" (difficulty 2700) is missing a human Python solution -- likely because no CodeForces user submitted a Python solution for this very hard problem. The 3 LLMs do have Python for all 55.

---

## File 1: `golden_sample_problems.csv` -- Problem-level view

**One row per problem (55 rows).** Use this for overview plots.

| Column | What it means | Example |
| --- | --- | --- |
| `problem_id` | CodeForces problem ID | `1381/B` |
| `problem_name` | Problem title | `Unmerge` |
| `difficulty_rating` | CodeForces difficulty (800 = easiest) | `1800` |
| `tags` | Problem categories, pipe-separated | `dp\|greedy` |
| `solved_count` | How many people solved it on CodeForces | `9471` |
| `total_samples` | Number of code samples for this problem | `61` |

### How to load it

```python
import pandas as pd

problems = pd.read_csv("golden_sample_problems.csv")
print(problems.shape)        # (55, 11)
print(problems.columns.tolist())
```

### What mockups you can make with this file

- **Difficulty distribution:** Histogram of `difficulty_rating` to show the range of problems in the golden sample (800 to 2700)
- **Problem tag breakdown:** Bar chart of the most common `tags` (split on `|`)
- **Sample count by problem:** Bar chart of `total_samples` to show data density per problem

---

## File 2: `golden_sample_metadata.csv` -- Sample-level view

**One row per code sample (2,623 rows).** This is the main file for visualization mockups.

| Column | What it means | Example | City of Agents mapping |
| --- | --- | --- | --- |
| `problem_id` | Which problem | `1381/B` | Groups buildings into "city blocks" |
| `model` | Who wrote the code | `human` / `gpt` / `codellama` / `llama3.1` | **Color** of the building |
| `model_display` | Display name | `gpt4o` | Use this for labels |
| `language` | Programming language | `python` / `java` / `cpp` | Could be a filter or facet |
| `difficulty_rating` | Problem difficulty | `1800` | Could sort or group city blocks |
| `loc` | Lines of code | `45` | Proxy for **verbosity** (building base size) |
| `token_count_approx` | Word count of the code | `120` | Another proxy for **verbosity** |
| `feature_maintainabilityIndex` | Code quality score (0-100) | `67.3` | |
| `feature_avgFunctionLength` | Avg lines per function | `12.5` | |
| `feature_avgIdentifierLength` | Avg variable name length | `5.2` | |
| `feature_avgLineLength` | Avg characters per line | `28.7` | |
| `feature_emptyLinesDensity` | % of lines that are blank | `0.08` | |
| `feature_functionDefinitionDensity` | % of lines that define functions | `0.05` | |
| `feature_maxDecisionTokens` | Deepest nesting of if/for/while | `3` | Proxy for **complexity** (building height) |
| `feature_whiteSpaceRatio` | % whitespace in the code | `0.25` | |

### How to load it

```python
import pandas as pd

samples = pd.read_csv("golden_sample_metadata.csv")
print(samples.shape)          # (2623, 28)
print(samples.columns.tolist())

# Quick look at model distribution
print(samples['model'].value_counts())
# human       2358
# gpt          105
# codellama     80
# llama3.1      80
```

---

## Connecting the data to the City of Agents proposal

The proposal (Section 4.3) maps code metrics to city dimensions:

| City dimension | Proposal metric | Available in metadata CSV |
| --- | --- | --- |
| **Building height** (Z-axis) | Max AST Depth | `feature_maxDecisionTokens` (approximate proxy). Full AST depth comes from Tree-sitter (task 1.7) |
| **Building base area** | Token Count (verbosity) | `token_count_approx` or `loc` |
| **Building color** | AI Model | `model` / `model_display` |

The features in the CSV are a starting point. Once Tree-sitter (task 1.7) produces proper AST depths, those can replace `feature_maxDecisionTokens` for the height dimension.

---

## Suggested mockup ideas (simplest first)

### Mockup 1: Side-by-side comparison (one problem)

Pick one problem (e.g., problem `1381/B` "Unmerge") and compare the 4 models on Python:

```python
import pandas as pd
import plotly.express as px

samples = pd.read_csv("golden_sample_metadata.csv")

# Filter to one problem, one language, one sample per model
problem = samples[
    (samples['problem_id'] == '1381/B') &
    (samples['language'] == 'python')
].drop_duplicates(subset='model')

fig = px.bar(
    problem,
    x='model_display',
    y='loc',
    color='model_display',
    title='Lines of Code: Problem 1381/B "Unmerge" (Python)',
    labels={'loc': 'Lines of Code', 'model_display': 'Model'}
)
fig.show()
```

### Mockup 2: Verbosity vs complexity scatter (all models)

```python
# One dot per model per problem (Python only, one sample per combo)
python_samples = samples[samples['language'] == 'python'].drop_duplicates(
    subset=['problem_id', 'model']
)

fig = px.scatter(
    python_samples,
    x='token_count_approx',
    y='feature_maxDecisionTokens',
    color='model_display',
    hover_data=['problem_name', 'difficulty_rating'],
    title='Token Count vs Decision Depth by Model',
    labels={
        'token_count_approx': 'Token Count (Verbosity)',
        'feature_maxDecisionTokens': 'Max Decision Tokens (Complexity)',
        'model_display': 'Model'
    }
)
fig.show()
```

This is the 2D version of the "City of Agents" -- verbosity on X, complexity on Y, color by model. If models cluster differently, that supports the "universal fingerprint" hypothesis.

### Mockup 3: Distribution comparison across models

```python
fig = px.box(
    python_samples,
    x='model_display',
    y='loc',
    color='model_display',
    title='Lines of Code Distribution by Model (Python)',
    labels={'loc': 'Lines of Code', 'model_display': 'Model'}
)
fig.show()
```

Try this with different Y columns: `loc`, `token_count_approx`, `feature_maintainabilityIndex`, `feature_avgFunctionLength`.

### Mockup 4: Feature heatmap across models

```python
feature_cols = [c for c in samples.columns if c.startswith('feature_')]

# Average features per model (Python only)
model_features = python_samples.groupby('model_display')[feature_cols].mean()

fig = px.imshow(
    model_features,
    title='Average Code Features by Model',
    labels={'color': 'Value'},
    aspect='auto'
)
fig.show()
```

---

## Important notes about the data

### Human has many more samples than LLMs

Human has ~2,358 samples vs ~80-105 per LLM. This is because many human submissions exist per problem on CodeForces. For fair comparison in mockups, **pick one representative sample per model per problem** using `drop_duplicates(subset=['problem_id', 'model'])` or take the first/median.

### Cross-model comparison = Python only

GPT-4o, CodeLlama, and Llama3.1 only have Python solutions in the golden sample. Java and C++ are human-only. This is because our problem-matching strategy (function name matching) only works reliably on Python code -- Java and C++ competitive programming solutions use generic names like `main()`.

So there are two types of comparisons you can make:

| Comparison | Filter | What it answers |
| --- | --- | --- |
| **Cross-model** (RQ1) | `language == 'python'` | Do different models produce distinct "city skylines"? |
| **Cross-language** (RQ2) | `model == 'human'` | Does human coding style change across Python/Java/C++? |

For cross-language LLM comparison (e.g., GPT-4o Python vs GPT-4o Java), use the full CoDet-M4 dataset -- that data exists but can't be matched to specific problems.

### Some feature values may be missing

The `feature_*` columns come from the HuggingFace dataset. Samples sourced from raw JSON files (some Llama3.1 entries) may have NaN values for features. Use `dropna()` or filter when plotting.

---

## Files you do NOT need for mockups

- `code/` -- you do not need to read actual code files for mockups; the CSV has pre-computed metrics
- `golden_sample_code.jsonl` -- this is for the GNN pipeline
- `matching_report.json` -- internal matching data
- `golden_sample_master.json` -- documentation/provenance, not for plotting

---

## Quick reference: what each golden sample file is for

| File | Format | Rows | Use |
| --- | --- | --- | --- |
| `golden_sample_problems.csv` | CSV | 55 | Problem-level overview plots |
| `golden_sample_metadata.csv` | CSV | 2,623 | **Main file for mockups** -- all metrics, no raw code |
| `golden_sample_code.jsonl` | JSONL | 2,623 | Raw code text (for GNN pipeline, not viz) |
| `golden_sample_master.json` | JSON | 1 | Documentation/provenance metadata |
| `code/` | directories | 55 dirs | Actual code files (for Tree-sitter/Joern, not viz) |
