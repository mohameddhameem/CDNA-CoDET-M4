---
title: Draft Experiment Matrix
datetime: 2026-02-18 16:47:18 +08
author: simkimsia
---

## Goal

Compare `GNN` against two baselines on CoDet-M4 (CodeForces - python subset) for 6-class authorship attribution.

## Fixed Protocol (All Runs)

- Dataset: CoDet-M4 CodeForces subset
- Labels: `human`, `gpt4o`, `codellama`, `llama3.1`, `nxcode`, `qwen`
- Splits: one fixed train/val/test split shared by all runs
- Primary metric: `Macro-F1`
- Secondary metrics: `Accuracy`, per-class F1, confusion matrix
- Evaluation rule: no tuning on test set

## Run Matrix

| ID | Model | Input | Core Settings | Why |
| --- | --- | --- | --- | --- |
| B1 | XGBoost | Flat structural features | Tune on validation split only | Structural baseline without graphs |
| B2 | UniXcoder (fine-tuned) | Raw code tokens/text | 5 epochs, lr `3e-4`, wd `1e-3`, linear scheduler, effective batch `256` (use grad accumulation if needed) | Strong PLM baseline |
| M1 | GNN (main) | Graph representation (CPG/AST/CFG/PDG, per Shenghua's final design) | Single agreed architecture + training recipe | Proposed method |
| A1 | GNN (AST-only) | AST only | Same as M1 except graph scope | Ablation: effect of richer graph signals |
| A2 | GNN (untyped edges) | Same graph as M1 but remove edge types | Same as M1 | Ablation: effect of relation types |

## Comparison Questions

- `M1 vs B1`: does graph modeling beat flat structural features?
- `M1 vs B2`: can GNN match or beat a strong code encoder baseline?
- `M1 vs A1/A2`: which graph components are actually helping?

## Minimum Viable Plan (If Time Is Tight)

Run only:

1. `B1` (XGBoost)
2. `B2` (UniXcoder)
3. `M1` (GNN main)

Drop order when you need to cut work:

1. Drop `A2` first
2. Drop `A1` second
3. Keep `B1`, `B2`, `M1` unless timeline is critical

If timeline is very critical:

1. Run single seed for `B1`, `B2`, `M1` first
2. Add multi-seed reruns only for the top two models

## Suggested Outputs Per Run

- `metrics.json` (Macro-F1, Accuracy, per-class F1)
- `confusion_matrix.csv` (or image)
- `train_log.csv` (for traceability)
- Short note on runtime + peak GPU memory
