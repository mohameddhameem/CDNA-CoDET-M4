# Full Prompt for Claude Code: Preliminary Structural Analysis

Copy and paste the entire prompt below into Claude Code:

---

## PROMPT START

I need to run a comprehensive preliminary analysis to determine if structural code features can distinguish between different LLMs (not just AI vs Human). This is for a research project on LLM code authorship attribution using Code Property Graphs and Graph Neural Networks.

### Context

We have structural features extracted from code samples in the CoDet-M4 dataset. The samples come from 6 sources:
- **Human** (human-written code)
- **GPT-4o** (or labeled as "gpt")
- **CodeLlama** (or labeled as "codellama")  
- **Llama3.1** (or labeled as "llama3.1")
- **Nxcode** (or labeled as "nxcode")
- **CodeQwen1.5** (or labeled as "qwen1.5")

### Research Question

**Can we distinguish which LLM generated code based on structural features alone?**

Important: We need to show LLMs differ from EACH OTHER, not just from humans. The human vs AI distinction is well-established — our contribution is multi-class LLM attribution.

### Tasks

Please perform the following analyses in order:

---

#### Task 1: Data Overview

1. Load the structural features dataset
2. Report sample counts per class (human, gpt, codellama, llama3.1, nxcode, qwen1.5)
3. List all available features
4. Check for missing values
5. Create two filtered datasets:
   - `df_all`: All 6 classes
   - `df_llm_only`: Exclude human samples (5 LLM classes only)

---

#### Task 2: LLM-Only Statistical Analysis (MOST IMPORTANT)

Using `df_llm_only` (human samples EXCLUDED):

1. Run Kruskal-Wallis H-test for each structural feature across the 5 LLM classes
2. Calculate effect size (eta-squared: η² = H / (n-1)) for each feature
3. Classify effect sizes: Small (<0.01), Medium (0.01-0.06), Large (>0.06)
4. For features with large effect size, run pairwise Mann-Whitney U tests with Bonferroni correction
5. Calculate Cliff's delta (effect size) for significant LLM pairs

**Output:**
- Table: Feature | H-statistic | p-value | η² | Effect Size Category
- Table: Feature | LLM Pair | Mann-Whitney U | Adjusted p-value | Cliff's Delta
- Summary: "X out of Y features show significant LLM-to-LLM differences with medium+ effect size"

---

#### Task 3: Dimensionality Reduction Visualization

1. Standardize all numeric features (z-score normalization)
2. Apply t-SNE (perplexity=30, random_state=42) to `df_all` and `df_llm_only`
3. Apply UMAP (n_neighbors=15, min_dist=0.1, random_state=42) to `df_all` and `df_llm_only`
4. Calculate silhouette score for both datasets using LLM labels

**Output:**
- Create a 2x2 figure:
  - Top-left: t-SNE (all 6 classes)
  - Top-right: t-SNE (LLM-only, 5 classes)
  - Bottom-left: UMAP (all 6 classes)
  - Bottom-right: UMAP (LLM-only, 5 classes)
- Use distinct, colorblind-friendly colors for each class
- Add legend and title with silhouette score
- Save as `clustering_visualization.png`

---

#### Task 4: Random Forest Classification (LLM-Only)

Using `df_llm_only`:

1. Prepare features (X) and labels (y)
2. Train RandomForestClassifier with 5-fold stratified cross-validation
3. Use parameters: n_estimators=100, random_state=42

**Output:**
- Report: Mean accuracy ± std, Mean macro F1 ± std
- Compare to random baseline (20% for 5 classes)
- Print: "Classifier achieves X.X% accuracy, which is Y.Yx above random baseline"
- Extract and plot feature importance (top 15 features)
- Save as `feature_importance.png`

---

#### Task 5: Confusion Matrix Analysis

1. Train final RF model on full `df_llm_only` dataset (or use cross-val predictions)
2. Generate confusion matrix (normalized by true labels)
3. Identify:
   - Which LLM pairs have highest confusion (most misclassified)?
   - Which LLMs are most distinctively recognized?
   - Are models from similar families confused more often?

**Output:**
- Heatmap of normalized confusion matrix
- Annotation showing confusion percentages
- Save as `confusion_matrix.png`
- Text summary of confusion patterns

---

#### Task 6: Distribution Visualization

For the top 5 most important features (from Task 4):

1. Create violin plots showing distribution per LLM (exclude human)
2. One subplot per feature
3. Add median markers
4. Order LLMs consistently across all plots

**Output:**
- Save as `distribution_plots.png`

---

#### Task 7: Summary Report

Generate a markdown summary with:

1. **Data Overview**: Sample counts, feature counts
2. **Key Statistical Findings**: 
   - How many features distinguish LLMs from each other?
   - Which features have largest effect sizes?
3. **Clustering Assessment**:
   - Silhouette scores
   - Visual assessment of cluster separation
4. **Classification Performance**:
   - Accuracy vs baseline
   - Most/least distinguishable LLMs
5. **Top Discriminative Features**: List top 5 with brief interpretation
6. **Conclusion**: 
   - Strong/Moderate/Weak evidence for our hypothesis?
   - Recommended framing for our research proposal

Save as `preliminary_findings.md`

---

#### Task 8: Generate Slide Content

Based on the analysis, write a paragraph suitable for our research proposal slide (replacing a placeholder). Use this format:

> "Our preliminary analysis of [N] Python samples shows that [FINDING about LLM-to-LLM differences]. A Random Forest classifier achieves [X]% accuracy on 5-class LLM attribution using only structural features—[Y]x above random baseline. Top discriminative features include [feature1], [feature2], and [feature3]. [One sentence about what this means for our CPG-GNN approach]."

---

### Additional Notes

- If any required libraries are missing, install them (sklearn, umap-learn, seaborn, etc.)
- Use seaborn or matplotlib for visualizations
- Ensure all plots have proper labels, titles, and are publication-ready
- If the dataset file path is unclear, ask me to provide it
- Save all output files to the current working directory

### What I'm Hoping to See (Success Criteria)

| Analysis | Good Sign | Concerning Sign |
|----------|-----------|-----------------|
| LLM-only Kruskal-Wallis | p < 0.001, η² > 0.06 for 5+ features | Small effects across all features |
| Silhouette Score (LLM-only) | > 0.1 | < 0.05 |
| RF Accuracy (LLM-only) | > 35% (1.75x baseline) | < 25% (near random) |
| Feature Importance | AST/structural features in top 5 | Only surface features matter |

Please proceed with the analysis and show me results as you go.

## PROMPT END

---

# Notes for Running This Prompt

1. **Before running**: Make sure Claude Code has access to your structural features CSV/dataframe
2. **If it asks for file path**: Provide the path to your extracted features data
3. **Expected runtime**: 5-15 minutes depending on dataset size
4. **Output files**: You should get 4 PNG files + 1 markdown summary

# Alternative: Running Step-by-Step

If you prefer to run tasks one at a time (useful for debugging), you can split at each "#### Task N" section. But I recommend running the full prompt first — Claude Code will handle the sequencing.
