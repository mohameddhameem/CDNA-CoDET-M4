# Preliminary Structural Analysis for CPG-GNN LLM Attribution

## Objective

Provide empirical evidence that **different LLMs produce structurally distinguishable code patterns** that can be captured by Code Property Graph (CPG) features. This supports our core hypothesis that explicit structural modeling via GNNs can identify LLM "fingerprints."

---

## Key Research Question

> Can we distinguish which LLM generated a piece of code based on structural features alone?

**NOT:** Can we distinguish human from AI code? (This is binary detection, not our focus)

---

## Analysis Requirements

### 1. LLM-to-LLM Discrimination (Primary Evidence)

**Goal:** Show that LLMs differ from *each other*, not just from humans.

**Analysis:**
- **Exclude human samples** and analyze only LLM-generated code
- Run Kruskal-Wallis test on LLM-only subset (5 classes: GPT-4o, CodeLlama, Llama3.1, Nxcode, CodeQwen1.5)
- Calculate pairwise effect sizes (Cohen's d or Cliff's delta) between each LLM pair
- Identify features with **high between-LLM variance**

**Success criteria:**
- At least 5+ features with p < 0.001 and medium-to-large effect size for LLM-only comparison
- Pairwise tests showing significant differences between specific LLM pairs

**Prompt for Claude Code:**
```
Analyze structural features for LLM-only samples (exclude human).
1. Run Kruskal-Wallis test across 5 LLM classes
2. Calculate pairwise Mann-Whitney U tests with Bonferroni correction
3. Report effect sizes (Cliff's delta) for each LLM pair
4. Identify which features best discriminate BETWEEN LLMs
```

---

### 2. Cluster Separability Analysis

**Goal:** Visualize whether LLMs occupy distinct regions in structural feature space.

**Analysis:**
- Standardize all structural features (z-score)
- Apply dimensionality reduction (t-SNE or UMAP) 
- Color points by LLM source
- Calculate silhouette score for LLM clusters

**Success criteria:**
- Visual separation of at least some LLM clusters
- Silhouette score > 0.1 (weak but present structure)
- Clear identification of which LLMs are most/least distinguishable

**Prompt for Claude Code:**
```
Create t-SNE and UMAP visualizations of structural features:
1. Standardize features, apply t-SNE (perplexity=30) and UMAP (n_neighbors=15)
2. Create scatter plots colored by LLM source (use distinct colors)
3. Calculate silhouette score for LLM labels
4. Create separate plot for LLM-only samples (exclude human)
5. Save plots as PNG files
```

---

### 3. Feature Importance for LLM Discrimination

**Goal:** Identify which structural features best separate LLMs from each other.

**Analysis:**
- Train a simple Random Forest classifier on LLM-only data
- Extract feature importances
- Cross-validate to ensure features generalize

**Success criteria:**
- Classification accuracy significantly above random baseline (>20% for 5 classes, baseline is 20%)
- Identify top-5 discriminative features
- Structural features (AST-based) should rank higher than surface features (whitespace, comments)

**Prompt for Claude Code:**
```
Train a Random Forest classifier on LLM-only samples:
1. Use structural features as input, LLM label as target
2. 5-fold stratified cross-validation
3. Report accuracy, macro F1, and confusion matrix
4. Plot feature importance (top 15 features)
5. Compare: Do AST/CFG features outperform surface features?
```

---

### 4. Confusion Pattern Analysis

**Goal:** Understand which LLMs are structurally similar/different.

**Analysis:**
- From the classifier above, examine confusion matrix
- Identify which LLM pairs are most confused
- This informs our understanding of structural similarity

**What to look for:**
- Are models from the same family confused? (e.g., Llama variants)
- Is GPT-4o distinctly different from open-source models?
- Which pairs will be hardest to distinguish?

**Prompt for Claude Code:**
```
Analyze confusion patterns:
1. Generate normalized confusion matrix from RF classifier
2. Create heatmap visualization
3. Report which LLM pairs have highest confusion rates
4. Calculate pairwise classification accuracy for each LLM pair
```

---

### 5. Distribution Overlap Visualization

**Goal:** Show how feature distributions differ across LLMs.

**Analysis:**
- For top-5 discriminative features, create ridge plots or violin plots
- Show distribution shape, not just median
- Highlight where distributions separate vs. overlap

**Prompt for Claude Code:**
```
Create distribution visualizations for top discriminative features:
1. Violin plots or ridge plots showing distribution per LLM
2. One subplot per feature, LLMs on y-axis
3. Add median markers and interquartile range
4. Use colorblind-friendly palette
5. Focus on LLM-only samples
```

---

## Interpreting Results

### Strong Evidence (Supports Hypothesis)
- [ ] LLM-only Kruskal-Wallis shows significant differences (p < 0.001)
- [ ] At least 3 features with large effect size for LLM discrimination
- [ ] t-SNE/UMAP shows visible clustering by LLM
- [ ] RF classifier achieves >40% accuracy (2x random baseline)
- [ ] Structural features (AST depth, node counts) rank in top-5 importance

### Moderate Evidence (Proceed with Caution)
- [ ] Significant but small effect sizes
- [ ] Partial clustering with high overlap
- [ ] 25-40% classifier accuracy
- [ ] Mixed feature importance (surface features compete with structural)

### Weak Evidence (Reconsider Approach)
- [ ] No significant LLM-to-LLM differences after excluding human
- [ ] No visible clustering in reduced space
- [ ] Classifier near random baseline (~20%)
- [ ] Surface features dominate structural features

---

## Data Requirements

**Dataset:** CoDet-M4 Python subset
**Classes:** GPT-4o, CodeLlama, Llama3.1, Nxcode, CodeQwen1.5, Human
**Sample size:** Report N per class

**Features to extract (if not already):**
- AST-based: node_count, max_depth, node_type_distribution, branching_factor
- CFG-based: cyclomatic_complexity, num_branches, max_nesting_depth
- PDG-based: data_dependency_count, control_dependency_count
- Surface: LOC, comment_ratio, whitespace_ratio, identifier_length_stats

---

## Output Artifacts

After running analysis, you should have:

1. `llm_only_statistical_tests.csv` - Kruskal-Wallis + pairwise results
2. `tsne_umap_clustering.png` - Dimensionality reduction plots
3. `feature_importance.png` - RF feature importance bar chart
4. `confusion_matrix.png` - Classification confusion heatmap
5. `distribution_plots.png` - Violin/ridge plots for top features
6. `preliminary_findings.md` - Summary of evidence

---

## Example Framing for Slide 18

Based on analysis results, update the placeholder with something like:

**If strong evidence:**
> "Our preliminary analysis of [N] Python samples shows that structural features differ significantly across LLMs even when excluding human code (Kruskal-Wallis p < 0.001). A Random Forest classifier achieves [X]% accuracy on 5-class LLM attribution using only structural features—[Y]x above random baseline. Top discriminative features include [feature1], [feature2], and [feature3], all of which are captured by CPG representations."

**If moderate evidence:**
> "Our preliminary analysis suggests structural differences exist between LLMs, though with substantial overlap. While human code is easily distinguishable, LLM-to-LLM separation shows [silhouette score] clustering strength. This motivates our use of GNNs to learn more complex structural patterns beyond hand-crafted features."

---

## Next Steps After Analysis

1. If evidence is strong → Proceed with confidence, update slide 18
2. If evidence is moderate → Emphasize that GNNs can learn patterns beyond hand-crafted features
3. If evidence is weak → Discuss with team; may need to reframe contribution as "exploration" rather than "we expect to beat baselines"
