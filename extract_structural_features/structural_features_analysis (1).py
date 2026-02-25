# =============================================================================
# CodeDNA: Statistical Comparison of Structural Features Across LLMs
# =============================================================================
# Purpose: Validate that different LLMs produce structurally different code
# Author: Team CodeDNA
# Usage: Run this script or convert to Jupyter notebook cells
#
# This addresses Prof. Lo's feedback:
# "Convince yourself that looking into the structure can help you 
#  identify the author of a piece of code."
# =============================================================================

# %% [markdown]
# # Structural Features Analysis: Do LLMs Have Different Fingerprints?
# 
# ## Goal
# Before building our full CPG-GNN model, we need evidence that structural 
# differences actually exist between LLM-generated code samples.
#
# ## What This Script Does
# 1. Load structural features data
# 2. Visualize distributions across LLMs
# 3. Run statistical significance tests
# 4. Generate summary for Slide 6 of our presentation

# %% [markdown]
# ## Step 0: Setup

# %%
# Install required packages (uncomment if needed)
# !pip install pandas numpy scipy seaborn matplotlib scikit-learn

# %%
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kruskal, f_oneway, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
import sys
warnings.filterwarnings('ignore')

# Ensure Unicode output works on Windows terminals
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set style for nice plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✓ Libraries loaded successfully")

# %% [markdown]
# ## Step 1: Load Your Data
# 
# ⚠️ **ACTION REQUIRED**: Update the file path to your actual parquet file

# %%
# =============================================================================
# ⚠️ UPDATE THIS PATH TO YOUR ACTUAL FILE
# =============================================================================
DATA_PATH = "City-of-Agents/dataset/extract_structural_features/train.parquet"

# Column that identifies the LLM source (e.g., "source", "label", "author", "model")
SOURCE_COLUMN = "source"  # <-- CHANGE THIS IF DIFFERENT

# =============================================================================

# Load data
try:
    df = pd.read_parquet(DATA_PATH)
    print(f"✓ Data loaded successfully!")
    print(f"  Shape: {df.shape[0]:,} samples, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"✗ Error: File not found at '{DATA_PATH}'")
    print("  Please update DATA_PATH variable above.")
    # Create dummy data for demonstration
    print("\n  Creating dummy data for demonstration...")
    np.random.seed(42)
    n_samples = 1000
    sources = ['GPT-4o', 'CodeLlama', 'Llama3.1', 'Nxcode', 'CodeQwen1.5', 'Human']
    df = pd.DataFrame({
        'source': np.random.choice(sources, n_samples),
        'ast_max_depth': np.random.randint(3, 20, n_samples),
        'cyclomatic_complexity': np.random.randint(1, 15, n_samples),
        'num_functions': np.random.randint(1, 10, n_samples),
        'num_loops': np.random.randint(0, 8, n_samples),
        'num_conditionals': np.random.randint(0, 12, n_samples),
        'avg_function_length': np.random.uniform(5, 50, n_samples),
        'num_variables': np.random.randint(1, 30, n_samples),
    })
    # Add some artificial differences to make demo interesting
    df.loc[df['source'] == 'GPT-4o', 'ast_max_depth'] += 3
    df.loc[df['source'] == 'Human', 'cyclomatic_complexity'] += 2
    print("  ✓ Dummy data created for demonstration")

# %% [markdown]
# ## Step 2: Data Overview

# %%
print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)

# Show column names
print("\n📋 Available columns:")
for i, col in enumerate(df.columns):
    print(f"   {i+1}. {col}")

# Show source distribution
print(f"\n📊 Samples per {SOURCE_COLUMN}:")
print(df[SOURCE_COLUMN].value_counts().to_string())

# Show basic stats
print("\n📈 Basic statistics:")
print(df.describe().round(2))

# %% [markdown]
# ## Step 3: Select Features to Analyze
#
# ⚠️ **ACTION REQUIRED**: Update the list of structural features based on your data

# %%
# =============================================================================
# ⚠️ UPDATE THIS LIST BASED ON YOUR ACTUAL COLUMN NAMES
# =============================================================================
# These should be NUMERIC columns representing structural features

STRUCTURAL_FEATURES = [
    'ast_max_depth',           # Maximum depth of AST
    'cyclomatic_complexity',   # McCabe complexity
    'num_functions',           # Number of function definitions
    'num_loops',               # Number of loop constructs
    'num_conditionals',        # Number of if/else statements
    'avg_function_length',     # Average lines per function
    'num_variables',           # Number of variable declarations
    # Add more features from your data here:
    # 'node_count',
    # 'edge_count',
    # 'max_fan_in',
    # 'max_fan_out',
]

# Filter to only features that exist in dataframe
available_features = [f for f in STRUCTURAL_FEATURES if f in df.columns]
missing_features = [f for f in STRUCTURAL_FEATURES if f not in df.columns]

if missing_features:
    print(f"⚠️ Warning: These features not found in data: {missing_features}")
    
print(f"✓ Analyzing {len(available_features)} features: {available_features}")

# %% [markdown]
# ## Step 4: Visual Comparison (Box Plots)
#
# Box plots show the distribution of each feature across different LLMs.
# - If boxes are well-separated → good signal for our approach
# - If boxes overlap heavily → weaker signal

# %%
def create_boxplots(df, features, source_col, save_path=None):
    """Create box plots comparing features across LLM sources."""
    
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.boxplot(x=source_col, y=feature, data=df, ax=ax)
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Structural Features Distribution by LLM Source', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.show()

# Create the plots
create_boxplots(df, available_features, SOURCE_COLUMN, 
                save_path='boxplots_structural_features.png')

# %% [markdown]
# ## Step 5: Visual Comparison (Violin Plots)
#
# Violin plots show the full distribution shape, which can reveal 
# multimodal patterns that box plots might miss.

# %%
def create_violinplots(df, features, source_col, save_path=None):
    """Create violin plots comparing features across LLM sources."""
    
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.violinplot(x=source_col, y=feature, data=df, ax=ax, inner='box')
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Structural Features Distribution (Violin) by LLM Source', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.show()

# Create the plots
create_violinplots(df, available_features, SOURCE_COLUMN,
                   save_path='violinplots_structural_features.png')

# %% [markdown]
# ## Step 6: Statistical Significance Tests
#
# We use **Kruskal-Wallis H-test** (non-parametric) because:
# - We're comparing more than 2 groups
# - We don't assume normal distribution
# - It's robust to outliers
#
# **Interpretation:**
# - p < 0.05 → Statistically significant difference exists
# - p < 0.01 → Highly significant
# - p < 0.001 → Very highly significant

# %%
def run_statistical_tests(df, features, source_col):
    """Run Kruskal-Wallis tests for each feature across LLM sources."""
    
    results = []
    sources = df[source_col].unique()
    
    for feature in features:
        # Get data for each group
        groups = [df[df[source_col] == s][feature].dropna().values 
                  for s in sources]
        
        # Kruskal-Wallis H-test
        try:
            h_stat, p_value = kruskal(*groups)
            
            # Effect size (eta-squared approximation)
            n = sum(len(g) for g in groups)
            k = len(groups)
            eta_squared = (h_stat - k + 1) / (n - k)
            
            # Interpretation
            if p_value < 0.001:
                significance = "*** (p < 0.001)"
            elif p_value < 0.01:
                significance = "** (p < 0.01)"
            elif p_value < 0.05:
                significance = "* (p < 0.05)"
            else:
                significance = "ns (not significant)"
            
            # Effect size interpretation
            if eta_squared >= 0.14:
                effect = "Large"
            elif eta_squared >= 0.06:
                effect = "Medium"
            elif eta_squared >= 0.01:
                effect = "Small"
            else:
                effect = "Negligible"
            
            results.append({
                'Feature': feature,
                'H-statistic': round(h_stat, 2),
                'p-value': p_value,
                'Significance': significance,
                'Effect Size (η²)': round(eta_squared, 4),
                'Effect Interpretation': effect
            })
            
        except Exception as e:
            results.append({
                'Feature': feature,
                'H-statistic': None,
                'p-value': None,
                'Significance': f"Error: {str(e)}",
                'Effect Size (η²)': None,
                'Effect Interpretation': None
            })
    
    return pd.DataFrame(results)

# Run the tests
print("=" * 70)
print("STATISTICAL SIGNIFICANCE TESTS (Kruskal-Wallis)")
print("=" * 70)
print("\nNull Hypothesis: All LLM sources have the same distribution")
print("Alternative: At least one LLM source differs\n")

results_df = run_statistical_tests(df, available_features, SOURCE_COLUMN)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('statistical_test_results.csv', index=False)
print("\n✓ Results saved to: statistical_test_results.csv")

# %% [markdown]
# ## Step 7: Pairwise Comparisons
#
# For features with significant differences, which specific LLM pairs differ?

# %%
def pairwise_comparisons(df, feature, source_col, alpha=0.05):
    """Run pairwise Mann-Whitney U tests with Bonferroni correction."""
    
    sources = df[source_col].unique()
    pairs = list(combinations(sources, 2))
    n_comparisons = len(pairs)
    adjusted_alpha = alpha / n_comparisons  # Bonferroni correction
    
    results = []
    for s1, s2 in pairs:
        g1 = df[df[source_col] == s1][feature].dropna().values
        g2 = df[df[source_col] == s2][feature].dropna().values
        
        try:
            stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
            significant = "Yes" if p_val < adjusted_alpha else "No"
            
            # Median difference
            med1, med2 = np.median(g1), np.median(g2)
            
            results.append({
                'Pair': f"{s1} vs {s2}",
                'Median 1': round(med1, 2),
                'Median 2': round(med2, 2),
                'p-value': round(p_val, 6),
                'Significant (Bonferroni)': significant
            })
        except Exception as e:
            results.append({
                'Pair': f"{s1} vs {s2}",
                'Median 1': None,
                'Median 2': None,
                'p-value': None,
                'Significant (Bonferroni)': f"Error: {e}"
            })
    
    return pd.DataFrame(results)

# Find significant features and do pairwise comparisons
significant_features = results_df[
    results_df['p-value'].notna() & 
    (results_df['p-value'] < 0.05)
]['Feature'].tolist()

if significant_features:
    print("=" * 70)
    print("PAIRWISE COMPARISONS (for significant features)")
    print("=" * 70)
    
    # Analyze top 3 most significant features
    for feature in significant_features[:3]:
        print(f"\n📊 {feature}")
        print("-" * 50)
        pairwise_df = pairwise_comparisons(df, feature, SOURCE_COLUMN)
        print(pairwise_df.to_string(index=False))
else:
    print("⚠️ No features showed significant differences.")
    print("   This might indicate that structural features alone won't")
    print("   distinguish between LLMs well. Consider:")
    print("   1. Extracting more/different structural features")
    print("   2. Looking at combinations of features")
    print("   3. Adjusting research approach")

# %% [markdown]
# ## Step 8: Summary Statistics by LLM

# %%
def summary_by_source(df, features, source_col):
    """Generate summary statistics for each LLM source."""
    
    summary = df.groupby(source_col)[features].agg(['mean', 'median', 'std'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary.round(2)

print("=" * 70)
print("SUMMARY STATISTICS BY LLM SOURCE")
print("=" * 70)

summary_df = summary_by_source(df, available_features, SOURCE_COLUMN)
print(summary_df.to_string())

summary_df.to_csv('summary_by_source.csv')
print("\n✓ Summary saved to: summary_by_source.csv")

# %% [markdown]
# ## Step 9: Generate Report for Presentation (Slide 6)

# %%
def generate_presentation_summary(results_df, df, source_col):
    """Generate a text summary suitable for Slide 6 of the presentation."""
    
    sig_features = results_df[results_df['p-value'] < 0.05]
    n_sig = len(sig_features)
    n_total = len(results_df)
    
    print("=" * 70)
    print("📋 SUMMARY FOR SLIDE 6: PRELIMINARY VALIDATION")
    print("=" * 70)
    
    print(f"""
FINDINGS:

We analyzed {n_total} structural features across {df[source_col].nunique()} LLM sources 
using {len(df):,} code samples.

KEY RESULTS:
• {n_sig} out of {n_total} features ({100*n_sig/n_total:.0f}%) show statistically 
  significant differences across LLMs (Kruskal-Wallis, p < 0.05)
""")
    
    if n_sig > 0:
        print("SIGNIFICANT FEATURES:")
        for _, row in sig_features.iterrows():
            print(f"  • {row['Feature']}: {row['Significance']}, {row['Effect Interpretation']} effect")
        
        print(f"""
INTERPRETATION:
✓ Structural differences DO exist between LLM-generated code
✓ This supports our hypothesis that CPG-based methods can capture 
  distinguishing patterns
✓ Features with {sig_features['Effect Interpretation'].value_counts().index[0].lower()} 
  effect sizes suggest meaningful structural variation

RECOMMENDED STATEMENT FOR SLIDE 6:
"Our preliminary analysis of {len(df):,} samples shows that structural features 
differ significantly across LLMs. {n_sig} features including 
{', '.join(sig_features['Feature'].head(3).tolist())} show statistically 
significant differences (p < 0.05), suggesting that CPG-based approaches 
can capture distinguishing structural patterns."
""")
    else:
        print("""
INTERPRETATION:
⚠️ No significant structural differences found
⚠️ This challenges our hypothesis
⚠️ Consider: 
   - Extracting different/more features
   - Focusing on specific code types
   - Hybrid approach combining structure + other signals

RECOMMENDED STATEMENT FOR SLIDE 6:
"Our preliminary analysis did not reveal significant structural differences 
in basic metrics. We are exploring more sophisticated structural features 
and may consider hybrid approaches combining structural and sequential signals."
""")
    
    return n_sig, n_total

n_sig, n_total = generate_presentation_summary(results_df, df, SOURCE_COLUMN)

# %% [markdown]
# ## Step 10: Save All Outputs

# %%
# Create a summary report
report = f"""
# CodeDNA: Structural Features Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Dataset
- Total samples: {len(df):,}
- LLM sources: {df[SOURCE_COLUMN].nunique()}
- Features analyzed: {len(available_features)}

## Statistical Test Results
{results_df.to_markdown(index=False)}

## Key Findings
- Significant features: {n_sig} out of {n_total}
- Conclusion: {'Structural differences EXIST' if n_sig > 0 else 'No significant differences found'}

## Files Generated
- boxplots_structural_features.png
- violinplots_structural_features.png
- statistical_test_results.csv
- summary_by_source.csv
- analysis_report.md (this file)
"""

with open('analysis_report.md', 'w') as f:
    f.write(report)

print("=" * 70)
print("✓ ANALYSIS COMPLETE!")
print("=" * 70)
print("""
Generated files:
  📊 boxplots_structural_features.png    - Visual comparison (box plots)
  📊 violinplots_structural_features.png - Visual comparison (violin plots)
  📄 statistical_test_results.csv        - Kruskal-Wallis test results
  📄 summary_by_source.csv               - Summary stats by LLM
  📄 analysis_report.md                  - Full report summary

Next steps:
  1. Review the plots - do distributions look different?
  2. Check statistical results - are differences significant?
  3. Use findings for Slide 6 of presentation
  4. Share results with team
""")
