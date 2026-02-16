"""
Generate synthetic data for 'The City of Agents' research paper.
Simulates coding styles of different LLMs to test visualization hypothesis.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Constants
N_ROWS = 150
TASK_IDS = list(range(1, 11))
MODELS = ["GPT-4o", "Claude 3.5", "Llama 3"]
LANGUAGES = ["Python", "Java"]
JAVA_TOKEN_BONUS = 20

# Model-specific distribution parameters (mean, std) for (Max_Depth, Token_Count)
MODEL_PARAMS = {
    "GPT-4o": {"max_depth": (12, 2), "token_count": (100, 15)},      # Dense architect
    "Llama 3": {"max_depth": (5, 1), "token_count": (180, 30)},      # Sprawl architect
    "Claude 3.5": {"max_depth": (8, 2), "token_count": (140, 20)},   # Balanced architect
}

# Generate random assignments for each row
task_ids = np.random.choice(TASK_IDS, size=N_ROWS)
models = np.random.choice(MODELS, size=N_ROWS)
languages = np.random.choice(LANGUAGES, size=N_ROWS)

# Generate Token_Count and Max_Depth per row based on model
token_counts = np.zeros(N_ROWS)
max_depths = np.zeros(N_ROWS)

for i in range(N_ROWS):
    model = models[i]
    params = MODEL_PARAMS[model]
    max_depths[i] = np.random.normal(params["max_depth"][0], params["max_depth"][1])
    token_counts[i] = np.random.normal(params["token_count"][0], params["token_count"][1])

# Language adjustment: Java gets +20 Token_Count
java_mask = languages == "Java"
token_counts[java_mask] += JAVA_TOKEN_BONUS

# Clip to plausible positive values
max_depths = np.clip(max_depths, 1, None)
token_counts = np.clip(token_counts, 1, None)

# Round for readability (optional; keep floats if you prefer)
max_depths = np.round(max_depths, 1)
token_counts = np.round(token_counts, 1)

# Build DataFrame
df = pd.DataFrame({
    "Task_ID": task_ids,
    "Model": models,
    "Language": languages,
    "Token_Count": token_counts,
    "Max_Depth": max_depths,
})

# Reorder columns to match spec
df = df[["Task_ID", "Model", "Language", "Token_Count", "Max_Depth"]]

# Save to CSV
output_path = "simulation_data.csv"
df.to_csv(output_path, index=False)
print(f"Saved {N_ROWS} rows to {output_path}\n")
print("First 5 rows:")
print(df.head().to_string(index=False))
