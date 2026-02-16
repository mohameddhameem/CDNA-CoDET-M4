# %% [markdown]
# # City of Agents - Google Colab Setup
# 
# This notebook helps you set up the **City of Agents** project in Google Colab.
# 
# ## Quick Start
# 
# Run the cells below in order to:
# 1. Clone the GitHub repository
# 2. Install the package and dependencies
# 3. Verify the setup with imports
# 4. Run example visualizations

# %% [code]
# Clone the repository (replace URL with your GitHub repo)
import os

REPO_URL = "https://github.com/mohameddhameem/City-of-Agents-1.git"
REPO_NAME = "City-of-Agents-1"

if not os.path.exists(REPO_NAME):
    !git clone {REPO_URL}
    print(f"✅ Repository cloned")
else:
    print(f"✅ Repository already exists")

os.chdir(REPO_NAME)
print(f"📂 Current: {os.getcwd()}")

# %% [code]
# Install the package in editable mode
!pip install -e . -q
!pip install -q datasets transformers==4.47.0 ogb tqdm

print("✅ Package installed!")

# %% [code]
# Verify imports
try:
    import city_of_agents
    from city_of_agents import ast2pyg
    from city_of_agents.utils import pyg_creator
    from city_of_agents.builders import city
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")

# %% [markdown]
# ## Generate Sample Data

# %% [code]
# Generate simulation data
import numpy as np
import pandas as pd

np.random.seed(42)
N = 150

df = pd.DataFrame({
    "Task_ID": np.random.choice(range(1, 11), N),
    "Model": np.random.choice(["GPT-4o", "Claude 3.5", "Llama 3"], N),
    "Language": np.random.choice(["Python", "Java"], N),
    "Token_Count": np.clip(np.random.normal(120, 30, N), 1, None),
    "Max_Depth": np.clip(np.random.normal(8, 2, N), 1, None)
})

df.to_csv("simulation_data.csv", index=False)
print(f"✅ Generated {len(df)} data points")
df.head()

# %% [markdown]
# ## Create Visualization

# %% [code]
# Create 3D visualization
import plotly.graph_objects as go

df = pd.read_csv("simulation_data.csv")
MODEL_TO_X = {"GPT-4o": 0, "Claude 3.5": 4, "Llama 3": 8}
MODEL_COLORS = {"GPT-4o": "red", "Claude 3.5": "blue", "Llama 3": "green"}

np.random.seed(42)
df["x"] = df["Model"].map(MODEL_TO_X) + np.random.uniform(-0.3, 0.3, len(df))
df["y"] = df["Task_ID"] + np.random.uniform(-0.2, 0.2, len(df))

fig = go.Figure()
for model, color in MODEL_COLORS.items():
    sub = df[df["Model"] == model]
    fig.add_trace(go.Scatter3d(
        x=sub["x"], y=sub["y"], z=sub["Max_Depth"],
        mode="markers", marker=dict(size=8, color=color),
        name=model
    ))

fig.update_layout(
    title="City of Agents",
    scene=dict(xaxis_title="Model", yaxis_title="Task", zaxis_title="Depth"),
    height=600
)
fig.show()

# %% [markdown]
# ## Next Steps
# 
# - Explore other notebooks: `Experiment_1.ipynb`, `Experiment_Models.ipynb`
# - Use the golden dataset in `dataset/golden sample data/`
# - Try different builders: `city_buildings`, `city_research`, `city_v2`
# 
# ### Available Modules
# 
# **Core modules:**
# ```python
# from city_of_agents.ast2pyg import ast_to_pyg_data
# from city_of_agents.joern import JoernRunner
# from city_of_agents.codeclip import Pretrain, Downstream
# ```
# 
# **Builders:**
# ```python
# from city_of_agents.builders import city, city_v2, city_buildings, city_research
# ```
# 
# **Utilities:**
# ```python
# from city_of_agents.utils import clean_data, create_feature, pyg_creator
# ```
