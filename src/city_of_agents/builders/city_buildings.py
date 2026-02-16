"""
Build the 'City of Agents' 3D Visualization (Solid Buildings Edition).
Constructs 3D rectangular prisms (meshes) for each data point.
"""

import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os
import numpy as np

# --- Configuration ---
# Cyberpunk / Neon City Palette
MODEL_COLORS = {
    "GPT-4o": "#A56EFF",      # Purple
    "Claude 3.5": "#FF6B6B",  # Red/Orange
    "Llama 3": "#00F0FF",     # Cyan
}
# Separate the "districts" along the X axis
MODEL_X_POS = {"GPT-4o": 0, "Claude 3.5": 6, "Llama 3": 12}

# Load data
df = pd.read_csv("simulation_data.csv")

# --- Data Preparation ---
np.random.seed(42)
x_jitter = 0.3
y_jitter = 0.3

# Calculate coordinates
df["base_x"] = df["Model"].map(MODEL_X_POS)
df["x"] = df["base_x"] + np.random.uniform(-x_jitter, x_jitter, size=len(df))
df["y"] = df["Task_ID"] + np.random.uniform(-y_jitter, y_jitter, size=len(df))
df["height"] = df["Max_Depth"]

# Calculate building "footprint" (width) based on Token Count
# We scale it to be between 0.3 (thin tower) and 0.8 (thick tower)
tc_min, tc_max = df["Token_Count"].min(), df["Token_Count"].max()
df["width"] = df["Token_Count"].apply(
    lambda t: 0.3 + 0.5 * (t - tc_min) / (tc_max - tc_min)
)

# --- Helper: The Building Constructor ---
def create_city_mesh(subset, color, name):
    """
    Generates a single Mesh3d trace containing ALL buildings for a specific model.
    This is much faster than creating 100 separate traces.
    """
    x_coords = []
    y_coords = []
    z_coords = []
    i_indices = []
    j_indices = []
    k_indices = []

    # We build each building by defining 8 vertices (corners) and 12 triangles (faces)
    current_vertex_index = 0

    for _, row in subset.iterrows():
        x, y, h, w = row["x"], row["y"], row["height"], row["width"]
        half_w = w / 2

        # Define 8 corners of the building
        # Bottom 4 (z=0), Top 4 (z=h)
        # Order: BL, BR, TR, TL (Bottom Left, Bottom Right, etc.)

        # Bottom face (z=0)
        x_coords.extend([x - half_w, x + half_w, x + half_w, x - half_w])
        y_coords.extend([y - half_w, y - half_w, y + half_w, y + half_w])
        z_coords.extend([0, 0, 0, 0])

        # Top face (z=h)
        x_coords.extend([x - half_w, x + half_w, x + half_w, x - half_w])
        y_coords.extend([y - half_w, y - half_w, y + half_w, y + half_w])
        z_coords.extend([h, h, h, h])

        # Define the triangles connecting these vertices (Mesh topology)
        # Vertices 0-3: Bottom, 4-7: Top
        base = current_vertex_index

        # Top Face (4,5,6,7) -> Triangles (4,5,6) and (4,6,7)
        i_indices.extend([base + 4, base + 4])
        j_indices.extend([base + 5, base + 6])
        k_indices.extend([base + 6, base + 7])

        # Bottom Face (0,1,2,3) -> Triangles (0,2,3) and (0,1,2)
        i_indices.extend([base + 0, base + 0])
        j_indices.extend([base + 2, base + 1])
        k_indices.extend([base + 3, base + 2])

        # Front (0,1,5,4)
        i_indices.extend([base + 0, base + 0])
        j_indices.extend([base + 1, base + 5])
        k_indices.extend([base + 5, base + 4])

        # Right (1,2,6,5)
        i_indices.extend([base + 1, base + 1])
        j_indices.extend([base + 2, base + 6])
        k_indices.extend([base + 6, base + 5])

        # Back (2,3,7,6)
        i_indices.extend([base + 2, base + 2])
        j_indices.extend([base + 3, base + 7])
        k_indices.extend([base + 7, base + 6])

        # Left (3,0,4,7)
        i_indices.extend([base + 3, base + 3])
        j_indices.extend([base + 0, base + 4])
        k_indices.extend([base + 4, base + 7])

        current_vertex_index += 8

    return go.Mesh3d(
        x=x_coords, y=y_coords, z=z_coords,
        i=i_indices, j=j_indices, k=k_indices,
        color=color,
        opacity=0.9,
        flatshading=True,
        name=name,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
        hoverinfo="skip",
    )


# --- Build the City ---
fig = go.Figure()

# 1. Generate Building Meshes per Model
for model, color in MODEL_COLORS.items():
    subset = df[df["Model"] == model]
    mesh_trace = create_city_mesh(subset, color, model)
    fig.add_trace(mesh_trace)

    # 2. Add "Ghost" Scatter points for Hover info (invisible but interactive)
    fig.add_trace(go.Scatter3d(
        x=subset["x"],
        y=subset["y"],
        z=subset["height"],
        mode="markers",
        marker=dict(size=1, color=color, opacity=0),
        name=model,
        text=[
            f"<b>{row['Model']}</b> ({row['Language']})<br>"
            f"Task: {row['Task_ID']}<br>"
            f"Height (Depth): {row['Max_Depth']:.1f}<br>"
            f"Width (Tokens): {row['Token_Count']:.0f}"
            for _, row in subset.iterrows()
        ],
        hoverinfo="text",
    ))

# 3. Add Ground Tiles (The District Floors)
# Mesh3d needs triangular faces: quad = two triangles (0,1,2) and (0,2,3)
for model, center_x in MODEL_X_POS.items():
    fig.add_trace(go.Mesh3d(
        x=[center_x - 2.5, center_x - 2.5, center_x + 2.5, center_x + 2.5],
        y=[0, 11, 11, 0],
        z=[-0.1, -0.1, -0.1, -0.1],
        i=[0, 0], j=[1, 3], k=[2, 2],
        color=MODEL_COLORS[model],
        opacity=0.2,
        hoverinfo="skip",
        showlegend=False,
    ))

# --- Layout Configuration ---
fig.update_layout(
    title=dict(
        text="<b>CITY OF AGENTS</b><br><sup>Each Building is a Task Execution · Height = Complexity · Thickness = Cost</sup>",
        font=dict(size=24, color="white", family="Arial"),
        y=0.95,
    ),
    template="plotly_dark",
    scene=dict(
        xaxis=dict(
            title="",
            tickvals=list(MODEL_X_POS.values()),
            ticktext=list(MODEL_X_POS.keys()),
            backgroundcolor="rgba(10,10,10,1)",
            gridcolor="#333",
            showbackground=True,
            tickfont=dict(size=14, color="white"),
        ),
        yaxis=dict(
            title="Task ID",
            backgroundcolor="rgba(10,10,10,1)",
            gridcolor="#333",
            showbackground=True,
            tickfont=dict(color="gray"),
        ),
        zaxis=dict(
            title="Structural Depth",
            backgroundcolor="rgba(10,10,10,1)",
            gridcolor="#333",
            showbackground=True,
            tickfont=dict(color="gray"),
        ),
        aspectmode="manual",
        aspectratio=dict(x=2.2, y=1.5, z=0.8),
        camera=dict(
            eye=dict(x=0, y=-2.0, z=0.8),
            center=dict(x=0, y=0, z=-0.2),
        ),
    ),
    margin=dict(l=0, r=0, b=0, t=80),
    legend=dict(
        yanchor="top", y=0.95,
        xanchor="right", x=0.95,
        bgcolor="rgba(0,0,0,0.5)",
    ),
)

# Save
out_path = os.path.abspath("city_map_buildings.html")
fig.write_html(out_path, auto_open=False)
webbrowser.open(out_path)
print(f"City built. Opened: {out_path}")
