"""
Build the 'City of Agents' 3D visualization (High-Contrast / Dark Mode).
Features:
- 'Lollipop' chart style for better depth perception.
- Floor shadows to anchor data points.
- Distinct 'District' zones for each Model.
"""

import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os
import numpy as np

# --- Configuration ---
# Neon/High-contrast palette for dark mode
MODEL_COLORS = {
    "GPT-4o": "#A56EFF",      # Neon Purple
    "Claude 3.5": "#FF6B6B",  # Neon Red/Orange
    "Llama 3": "#4CC9F0"      # Cyan/Blue
}
MODEL_X_POS = {"GPT-4o": 0, "Claude 3.5": 5, "Llama 3": 10}
LANG_SYMBOLS = {"Python": "circle", "Java": "diamond"}

# Load data
df = pd.read_csv("simulation_data.csv")

# --- Data Preparation ---
np.random.seed(42)
# Tighter jitter to keep districts clean
x_jitter = 0.25
y_jitter = 0.20

df["base_x"] = df["Model"].map(MODEL_X_POS)
df["x"] = df["base_x"] + np.random.uniform(-x_jitter, x_jitter, size=len(df))
df["y"] = df["Task_ID"] + np.random.uniform(-y_jitter, y_jitter, size=len(df))
df["z"] = df["Max_Depth"]

# Scale size: Standardize min/max to a readable range (8px to 25px)
tc_min, tc_max = df["Token_Count"].min(), df["Token_Count"].max()
df["size_scaled"] = df["Token_Count"].apply(
    lambda x: 8 + 17 * (x - tc_min) / (tc_max - tc_min)
)

# Create the Figure
fig = go.Figure()

# 1. Draw "District Floors" (Rectangles on the ground) to group models visually
# Mesh3d needs triangular faces: quad = two triangles (0,1,2) and (0,2,3)
for model, center_x in MODEL_X_POS.items():
    x_verts = [center_x - 2, center_x - 2, center_x + 2, center_x + 2]
    y_verts = [0, 11, 11, 0]
    z_verts = [0, 0, 0, 0]
    fig.add_trace(go.Mesh3d(
        x=x_verts,
        y=y_verts,
        z=z_verts,
        i=[0, 0], j=[1, 3], k=[2, 2],  # two triangles
        color=MODEL_COLORS[model],
        opacity=0.1,
        name=f"{model} Zone",
        hoverinfo="skip",
        showlegend=False,
    ))

# 2. Build the visual elements for each point
for model in MODEL_COLORS.keys():
    for lang in LANG_SYMBOLS.keys():
        mask = (df["Model"] == model) & (df["Language"] == lang)
        if not mask.any():
            continue

        sub = df[mask]
        color = MODEL_COLORS[model]
        symbol = LANG_SYMBOLS[lang]

        # A. The Pillars (Stems of the lollipops)
        x_lines = []
        y_lines = []
        z_lines = []
        for _, row in sub.iterrows():
            x_lines.extend([row["x"], row["x"], None])
            y_lines.extend([row["y"], row["y"], None])
            z_lines.extend([0, row["z"], None])

        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode="lines",
            line=dict(color=color, width=3),
            opacity=0.4,
            showlegend=False,
            hoverinfo="skip",
        ))

        # B. The Floor Shadows (Anchors)
        fig.add_trace(go.Scatter3d(
            x=sub["x"], y=sub["y"], z=[0] * len(sub),
            mode="markers",
            marker=dict(
                size=4,
                color="gray",
                symbol="circle",
                opacity=0.5,
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

        # C. The Data Markers (The "Heads" of the lollipops)
        fig.add_trace(go.Scatter3d(
            x=sub["x"],
            y=sub["y"],
            z=sub["z"],
            mode="markers",
            marker=dict(
                size=sub["size_scaled"],
                color=color,
                symbol=symbol,
                line=dict(width=1, color="white"),
                opacity=1.0,
            ),
            name=f"{model} · {lang}",
            text=[
                f"<b>{row['Model']}</b> ({row['Language']})<br>"
                f"Task ID: {row['Task_ID']}<br>"
                f"Complexity (Depth): {row['z']:.1f}<br>"
                f"Tokens: {row['Token_Count']:.0f}"
                for _, row in sub.iterrows()
            ],
            hoverinfo="text",
        ))

# --- Layout Styling ---
fig.update_layout(
    title=dict(
        text="<b>CITY OF AGENTS</b><br><sup>Height = Complexity · Size = Token Cost · Zones = Models</sup>",
        font=dict(size=20, color="white"),
        x=0.5,
    ),
    template="plotly_dark",
    scene=dict(
        xaxis=dict(
            title="",
            tickvals=list(MODEL_X_POS.values()),
            ticktext=list(MODEL_X_POS.keys()),
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="#444",
            showbackground=False,
            tickfont=dict(size=14, color="white"),
        ),
        yaxis=dict(
            title="Task ID (1-10)",
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="#444",
            showbackground=False,
            tickfont=dict(color="gray"),
        ),
        zaxis=dict(
            title="Max Depth",
            backgroundcolor="rgba(20,20,20,0.8)",
            gridcolor="#444",
            tickfont=dict(color="gray"),
        ),
        aspectmode="manual",
        aspectratio=dict(x=2, y=1.5, z=0.8),
        camera=dict(
            eye=dict(x=0, y=-1.8, z=0.6),
            center=dict(x=0, y=0, z=-0.2),
        ),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=0.05,
        xanchor="center", x=0.5,
        bgcolor="rgba(0,0,0,0.5)",
        font=dict(size=12),
    ),
    margin=dict(l=0, r=0, b=0, t=60),
)

# Save
out_path = os.path.abspath("city_map_v2.html")
fig.write_html(out_path, auto_open=False)
webbrowser.open(out_path)
print(f"City built. Opened: {out_path}")
