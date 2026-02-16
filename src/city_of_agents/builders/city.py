"""
Build the 'City of Agents' 3D visualization from simulation_data.csv.
Maps software metrics to city dimensions with pillars (drop lines) and styled markers.
"""

import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os
import numpy as np

# Load data
df = pd.read_csv("simulation_data.csv")

# Model -> numeric X position (spread out so "districts" don't overlap) and color
MODEL_TO_X = {"GPT-4o": 0, "Claude 3.5": 4, "Llama 3": 8}
MODEL_COLORS = {"GPT-4o": "red", "Claude 3.5": "blue", "Llama 3": "green"}
LANG_SYMBOLS = {"Python": "circle", "Java": "diamond"}

# Add numeric X and Y with larger jitter so pillars are easier to distinguish
np.random.seed(42)
x_jitter = 0.35
y_jitter = 0.2
df["x"] = df["Model"].map(MODEL_TO_X) + np.random.uniform(-x_jitter, x_jitter, size=len(df))
df["y"] = df["Task_ID"] + np.random.uniform(-y_jitter, y_jitter, size=len(df))
df["z"] = df["Max_Depth"]

# Scale Token_Count to marker size — wider range so size differences are easier to read
tc_min, tc_max = df["Token_Count"].min(), df["Token_Count"].max()
df["size"] = 6 + 22 * (df["Token_Count"] - tc_min) / (tc_max - tc_min) if tc_max > tc_min else 14

# Build drop lines: for each point, line from (x, y, 0) to (x, y, z); separate segments with NaN
n = len(df)
x_line = np.full(n * 3, np.nan)
y_line = np.full(n * 3, np.nan)
z_line = np.full(n * 3, np.nan)
for i in range(n):
    r = i * 3
    x_line[r] = df["x"].iloc[i]
    x_line[r + 1] = df["x"].iloc[i]
    y_line[r] = df["y"].iloc[i]
    y_line[r + 1] = df["y"].iloc[i]
    z_line[r] = 0
    z_line[r + 1] = df["z"].iloc[i]

# Trace: drop lines (pillars) — thin and subtle so markers stand out
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode="lines",
            line=dict(color="rgba(0.55, 0.55, 0.6, 0.25)", width=1),
            name="Pillars",
            showlegend=False,
            hoverinfo="skip",
        )
    ]
)

# Scatter traces: one per (Model, Language) for color + symbol
for model in ["GPT-4o", "Claude 3.5", "Llama 3"]:
    for lang in ["Python", "Java"]:
        mask = (df["Model"] == model) & (df["Language"] == lang)
        if not mask.any():
            continue
        sub = df.loc[mask]
        fig.add_trace(
            go.Scatter3d(
                x=sub["x"],
                y=sub["y"],
                z=sub["z"],
                mode="markers",
                marker=dict(
                    size=sub["size"],
                    color=MODEL_COLORS[model],
                    symbol=LANG_SYMBOLS[lang],
                    line=dict(width=1.2, color="rgba(0.2,0.2,0.2,0.8)"),
                    opacity=0.95,
                ),
                name=f"{model} ({lang})",
                text=[
                    f"<b>{model}</b> · {lang}<br>Task {t} · Max_Depth: {z:.1f}<br>Token_Count: {tc:.0f}"
                    for t, z, tc in zip(sub["Task_ID"], sub["z"], sub["Token_Count"])
                ],
                hoverinfo="text",
            )
        )

# Layout: clearer labels, default view that separates the three "districts"
fig.update_layout(
    title=dict(
        text="The City of Agents: Cross-Language Fingerprints<br><sub>Marker size = Token count · Circle = Python, Diamond = Java · Height = Max_Depth (structural complexity)</sub>",
        font=dict(size=18),
        x=0.5,
        xanchor="center",
    ),
    scene=dict(
        xaxis=dict(
            title="Model",
            tickvals=[0, 4, 8],
            ticktext=list(MODEL_TO_X.keys()),
            title_font=dict(size=14),
            tickfont=dict(size=12),
            backgroundcolor="rgba(0.97, 0.97, 0.98, 0.95)",
            gridcolor="rgba(0.85, 0.85, 0.88, 0.6)",
        ),
        yaxis=dict(
            title="Task ID",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            backgroundcolor="rgba(0.97, 0.97, 0.98, 0.95)",
            gridcolor="rgba(0.85, 0.85, 0.88, 0.6)",
        ),
        zaxis=dict(
            title="Max_Depth",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            backgroundcolor="rgba(0.97, 0.97, 0.98, 0.95)",
            gridcolor="rgba(0.85, 0.85, 0.88, 0.6)",
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.5, y=1.4, z=1.1), center=dict(x=0, y=0, z=0)),
    ),
    margin=dict(l=0, r=0, b=0, t=80),
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=1.01,
        font=dict(size=11),
        title=dict(text="Model · Language", font=dict(size=12)),
    ),
    paper_bgcolor="white",
)

# Save and open in browser
out_path = os.path.abspath("city_map.html")
fig.write_html(out_path, auto_open=False)
webbrowser.open(out_path)
print(f"Saved and opened: {out_path}")
