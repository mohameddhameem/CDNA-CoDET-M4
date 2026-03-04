"""
Build the 'City of Agents' 3D Visualization (Research Edition).
- Visuals: Solid 'Buildings' (Meshes) representing code structure.
- Interaction: Improved 'Hitbox' system for reliable hovering.
- Metrics: Adds 'Structural Efficiency' to analyze Model Verbosity vs. Complexity.
"""

import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os
import numpy as np

# --- 1. Research Configuration ---
# Color by model + language so Java vs Python are visually distinct
# (Model, Language) -> hex; Java = richer shade, Python = lighter tint
MODEL_LANG_COLORS = {
    ("GPT-4o", "Java"): "#8B5CF6",
    ("GPT-4o", "Python"): "#C4B5FD",
    ("Claude 3.5", "Java"): "#DC2626",
    ("Claude 3.5", "Python"): "#FCA5A5",
    ("Llama 3", "Java"): "#0891B2",
    ("Llama 3", "Python"): "#67E8F9",
}
MODEL_COLORS = {
    "GPT-4o": "#A56EFF",
    "Claude 3.5": "#FF6B6B",
    "Llama 3": "#00F0FF",
}
MODEL_X_POS = {"GPT-4o": 0, "Claude 3.5": 6, "Llama 3": 12}

# Load Data
df = pd.read_csv("simulation_data.csv")

# --- 2. Metric Engineering (The Research Logic) ---
# Question: Which model achieves high structural complexity (depth) with low cost (tokens)?

# a. Normalize dimensions
np.random.seed(42)
df["base_x"] = df["Model"].map(MODEL_X_POS)
# Jitter X/Y to spread buildings out like a real city map
df["x"] = df["base_x"] + np.random.uniform(-0.4, 0.4, size=len(df))
df["y"] = df["Task_ID"] + np.random.uniform(-0.3, 0.3, size=len(df))

# b. Height = Structural Complexity (AST Max Depth)
df["height"] = df["Max_Depth"]

# c. Width = Token Cost (Scaled)
# Thicker building = More tokens used (potentially "bloat")
tc_min, tc_max = df["Token_Count"].min(), df["Token_Count"].max()
df["width"] = df["Token_Count"].apply(
    lambda t: 0.2 + 0.6 * (t - tc_min) / (tc_max - tc_min)
)

# d. Research Metric: Efficiency (Tokens per Unit of Depth)
# Lower is better (more structure per token)
df["Efficiency_Ratio"] = df["Token_Count"] / df["Max_Depth"]
df["Efficiency_Label"] = df["Efficiency_Ratio"].apply(
    lambda x: "Concise" if x < 15 else "Verbose" if x > 25 else "Balanced"
)

# --- 3. Geometry Builder (The Mesh) ---
def create_city_mesh(subset, color, name):
    """
    Constructs 3D rectangular prisms for visual density.
    Hover text is attached to each face so hovering any part of a building shows its tooltip.
    """
    x_c, y_c, z_c = [], [], []
    i_ind, j_ind, k_ind = [], [], []
    hover_texts = []  # one per triangle (12 per building)
    curr_idx = 0

    for _, row in subset.iterrows():
        x, y, h, w = row['x'], row['y'], row['height'], row['width']
        hw = w / 2

        # Tooltip: clear sections, no raw HTML tags; plain separator
        lang = row["Language"]
        tip = (
            f"<b>{row['Model']}</b> · {lang}<br>"
            f"Task ID: {row['Task_ID']}<br>"
            f"———————————————<br>"
            f"Structure (depth): <b>{row['Max_Depth']:.1f}</b><br>"
            f"Token cost: <b>{row['Token_Count']:.0f}</b><br>"
            f"Efficiency: <b>{row['Efficiency_Label']}</b> ({row['Efficiency_Ratio']:.1f} tok/depth)<br>"
            f"<i>Lower tok/depth = more structure per token</i>"
        )
        hover_texts.extend([tip] * 12)

        # 8 Vertices (Corners)
        # Bottom (z=0) -> Top (z=h)
        # Order: BL, BR, TR, TL relative to center
        x_c.extend([x-hw, x+hw, x+hw, x-hw, x-hw, x+hw, x+hw, x-hw])
        y_c.extend([y-hw, y-hw, y+hw, y+hw, y-hw, y-hw, y+hw, y+hw])
        z_c.extend([0, 0, 0, 0, h, h, h, h])

        # 12 Triangles (6 Faces)
        # Top, Bottom, Front, Right, Back, Left
        # (Standard Cube Topology indices)
        indices = [
            (4, 5, 6), (4, 6, 7),   # Top
            (0, 2, 1), (0, 3, 2),   # Bottom
            (0, 1, 5), (0, 5, 4),   # Front
            (1, 2, 6), (1, 6, 5),   # Right
            (2, 3, 7), (2, 7, 6),   # Back
            (3, 0, 4), (3, 4, 7)    # Left
        ]
        for tri in indices:
            i_ind.append(curr_idx + tri[0])
            j_ind.append(curr_idx + tri[1])
            k_ind.append(curr_idx + tri[2])

        curr_idx += 8

    return go.Mesh3d(
        x=x_c, y=y_c, z=z_c,
        i=i_ind, j=j_ind, k=k_ind,
        color=color,
        opacity=0.85,
        flatshading=True,
        name=name,
        lighting=dict(ambient=0.8, diffuse=1.0, specular=0.15),
        hoverinfo="text",
        hovertext=hover_texts,
    )


fig = go.Figure()

# --- 4. Build Layers ---

for (model, lang), color in MODEL_LANG_COLORS.items():
    sub = df[(df["Model"] == model) & (df["Language"] == lang)]
    if sub.empty:
        continue
    name = f"{model} ({lang})"

    # Layer A: Buildings (same hex as legend; brighter lighting so swatch matches)
    mesh = create_city_mesh(sub, color, name)
    mesh.legendgroup = name
    fig.add_trace(mesh)

    # Layer B: Invisible hitboxes (same legendgroup = one legend entry, click isolates)
    scatter = go.Scatter3d(
        x=sub['x'],
        y=sub['y'],
        z=sub['height'] / 2,
        mode='markers',
        marker=dict(
            size=sub['width'] * 30,
            color=color,
            opacity=0,
            symbol='square'
        ),
        name=name,
        legendgroup=name,
        hoverinfo='skip',
    )
    fig.add_trace(scatter)

# Layer C: District Foundations (The Floor) — quad as two triangles
for model, center_x in MODEL_X_POS.items():
    fig.add_trace(go.Mesh3d(
        x=[center_x - 2.5, center_x - 2.5, center_x + 2.5, center_x + 2.5],
        y=[0, 11, 11, 0],
        z=[-0.1, -0.1, -0.1, -0.1],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color=MODEL_COLORS[model],
        opacity=0.15,
        hoverinfo='skip',
        showlegend=False
    ))

# --- 5. Layout & Camera ---
fig.update_layout(
    title=dict(
        text="<b>CITY OF AGENTS: Structural Efficiency Map</b><br><sup>Tall = Deep Logic · Thick = High Cost · Hover for Efficiency Metrics</sup>",
        font=dict(size=20, color="white"),
        y=0.95,
        x=0.5,
        xanchor="center"
    ),
    template="plotly_dark",
    scene=dict(
        xaxis=dict(
            title="",
            tickvals=list(MODEL_X_POS.values()),
            ticktext=list(MODEL_X_POS.keys()),
            showgrid=False
        ),
        yaxis=dict(title="Task ID", gridcolor="#333"),
        zaxis=dict(title="Complexity (Depth)", gridcolor="#333"),
        aspectmode="manual",
        aspectratio=dict(x=2.2, y=1.5, z=0.7),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.6))
    ),
    margin=dict(l=0, r=0, b=0, t=80),
    legend=dict(orientation="h", y=0.05, x=0.5, xanchor="center")
)

# Output
out_path = os.path.abspath("city_map_research.html")
fig.write_html(out_path, auto_open=False)

# Inject legend click-to-isolate: one click = show only that group, click again = show all
LEGEND_ISOLATE_SCRIPT = """
<script type="text/javascript">
/* legend-click-isolate */
(function() {
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd || typeof Plotly === 'undefined') return;
    var NUM_GROUPS = 6, TRACES_PER_GROUP = 2, BUILDING_TRACES = NUM_GROUPS * TRACES_PER_GROUP, FLOOR_TRACES = 3;
    var isolatedGroup = null;
    gd.on('plotly_legendclick', function(evt) {
        evt.preventDefault();
        var curveNumber = evt.curveNumber;
        if (curveNumber == null || curveNumber >= BUILDING_TRACES) return false;
        var groupIndex = Math.floor(curveNumber / TRACES_PER_GROUP);
        var vis = [];
        var i;
        if (isolatedGroup === groupIndex) {
            isolatedGroup = null;
            for (i = 0; i < BUILDING_TRACES; i++) vis.push(true);
        } else {
            isolatedGroup = groupIndex;
            for (i = 0; i < BUILDING_TRACES; i++)
                vis.push(i >= groupIndex * TRACES_PER_GROUP && i < (groupIndex + 1) * TRACES_PER_GROUP);
        }
        for (i = 0; i < FLOOR_TRACES; i++) vis.push(true);
        Plotly.restyle(gd, {visible: vis});
        return false;
    });
})();
</script>
"""

with open(out_path, "r", encoding="utf-8") as f:
    html = f.read()
# Insert our script right before </body> (unique marker so we don't double-inject)
if "legend-click-isolate" not in html:
    html = html.replace("</body>", LEGEND_ISOLATE_SCRIPT + "\n</body>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

webbrowser.open(out_path)
print(f"Visualization ready: {out_path}")
