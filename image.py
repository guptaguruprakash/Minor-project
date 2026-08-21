import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ============================================================
# Figure setup
# ============================================================
fig, ax = plt.subplots(figsize=(7.3, 10.3), dpi=100)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ============================================================
# Colors — matching the KNN training pipeline
# ============================================================
green = "#D9EAD3"
yellow = "#FFF2CC"
peach = "#FCE4D6"
purple = "#E4DFEC"
blue = "#D9EAF7"
gray = "#E6E6E6"

# ============================================================
# Helper function for rounded boxes
# ============================================================
def add_box(x, y, width, height, color, text, fontsize=10):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.4,
        edgecolor="black",
        facecolor=color
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        family="Arial",
        color="black",
        linespacing=1.15
    )


# ============================================================
# Helper function for arrows
# ============================================================
def arrow(x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=1.5,
            mutation_scale=10
        )
    )


# ============================================================
# Title
# ============================================================
ax.text(
    0.5, 0.965,
    "KNN-BASED EXERCISE AND POSTURE DETECTION PIPELINE",
    ha="center",
    va="center",
    fontsize=12.5,
    fontweight="bold",
    family="Arial"
)


# ============================================================
# Main vertical pipeline
# ============================================================

# 1. Motion Sensor
add_box(
    0.28, 0.855, 0.44, 0.060,
    gray,
    "Motion Sensor"
)

arrow(0.50, 0.855, 0.50, 0.830)


# 2. Microcontroller
add_box(
    0.28, 0.765, 0.44, 0.060,
    green,
    "Microcontroller"
)

arrow(0.50, 0.765, 0.50, 0.740)


# 3. Data Processing
add_box(
    0.28, 0.675, 0.44, 0.060,
    green,
    "Data Processing"
)

arrow(0.50, 0.675, 0.50, 0.650)


# 4. KNN Model
add_box(
    0.28, 0.585, 0.44, 0.060,
    yellow,
    "KNN Model"
)


# ============================================================
# Branching from KNN model
# ============================================================

# Vertical line from KNN
ax.plot(
    [0.50, 0.50],
    [0.585, 0.515],
    color="black",
    linewidth=1.5
)

# Horizontal branch
ax.plot(
    [0.22, 0.78],
    [0.515, 0.515],
    color="black",
    linewidth=1.5
)

# ============================================================
# Rep Counting Algorithm
# ============================================================
add_box(
    0.12, 0.425, 0.32, 0.070,
    peach,
    "Rep Counting\nAlgorithm"
)

# Arrow down into left branch
arrow(0.22, 0.515, 0.22, 0.495)


# ============================================================
# Posture Detection Algorithm
# ============================================================
add_box(
    0.56, 0.425, 0.32, 0.070,
    peach,
    "Posture Detection\nAlgorithm"
)

# Arrow down into right branch
arrow(0.78, 0.515, 0.78, 0.495)


# ============================================================
# Combine branches
# ============================================================

# Left vertical
ax.plot(
    [0.22, 0.22],
    [0.425, 0.365],
    color="black",
    linewidth=1.5
)

# Right vertical
ax.plot(
    [0.78, 0.78],
    [0.425, 0.365],
    color="black",
    linewidth=1.5
)

# Bottom horizontal connection
ax.plot(
    [0.22, 0.78],
    [0.365, 0.365],
    color="black",
    linewidth=1.5
)

# Arrow toward feedback system
arrow(0.50, 0.365, 0.50, 0.315)


# ============================================================
# Feedback System
# ============================================================
add_box(
    0.28, 0.235, 0.44, 0.060,
    blue,
    "Feedback System"
)


# ============================================================
# Caption
# ============================================================
ax.text(
    0.5, 0.065,
    "Figure 4.X: KNN-based exercise and posture detection pipeline.",
    ha="center",
    va="center",
    fontsize=11.5,
    family="serif",
    color="black"
)


# ============================================================
# Save
# ============================================================
plt.savefig(
    "knn_exercise_posture_pipeline.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()