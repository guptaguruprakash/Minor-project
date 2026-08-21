import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


fig, ax = plt.subplots(figsize=(7.3, 10.3), dpi=100)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

colors = {
    "gray": "#E6E6E6",
    "green": "#D9EAD3",
    "yellow": "#FFF2CC",
    "peach": "#FCE4D6",
    "purple": "#E4DFEC",
    "blue": "#D9EAF7",
}


def add_box(y, height, color, text, fontsize=9.5):
    x = 0.08
    width = 0.84
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.4,
        edgecolor="black",
        facecolor=colors[color],
    )
    ax.add_patch(box)
    ax.text(
        0.5,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        family="Arial",
        color="black",
        linespacing=1.05,
    )


def arrow(y1, y2):
    ax.annotate(
        "",
        xy=(0.5, y2),
        xytext=(0.5, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=1.5,
            mutation_scale=10,
        ),
    )


ax.text(
    0.5,
    0.99,
    "KNN MODEL TRAINING PIPELINE",
    ha="center",
    va="center",
    fontsize=12.5,
    fontweight="bold",
    family="Arial",
)

steps = [
    (0.905, 0.060, "gray", "Labeled sensor CSV data\ndata/train.csv"),
    (0.815, 0.060, "green", "Load and clean rows\nremove repeated headers and invalid labels"),
    (0.725, 0.060, "green", "Extract inputs and labels\n12 MPU6050 features + exercise/posture label"),
    (0.635, 0.070, "yellow", "Group by label\nbicep_bad, bicep_good, bicep_idle\nsquat_bad, squat_good, squat_idle"),
    (0.545, 0.060, "yellow", "Select representative prototypes\n64 default or 128 configured per label"),
    (0.455, 0.060, "peach", "Calculate feature statistics\nmeans and standard deviations"),
    (0.365, 0.060, "peach", "Standardize features\nz = (x - mean) / scale"),
    (0.275, 0.075, "purple", "Generate knn_model.h\nnormalized prototypes, labels, means, scales\nstored in ESP32 program memory", 8.8),
    (0.185, 0.060, "blue", "ESP32 inference\nKNN with k = 5 and 12 normalized features"),
    (0.095, 0.060, "blue", "Live output\nexercise, posture, confidence, repetitions"),
]

for step in steps:
    add_box(*step)

for index in range(len(steps) - 1):
    current_y, current_height = steps[index][0], steps[index][1]
    next_y = steps[index + 1][0] + steps[index + 1][1]
    arrow(current_y, next_y)

output_path = os.path.join(
    os.path.dirname(__file__),
    "Minor_Project_Report_Formatting",
    "Graphics",
    "knn_training_pipeline.png",
)
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
