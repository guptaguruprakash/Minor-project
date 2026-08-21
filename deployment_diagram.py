import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

colors = {
    "gray": "#E6E6E6",
    "green": "#D9EAD3",
    "yellow": "#FFF2CC",
    "peach": "#FCE4D6",
    "blue": "#D9EAF7",
}


def device(x, y, width, height, title, items, color):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.5, edgecolor="black", facecolor=colors[color],
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height - 0.07, "<<device>>", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + width / 2, y + height - 0.105, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.plot([x + 0.03, x + width - 0.03], [y + height - 0.15] * 2, color="#999999", linewidth=0.8)
    line_width = 27 if width >= 0.28 else 24
    body_lines = []
    for item in items:
        wrapped = textwrap.wrap(item, width=line_width, break_long_words=False)
        body_lines.append(f"• {wrapped[0]}")
        body_lines.extend(f"  {line}" for line in wrapped[1:])
    ax.text(
        x + 0.05,
        y + height - 0.19,
        "\n".join(body_lines),
        ha="left",
        va="top",
        fontsize=7.5,
        linespacing=1.1,
    )


def arrow(start, end, label, label_offset=(0, 0.025)):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5, mutation_scale=12))
    midpoint = ((start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1])
    ax.text(*midpoint, label, ha="center", va="center", fontsize=7.8, color="#333333", bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc"))


device(0.04, 0.46, 0.28, 0.40, "ESP32 Microcontroller", [
    "FreeRTOS: queues + mutex", "MPU6050 x2: 0x68, 0x69", "Embedded UI: OLED + buzzer", "SH1106 0x3C | GPIO 25", "Collector: Core 0 | 50 Hz", "ML / Wi-Fi / OLED: Core 1", "KNN: k=5 | 12 features", "HTTP :80 | WebSocket :81",
], "green")
device(0.68, 0.46, 0.28, 0.40, "Backend Server (PC)", [
    "FastAPI data service", "POST /api/sensor-data", "GET /api/live-prediction", "CSV training-data storage", "Live updates: /ws/live", "Browser dashboard",
], "peach")
device(0.37, 0.06, 0.26, 0.27, "User Phone / Laptop", [
    "ESP32 live dashboard", "FastAPI analytics dashboard",
], "blue")

network = FancyBboxPatch((0.39, 0.62), 0.22, 0.15, boxstyle="round,pad=0.02,rounding_size=0.05", linewidth=1.5, edgecolor="black", facecolor=colors["yellow"])
ax.add_patch(network)
ax.text(0.5, 0.695, "Home Wi-Fi\nNetwork", ha="center", va="center", fontsize=12, fontweight="bold")

arrow((0.32, 0.66), (0.39, 0.68), "Wi-Fi", (0, 0.035))
arrow((0.61, 0.68), (0.68, 0.66), "Wi-Fi", (0, 0.035))
arrow((0.32, 0.47), (0.68, 0.47), "Collection POST + live prediction GET", (0, 0.045))
arrow((0.18, 0.46), (0.46, 0.33), "ESP32 dashboard\nHTTP :80 + WebSocket :81", (0, -0.02))
arrow((0.82, 0.46), (0.54, 0.33), "FastAPI dashboard\nHTTP + WebSocket /ws/live", (0, -0.02))

ax.text(0.5, 0.94, "SMART GYM ASSISTANT - APPLICATION DEPLOYMENT", ha="center", va="center", fontsize=14, fontweight="bold")

output_path = os.path.join(os.path.dirname(__file__), "Minor_Project_Report_Formatting", "Graphics", "deployment_diagram.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
