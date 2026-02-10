"""Generate blog charts from the 48-game expanded-rules run."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = "/Users/visakhmadathil/projects/catan-bench/charts"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ── Model names & palette ──
MODELS = [
    "Gemini 3 Flash Preview",
    "Claude 4.5 Sonnet",
    "Claude 4.5 Haiku",
    "Gemini 2.5 Flash",
]

# Muted, editorial palette
COLORS = {
    "Gemini 3 Flash Preview": "#2d3a8c",   # deep indigo
    "Claude 4.5 Sonnet":     "#b45309",    # warm amber
    "Claude 4.5 Haiku":      "#0f766e",    # dark teal
    "Gemini 2.5 Flash":      "#94a3b8",    # cool slate
}

# ── Shared style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d1d5db",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


# ═══════════════════════════════════════════════════════════════════
# CHART 1: Hero — Win rate (horizontal bars, no CIs)
# ═══════════════════════════════════════════════════════════════════

win_rates = [72.9, 16.7, 8.3, 2.1]
wins =      [35, 8, 4, 1]

fig, ax = plt.subplots(figsize=(9, 3.8))

y_pos = np.arange(len(MODELS))[::-1]
bars = ax.barh(
    y_pos, win_rates,
    color=[COLORS[m] for m in MODELS],
    height=0.52,
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)

# Labels on/beside bars
for i, (wr, w) in enumerate(zip(win_rates, wins)):
    label = f"{wr:.0f}%  ({w}/48)"
    if wr > 20:
        ax.text(3, y_pos[i], label, va="center", ha="left",
                fontweight="600", fontsize=13, color="white", zorder=5)
    else:
        ax.text(wr + 1.5, y_pos[i], label, va="center", ha="left",
                fontweight="600", fontsize=12.5, color="#374151", zorder=5)

ax.set_yticks(y_pos)
ax.set_yticklabels(MODELS, fontsize=13, fontweight="500", color="#1f2937")
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.tick_params(axis="x", labelsize=10)
ax.grid(axis="x", alpha=0.12, color="#9ca3af", zorder=0)
ax.set_title("Win Rate", fontsize=20, fontweight="700",
             pad=12, loc="left", color="#111827")

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/hero_win_rates.png")
plt.close()
print("Saved hero_win_rates.png")


# ═══════════════════════════════════════════════════════════════════
# CHART 2: Resource production gap (grouped bar)
# ═══════════════════════════════════════════════════════════════════

resources = {
    "Gemini 3 Flash Preview": {"W": 39.7, "B": 33.0, "S": 32.7, "H": 53.6, "O": 42.2},
    "Claude 4.5 Sonnet":      {"W": 26.2, "B": 17.9, "S": 17.7, "H": 25.3, "O": 16.9},
    "Claude 4.5 Haiku":       {"W": 23.0, "B": 20.6, "S": 24.8, "H": 23.1, "O": 17.2},
    "Gemini 2.5 Flash":       {"W": 22.6, "B":  7.6, "S": 14.8, "H": 18.1, "O": 27.8},
}

res_labels = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]
res_keys = ["W", "B", "S", "H", "O"]
# Muted resource colors
res_colors = ["#15803d", "#b91c1c", "#65a30d", "#ca8a04", "#64748b"]

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(MODELS))
width = 0.15
offsets = np.arange(len(res_keys)) - 2

for j, (rk, rl, rc) in enumerate(zip(res_keys, res_labels, res_colors)):
    vals = [resources[m][rk] for m in MODELS]
    ax.bar(x + offsets[j] * width, vals, width * 0.88,
           label=rl, color=rc, alpha=0.8, zorder=3)

# Total annotation
for i, m in enumerate(MODELS):
    total = sum(resources[m].values())
    peak = max(resources[m].values())
    ax.text(i, peak + 3, f"{total:.0f}",
            ha="center", fontweight="700", fontsize=13, color="#1f2937")

ax.set_ylim(0, 68)
ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=12, fontweight="500", color="#1f2937")
ax.set_ylabel("Avg per game", fontsize=11, color="#6b7280")
ax.legend(loc="upper right", framealpha=0.95, fontsize=10,
          edgecolor="#e5e7eb", fancybox=False)
ax.grid(axis="y", alpha=0.12, color="#9ca3af", zorder=0)
ax.set_title("Resources Collected", fontsize=20, fontweight="700",
             pad=12, loc="left", color="#111827")

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/resource_production.png")
plt.close()
print("Saved resource_production.png")


# ═══════════════════════════════════════════════════════════════════
# CHART 3: Build strategy — settlements vs cities (scatter)
# ═══════════════════════════════════════════════════════════════════

avg_settlements = [0.9, 1.6, 1.6, 1.2]
avg_cities =      [2.6, 1.0, 1.2, 1.0]
avg_vp =          [9.2, 5.4, 5.0, 4.5]

fig, ax = plt.subplots(figsize=(8, 6))

# Plot points
for i, m in enumerate(MODELS):
    size = avg_vp[i] * 45
    ax.scatter(avg_settlements[i], avg_cities[i], s=size,
               color=COLORS[m], alpha=0.9, edgecolors="white",
               linewidth=1.5, zorder=4)

# Labels with manual offsets to avoid overlap
label_cfg = {
    "Gemini 3 Flash Preview": {"dx": 0.08, "dy": 0.12, "ha": "left"},
    "Claude 4.5 Haiku":       {"dx": 0.08, "dy": 0.12, "ha": "left"},
    "Claude 4.5 Sonnet":      {"dx": 0.08, "dy": -0.22, "ha": "left"},
    "Gemini 2.5 Flash":       {"dx": -0.08, "dy": -0.22, "ha": "right"},
}

for i, m in enumerate(MODELS):
    cfg = label_cfg[m]
    ax.annotate(
        f"{m}\n{avg_vp[i]} avg VP",
        (avg_settlements[i], avg_cities[i]),
        xytext=(avg_settlements[i] + cfg["dx"], avg_cities[i] + cfg["dy"]),
        fontsize=10.5, fontweight="500", color=COLORS[m],
        ha=cfg["ha"],
    )

ax.set_xlabel("Avg Settlements", fontsize=12, color="#374151")
ax.set_ylabel("Avg Cities", fontsize=12, color="#374151")
ax.set_xlim(0.4, 2.2)
ax.set_ylim(0.4, 3.2)
ax.grid(alpha=0.12, color="#9ca3af", zorder=0)
ax.set_title("Build Strategy", fontsize=20, fontweight="700",
             pad=20, loc="left", color="#111827")

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/build_strategy.png")
plt.close()
print("Saved build_strategy.png")

print(f"\nAll charts saved to {OUT_DIR}/")
