import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.use('Agg')
os.makedirs("figures", exist_ok=True)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ============ 读取数据 ============
df = pd.read_csv("results/exp4_results.csv")
df = df[df["prediction"] != "unknown"]

categories     = ["face", "animal", "object", "nature"]
labels_display = ["Face", "Animal", "Object", "Nature"]
colors         = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

accuracy_list = []
real_acc_list = []
fake_acc_list = []
f1_list       = []

for cat in categories:
    subset   = df[df["category"] == cat]
    real_sub = subset[subset["ground_truth"] == "real"]
    fake_sub = subset[subset["ground_truth"] == "fake"]

    acc      = subset["correct"].mean()
    real_acc = (real_sub["prediction"] == "real").mean()
    fake_acc = (fake_sub["prediction"] == "fake").mean()

    tp = ((subset["prediction"] == "fake") &
          (subset["ground_truth"] == "fake")).sum()
    fp = ((subset["prediction"] == "fake") &
          (subset["ground_truth"] == "real")).sum()
    fn = ((subset["prediction"] == "real") &
          (subset["ground_truth"] == "fake")).sum()
    f1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

    accuracy_list.append(acc)
    real_acc_list.append(real_acc)
    fake_acc_list.append(fake_acc)
    f1_list.append(f1)

x = np.arange(len(categories))

# ============ 图一：整体准确率柱状图 ============
fig1, ax1 = plt.subplots(figsize=(8, 5))

bars = ax1.bar(x, accuracy_list, width=0.5,
               color=colors, edgecolor="white", linewidth=1.2)

ax1.set_xticks(x)
ax1.set_xticklabels(labels_display, fontsize=13)
ax1.set_ylabel("Detection Accuracy", fontsize=12)
ax1.set_title("Overall Detection Accuracy by Image Category",
              fontsize=13, fontweight="bold")
ax1.set_ylim(0.4, 0.85)
ax1.axhline(y=0.5, color="gray", linestyle="--",
            linewidth=1.2, alpha=0.6, label="Random baseline (50%)")
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, accuracy_list):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.008,
             f"{val:.1%}",
             ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig1_overall_accuracy.png",
            dpi=150, bbox_inches="tight")
print("✅ 图一已保存：figures/fig1_overall_accuracy.png")

# ============ 图二：真实图 vs AI图（修复图例位置）============
fig2, ax2 = plt.subplots(figsize=(10, 6))

width = 0.3
b1 = ax2.bar(x - width/2, real_acc_list, width=width,
             label="Real Photo Detection",
             color="#4C72B0", edgecolor="white", linewidth=1.2)
b2 = ax2.bar(x + width/2, fake_acc_list, width=width,
             label="AI Image Detection",
             color="#DD8452", edgecolor="white", linewidth=1.2)

ax2.set_xticks(x)
ax2.set_xticklabels(labels_display, fontsize=13)
ax2.set_ylabel("Detection Accuracy", fontsize=12)
ax2.set_title("Real vs AI Detection Accuracy by Category",
              fontsize=13, fontweight="bold")

# y轴上限留出图例空间
ax2.set_ylim(0.0, 1.18)

ax2.axhline(y=0.5, color="gray", linestyle="--",
            linewidth=1.2, alpha=0.6)

# 图例放右上角，不与柱子重叠
ax2.legend(
    fontsize=11,
    loc="upper right",
    framealpha=0.9,
    edgecolor="#cccccc"
)
ax2.grid(axis="y", alpha=0.3)

# 数值标注
for bar in list(b1) + list(b2):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.012,
             f"{bar.get_height():.1%}",
             ha="center", fontsize=10, fontweight="bold")

# Random baseline 文字标注
ax2.text(3.62, 0.505, "Random baseline (50%)",
         fontsize=9, color="gray", va="bottom")

plt.tight_layout()
plt.savefig("figures/fig2_real_vs_ai_accuracy.png",
            dpi=150, bbox_inches="tight")
print("✅ 图二已保存：figures/fig2_real_vs_ai_accuracy.png")

# ============ 图三：热力图（修复F1格式）============
fig3, ax3 = plt.subplots(figsize=(8, 4))

metrics      = ["Overall Acc", "Real Acc", "AI Acc", "F1-Score"]
heatmap_data = np.array([
    accuracy_list,
    real_acc_list,
    fake_acc_list,
    f1_list
])

im = ax3.imshow(
    heatmap_data,
    cmap="Blues",
    aspect="auto",
    vmin=0.0,
    vmax=1.0
)

ax3.set_xticks(np.arange(len(labels_display)))
ax3.set_yticks(np.arange(len(metrics)))
ax3.set_xticklabels(labels_display, fontsize=13)
ax3.set_yticklabels(metrics, fontsize=12)

for i in range(len(metrics)):
    for j in range(len(labels_display)):
        val = heatmap_data[i, j]
        text_color = "white" if val > 0.6 else "#333333"

        # F1-Score 行（i=3）用小数格式，其余行用百分比
        display_val = f"{val:.3f}" if i == 3 else f"{val:.1%}"

        ax3.text(j, i, display_val,
                 ha="center", va="center",
                 fontsize=12, fontweight="bold",
                 color=text_color)

cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label("Score", fontsize=11)
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

ax3.set_title(
    "Detection Performance Heatmap by Category",
    fontsize=13, fontweight="bold", pad=12
)

# 格子边界
ax3.set_xticks(np.arange(len(labels_display)) - 0.5, minor=True)
ax3.set_yticks(np.arange(len(metrics)) - 0.5, minor=True)
ax3.grid(which="minor", color="white", linewidth=2)
ax3.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.savefig("figures/fig3_heatmap.png",
            dpi=150, bbox_inches="tight")
print("✅ 图三（热力图）已保存：figures/fig3_heatmap.png")

# ============ 打印汇总表 ============
print("\n" + "=" * 60)
print("汇总表（用于报告）：")
print("=" * 60)
print(f"{'Category':<10} {'Accuracy':>10} {'F1-Score':>10} "
      f"{'Real Acc':>10} {'AI Acc':>10}")
print("-" * 60)
for i, cat in enumerate(categories):
    print(f"{labels_display[i]:<10} "
          f"{accuracy_list[i]:>10.1%} "
          f"{f1_list[i]:>10.3f} "
          f"{real_acc_list[i]:>10.1%} "
          f"{fake_acc_list[i]:>10.1%}")

