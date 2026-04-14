import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

fig, ax = plt.subplots(figsize=(16, 5))
ax.axis("off")

columns = ["Type", "Name", "Description", "Face", "Animal", "Object", "Nature", "Total", "Ratio"]
rows = [
    ["Type A", "Over-Perfection",
     "SD generates highly realistic images;\nMLLM mistakes quality for authenticity",
     "50", "51", "—", "12", "113", "45.7%"],
    ["Type B", "Semantic Neglect",
     "Impossible features (e.g. yellow penguin)\nnot flagged; vision-reasoning decoupled",
     "—", "5", "—", "—", "5", "2.0%"],
    ["Type C", "Category Mismatch",
     "Urban/industrial/food scenes lack\nstructural artifact reference",
     "—", "—", "—", "65", "65", "26.3%"],
    ["Type D", "Text Artifact Neglect",
     "Garbled text (SD artifact) present\nbut not identified as evidence",
     "—", "—", "64", "—", "64", "25.9%"],
    ["Total", "", "All missed detections",
     "50", "56", "64", "77", "247", "100%"],
]

colors_row = [
    ["#E6F1FB", "#E6F1FB", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#E6F1FB", "#E6F1FB"],
    ["#EAF3DE", "#EAF3DE", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#EAF3DE", "#EAF3DE"],
    ["#FAEEDA", "#FAEEDA", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#FAEEDA", "#FAEEDA"],
    ["#FCEBEB", "#FCEBEB", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#f9f9f9", "#FCEBEB", "#FCEBEB"],
    ["#eeeeee"] * 9,
]

table = ax.table(
    cellText=rows,
    colLabels=columns,
    cellLoc="center",
    loc="center",
    cellColours=colors_row
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.8)

# 列宽
col_widths = [0.06, 0.13, 0.28, 0.06, 0.07, 0.07, 0.07, 0.07, 0.07]
for i, w in enumerate(col_widths):
    for j in range(len(rows) + 1):
        table[j, i].set_width(w)

# 表头加粗
for j in range(len(columns)):
    table[0, j].set_text_props(fontweight="bold")
    table[0, j].set_facecolor("#dddddd")

# Description 列左对齐
for i in range(1, len(rows) + 1):
    table[i, 2].set_text_props(ha="left")
    table[i, 2]._loc = "left"

ax.set_title(
    "Table: Visual Artifact Error Taxonomy for Missed Detections (n=247)",
    fontsize=13, fontweight="bold", pad=10
)

plt.tight_layout()
plt.savefig("figures/table_taxonomy.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print("✅ 已保存：figures/table_taxonomy.png")