import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib
matplotlib.use('Agg')

# ============ 手动指定每种类型最具代表性的2张图 ============
# 不再用 head(2)，直接指定最能体现错误特征的图片
SELECTED_CASES = {
    "Type A: Over-Perfection": [
        {"category": "face", "label": "ai", "filename": "ai_face_013.jpg"},
        {"category": "face", "label": "ai", "filename": "ai_face_000.jpg"},
    ],
    "Type B: Semantic Neglect": [
        {"category": "animal", "label": "ai", "filename": "ai_004.png"},   # 黄色企鹅
        {"category": "animal", "label": "ai", "filename": "ai_003.png"},   # 其他语义错误动物
    ],
    "Type C: Category Mismatch": [
        {"category": "nature", "label": "ai", "filename": "ai_017.png"},   # 山坡建筑群
        {"category": "nature", "label": "ai", "filename": "ai_000.png"},   # 军舰
    ],
    "Type D: Text Artifact Neglect": [
        {"category": "object", "label": "ai", "filename": "ai_095.png"},   # 护肤品瓶（ILnDCN）
        {"category": "object", "label": "ai", "filename": "ai_006.png"},   # 锁（PaDIOo）
    ],
}

TYPE_CONFIG = {
    "Type A: Over-Perfection": {
        "short": "Type A: Over-Perfection",
        "desc":  "Image too realistic;\nno obvious artifacts detected",
        "color": "#4C72B0"
    },
    "Type B: Semantic Neglect": {
        "short": "Type B: Semantic Neglect",
        "desc":  "Semantically impossible\nfeatures ignored by MLLM",
        "color": "#55A868"
    },
    "Type C: Category Mismatch": {
        "short": "Type C: Category Mismatch",
        "desc":  "Scene lacks structural\nreference for detection",
        "color": "#DD8452"
    },
    "Type D: Text Artifact Neglect": {
        "short": "Type D: Text Artifact Neglect",
        "desc":  "Garbled text (SD artifact)\nnot flagged by MLLM",
        "color": "#C44E52"
    }
}

type_list = list(TYPE_CONFIG.keys())

fig, axes = plt.subplots(2, 4, figsize=(22, 13))

fig.subplots_adjust(
    top=0.76,
    bottom=0.04,
    left=0.03,
    right=0.97,
    hspace=0.28,
    wspace=0.12
)

# 大标题
fig.text(
    0.5, 0.97,
    "Representative Missed Detection Cases by Error Type",
    ha="center", va="top",
    fontsize=22, fontweight="bold"
)
fig.text(
    0.5, 0.925,
    "(AI-Generated Images Incorrectly Classified as Real)",
    ha="center", va="top",
    fontsize=15, color="#555555"
)

# 每列彩色标题条
for col, error_type in enumerate(type_list):
    config   = TYPE_CONFIG[error_type]
    ax_ref   = axes[0][col]
    pos      = ax_ref.get_position()
    x_center = pos.x0 + pos.width / 2

    fig.text(
        x_center, 0.865,
        config["short"],
        ha="center", va="center",
        fontsize=16, fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=config["color"],
            edgecolor="none",
            alpha=1.0
        )
    )
    fig.text(
        x_center, 0.815,
        config["desc"],
        ha="center", va="center",
        fontsize=11, color="#333333",
        style="italic"
    )

# 图片
for col, error_type in enumerate(type_list):
    config  = TYPE_CONFIG[error_type]
    samples = SELECTED_CASES[error_type]

    for row_idx, sample in enumerate(samples):
        ax = axes[row_idx][col]

        img_path = (f"data/{sample['category']}/"
                    f"{sample['label']}/{sample['filename']}")

        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
        else:
            # 文件不存在时显示提示
            ax.text(0.5, 0.5,
                    f"Not Found:\n{sample['filename']}",
                    ha="center", va="center",
                    fontsize=11, color="gray",
                    transform=ax.transAxes)
            ax.set_facecolor("#f5f5f5")
            print(f"⚠️  图片不存在：{img_path}")

        ax.set_title(
            f"{sample['category'].upper()}  |  {sample['filename']}",
            fontsize=13,
            color=config["color"],
            fontweight="bold",
            pad=6
        )

        ax.set_xlabel(
            "MLLM Prediction: Real  ✗",
            fontsize=12,
            color="red",
            labelpad=5
        )

        for spine in ax.spines.values():
            spine.set_edgecolor(config["color"])
            spine.set_linewidth(3.5)

        ax.set_xticks([])
        ax.set_yticks([])

plt.savefig(
    "figures/fig5_taxonomy_examples.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()
print("✅ 已保存：figures/fig5_taxonomy_examples.png")
