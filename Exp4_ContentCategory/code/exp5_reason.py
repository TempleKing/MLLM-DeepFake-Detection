import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

import logging
import warnings
import matplotlib

import os

matplotlib.use('Agg')
logging.getLogger('matplotlib.font_manager').disabled = True
warnings.filterwarnings("ignore")

os.makedirs("figures/case_studies", exist_ok=True)

# 三个框的固定框线颜色：蓝、橙、绿
BOX_BORDERS = ["#4C72B0", "#DD8452", "#55A868"]

CASES = [
    {
        "type":     "Type A: Over-Perfection",
        "img_path": "data/face/ai/ai_face_013.jpg",
        "color":    "#4C72B0",
        "boxes": [
            {
                "title":   "Artifact Spotlight",
                "content": [
                    "Subject: AI-generated human face (FFHQ-style portrait)",
                    "Skin texture is unnaturally flawless — zero visible pores,",
                    "  blemishes, or tonal variation across the entire face",
                    "Hair strands near the temple boundary merge unnaturally",
                    "Background bokeh is perfectly uniform, unlike real optics"
                ]
            },
            {
                "title":   "Why MLLM Failed",
                "content": [
                    "• Equated high visual quality with photographic authenticity",
                    "• Praised 'consistent lighting' and 'natural skin texture'",
                    "• No single obvious artifact to trigger a Fake verdict",
                    "• Diffusion models produce faces that exceed MLLM's",
                    "  quality-based detection threshold"
                ]
            },
            {
                "title":   "Human Verdict",
                "content": [
                    "• Skin is too perfect — real faces always have",
                    "  micro-imperfections, pores, and subtle asymmetry",
                    "• The overall appearance resembles a digital render,",
                    "  not a candid or studio photograph"
                ]
            }
        ]
    },
    {
        "type":     "Type B: Semantic Neglect",
        "img_path": "data/animal/ai/ai_004.png",
        "color":    "#55A868",
        "boxes": [
            {
                "title":   "Artifact Spotlight",
                "content": [
                    "Subject: Emperor penguin with bright yellow body",
                    "Penguins are black-and-white birds; yellow coloration",
                    "  on the body does not exist in any known species",
                    "The color is not a lighting effect — it is a fundamental",
                    "  violation of real-world biology"
                ]
            },
            {
                "title":   "Why MLLM Failed",
                "content": [
                    "• Analyzed visual features (texture, posture, lighting)",
                    "  in isolation from real-world semantic knowledge",
                    "• Rated 'feather texture' and 'natural stance' as authentic",
                    "• Failed to cross-reference appearance with biological facts",
                    "• Vision perception and common-sense reasoning are decoupled"
                ]
            },
            {
                "title":   "Human Verdict",
                "content": [
                    "• Any person knows penguins are black and white —",
                    "  a yellow penguin body is biologically impossible",
                    "• The impossible color alone is sufficient to conclude",
                    "  this image was AI-generated"
                ]
            }
        ]
    },
    {
        "type":     "Type C: Category Mismatch",
        "img_path": "data/nature/ai/ai_017.png",
        "color":    "#DD8452",
        "boxes": [
            {
                "title":   "Artifact Spotlight",
                "content": [
                    "Subject: Dense hillside residential area (buildings + road)",
                    "Labeled as 'nature' but contains no natural landscape",
                    "Building colors are oversaturated and inconsistent:",
                    "  magenta awnings and vivid red roofs in unrealistic combinations",
                    "Spatial perspective of hillside buildings has subtle distortions"
                ]
            },
            {
                "title":   "Why MLLM Failed",
                "content": [
                    "• No biological reference to anchor artifact detection",
                    "• Complex multi-element scene obscures individual anomalies",
                    "• Model evaluated global scene quality, not content anomalies",
                    "• 'Nature' detection heuristics are not calibrated for",
                    "  urban or industrial imagery"
                ]
            },
            {
                "title":   "Human Verdict",
                "content": [
                    "• Building color combinations are unrealistically vivid —",
                    "  magenta and saturated red roofs together look rendered",
                    "• The scene composition feels too dense and uniform,",
                    "  lacking the organic randomness of a real neighborhood"
                ]
            }
        ]
    },
    {
        "type":     "Type D: Text Artifact Neglect",
        "img_path": "data/object/ai/ai_095.png",
        "color":    "#C44E52",
        "boxes": [
            {
                "title":   "Artifact Spotlight",
                "content": [
                    "Subject: Two skincare/lotion bottles with product labels",
                    "All text on both bottles is completely garbled:",
                    "  Large bottle: 'ILorm', 'IDorm', random character lines",
                    "  Small bottle: 'ILnDCN' at top — not a real word or brand",
                    "Text layout mimics real labels but contains zero meaning"
                ]
            },
            {
                "title":   "Why MLLM Failed",
                "content": [
                    "• Focused on bottle shape, surface texture, and lighting",
                    "• Described text as 'slightly blurry but consistent with",
                    "  product packaging' — did not flag it as garbled",
                    "• Text artifact detection is not a prioritized signal",
                    "  in MLLM's current forensic reasoning pipeline"
                ]
            },
            {
                "title":   "Human Verdict",
                "content": [
                    "• Any reader would immediately notice: 'ILnDCN' and",
                    "  'IDorm' are not real product names or ingredients",
                    "• Garbled text is a well-known SD generation failure —",
                    "  this alone is definitive proof of AI generation"
                ]
            }
        ]
    }
]


def draw_case(case, save_path):
    print(f"绘制：{case['type']}")

    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(
        1, 2,
        width_ratios=[1, 1.35],
        figure=fig,
        left=0.03, right=0.97,
        top=0.88,  bottom=0.03,
        wspace=0.05
    )

    # ---- 左侧：图片 ----
    ax_img = fig.add_subplot(gs[0, 0])

    if os.path.exists(case["img_path"]):
        img = mpimg.imread(case["img_path"])
        ax_img.imshow(img, aspect="equal")
    else:
        ax_img.text(0.5, 0.5,
                    f"Not found:\n{case['img_path']}",
                    ha="center", va="center",
                    color="red", fontsize=11)
        ax_img.set_facecolor("#eeeeee")

    cat   = case["img_path"].split("/")[1].upper()
    fname = os.path.basename(case["img_path"])

    ax_img.set_title(
        f"{cat}  |  {fname}\n"
        f"Ground Truth: AI-Generated    MLLM Prediction: Real  ✗",
        fontsize=12, fontweight="bold",
        color=case["color"], pad=8
    )
    for spine in ax_img.spines.values():
        spine.set_edgecolor(case["color"])
        spine.set_linewidth(3)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # ---- 右侧：3个文字框 ----
    n_boxes = len(case["boxes"])

    def line_count(box):
        return len(box["content"]) + 2

    ratios  = [line_count(b) for b in case["boxes"]]
    gs_text = gs[0, 1].subgridspec(
        n_boxes, 1,
        hspace=0.08,
        height_ratios=ratios
    )

    for i, box in enumerate(case["boxes"]):
        ax = fig.add_subplot(gs_text[i, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")          # 白色背景

        for spine in ax.spines.values():
            spine.set_edgecolor(BOX_BORDERS[i])   # 蓝、橙、绿
            spine.set_linewidth(2.5)

        title_line = f"[ {box['title']} ]"
        content_str = "\n".join(box["content"])
        full_text   = f"{title_line}\n\n{content_str}"

        ax.text(
            0.025, 0.97,
            full_text,
            fontsize=11.5,
            va="top", ha="left",
            transform=ax.transAxes,
            linespacing=1.6,
            fontfamily="DejaVu Sans"
        )

    fig.suptitle(
        case["type"],
        fontsize=17,
        fontweight="bold",
        color=case["color"],
        y=0.97
    )

    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor="white")
    print(f"  ✅ 已保存：{save_path}")
    plt.close()


# ============ 生成四张图 ============
type_labels = ["typeA", "typeB", "typeC", "typeD"]

for case, label in zip(CASES, type_labels):
    save_path = f"figures/case_studies/case_{label}.png"
    draw_case(case, save_path)

print("\n✅ 全部完成！保存在 figures/case_studies/")