
import os
import base64
import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
import time
import io

# ============ 配置 ============
API_KEY = "sk-xxxxxxxxxxxxxxx"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-vl-max"
RESULT_PATH = "results/exp4_results.csv"

PROMPT = """You are an expert forensic analyst. Your task is to 
determine whether this image is a real photograph or AI-generated.

Analyze the following aspects carefully:

1. TEXTURE: 
   - AI: overly smooth, uniform, or waxy surfaces
   - Real: natural variations, subtle imperfections

2. STRUCTURE:
   - AI: slightly distorted fingers, asymmetric facial features
   - Real: anatomically consistent body parts

3. LIGHTING:
   - AI: inconsistent shadows, multiple light sources
   - Real: single coherent light source, natural shadows

4. BACKGROUND:
   - AI: blurring in wrong areas, repetitive patterns
   - Real: natural depth of field, organic variety

5. OVERALL QUALITY:
   - AI: too perfect, looks "rendered"
   - Real: natural noise, organic feel

Analyze step by step:
Step 1: Examine textures and surfaces
Step 2: Check structural details and anatomy
Step 3: Evaluate lighting and shadows
Step 4: Assess background and depth
Step 5: Give final judgment with main reason

End with exactly one of:
FINAL: Real
FINAL: Fake"""

CATEGORIES = ["face", "animal", "object", "nature"]
# ==============================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
os.makedirs("results", exist_ok=True)


def encode_image(image_path):
    """图片转 base64，统一转为 JPEG 格式"""
    img = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def query_qwen_vl(image_path, prompt, max_retries=3):
    """调用 Qwen-VL，带重试机制"""
    for attempt in range(max_retries):
        try:
            img_base64 = encode_image(image_path)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"\n  API错误（第{attempt + 1}次）：{e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return None


def parse_response(response_text):
    """
    从模型回答中提取 Real/Fake
    优先级：
    1. 找明确的 FINAL 标记（最准确）
    2. 找不到则只看最后3行关键词（兜底）
    3. 还找不到则返回 unknown
    """
    if response_text is None:
        return "unknown"

    text = response_text.upper()

    # 第一优先级：找明确的 FINAL 标记
    if "FINAL: FAKE" in text:
        return "fake"
    if "FINAL: REAL" in text:
        return "real"

    # 第二优先级：只看最后3行，避免被分析过程中的词干扰
    last_lines = text.strip().split("\n")[-3:]
    last_text = " ".join(last_lines)
    if "FAKE" in last_text:
        return "fake"
    if "REAL" in last_text:
        return "real"

    return "unknown"


# ============ 断点续传 ============
if os.path.exists(RESULT_PATH):
    existing_df = pd.read_csv(RESULT_PATH)
    done_set = set(zip(
        existing_df["category"],
        existing_df["label"],
        existing_df["filename"]
    ))
    all_results = existing_df.to_dict("records")
    print(f"断点续传：已有 {len(all_results)} 条结果，跳过已完成的图片")
else:
    done_set = set()
    all_results = []
    print("未找到已有结果，从头开始")

# ============ 主程序：逐类别推理 ============
for category in CATEGORIES:
    for label in ["real", "ai"]:
        folder = f"data/{category}/{label}"

        if not os.path.exists(folder):
            print(f"\n⚠️  文件夹不存在，跳过：{folder}")
            continue

        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # 过滤掉已处理的
        remaining = [f for f in files
                     if (category, label, f) not in done_set]

        print(f"\n【{category}/{label}】"
              f"共 {len(files)} 张，"
              f"已完成 {len(files) - len(remaining)} 张，"
              f"剩余 {len(remaining)} 张")

        if not remaining:
            print("  全部已完成，跳过")
            continue

        for fname in tqdm(remaining, desc="  推理中"):
            fpath = os.path.join(folder, fname)
            ground_truth = "real" if label == "real" else "fake"

            response = query_qwen_vl(fpath, PROMPT)
            prediction = parse_response(response)
            is_correct = (prediction == ground_truth)

            all_results.append({
                "category":     category,
                "label":        label,
                "filename":     fname,
                "ground_truth": ground_truth,
                "prediction":   prediction,
                "correct":      is_correct,
                "response":     response
            })
            done_set.add((category, label, fname))

            # 每10张保存一次，防止中途崩溃丢数据
            if len(all_results) % 10 == 0:
                pd.DataFrame(all_results).to_csv(RESULT_PATH, index=False)

            time.sleep(0.5)  # 避免请求过快

# ============ 最终保存 ============
pd.DataFrame(all_results).to_csv(RESULT_PATH, index=False)
print(f"\n{'=' * 50}")
print(f"✅ 推理完成！结果保存到 {RESULT_PATH}")

df = pd.read_csv(RESULT_PATH)
df_valid = df[df["prediction"] != "unknown"]
unknown_count = len(df) - len(df_valid)

print(f"总计处理：{len(df)} 张")
print(f"有效预测：{len(df_valid)} 张")
if unknown_count > 0:
    print(f"⚠️  unknown：{unknown_count} 张（解析失败，不影响有效结果）")

# 快速预览各类别结果
print(f"\n快速预览：")
for cat in CATEGORIES:
    subset = df_valid[df_valid["category"] == cat]
    if len(subset) > 0:
        acc = subset["correct"].mean()
        print(f"  {cat}: {acc:.1%} 准确率（{len(subset)} 张有效）")
