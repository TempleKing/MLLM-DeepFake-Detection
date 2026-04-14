import gradio as gr
import base64
import json
import os
from openai import OpenAI
import httpx
from PIL import Image, ImageDraw

# ----------------- 升级版提示词 -----------------
# 1. 增加了分类审查逻辑 (Face, Animal, Object, Landscape)
# 2. 优化了 Taxonomy 维度，优先关注物理和结构逻辑
# 3. 强制模型输出 artifacts 的 2D 坐标用于画框
SYSTEM_PROMPT = """You are an expert forensic image analyst. Your task is to determine whether this image is a real photograph or AI-generated.

First, identify the main subject category: [Human Face, Animal, Object, or Landscape/Architecture].
Then, evaluate the image focusing on category-specific AI artifacts:

- HUMAN FACE: Check pupil symmetry, mismatched earrings, teeth structure, hair strands blending unnaturally into skin/clothing, and asymmetric ears.
- ANIMAL: Check leg joints/directions, incorrect number of toes/limbs, merging tails, and unnatural fur/scale transitions.
- OBJECT: Check text/gibberish letters, straight line straightness, geometric symmetry, and nonsensical intersecting parts (e.g., handles blending into cups).
- LANDSCAPE/ARCHITECTURE: Check structural logic of buildings, window alignments, tree branches floating/merging, and contradictory shadow directions.

Evaluate these 5 dimensions:
1. Physical/Structural Logic (Anatomy, physics, symmetry)
2. Lighting & Shadows (Consistency, light sources)
3. Texture & Material (Over-smoothed, waxy, or unnatural noise)
4. Background & Depth (Nonsensical blurring, edge blending)
5. Overall Coherence (Does it make physical sense?)

IMPORTANT: You MUST output STRICTLY as a JSON object. Do not include markdown code blocks.
The JSON must strictly follow this structure:
{
    "category": "<String: Human Face, Animal, Object, or Landscape/Architecture>",
    "probability": <float between 0.0 and 1.0 representing the probability of being AI-generated>,
    "scores": {
        "Physical/Structural Logic": <float between 0.0 and 1.0>,
        "Lighting & Shadows": <float between 0.0 and 1.0>,
        "Texture & Material": <float between 0.0 and 1.0>,
        "Background & Depth": <float between 0.0 and 1.0>,
        "Overall Coherence": <float between 0.0 and 1.0>
    },
    "artifacts": [
        {
            "label": "<Short string, e.g., '6 fingers' or 'Warped window'>",
            "box_2d": [<ymin>, <xmin>, <ymax>, <xmax>] 
        }
    ],
    "report": "<Your step-by-step analysis and the FINAL judgment string>"
}

Note for box_2d: Coordinates MUST be normalized floats between 0.00 and 1.00 (e.g., [0.25, 0.40, 0.50, 0.60]). If no distinct artifacts are found, return an empty list [].
"""


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def draw_bounding_boxes(image_path, artifacts):
    """在图像上绘制模型返回的高亮错误框"""
    try:
        img = Image.open(image_path).convert("RGBA")
        # 创建一个可以绘制半透明图层的遮罩
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        width, height = img.size

        for art in artifacts:
            box = art.get("box_2d", [])
            label = art.get("label", "Artifact")

            if len(box) == 4:
                ymin, xmin, ymax, xmax = box

                # 防御性编程：如果 Qwen 返回了 0-1000 格式的坐标，将其归一化
                if ymax > 1.0 or xmax > 1.0:
                    ymin, xmin, ymax, xmax = ymin / 1000, xmin / 1000, ymax / 1000, xmax / 1000

                # 将归一化坐标转换为实际像素坐标
                abs_xmin = max(0, int(xmin * width))
                abs_ymin = max(0, int(ymin * height))
                abs_xmax = min(width, int(xmax * width))
                abs_ymax = min(height, int(ymax * height))

                # 绘制半透明橙色填充框
                draw.rectangle(
                    [abs_xmin, abs_ymin, abs_xmax, abs_ymax],
                    fill=(255, 165, 0, 70),  # 半透明橙色
                    outline="darkorange",
                    width=3
                )

                # 绘制标签文本背景
                try:
                    text_bbox = draw.textbbox((abs_xmin, max(0, abs_ymin - 15)), label)
                    draw.rectangle(text_bbox, fill="darkorange")
                except AttributeError:
                    # 兼容老版本 Pillow
                    draw.rectangle([abs_xmin, max(0, abs_ymin - 15), abs_xmin + len(label) * 6, abs_ymin],
                                   fill="darkorange")

                # 绘制标签文字
                draw.text((abs_xmin, max(0, abs_ymin - 15)), label, fill="white")

        # 将绘制好的半透明层与原图合并
        result_img = Image.alpha_composite(img, overlay)
        return result_img.convert("RGB")

    except Exception as e:
        print(f"Error drawing boxes: {e}")
        # 如果绘制失败，直接返回原图
        return Image.open(image_path).convert("RGB")


def analyze_image(image_path, api_key):
    if not image_path:
        return None, "Please upload an image.", "Error: No image uploaded.", "Error: No image uploaded."
    if not api_key:
        return None, "Please provide a Qwen API Key.", "Error: Missing API Key.", "Error: Missing API Key."

    try:
        custom_http_client = httpx.Client(verify=False)

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=custom_http_client,
            timeout=60.0
        )

        base64_image = encode_image_to_base64(image_path)

        # Call qwen-vl-max
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
        )

        result_text = response.choices[0].message.content.strip()

        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "", 1)
        if result_text.endswith("```"):
            result_text = result_text.rsplit("```", 1)[0]

        data = json.loads(result_text.strip())

        # 1. 解析坐标并在图片上画框
        artifacts = data.get("artifacts", [])
        annotated_image = draw_bounding_boxes(image_path, artifacts)

        # 2. 获取分类并生成 HTML 概率条
        category = data.get("category", "Unknown Category")
        prob_percent = data.get("probability", 0) * 100
        color = "red" if prob_percent > 50 else "green"
        prob_html = f"""
        <div style='margin-bottom: 10px; padding: 8px; background-color: #fff7ed; border-left: 4px solid orange; color: #9a3412; font-weight: bold;'>
            Detected Subject Category: {category}
        </div>
        <div style='margin-top: 10px;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                <span style='font-weight: bold; color: #374151;'>AI-Generated Probability</span>
                <span style='font-weight: bold; font-size: 1.25rem; color: {color};'>{prob_percent:.1f}%</span>
            </div>
            <div style='height: 16px; width: 100%; background-color: #e5e7eb; border-radius: 9999px; overflow: hidden;'>
                <div style='height: 100%; width: {prob_percent}%; background-color: {color}; border-radius: 9999px;'></div>
            </div>
        </div>
        """

        # 3. 生成 Taxonomy Markdown 表格 (替换 Dataframe)
        scores = data.get("scores", {})
        score_md = "| Dimension | Contradiction Score (0.0 - 1.0) |\n| :--- | :--- |\n"
        for dim, score in scores.items():
            score_md += f"| **{dim}** | {score} |\n"

        # 4. Extract the Report text
        report = data.get("report", "No report generated.")

        # 注意：这里返回的是 annotated_image (PIL 对象)
        return annotated_image, prob_html, score_md, report

    except json.JSONDecodeError:
        error_msg = f"Failed to parse Qwen output as JSON. Raw output:\n{result_text}"
        return image_path, "Error parsing JSON.", error_msg, error_msg
    except Exception as e:
        return image_path, f"API Error: {str(e)}", f"An error occurred: {str(e)}", f"An error occurred: {str(e)}"


# --- Gradio UI Layout ---
custom_theme = gr.themes.Default(primary_hue="orange")

with gr.Blocks(theme=custom_theme, title="DeepFake Forensic Analyzer") as demo:
    gr.Markdown("# DeepFake Forensic Analyzer")
    gr.Markdown(
        "Upload an image to inspect physical logic, lighting, and textures using Qwen-VL. The model will auto-detect the subject category and highlight artifacts.")

    with gr.Row():
        api_key_input = gr.Textbox(
            label="Qwen (DashScope) API Key",
            placeholder="sk-...",
            type="password"
        )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Input Image")
            analyze_btn = gr.Button("Run Qwen Analysis", variant="primary")

        with gr.Column(scale=1):
            # 修改点：将 type 改为 "pil"，以接收后端画好框的图片对象
            image_output = gr.Image(type="pil", label="Detection Visualization", interactive=False)
            prob_output = gr.HTML(label="AI Probability")

    with gr.Row():
        with gr.Tabs():
            with gr.TabItem("Taxonomy Scoring"):
                score_output = gr.Markdown("Awaiting analysis...")
            with gr.TabItem("Analysis Report"):
                report_output = gr.Textbox(
                    lines=10,
                    label="Step-by-Step Analysis",
                    interactive=False
                )

    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input, api_key_input],
        outputs=[image_output, prob_output, score_output, report_output],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(share=False)