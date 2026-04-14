# Multi-generator AI Image Detection Based on GenImage
**DSAI5201 Group Project (Group 7)**

## 📌 Project Overview
This project evaluates the performance of Multimodal Large Language Models (MLLMs), specifically **Qwen3-VL**, in detecting AI-generated images across various generators and content categories. Based on the **GenImage** dataset, we conducted a systematic evaluation pipeline including baseline comparison, prompt tuning, robustness testing, and failure case taxonomy.

### 🚀 Key Features
- **Comprehensive Benchmarking:** Comparison between traditional methods (FFT, CLIP) and state-of-the-art MLLMs.
- **Multi-Generator Evaluation:** Testing across 8 different generators (Stable Diffusion, Midjourney, StyleGAN, etc.).
- **Interpretability:** Analysis of MLLM reasoning through a custom-built **Artifact Taxonomy**.
- **Interactive Demo:** A Gradio-based "DeepFake Analyzer" for real-time detection and reasoning.

---

## 📂 Project Structure
As the project consists of five independent experiments, the repository is organized by experiment stages:

- `Exp1_Baselines/` : FFT and CLIP baseline implementation.
- `Exp2_PromptTuning/` : Ablation study on different prompt strategies.
- `Exp3_Generators/` : Evaluation across 8 AI generators.
- `Exp4_ContentCategory/` : Detection difficulty across image categories.
- `Demo/` : Source code for the Gradio DeepFake Analyzer.
- `sample_data/` : Sample images for testing code execution.
- `DSAI5201_group7_report.pdf` : Full technical report.

> **Note on Datasets:** Due to GitHub's file size limits, only a small subset of sample data is included in `sample_data/` to verify the code execution. The full dataset used in our experiments is based on the official [GenImage Dataset](https://github.com/GenImage-Dataset/GenImage).

---

## 📺 Demo Presentation
Our Gradio-based analyzer provides real-time detection and interpretable reasoning.

> [!TIP]
> **[Watch the Demo Video Here](./Demo/demo.mp4)**

---

## 📊 Core Experimental Findings

> [!IMPORTANT]
> **Key takeaways from our 5-stage experiment pipeline:**
> * **Interpretability:** MLLMs provide human-readable reasoning that traditional "black-box" baselines (FFT/CLIP) lack.
> * **Prompt Impact:** Chain-of-Thought (CoT) prompting significantly mitigates the "Real-Image Bias".
> * **The "Lighting" Challenge:** According to our Artifact Taxonomy, Lighting Consistency is the most common reason for MLLM misjudgment.
> * **Content Sensitivity:** Detection accuracy is highest for Faces but lowest for complex Nature/Scenery images.

---

## 🛠️ Quick Start

**Environment Setup**
Please ensure you have configured your local environment according to the script requirements (e.g., Gradio, PyTorch, Qwen-VL API).

**Running the Demo**
To launch our interactive detection system, navigate to the Demo folder and run:
```bash
cd Demo
python app_demo.py

## 👥 Statement of Contribution
| Name | Core Tasks |
| :--- | :--- |
| **Gao Jing** | Report framework, FFT/CLIP baselines, MLLM benchmarking. |
| **Zhao Kangzhe** | Prompt tuning strategy, quantitative visualization. |
| **Sun Yaqi** | Cross-generator experiments, Error taxonomy formulation. |
| **Yang Qi** | Category analysis, Gradio system development. |