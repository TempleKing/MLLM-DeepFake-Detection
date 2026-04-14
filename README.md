**DSAI5201 Project**

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

> **Note on Datasets:** Due to GitHub's file size limits, only a small subset of sample data is included in `sample_data/` to verify the code execution. The full dataset used in our experiments is based on the official [GenImage Dataset](https://github.com/GenImage-Dataset/GenImage).

---

## 📺 Demo Presentation
Our Gradio-based analyzer provides real-time detection and interpretable reasoning.

> [!TIP]
> **Interactive Demo Video:**
> <br>
![demo](https://github.com/user-attachments/assets/39e0e048-8d15-4bf8-82eb-83b0f3417ba9)


---

## 📊 Core Experimental Findings

> [!IMPORTANT]
> **Key takeaways from our comprehensive evaluation pipeline:**
> * **Interpretability vs. Hallucination:** While MLLMs (like Qwen3-32B) match traditional baselines (CLIP) in accuracy, their multi-step reasoning can counter-productively lead to **"Logical Over-rationalization"**—using real-world physics to justify obvious AI artifacts.
> * **Prompt Polarization & Multi-Agent Solution:** Our ablation study reveals that an *Expert Persona* prompt maximizes fake detection but triggers a "Paranoia Effect" (high false positives). Conversely, a *Knowledge Checklist* prompt eliminates false alarms but causes "Attention Narrowing". We propose a **cascaded multi-agent architecture** to balance this trade-off.
> * **Generator Evolution Challenge:** Advanced diffusion models are significantly harder to detect than traditional GANs. Notably, **Midjourney** drops the MLLM's detection accuracy to near random guessing (50.5%).
> * **Content Category Sensitivity:** MLLMs exhibit a strong "Real-Image Bias". Detection accuracy is highest for **Faces (72.5%)** due to sensitivity to structural anomalies, but plummets for complex **Nature scenes (59.5%)**, where models suffer from "Complexity-driven Misguidance".
---

## 🧪 Experimental Pipeline & Reproduction

Our research follows a step-by-step pipeline from baseline benchmarking to advanced interpretability analysis.

### 1. Data Preparation
To run the scripts and notebooks, please organize your local images as follows:
- Place a subset of images in the `exp_/` folder.
- Ensure the folder contains both `Real` and `Fake` subdirectories (consistent with the file structure).

### 2. Step-by-Step Reproduction
We recommend navigating folders in order.

### 3. Requirements
- **API Key**: A valid Qwen (DashScope) API Key is required in the notebooks.
- **Packages**: `pip install openai httpx matplotlib pillow gradio`

---

## 👥 Statement of Contribution
| Name | Core Tasks |
| :--- | :--- |
| **Gao Jing** | Report framework, FFT/CLIP baselines, MLLM benchmarking. |
| **Zhao Kangzhe** | Prompt tuning strategy, quantitative visualization. |
| **Sun Yaqi** | Cross-generator experiments, Error taxonomy formulation. |
| **Yang Qi** | Category analysis, Gradio system development. |
