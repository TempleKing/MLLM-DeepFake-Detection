import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

import random
import time
import cv2
import base64
import numpy as np
import torch
import kagglehub
import httpx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import shutil
from tqdm import tqdm

# Configuration & Global Variables
SILICONFLOW_API_KEY = " "
NUM_REAL = 250  
NUM_AI = 250    

MODELS = {
    "Qwen2.5-72B": {"id": "Qwen/Qwen2.5-VL-72B-Instruct"}, 
    "Qwen3-32B":   {"id": "Qwen/Qwen3-VL-32B-Instruct"},   
    "Qwen3-8B":    {"id": "Qwen/Qwen3-VL-8B-Instruct"},    
    "GLM-4.6V":    {"id": "zai-org/GLM-4.6V"}              
}

PROMPT = """You are an expert AI image forensic analyst. 
Please carefully analyze this image to determine if it is a 'Real' photo or an 'AI-generated' image. 
First, briefly explain your reasoning by pointing out specific visual evidences (e.g., lighting, structural logic, textures, or artifacts). 
Finally, you MUST conclude your response with exactly: 'Final Answer: Real' or 'Final Answer: Fake'."""

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.family'] = 'DejaVu Sans'

client_sf = OpenAI(
    api_key=SILICONFLOW_API_KEY, 
    base_url="https://api.siliconflow.cn/v1", 
    timeout=240.0, 
    http_client=httpx.Client(trust_env=False) 
)

# Dataset Preparation Functions
def setup_and_download_kaggle_data():
    print("Start downloading dataset using kagglehub")
    path = kagglehub.dataset_download("yangsangtai/tiny-genimage")
    print(f"Download and extraction completed automatically! Path: {path}")
    return path

def build_test_dataset(base_dir, num_real, num_ai):
    real_imgs, ai_imgs = [], []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                f_path = os.path.join(root, file)
                if 'real' in root.lower() or 'nature' in root.lower(): real_imgs.append(f_path)
                elif 'ai' in root.lower() or 'fake' in root.lower(): ai_imgs.append(f_path)
    
    random.seed(2026) # Fixed seed to ensure reproducibility
    dataset = [(p, 0) for p in random.sample(real_imgs, min(num_real, len(real_imgs)))] + \
              [(p, 1) for p in random.sample(ai_imgs, min(num_ai, len(ai_imgs)))]
    random.shuffle(dataset)
    print(f"Dataset extraction complete: {num_real} real images, {num_ai} AI images.")
    return dataset

# Traditional Baselines (FFT & CLIP)
def extract_fft_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return np.zeros(2)
    img = cv2.resize(img, (256, 256))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return np.array([np.mean(magnitude_spectrum), np.std(magnitude_spectrum)])

def run_fft_baseline(dataset):
    print("Running FFT Baseline")
    X, y_true = [], []
    for path, label in tqdm(dataset, desc="FFT Processing"):
        X.append(extract_fft_features(path))
        y_true.append(label)
    
    X, y_true = np.array(X), np.array(y_true)
    clf = SVC(kernel='rbf')
    train_size = int(len(X) * 0.8)
    if train_size > 0:
        clf.fit(X[:train_size], y_true[:train_size]) 
        y_pred = clf.predict(X)
    else:
        y_pred = np.zeros_like(y_true)
    return y_true, y_pred

def run_clip_baseline(dataset):
    print("Running CLIP Baseline")
    model_id = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    y_true, y_pred = [], []
    choices = ["a real photo", "an AI generated image"]
    
    for path, label in tqdm(dataset, desc="CLIP Processing"):
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(text=choices, images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            pred = torch.argmax(probs, dim=1).item()
            y_pred.append(pred)
            y_true.append(label)
        except Exception as e:
            print(f"Error processing {path} with CLIP: {e}")
            
    return y_true, y_pred

# MLLM Execution Functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_answer(text):
    text_clean = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip().lower()
    if "final answer: fake" in text_clean or "is ai-generated" in text_clean: return 1
    elif "final answer: real" in text_clean or "is a real" in text_clean: return 0
    last_bit = text_clean[-60:]
    return 1 if "fake" in last_bit or "ai" in last_bit else 0 

def get_prediction(image_path, model_id):
    start_time = time.time()
    raw_text = ""
    base64_img = encode_image(image_path)
    
    max_retries = 5 
    for attempt in range(max_retries):
        try:
            response = client_sf.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}],
                temperature=0.1
            )
            raw_text = response.choices[0].message.content
            break 
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5) 
            else:
                raw_text = f"Error after retries: {e}"
                
    end_time = time.time()
    return parse_answer(raw_text), raw_text, end_time - start_time

# Plotting Functions
def plot_baseline_results(metrics_dict):
    methods = list(metrics_dict.keys())
    acc_scores = [metrics_dict[m]['acc'] for m in methods]
    f1_scores = [metrics_dict[m]['f1'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, acc_scores, width, label='Accuracy', color='skyblue')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='salmon')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison of Detection Methods (Baseline)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    fig.tight_layout()
    plt.savefig('experiment1_baseline_comparison.png', dpi=300)
    print("Saved experiment1_baseline_comparison.png")

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predict: Real', 'Predict: AI'], 
                yticklabels=['Actual: Real', 'Actual: AI'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")

# Main Execution Pipeline
if __name__ == "__main__":
    # Setup Dataset
    dataset_dir = setup_and_download_kaggle_data()
    dataset = build_test_dataset(dataset_dir, num_real=NUM_REAL, num_ai=NUM_AI) 
    if len(dataset) < 10: exit("Dataset extraction failed.")
    
    baseline_results = {}
    
    # Run FFT & CLIP Baselines 
    y_true_fft, y_pred_fft = run_fft_baseline(dataset)
    baseline_results['FFT'] = {'acc': accuracy_score(y_true_fft, y_pred_fft), 'f1': f1_score(y_true_fft, y_pred_fft)}
    
    y_true_clip, y_pred_clip = run_clip_baseline(dataset)
    baseline_results['CLIP'] = {'acc': accuracy_score(y_true_clip, y_pred_clip), 'f1': f1_score(y_true_clip, y_pred_clip)}

    # Setup for MLLM Execution 
    all_metrics, scatter_data, cm_dict = [], [], {}
    base_save_dir = "case_studies"
    os.makedirs(base_save_dir, exist_ok=True)

    # Run All MLLMs
    for model_alias, config in MODELS.items():
        print(f"\nRunning Model: {model_alias}...")
        y_true_mllm, y_pred_mllm, times = [], [], []
        
        # Create a specific error log file for the current model
        model_log_file = f"error_log_{model_alias}.txt"
        with open(model_log_file, "w", encoding="utf-8") as f:
            f.write(f"Error Reasoning Log for {model_alias}\n" + "="*40 + "\n")
            
        # Create a specific subfolder for the current model's image cases
        model_save_dir = os.path.join(base_save_dir, model_alias)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Reset counters at the beginning of each model's loop
        saved_correct = 0
        saved_error = 0
        
        for path, label in tqdm(dataset, desc=f"{model_alias}"):
            pred, raw_text, tm = get_prediction(path, config['id'])
            y_pred_mllm.append(pred)
            y_true_mllm.append(label)
            times.append(tm)
            
            actual_label_str = "Fake" if label == 1 else "Real"
            pred_label_str = "Fake" if pred == 1 else "Real"
            
            # (1) If the prediction is incorrect, log the error details to the model's text log
            if pred != label:
                with open(model_log_file, "a", encoding="utf-8") as f:
                    f.write(f"Image: {path} | True: {actual_label_str} | Pred: {pred_label_str}\nAnalysis:\n{raw_text}\n{'-'*40}\n")
            
            # (2) Save image and text analysis pairs into the model's specific subfolder
            is_correct = (pred == label)
            
            if is_correct:
                prefix = f"CorrectCase_{saved_correct}_{actual_label_str}"
                saved_correct += 1
            else:
                prefix = f"ErrorCase_{saved_error}_True{actual_label_str}_Pred{pred_label_str}"
                saved_error += 1
                
            ext = os.path.splitext(path)[1]
            # Copy the original image to the model's subfolder
            shutil.copy(path, os.path.join(model_save_dir, f"{prefix}{ext}"))
            # Save the corresponding detailed reasoning text
            with open(os.path.join(model_save_dir, f"{prefix}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Original File: {path}\nTrue Label: {actual_label_str}\nModel Predicted: {pred_label_str}\n" + "="*40 + f"\nMLLM Analysis Details:\n{raw_text}\n")
            
            time.sleep(5) # Prevent Rate Limiting
            
        # Calculate MLLM metrics
        acc = accuracy_score(y_true_mllm, y_pred_mllm)
        f1 = f1_score(y_true_mllm, y_pred_mllm, zero_division=0)
        avg_time = sum(times)/len(times) if times else 0
        
        # Save metrics for multi-model comparison charts
        all_metrics.extend([
            {'Model': model_alias, 'Metric': 'Accuracy', 'Score': acc},
            {'Model': model_alias, 'Metric': 'F1-Score', 'Score': f1}
        ])
        cm_dict[model_alias] = confusion_matrix(y_true_mllm, y_pred_mllm)
        scatter_data.append({'Model': model_alias, 'Accuracy': acc, 'Time (s)': avg_time})
        
        # Save baseline metrics specifically for Qwen3-32B to compare with FFT/CLIP later
        if model_alias == "Qwen3-32B":
            baseline_results['Qwen3-VL'] = {'acc': acc, 'f1': f1}
            plot_confusion_matrix(y_true_mllm, y_pred_mllm, title="Qwen3-VL Confusion Matrix", filename="cm_qwen.png")
            
        print(f"{model_alias} completed 500 images test: Acc={acc:.3f}, F1={f1:.3f}")

    # Plot Everything 
    print("\nGenerating plots")
    
    # Plot 1: Baseline comparison (FFT vs CLIP vs Qwen3-32B)
    plot_baseline_results(baseline_results)
    
    # Plot 2: All 4 MLLMs Bar Chart
    df_metrics = pd.DataFrame(all_metrics)
    plt.figure(figsize=(12, 6))
    g = sns.barplot(x='Metric', y='Score', hue='Model', data=df_metrics, palette='tab10', edgecolor='k')
    plt.title('Performance Comparison (500 Images)', fontweight='bold', pad=20)
    plt.ylim(0, 1.1)
    for container in g.containers: g.bar_label(container, fmt='%.3f', padding=3)
    plt.tight_layout()
    plt.savefig('500_metrics.png', dpi=300)

    # Plot 3: All 4 MLLMs Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(18, 4.5))
    axes_flat = axes.flatten() 
    for i, (name, cm) in enumerate(cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes_flat[i],
                    xticklabels=['Pred Real', 'Pred AI'], yticklabels=['True Real', 'True AI'])
        axes_flat[i].set_title(f'{name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('500_confusion_matrices.png', dpi=300)

    # Plot 4: Scatter Plot (Accuracy vs Speed)
    df_scatter = pd.DataFrame(scatter_data)
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x='Time (s)', y='Accuracy', hue='Model', style='Model', data=df_scatter, s=250, palette='tab10')
    plt.title('Accuracy vs. Inference Speed (500 Images)')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig('500_speed_acc.png', dpi=300)
