import json
import csv
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns 

from transformers import AutoProcessor, MllamaForConditionalGeneration
from peft import PeftModel

# --- DYNAMIC PATH CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MODEL_PATH = SCRIPT_DIR / "models" / "base_model"
LORA_ADAPTER_PATH = SCRIPT_DIR / "outputs" / "lora_weights" / "final"
TEST_DATASET_ROOT = SCRIPT_DIR / "data" / "test"
RESULTS_OUTPUT_DIR = SCRIPT_DIR / "reports"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Label mapping for industrial defect categories
QUALITY_LABELS = {
    "0_good": "good",
    "1_scratch": "scratch",
    "2_paint": "paint",
    "3_over-coupling": "over-coupling",
    "4_lacking": "lacking"
}

print("🧠 Loading Vision Processor...")
processor = AutoProcessor.from_pretrained(str(BASE_MODEL_PATH))

print("⚙️ Loading Base Llama 3.2 Vision weights (FP16)...")
base_model = MllamaForConditionalGeneration.from_pretrained(
    str(BASE_MODEL_PATH),
    device_map="auto",
    torch_dtype=torch.float16 
)

print("🔥 Injecting Fine-Tuned LoRA Knowledge...")
model = PeftModel.from_pretrained(base_model, str(LORA_ADAPTER_PATH))
model.eval() 

def preprocess_image(image_path):
    """Resizes and converts BGR image to RGB PIL format."""
    raw_img = cv2.imread(str(image_path))
    if raw_img is None: return None
    
    h, w = raw_img.shape[:2]
    max_resolution = 1024 
    if max(h, w) > max_resolution:
        scale_factor = max_resolution / max(h, w)
        raw_img = cv2.resize(raw_img, (int(w * scale_factor), int(h * scale_factor)))
    
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

# Industrial Quality Control Prompt
INSPECTION_PROMPT = """
You are an industrial quality control computer vision system.
Analyze the screw image and classify it into EXACTLY ONE of these 5 categories:
- good
- scratch
- paint
- over-coupling
- lacking

Respond ONLY with the category word. No punctuation, no JSON, no explanations.
"""

def perform_inspection(image_path):
    """Runs inference on a single image and returns the predicted label."""
    img_pil = preprocess_image(image_path)
    if img_pil is None: return "error"

    chat_messages =}
    ]
    
    formatted_prompt = processor.apply_chat_template(chat_messages, add_generation_prompt=True)
    model_inputs = processor(images=img_pil, text=formatted_prompt, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad(): 
            inference_output = model.generate(
                **model_inputs, 
                max_new_tokens=15, 
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id 
            )
            
        raw_response = processor.decode(
            inference_output[model_inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip().lower()
        
    except Exception as e:
        print(f"\n❌ {e}")
        raise e 
        
    finally:
        torch.cuda.empty_cache()

    # Keyword-based classification logic
    for category in QUALITY_LABELS.values():
        if category in raw_response: return category
    return "good" # Fallback category

def main():
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    performance_stats = defaultdict(lambda: {"total": 0, "correct": 0, "failure": 0})
    csv_data =]
    
    test_files =
    print(f"\n📂 Scanning dataset at: {TEST_DATASET_ROOT}")
    for folder, logic_label in QUALITY_LABELS.items():
        folder_path = TEST_DATASET_ROOT / folder
        if folder_path.exists():
            images =
            print(f"   -> {folder}: {len(images)} images")
            for img in images: test_files.append((img, logic_label))
        else:
            print(f"   ⚠️ Folder missing: {folder}")

    if not test_files:
        print("❌ Error: No test images found.")
        return

    print(f"\n🚀 Starting Quality Inspection on {len(test_files)} components...")
    
    actual_labels =
    predicted_labels =

    for idx, (img_path, ground_truth) in enumerate(test_files):
        prediction = perform_inspection(img_path)

        match = (prediction == ground_truth)
        status_icon = "✅ PASS" if match else "❌ FAIL"
        
        performance_stats[ground_truth]["total"] += 1
        if match: performance_stats[ground_truth]["correct"] += 1
        else: performance_stats[ground_truth]["failure"] += 1
        
        actual_labels.append(ground_truth)
        predicted_labels.append(prediction)
        
        print(f"[{idx+1:03d}/{len(test_files)}] {img_path.name[:20]:<20} | {ground_truth:<15} | {prediction:<15} | {status_icon}")
        csv_data.append([img_path.name, ground_truth, prediction])

    # Save results to CSV
    csv_report_path = RESULTS_OUTPUT_DIR / "inspection_metrics.csv"
    with open(csv_report_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_data)

    # --- PROFESSIONAL CONFUSION MATRIX GENERATION ---
    print("\n📊 Generating high-fidelity confusion matrix...")
    sorted_labels = sorted(list(QUALITY_LABELS.values()))
    label_indices = {lbl: i for i, lbl in enumerate(sorted_labels)}
    cm_matrix = np.zeros((len(sorted_labels), len(sorted_labels)), dtype=int)
    
    for t, p in zip(actual_labels, predicted_labels):
        if t in label_indices and p in label_indices:
            cm_matrix[label_indices[t]][label_indices[p]] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted_labels, yticklabels=sorted_labels,
                cbar_kws={'label': 'Count'}, annot_kws={"size": 14}, linewidths=.5) 

    plt.title("Confusion Matrix - Llama 3.2 Vision (Fine-Tuned)", pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=14, labelpad=10)
    plt.xlabel('Model Prediction', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plot_path = RESULTS_OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300) 
    
    print("\n" + "="*75)
    print("🎯 EVALUATION COMPLETE")
    print("="*75)
    print(f"📂 Report Folder: {RESULTS_OUTPUT_DIR.resolve()}")
    print(f"📈 Chart: {plot_path.name}")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
