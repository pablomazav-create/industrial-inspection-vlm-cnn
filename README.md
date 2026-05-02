# Industrial Visual Inspection: CNN vs. VLM Architectures
This project explores the integration of Deep Learning for automated quality control in manufacturing lines, comparing traditional supervised learning with next-generation Vision-Language Models.

## 📌 Project Overview
As part of my Undergraduate Thesis (TFG), I developed a comparative framework to evaluate:
1. **CNNs (ResNet/YOLO):** Optimized for high-speed, specific defect detection.
2. **VLMs (Vision-Language Models):** Leveraged for descriptive, zero-shot anomaly detection and reasoning.

## 📊 Key Results
| Architecture | Accuracy | Latency | Use Case |
| :--- | :--- | :--- | :--- |
| **Custom CNN** | 98.2% | ~12ms | Real-time line inspection |
| **VLM (Zero-shot)** | 92.5% | ~150ms | Complex anomaly description |

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, OpenCV, Transformers (Hugging Face).
- **Models:** YOLOv8 / ResNet + CLIP-based VLMs.
