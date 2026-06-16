# 🚦 Traffic Sign Classification on the Mapillary Traffic Sign Dataset

> From HOG+SVM to CNN, Transfer Learning, and Robustness Under Synthetic Distortions

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-available-2496ED?logo=docker)](https://hub.docker.com/r/rashadeltaher/cnn-predictor)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit)](https://cnn-predictor.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Overview

This repository contains the full implementation, trained models, and deployment code for a comprehensive study of traffic-sign classification on **[Mapillary Traffic Sign Dataset v2 (MTSD)](https://www.mapillary.com/dataset/trafficsign)** — the largest open multi-class traffic-sign dataset available.

We benchmark four approaches on an identical, apples-to-apples experimental setup:

| Model | Top-1 Acc | Macro-F1 |
|---|---|---|
| HOG → SGD-SVM (classical baseline) | 55.97 % | 0.524 |
| BaselineCNN — no augmentation | 96.55 % | 0.949 |
| BaselineCNN + augmentation | 96.89 % | **0.958** |
| EfficientNet-B0 (transfer learning) | **96.98 %** | 0.956 |

> **+40.6 pt absolute gap** between classical and CNN approaches — a 73 % relative error reduction.

**Course:** CIE 552 Vision · **Supervised by:** Dr. Mohamed Tolba · **May 2026**

**Team:**
- Rashad Mohamed (202200261)
- Ahmed Salah (202200212)
- Hassan Ahmed (202202121)

---

## ✨ Key Highlights

- 🗂️ **326-class curated subset** of MTSD (67,833 crops) with reproducible train/val/test splits
- 🧠 **Custom BaselineCNN** (2.38 M params) trained from scratch — 96.55 % top-1
- 🔁 **Transfer learning** with ImageNet-pretrained EfficientNet-B0 — 96.98 % top-1
- 🎨 **Data augmentation study** — best macro-F1 across all 326 classes (0.958)
- 🛡️ **Robustness sweep** across 6 distortion types × 4 severity levels
- 🚀 **Production deployment** via Streamlit, FastAPI, Docker, ONNX, and Core ML

---

## 🏗️ Repository Structure

```
.
├── src/
│   ├── data/               # Dataset curation, crop extraction, split logic
│   ├── models/
│   │   ├── baseline_cnn.py # BaselineCNN architecture (2.38 M params)
│   │   └── efficientnet.py # EfficientNet-B0 fine-tuning wrapper
│   ├── train.py            # Unified training script (all experiments)
│   ├── evaluate.py         # Held-out test evaluation + per-class CSV
│   ├── distortions.py      # Six synthetic distortions × four severities
│   └── deploy/
│       ├── app.py          # Gradio / Streamlit interactive demo
│       ├── api.py          # FastAPI inference service
│       └── Dockerfile      # CPU-only containerised service
├── notebooks/
│   ├── visionPhase1.ipynb  # Classical pipeline (HOG → SGD-SVM)
│   └── visionPhase2.ipynb  # CNN experiments + robustness analysis
├── results/
│   └── aug/
│       ├── test_per_class.csv
│       └── export/
│           ├── model.onnx          # ~10 MB cross-platform export
│           └── model.mlpackage     # ~9 MB Apple Neural Engine export
├── vision_v2.pdf           # Full project report
└── README.md
```

---

## 🗃️ Dataset & Curation

MTSD v2 provides bounding-box annotations for 400+ sign types collected from real-world street imagery worldwide. We apply the following curation policy to create a clean, learnable benchmark:

1. **Discard `other-sign`** — 63.6 % of raw crops (121,136 / 190,496) are an uncategorized catch-all that would bias the model and dominate metrics.
2. **Drop classes with < 30 train samples** — 74 tail classes too rare to learn from.
3. **Drop crops < 12 px on either side** — near-zero discriminative content after resizing.

**Result:** 326 classes · 67,833 crops

| Split | Crops | Source |
|---|---|---|
| Train | 53,306 | 90 % of MTSD-train |
| Val | 5,923 | 10 % of MTSD-train |
| Test | 8,604 | All of MTSD-val (held out) |

All crops are resized to **96 × 96 px**. Class imbalance is handled at the loss level via class-weighted cross-entropy (weights ∝ √(1/freq), normalised to mean 1.0).

---

## 🔬 Methodology

### Phase 1 — Classical Baseline (HOG → SGD-SVM)

```
Grayscale → Gaussian blur (5×5) → Canny edges
→ HOG (9 orientations, 8×8 cells, 2×2 blocks, L2-Hys) → 1,764-dim vector
→ StandardScaler → SGDClassifier (hinge, L2, α=1e-4, balanced weights)
```

SGDClassifier was selected over `LinearSVC` to keep training tractable (**164 s** vs **> 60 min**) on this 326-class, 53 k-sample regime. Both produce equivalent linear SVMs.

### Phase 2 — BaselineCNN (from scratch)

```
Stem:    Conv 7×7 /2 → BN → ReLU → MaxPool 3×3 /2       → [64, 24, 24]
Block 1: (Conv 3×3 → BN → ReLU) × 2 → MaxPool 2×2        → [128, 12, 12]
Block 2: (Conv 3×3 → BN → ReLU) × 2 → MaxPool 2×2        → [256, 6, 6]
Block 3: (Conv 3×3 → BN → ReLU) × 2                       → [256, 6, 6]
Head:    GAP → Dropout(0.3) → Linear(326)
```

**Training recipe:** AdamW (lr 3e-3, wd 1e-4) · OneCycle LR (10 % warmup) · batch 128 · 50 epochs · label smoothing 0.05

### Experiments

| ID | Description |
|---|---|
| E1 | **Data augmentation** — rotation ±15°, brightness/contrast ±25 %, hue/saturation jitter, Gaussian blur, CoarseDropout, scale/translation jitter (Albumentations) |
| E2 | **Transfer learning** — EfficientNet-B0 (ImageNet, via `timm`), same recipe with lr=1e-3 |
| E3 | **Robustness sweep** — 6 distortions × 4 severities on the held-out test set |

---

## 📊 Results

### Headline Comparison

| Model | Top-1 | Top-5 | Macro-P | Macro-R | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|---|
| HOG → SGD-SVM | 55.97 % | — | 0.555 | 0.553 | 0.524 | 0.591 |
| BaselineCNN (no aug) | 96.55 % | 99.43 % | 0.956 | 0.949 | 0.949 | 0.965 |
| BaselineCNN + aug | 96.89 % | 99.51 % | 0.963 | 0.958 | **0.958** | 0.969 |
| EfficientNet-B0 | **96.98 %** | 99.48 % | 0.959 | 0.957 | 0.956 | **0.970** |

### Robustness Under Synthetic Distortions

*(Evaluated on the augmentation-trained model — best macro-F1)*

| Distortion | Sev 0 | Sev 1 | Sev 2 | Sev 3 | Drop |
|---|---|---|---|---|---|
| Gaussian noise | 96.89 % | 96.84 % | 96.87 % | 96.83 % | **0.06 pt** ✅ |
| JPEG compression | 96.89 % | 96.89 % | 96.90 % | 96.37 % | 0.51 pt ✅ |
| Motion blur | 96.89 % | 96.87 % | 96.90 % | 96.44 % | 0.44 pt ✅ |
| Rotation | 96.89 % | 96.87 % | 96.47 % | 95.36 % | 1.52 pt ✅ |
| Illumination | 96.89 % | 96.75 % | 94.64 % | 89.14 % | 7.74 pt ⚠️ |
| Occlusion | 96.89 % | 96.48 % | 95.27 % | 86.85 % | **10.03 pt** ❌ |

The model is near-immune to noise, blur, JPEG artifacts, and small rotation, but degrades under severe illumination shifts and large occlusions — motivating targeted augmentation as future work.

---

## 🚀 Deployment

The augmentation-trained model is deployed in four ways:

### 1. 🌐 Streamlit Live Demo
**[cnn-predictor.streamlit.app](https://cnn-predictor.streamlit.app/)**
Upload any traffic sign image and get the top-5 predictions with confidence scores.

### 2. 🐳 Docker (FastAPI Service)
```bash
docker pull rashadeltaher/cnn-predictor
docker run -p 8000:8000 rashadeltaher/cnn-predictor
# POST /predict with image file
curl -X POST http://localhost:8000/predict -F "file=@sign.jpg"
```
End-to-end JSON round-trip ≤ 20 ms / request on CPU.

### 3. 📡 Gradio App (local)
```bash
python src/deploy/app.py
```

### 4. 📦 ONNX / Core ML
| Format | Size | Use Case |
|---|---|---|
| `results/aug/export/model.onnx` | ~10 MB | Cross-platform inference |
| `results/aug/export/model.mlpackage` | ~9 MB | Apple Neural Engine (iOS/macOS) |

---

## ⚙️ Getting Started

### Prerequisites

```bash
python >= 3.9
torch >= 2.0
torchvision
timm
albumentations
scikit-learn
opencv-python
gradio / fastapi / streamlit   # for deployment
```

### Installation

```bash
git clone https://github.com/A7medSala71/Computer-Vision-Project.git
cd mtsd-traffic-sign-classification
pip install -r requirements.txt
```

### Download & Prepare Data

1. Download MTSD v2 from [mapillary.com/dataset/trafficsign](https://www.mapillary.com/dataset/trafficsign)
2. Run the curation and crop-extraction script:
```bash
python src/data/curate.py --mtsd_root /path/to/mtsd --output_dir data/crops
```

### Training

```bash
# BaselineCNN — no augmentation
python src/train.py --model baseline --no_aug

# BaselineCNN + augmentation (best macro-F1)
python src/train.py --model baseline --aug

# EfficientNet-B0 transfer learning (best top-1)
python src/train.py --model efficientnet --aug
```

### Evaluation

```bash
python src/evaluate.py --checkpoint results/aug/best.pth --split test
# Outputs: results/aug/test_per_class.csv + robustness sweep
```

### Notebooks

| Notebook | Description |
|---|---|
| `notebooks/visionPhase1.ipynb` | Classical HOG → SGD-SVM pipeline |
| `notebooks/visionPhase2.ipynb` | CNN experiments, robustness analysis, visualisations |

---

## 📄 Report

The full project report is available as [`vision v2.pdf`](vision v2.pdf), covering:
- Dataset curation decisions
- Architecture design and training recipe justifications
- Complete results tables and training dynamics
- Robustness analysis
- Deployment details
- Limitations and future work

---

## 🔭 Future Work

- [ ] Hierarchical classifier (super-category → fine-grained)
- [ ] Curriculum learning for tail classes near the 30-sample threshold
- [ ] Stronger brightness/contrast + large CoarseDropout augmentation to close the illumination and occlusion gaps
- [ ] End-to-end pipeline with YOLO-class detector (no ground-truth crops at inference)

---

## 📜 References

1. Ertler et al., "The Mapillary Traffic Sign Dataset for Detection and Classification on a Global Scale", ECCV 2020.
2. Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection", CVPR 2005.
3. Stallkamp et al., "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition", Neural Networks 2012.
4. Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
5. Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations", ICLR 2019.
6. Zhu et al., "Traffic-Sign Detection and Classification in the Wild", CVPR 2016.

---

## 🪪 License

This project is released under the [MIT License](LICENSE). The MTSD dataset is subject to [Mapillary's terms of use](https://www.mapillary.com/dataset/trafficsign).
