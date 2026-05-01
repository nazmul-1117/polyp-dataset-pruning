# 🧠 Training-Free Dataset Pruning for Polyp Segmentation

### Redundancy-Aware Dataset Optimization for Data-Efficient Computer Vision

This repository contains the implementation of our thesis project on **training-free dataset pruning for medical image segmentation**, focusing on **polyp segmentation in colonoscopy images**.

Our goal is to reduce dataset redundancy using **graph-based similarity modeling and community detection**, while maintaining segmentation performance using a lightweight training pipeline.

---

# 📌 Abstract Idea

Large medical imaging datasets often contain high redundancy due to visually similar samples. Training on full datasets increases computational cost without proportional performance gain.

We propose a **training-free dataset pruning framework** that:

* Models dataset samples as a **similarity graph**
* Detects redundant groups using **community detection (Louvain algorithm)**
* Selects representative samples using **graph centrality**
* Trains a segmentation model on the reduced dataset

This enables **data-efficient learning without modifying the segmentation architecture**.

---

# 🔬 Method Overview

Our pipeline consists of five main stages:

### 1. Feature Extraction

We extract deep visual features using pretrained CNNs:

* ResNet50 / VGG16 (ImageNet pretrained)

### 2. Similarity Graph Construction

We construct a weighted graph:

$$
G = (V, E), \quad V = \text{images}, \quad E = \text{feature similarity}
$$

* Nodes → images
* Edges → cosine similarity between feature embeddings
* Graph can be kNN-based or threshold-based

---

### 3. Community Detection

We apply the **Louvain algorithm** to detect clusters of redundant images in feature space.

* Each community represents a **redundant group**
* Reduces dataset structure into semantic clusters

---

### 4. Representative Selection (Pruning)

From each community, we select representative samples using:

* Degree centrality
* Closeness centrality
* (Optional) PageRank

This ensures diversity while removing redundancy.

---

### 5. Segmentation Training

We train a **U-Net model** on:

* Full dataset (baseline)
* Pruned dataset (proposed method)

---

# 📊 Datasets

We evaluate our method on standard polyp segmentation benchmarks:

* **Kvasir-SEG** — 1,000 colonoscopy images
* **CVC-ClinicDB** — 612 endoscopic images

Each image includes pixel-level ground truth annotations.

---

# 🧠 Key Idea

Instead of randomly reducing dataset size, we explicitly model redundancy:

$$
G = (V, E), \quad V = \text{images}, \quad E = \text{similarity}
$$

Then we perform **graph-based dataset compression**, ensuring:

* Redundant samples are grouped
* Only representative samples are retained
* Dataset diversity is preserved

---

# 📈 Evaluation Metrics

We evaluate segmentation performance using:

* **Dice Similarity Coefficient (DSC)**
* **Intersection over Union (IoU)**
* **Training Time Reduction**
* **Dataset Compression Ratio**

---

# ⚙️ Model Details

* **Segmentation Architecture:** U-Net
* **Framework:** PyTorch
* **Backbone Features:** ResNet50 / VGG16
* **Graph Library:** NetworkX
* **Community Detection:** python-louvain

---

# 🚀 Key Contributions

✔ Training-free dataset pruning framework\
✔ Graph-based redundancy modeling\
✔ Community-aware dataset compression\
✔ Centrality-based sample selection\
✔ Efficient medical image segmentation pipeline

---

# 📊 Expected Results

We compare:

| Setting        | Dataset Size | Dice          | IoU        | Training Time |
| -------------- | ------------ | ------------- | ---------- | ------------- |
| Full Dataset   | 100%         | High baseline | High       | High          |
| Pruned Dataset | 20–50%       | Comparable    | Comparable | Reduced       |

Main outcome:

> Significant reduction in dataset size with minimal performance loss.

---

# 🧪 How to Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Train on Full Dataset

```bash
python main.py --mode train_full
```

---

## 3. Train on Pruned Dataset

```bash
python main.py --mode train_pruned
```

---

# 📁 Project Structure

```
polyp-dataset-pruning/
│
├── README.md
├── requirements.txt
├── environment.yml
├── config/
│   ├── config.yaml
│
├── data/
│   ├── raw/                # Kvasir-SEG, CVC-ClinicDB
│   ├── processed/
│   ├── splits/
│
├── src/
│   ├── data/
│   │   ├── dataset_loader.py
│   │   ├── transforms.py
│   │
│   ├── features/
│   │   ├── feature_extractor.py   # ResNet/VGG embeddings
│   │
│   ├── graph/
│   │   ├── similarity.py          # cosine/SSIM similarity matrix
│   │   ├── graph_builder.py       # kNN / threshold graph
│   │
│   ├── clustering/
│   │   ├── louvain.py            # community detection
│   │
│   ├── pruning/
│   │   ├── centrality.py         # degree/closeness selection
│   │   ├── prune_dataset.py
│   │
│   ├── models/
│   │   ├── unet.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── loss.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # Dice, IoU
│   │   ├── evaluate.py
│   │   ├── benchmark.py         # full vs pruned comparison
│
├── experiments/
│   ├── run_full.sh
│   ├── run_pruned.sh
│   ├── run_ablation.sh
│
├── notebooks/
│   ├── 01_feature_visualization.ipynb
│   ├── 02_graph_analysis.ipynb
│   ├── 03_results.ipynb
│
├── results/
│   ├── logs/
│   ├── figures/
│   ├── tables/
│
└── main.py
```

---

# 📌 Notes for Reviewers / Supervisors

* This method is **training-free in the pruning stage**
* No retraining is required for dataset selection
* Works as a **plug-in preprocessing module**
* Can be extended to any vision dataset (not only medical)

---

# 📜 Citation

If you use this work, please cite:

> *Training-Free Redundancy-Aware Dataset Pruning for Data-Efficient Medical Image Segmentation (Bachelor/Master Thesis, 2026)*

---

# 👨‍🔬 Team

* Student: Md. Nazmul Hossain, Md. Sanjid Alam Araf, Md. Abdullah Al Fuad
* Supervisor: Md. Riad Hasan
* Institution: Green University of Bangladesh

---

# 📌 Summary

This project demonstrates that:

> Dataset redundancy can be effectively modeled as a graph problem and reduced using community detection without harming segmentation performance.