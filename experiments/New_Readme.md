Proposed Thesis Topic:
“ASP-AL: Adaptive Semantic Pruning with Active Learning for Annotation-Efficient Medical Image Segmentation”

Core Goal:
Build a hybrid training-efficient framework that:

1. Removes redundant/duplicate medical images,
2. Filters low-quality images (blur/noise),
3. Preserves informative and diverse samples,
4. Reduces annotation cost and training complexity,
5. Maintains segmentation accuracy with a smaller dataset.

Main Research Direction:
We are extending the base PRIME-style pruning approach by replacing pixel-based similarity methods (SSIM) with semantic embeddings from DINOv2 and combining it with lightweight Active Learning.

Proposed Workflow:

Phase 1 — Quality Filtering

* Remove blurry images
* Remove noisy/corrupted frames
* Basic image quality assessment using OpenCV metrics

Phase 2 — Semantic Feature Extraction

* Use pretrained DINOv2 encoder
* Generate semantic embeddings for all medical images
* No heavy training required (inference only)

Phase 3 — Adaptive Semantic Pruning (ASP)

* Build similarity graph using cosine similarity
* Adaptive thresholding instead of fixed thresholds
* Use Louvain clustering/community detection
* Remove redundant images while keeping representative samples

Phase 4 — Diversity Preservation

* Keep cluster centroids
* Retain extra diverse/outlier samples
* Prevent loss of rare medical anomalies

Phase 5 — Lightweight Active Learning

* Train small segmentation model (likely UNet)
* Use uncertainty/entropy scoring
* Recover informative hard samples if needed

Main Datasets:

* Kvasir-SEG
* CVC-ClinicDB

Tech Stack:

* Python
* PyTorch
* OpenCV
* scikit-learn
* NetworkX
* python-louvain

Expected Outcome:

* ~50–60% dataset reduction
* Minimal Dice/mIoU drop
* Lower annotation cost
* Faster segmentation training
* Better dataset quality

Important Notes:

* We are NOT training DINOv2 from scratch.
* Most heavy computation comes only from segmentation evaluation.
* Entire project is feasible on Kaggle/Colab free GPU tiers.
* We should first stabilize the ASP pipeline before adding advanced active learning modules.

Potential Contribution:
A hybrid “Quality-Aware Semantic Pruning + Active Learning” framework for efficient medical image segmentation datasets.