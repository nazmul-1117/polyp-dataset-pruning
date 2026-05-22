# Refined Thesis Title Options

### Primary Recommended Title

**Adaptive Semantic Pruning (ASP): A Training-Free Dataset Pruning Framework for Medical Image Segmentation Using Self-Supervised Embeddings**

### Stronger Academic Variant

**Adaptive Semantic Graph Pruning for Annotation-Efficient Medical Image Segmentation via Self-Supervised Representation Learning**

### More Publication-Oriented Variant

**Training-Free Semantic Dataset Pruning for Medical Image Segmentation Using DINOv2 and Graph Community Detection**

---

# Research Problem Statement

Modern medical image segmentation datasets contain massive redundancy, especially in video-derived datasets such as colonoscopy sequences. Existing pruning approaches—including PRIME—primarily depend on low-level structural similarity metrics (e.g., SSIM), which fail to capture semantic equivalence under variations in viewpoint, illumination, deformation, and scale.

This results in:

* inefficient annotation expenditure,
* excessive training computation,
* over-representation of near-identical frames,
* and reduced scalability for clinical AI systems.

Your thesis addresses this by proposing a **training-free semantic pruning framework** that operates directly on unlabeled data using self-supervised embeddings and adaptive graph-based redundancy modeling.

---

# Core Novel Contributions

Your actual novelty is not “using DINOv2.”
The novelty comes from the *combination and automation* of the pipeline.

You can formally present your contributions as:

---

## Contribution 1 — Semantic Redundancy Modeling

Replace pixel/SSIM similarity with semantic embeddings extracted from:

DINOv2

This allows the framework to recognize medically similar structures despite visual perturbations.

### Why this matters

SSIM says:

> “These pixels changed.”

DINOv2 embeddings say:

> “This is still the same anatomical structure.”

That distinction is academically important.

---

## Contribution 2 — Adaptive Threshold Selection

Your strongest methodological innovation.

Instead of manually tuning similarity thresholds for each dataset:

[
\tau = \mu + \alpha \sigma
]

where:

* (\mu) = mean pairwise similarity,
* (\sigma) = standard deviation,
* (\alpha) = adaptive control parameter.

You can visualize this directly:

\tau = \mu + \alpha\sigma

This transforms the framework from:

* dataset-specific
  to:
* dataset-agnostic.

That is a significant research contribution.

---

## Contribution 3 — Graph-Based Semantic Community Detection

You model the dataset as a similarity graph:

[
G = (V, E)
]

where:

* nodes = images,
* edges = semantic similarity relationships.

Then apply:

Louvain Algorithm

to discover redundant semantic communities automatically.

This is much stronger academically than simple clustering because:

* graph topology preserves relational structure,
* communities emerge naturally,
* no fixed cluster count is needed.

---

## Contribution 4 — Diversity-Preserving Pruning

This is another potentially publishable angle.

Instead of naive centroid-only retention:

* preserve representative samples,
* retain edge-case diversity,
* protect rare anomalies.

This directly addresses a major weakness in aggressive pruning systems:

> accidental deletion of clinically important rare patterns.

You should emphasize this heavily in your thesis.

---

# Strong Technical Pipeline (Final Architecture)

Your pipeline can now be formalized as:

---

## Stage 1 — Data Acquisition

Datasets:

* Kvasir-SEG
* CVC-ClinicDB

---

## Stage 2 — Semantic Embedding Extraction

Input image:

[
I_i
]

Feature embedding:

[
z_i = f_{\text{DINOv2}}(I_i)
]

Visual form:

z_i = f_{\mathrm{DINOv2}}(I_i)

---

## Stage 3 — Similarity Matrix Construction

Cosine similarity:

[
S_{ij} = \frac{z_i \cdot z_j}{|z_i||z_j|}
]

S_{ij}=\frac{z_i\cdot z_j}{|z_i||z_j|}

---

## Stage 4 — Adaptive Graph Construction

Edge condition:

[
E_{ij} =
\begin{cases}
1 & \text{if } S_{ij} > \tau \
0 & \text{otherwise}
\end{cases}
]

E_{ij}=\begin{cases}1 & \text{if } S_{ij}>\tau\0 & \text{otherwise}\end{cases}

---

## Stage 5 — Community Detection

Apply:

* Louvain clustering
* modularity optimization

to discover semantic redundancy groups.

---

## Stage 6 — Representative Selection

For each community:

* compute centroid,
* retain representative node,
* optionally preserve top-(k) diverse samples.

Potential diversity score:

[
D_i = 1 - \frac{1}{|C|}\sum_{j \in C} S_{ij}
]

D_i = 1 - \frac{1}{|C|}\sum_{j\in C} S_{ij}

---

# Experimental Design (Very Important)

Your thesis becomes much stronger if experiments are structured around **ablation studies**.

You should compare:

| Method          | Similarity Type | Threshold | Clustering |
| --------------- | --------------- | --------- | ---------- |
| Random Sampling | None            | None      | None       |
| PRIME Baseline  | SSIM            | Manual    | Graph      |
| ASP (yours)     | DINOv2          | Adaptive  | Louvain    |

Then add ablations:

| Variant                    | Purpose                                  |
| -------------------------- | ---------------------------------------- |
| Without adaptive threshold | Tests automation benefit                 |
| Without diversity injector | Tests rare-case preservation             |
| SSIM + Louvain             | Isolates semantic embedding contribution |
| DINOv2 + KMeans            | Tests graph superiority                  |

This dramatically improves publication quality.

---

# Evaluation Metrics

You already identified the key segmentation metrics:

* Dice Coefficient
* mIoU

You should also add:

### Pruning Metrics

* Reduction Ratio (%)
* Annotation Savings
* GPU Training Time Reduction
* Storage Reduction

### Graph Metrics

* Graph Density
* Community Count
* Modularity Score

### Statistical Validation

Use:

* paired t-test,
* Wilcoxon signed-rank test,
* confidence intervals.

That elevates the thesis from engineering to research.

---

# Strong Thesis Hypothesis

You need a formal hypothesis section.

A strong version:

> Semantic graph-based pruning using self-supervised embeddings can remove over 50% of redundant medical images while preserving downstream segmentation performance within statistically insignificant margins.

---

# Your Biggest Research Strength

Your thesis is positioned at the intersection of:

* self-supervised learning,
* graph theory,
* data-centric AI,
* annotation efficiency,
* medical vision.

That combination is current and publication-relevant.

Importantly:

* you are not proposing “another segmentation model,”
* you are improving the *data pipeline itself*.

That is increasingly valuable in medical AI research.

---

# Potential Publication Venues

Depending on quality:

### Good Target Conferences/Journals

* MICCAI
* ISBI
* Computerized Medical Imaging and Graphics
* Biomedical Signal Processing and Control
* Scientific Reports

---

# Final Assessment

Your work is no longer just:

> “dataset reduction.”

It has evolved into:

> a semantic graph-learning framework for annotation-efficient medical AI.

That framing is substantially stronger for:

* thesis defense,
* publication,
* future PhD applications,
* and research positioning.
