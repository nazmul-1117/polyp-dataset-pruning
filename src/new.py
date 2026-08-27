# ================================================================
# TRAINING-FREE DATASET PRUNING FOR POLYP SEGMENTATION
# Via Community Detection in DINOv2 Similarity Networks
#
# Dataset: Kvasir-SEG
# Backbone: DINOv2 ViT-S/14
# Graph: NetworkX
# Community Detection: Louvain
# Segmentation: Compact U-Net
#
# Google Colab / Python 3.12+
# ================================================================


# ================================================================
# MODULE 0 — INSTALLATION & IMPORTS
# ================================================================


import os
import glob
import time
import random
import zipfile
import shutil
import math
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

import networkx as nx
import community as community_louvain

# from google.colab import files


# ================================================================
# MODULE 1 — GLOBAL CONFIGURATION
# ================================================================

class Config:
    """
    Central configuration for the complete experiment.
    """

    # -----------------------------
    # Dataset
    # -----------------------------
    IMAGE_SIZE = (224, 224)
    TRAIN_RATIO = 0.80
    RANDOM_SEED = 42

    # -----------------------------
    # Graph pruning
    # -----------------------------
    TAU = 0.75
    RETENTION_P = 0.30

    # -----------------------------
    # Training
    # -----------------------------
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # -----------------------------
    # DataLoader
    # -----------------------------
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # -----------------------------
    # U-Net
    # -----------------------------
    IN_CHANNELS = 3
    OUT_CHANNELS = 1

    # -----------------------------
    # DINOv2
    # -----------------------------
    DINO_MODEL_NAME = "dinov2_vits14"

    # -----------------------------
    # Paths
    # -----------------------------
    WORK_DIR = "/content/kvasir_pruning"
    EXTRACT_DIR = os.path.join(WORK_DIR, "dataset")

    ZIP_NAME = "Kvasir-SEG.zip"
    ZIP_PATH = Path("/content/kvasir-seg.zip")

    # ImageNet normalization used by DINOv2
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    AMP_ENABLED = torch.cuda.is_available()


# ================================================================
# MODULE 2 — REPRODUCIBILITY
# ================================================================

def set_seed(seed: int = Config.RANDOM_SEED):
    """
    Make the experiment as reproducible as practical.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms can reduce performance slightly,
    # but are useful for a thesis experiment.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

print("=" * 70)
print("DEVICE:", Config.DEVICE)
print("CUDA AVAILABLE:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 70)


# ================================================================
# MODULE 3 — DATASET ZIP EXTRACTION
# ================================================================

class KvasirArchiveManager:
    """
    Handles uploading and extracting Kvasir-SEG.zip.
    """

    def __init__(self, work_dir: str, extract_dir: str):
        self.work_dir = work_dir
        self.extract_dir = extract_dir

        os.makedirs(self.work_dir, exist_ok=True)

    def locate_zip(self, zip_name: str):
        """
        Look for the ZIP in common Colab locations.
        """

        candidates = [
            os.path.join("/content", zip_name),
            os.path.join(self.work_dir, zip_name),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

        # Recursive fallback
        matches = glob.glob(
            os.path.join("/content", "**", zip_name),
            recursive=True
        )

        if matches:
            return matches[0]

        return None

    def upload_if_needed(self, zip_name: str):
        """
        Select and copy the ZIP file if it cannot already be found.
        """

        zip_path = self.locate_zip(zip_name)

        if zip_path is not None:
            print(f"Found ZIP: {zip_path}")
            return zip_path

        print(f"{zip_name} was not found.")
        print("Please select your Kaggle Kvasir-SEG ZIP file.")

        # Open Windows file picker
        root = Tk()
        root.withdraw()

        source_path = filedialog.askopenfilename(
            title="Select Kvasir-SEG ZIP file",
            filetypes=[
                ("ZIP files", "*.zip"),
                ("All files", "*.*"),
            ],
        )

        root.destroy()

        if not source_path:
            raise FileNotFoundError(
                "No ZIP file was selected."
            )

        # Rename/copy into expected location
        target_path = os.path.join(
            self.work_dir,
            zip_name
        )

        shutil.copy2(source_path, target_path)

        print(f"Copied ZIP to: {target_path}")

        return target_path

    def extract(self, zip_path: str):
        """
        Safely extract ZIP archive.
        """

        if not os.path.isfile(zip_path):
            raise FileNotFoundError(
                f"ZIP file does not exist: {zip_path}"
            )

        if os.path.isdir(self.extract_dir):
            print(
                f"Extraction directory already exists: "
                f"{self.extract_dir}"
            )
            return self.extract_dir

        os.makedirs(self.extract_dir, exist_ok=True)

        print(f"Extracting:\n{zip_path}")
        print(f"To:\n{self.extract_dir}")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.extract_dir)

        print("Extraction completed.")

        return self.extract_dir


# Run archive preparation
archive_manager = KvasirArchiveManager(
    Config.WORK_DIR,
    Config.EXTRACT_DIR
)

zip_path = Config.ZIP_PATH

dataset_root = archive_manager.extract(zip_path)


# ================================================================
# MODULE 4 — Kvasir-SEG FILE DISCOVERY
# ================================================================

class KvasirFileParser:
    """
    Discovers image/mask pairs regardless of the exact nested
    directory structure inside the ZIP.
    """

    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    }

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def _all_image_files(self):
        files_found = []

        for path in glob.glob(
            os.path.join(self.root_dir, "**", "*"),
            recursive=True
        ):
            if not os.path.isfile(path):
                continue

            suffix = Path(path).suffix.lower()

            if suffix in self.IMAGE_EXTENSIONS:
                files_found.append(path)

        return files_found

    def _is_mask_path(self, path: str):
        """
        Detect typical Kvasir mask directories/names.
        """

        lower = path.lower()

        mask_keywords = [
            "mask",
            "masks",
            "ground_truth",
            "groundtruth",
            "segmentation",
            "segmentations"
        ]

        parts = Path(path).parts

        for part in parts:
            part_lower = part.lower()

            if part_lower in mask_keywords:
                return True

        filename = Path(path).stem.lower()

        mask_suffixes = [
            "_mask",
            "-mask",
            "mask"
        ]

        return any(
            filename.endswith(suffix)
            for suffix in mask_suffixes
        )

    def discover(self):
        """
        Pair image files with masks using filename stems.
        """

        all_files = self._all_image_files()

        if not all_files:
            raise RuntimeError(
                "No image files were found. "
                "Check the extracted ZIP structure."
            )

        mask_files = [
            p for p in all_files
            if self._is_mask_path(p)
        ]

        non_mask_files = [
            p for p in all_files
            if not self._is_mask_path(p)
        ]

        print(f"Total image-like files: {len(all_files)}")
        print(f"Potential mask files: {len(mask_files)}")
        print(f"Potential image files: {len(non_mask_files)}")

        # Index masks by normalized stem.
        mask_index = {}

        for mask_path in mask_files:
            stem = Path(mask_path).stem.lower()

            # Remove common mask suffixes.
            for suffix in [
                "_mask",
                "-mask",
                "_seg",
                "-seg"
            ]:
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]

            mask_index[stem] = mask_path

        pairs = []

        for image_path in non_mask_files:
            stem = Path(image_path).stem.lower()

            candidates = [
                stem,
                stem + "_mask",
                stem + "-mask",
            ]

            matched_mask = None

            for candidate in candidates:
                if candidate in mask_index:
                    matched_mask = mask_index[candidate]
                    break

            # Also search by exact stem after removing possible
            # image naming suffixes.
            if matched_mask is None:
                base = stem.replace("_image", "")
                base = base.replace("-image", "")

                if base in mask_index:
                    matched_mask = mask_index[base]

            if matched_mask is not None:
                pairs.append(
                    (image_path, matched_mask)
                )

        if not pairs:
            raise RuntimeError(
                "No image/mask pairs were found.\n"
                "Inspect the extracted dataset structure."
            )

        pairs.sort(key=lambda x: x[0])

        print(f"\nSuccessfully paired samples: {len(pairs)}")

        return pairs


parser = KvasirFileParser(dataset_root)
pairs = parser.discover()

print("\nExample pairs:")
for image_path, mask_path in pairs[:5]:
    print("IMAGE:", image_path)
    print("MASK :", mask_path)
    print()


# ================================================================
# MODULE 5 — TRAIN/TEST SPLIT
# ================================================================

class DatasetSplitter:
    """
    Performs the strict train/test split BEFORE graph processing.
    """

    def __init__(
        self,
        pairs,
        train_ratio=Config.TRAIN_RATIO,
        seed=Config.RANDOM_SEED
    ):
        self.pairs = list(pairs)
        self.train_ratio = train_ratio
        self.seed = seed

    def split(self):
        n = len(self.pairs)

        indices = np.arange(n)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)

        train_count = int(
            round(n * self.train_ratio)
        )

        train_indices = indices[:train_count]
        test_indices = indices[train_count:]

        train_pairs = [
            self.pairs[i]
            for i in train_indices
        ]

        test_pairs = [
            self.pairs[i]
            for i in test_indices
        ]

        return train_pairs, test_pairs


splitter = DatasetSplitter(pairs)

train_pairs, test_pairs = splitter.split()

print("=" * 70)
print("DATASET PARTITION")
print("=" * 70)
print("Total samples :", len(pairs))
print("Training      :", len(train_pairs))
print("Testing       :", len(test_pairs))
print(
    "Train ratio   :",
    len(train_pairs) / len(pairs)
)
print(
    "Test ratio    :",
    len(test_pairs) / len(pairs)
)
print("=" * 70)


# ================================================================
# MODULE 6 — PYTORCH DATASET
# ================================================================

class KvasirSegDataset(Dataset):
    """
    PyTorch dataset for Kvasir-SEG.

    Images:
        RGB -> resize -> tensor -> ImageNet normalization

    Masks:
        grayscale -> resize with nearest-neighbor -> binary tensor
    """

    def __init__(
        self,
        pairs,
        image_size=Config.IMAGE_SIZE,
        normalize=True
    ):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.normalize = normalize

        self.image_transform = T.Compose([
            T.Resize(
                self.image_size,
                interpolation=T.InterpolationMode.BILINEAR
            ),
            T.ToTensor(),
        ])

        self.normalize_transform = T.Normalize(
            mean=Config.IMAGENET_MEAN,
            std=Config.IMAGENET_STD
        )

        self.mask_resize = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sample {index}\n"
                f"Image: {image_path}\n"
                f"Mask : {mask_path}\n"
                f"Error: {exc}"
            )

        image = self.image_transform(image)

        if self.normalize:
            image = self.normalize_transform(image)

        mask = self.mask_resize(mask)
        mask = torch.from_numpy(
            np.asarray(mask, dtype=np.float32)
        )

        # Binary segmentation.
        mask = (mask > 127).float()

        # [H,W] -> [1,H,W]
        mask = mask.unsqueeze(0)

        return image, mask


train_dataset = KvasirSegDataset(train_pairs)
test_dataset = KvasirSegDataset(test_pairs)

print(
    "Train dataset:",
    len(train_dataset)
)

print(
    "Test dataset :",
    len(test_dataset)
)


# ================================================================
# MODULE 7 — DINOv2 FEATURE EXTRACTOR
# ================================================================

class DINOv2HybridFeatureExtractor:
    """
    Frozen DINOv2 ViT-S/14 feature extractor.

    Hybrid feature:
        CLS token:       384 dimensions
        GAP patch token: 384 dimensions
        --------------------------------
        Hybrid vector:   768 dimensions

    Final vector is L2-normalized.
    """

    def __init__(
        self,
        device=Config.DEVICE,
        model_name=Config.DINO_MODEL_NAME
    ):
        self.device = device
        self.model_name = model_name

        print("\nLoading DINOv2...")
        print("Model:", model_name)

        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv2 through torch.hub.\n"
                "Ensure Colab has internet access.\n"
                f"Original error: {exc}"
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Freeze every parameter.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        print("DINOv2 loaded successfully.")

    @torch.inference_mode()
    def extract_batch(self, images):
        """
        Extract a [B,768] hybrid feature matrix.
        """

        images = images.to(
            self.device,
            non_blocking=True
        )

        features = self.model.forward_features(images)

        # CLS token
        cls_token = features["x_norm_clstoken"]

        # Patch tokens
        patch_tokens = features["x_norm_patchtokens"]

        # Global average pooling over spatial patch dimension.
        gap_patch = patch_tokens.mean(dim=1)

        # Concatenate:
        # [B,384] + [B,384] -> [B,768]
        hybrid = torch.cat(
            [cls_token, gap_patch],
            dim=1
        )

        # L2 normalize for cosine similarity through dot product.
        hybrid = F.normalize(
            hybrid,
            p=2,
            dim=1
        )

        return hybrid

    @torch.inference_mode()
    def extract_dataset(
        self,
        dataset,
        batch_size=Config.BATCH_SIZE
    ):
        """
        Extract features for the complete training dataset.
        """

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )

        all_features = []

        start = time.perf_counter()

        for batch_idx, (images, _) in enumerate(loader):

            features = self.extract_batch(images)

            # Move to CPU to avoid unnecessary GPU memory usage.
            all_features.append(
                features.cpu()
            )

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Feature batches: "
                    f"{batch_idx + 1}/{len(loader)}"
                )

        features = torch.cat(
            all_features,
            dim=0
        )

        elapsed = time.perf_counter() - start

        print("\nFeature extraction complete.")
        print("Feature tensor shape:", tuple(features.shape))
        print(f"Extraction time: {elapsed:.2f} seconds")

        expected_dim = 768

        if features.shape[1] != expected_dim:
            raise ValueError(
                f"Expected 768-D features, "
                f"got {features.shape[1]}."
            )

        return features


# Instantiate and extract ONLY training features.
dino_extractor = DINOv2HybridFeatureExtractor()

train_features = dino_extractor.extract_dataset(
    train_dataset
)

print(
    "\nFinal training feature matrix:",
    train_features.shape
)


# ================================================================
# MODULE 8 — COSINE SIMILARITY MATRIX
# ================================================================

class CosineSimilarityGraphBuilder:
    """
    Builds an unweighted undirected NetworkX graph from
    DINOv2 cosine similarities.

    S_ij = feature_i dot feature_j

    W_ij = (S_ij + 1) / 2

    A_ij = 1 if W_ij >= tau
    """

    def __init__(
        self,
        tau=Config.TAU
    ):
        self.tau = tau

    def compute_similarity(self, features):
        """
        Compute full NxN cosine similarity matrix.

        Since the feature vectors have already been L2 normalized,
        matrix multiplication is equivalent to cosine similarity.
        """

        if not torch.is_floating_point(features):
            features = features.float()

        features = F.normalize(
            features,
            p=2,
            dim=1
        )

        similarity = torch.mm(
            features,
            features.T
        )

        # Keep memory footprint manageable.
        similarity = similarity.float()

        return similarity

    def build_graph(self, features):

        similarity = self.compute_similarity(
            features
        )

        print(
            "Cosine similarity matrix:",
            tuple(similarity.shape)
        )

        # W = (S + 1) / 2
        adjacency_probability = (
            similarity + 1.0
        ) / 2.0

        # Threshold.
        adjacency_binary = (
            adjacency_probability >= self.tau
        )

        # Remove self-loops.
        adjacency_binary.fill_diagonal_(False)

        # Make absolutely sure graph is symmetric.
        adjacency_binary = torch.logical_or(
            adjacency_binary,
            adjacency_binary.T
        )

        # Convert to numpy.
        adjacency_np = adjacency_binary.cpu().numpy()

        n = adjacency_np.shape[0]

        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        # Efficiently add only upper-triangular edges.
        upper_i, upper_j = np.where(
            np.triu(adjacency_np, k=1)
        )

        edges = list(
            zip(
                upper_i.tolist(),
                upper_j.tolist()
            )
        )

        graph.add_edges_from(edges)

        print("\nGraph construction complete.")
        print("Nodes :", graph.number_of_nodes())
        print("Edges :", graph.number_of_edges())

        possible_edges = n * (n - 1) / 2

        if possible_edges > 0:
            density = (
                graph.number_of_edges()
                / possible_edges
            )
        else:
            density = 0.0

        print(
            f"Graph density: {density:.6f}"
        )

        # Return matrix only if needed for analysis.
        return graph, similarity, adjacency_probability


graph_builder = CosineSimilarityGraphBuilder(
    tau=Config.TAU
)

train_graph, similarity_matrix, adjacency_probability = (
    graph_builder.build_graph(train_features)
)


# ================================================================
# MODULE 9 — LOUVAIN COMMUNITY DETECTION + PRUNING
# ================================================================

class LouvainDatasetPruner:
    """
    Performs:

    1. Louvain community detection.
    2. Local degree centrality calculation.
    3. Top-p selection within every community.
    """

    def __init__(
        self,
        retention_p=Config.RETENTION_P,
        random_state=Config.RANDOM_SEED
    ):
        self.retention_p = retention_p
        self.random_state = random_state

    def detect_communities(self, graph):

        if graph.number_of_nodes() == 0:
            raise ValueError(
                "Graph contains no nodes."
            )

        # Handle graph with no edges explicitly.
        if graph.number_of_edges() == 0:
            partition = {
                node: node
                for node in graph.nodes()
            }

            return partition

        partition = community_louvain.best_partition(
            graph,
            random_state=self.random_state
        )

        return partition

    def modularity(self, graph, partition):

        if graph.number_of_edges() == 0:
            return 0.0

        return community_louvain.modularity(
            partition,
            graph
        )

    def select_representatives(
        self,
        graph,
        partition
    ):
        """
        For every community:

            k_i = local degree centrality

        Retain top p%.

        At least one node is retained per community.
        """

        communities = {}

        for node, community_id in partition.items():
            communities.setdefault(
                community_id,
                []
            ).append(node)

        selected_global_indices = []

        community_statistics = []

        for community_id, nodes in communities.items():

            subgraph = graph.subgraph(nodes)

            # Degree centrality in local community subgraph.
            degree_centrality = nx.degree_centrality(
                subgraph
            )

            ranked_nodes = sorted(
                nodes,
                key=lambda node: (
                    degree_centrality[node],
                    graph.degree[node]
                ),
                reverse=True
            )

            requested_count = int(
                math.ceil(
                    len(nodes) * self.retention_p
                )
            )

            # At least one representative.
            selected_count = max(
                1,
                requested_count
            )

            selected_nodes = ranked_nodes[
                :selected_count
            ]

            selected_global_indices.extend(
                selected_nodes
            )

            community_statistics.append({
                "community_id": community_id,
                "community_size": len(nodes),
                "selected_count": len(selected_nodes),
                "retention_fraction": (
                    len(selected_nodes)
                    / len(nodes)
                )
            })

        selected_global_indices = sorted(
            set(selected_global_indices)
        )

        return (
            selected_global_indices,
            communities,
            community_statistics
        )

    def prune(self, graph):

        print("\nRunning Louvain community detection...")

        partition = self.detect_communities(
            graph
        )

        modularity_score = self.modularity(
            graph,
            partition
        )

        print(
            f"Louvain modularity Q = "
            f"{modularity_score:.6f}"
        )

        (
            selected_indices,
            communities,
            community_statistics
        ) = self.select_representatives(
            graph,
            partition
        )

        return {
            "partition": partition,
            "communities": communities,
            "selected_indices": selected_indices,
            "statistics": community_statistics,
            "modularity": modularity_score
        }


pruner = LouvainDatasetPruner(
    retention_p=Config.RETENTION_P
)

pruning_result = pruner.prune(
    train_graph
)

selected_indices = pruning_result[
    "selected_indices"
]

print("\n" + "=" * 70)
print("PRUNING RESULTS")
print("=" * 70)

print(
    "Number of Louvain communities:",
    len(pruning_result["communities"])
)

print(
    "Original training samples:",
    len(train_pairs)
)

print(
    "Selected training samples:",
    len(selected_indices)
)

reduction_ratio = (
    1.0
    - len(selected_indices) / len(train_pairs)
) * 100.0

print(
    f"Data reduction ratio: "
    f"{reduction_ratio:.2f}%"
)

print(
    f"Retention ratio: "
    f"{100.0 - reduction_ratio:.2f}%"
)

print("=" * 70)


# ================================================================
# MODULE 10 — CREATE PRUNED DATASET
# ================================================================

pruned_train_pairs = [
    train_pairs[i]
    for i in selected_indices
]

pruned_dataset = KvasirSegDataset(
    pruned_train_pairs
)

print(
    "Pruned dataset size:",
    len(pruned_dataset)
)


# ================================================================
# MODULE 11 — COMPACT U-NET
# ================================================================

class DoubleConv(nn.Module):
    """
    Basic U-Net convolutional block.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CompactUNet(nn.Module):
    """
    Compact encoder-decoder U-Net.

    Input:
        [B,3,224,224]

    Output:
        [B,1,224,224]
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1
    ):
        super().__init__()

        self.enc1 = DoubleConv(
            in_channels,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.enc4 = DoubleConv(
            128,
            256
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.bottleneck = DoubleConv(
            256,
            512
        )

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# ================================================================
# MODULE 12 — SEGMENTATION LOSSES
# ================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice loss.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.flatten(
            start_dim=1
        )

        targets = targets.flatten(
            start_dim=1
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    BCEWithLogits + Dice loss.
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )


# ================================================================
# MODULE 13 — U-NET TRAINER
# ================================================================

class UNetTrainer:
    """
    Handles training and wall-clock measurement.
    """

    def __init__(
        self,
        model,
        device=Config.DEVICE,
        learning_rate=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    ):
        self.model = model.to(device)
        self.device = device

        self.criterion = CombinedSegmentationLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=Config.AMP_ENABLED
        )

    def train_one_epoch(
        self,
        loader
    ):

        self.model.train()

        total_loss = 0.0
        sample_count = 0

        for images, masks in loader:

            images = images.to(
                self.device,
                non_blocking=True
            )

            masks = masks.to(
                self.device,
                non_blocking=True
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=Config.AMP_ENABLED
            ):
                logits = self.model(images)

                loss = self.criterion(
                    logits,
                    masks
                )

            self.scaler.scale(
                loss
            ).backward()

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            sample_count += batch_size

        return (
            total_loss
            / max(sample_count, 1)
        )

    def fit(
        self,
        dataset,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    ):

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=False
        )

        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)

        print(
            "Training samples:",
            len(dataset)
        )

        print(
            "Epochs:",
            epochs
        )

        print(
            "Batch size:",
            batch_size
        )

        start_time = time.perf_counter()

        epoch_losses = []

        for epoch in range(epochs):

            epoch_start = time.perf_counter()

            loss = self.train_one_epoch(
                loader
            )

            epoch_time = (
                time.perf_counter()
                - epoch_start
            )

            epoch_losses.append(loss)

            print(
                f"Epoch "
                f"{epoch + 1:02d}/{epochs} | "
                f"Loss: {loss:.5f} | "
                f"Time: {epoch_time:.2f}s"
            )

        total_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"\nTotal training time: "
            f"{total_time:.2f}s "
            f"({total_time / 60:.2f} min)"
        )

        return {
            "model": self.model,
            "training_time": total_time,
            "epoch_losses": epoch_losses
        }


# ================================================================
# MODULE 14 — SEGMENTATION EVALUATOR
# ================================================================

class SegmentationEvaluator:
    """
    Calculates mean Dice and mean IoU over the untouched test set.
    """

    def __init__(
        self,
        model,
        device=Config.DEVICE,
        threshold=0.5
    ):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold

    @staticmethod
    def dice_score(
        prediction,
        target,
        smooth=1e-7
    ):

        prediction = prediction.flatten(
            start_dim=1
        )

        target = target.flatten(
            start_dim=1
        )

        intersection = (
            prediction * target
        ).sum(dim=1)
        

        denominator = (
            prediction.sum(dim=1)
            + target.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + smooth
        ) / (
            denominator
            + smooth
        )

        return dice

    @staticmethod
    def iou_score(
        prediction,
        target,
        smooth=1e-7
    ):

        prediction = prediction.flatten(
            start_dim=1
        )

        target = target.flatten(
            start_dim=1
        )

        intersection = (
            prediction * target
        ).sum(dim=1)

        union = (
            prediction.sum(dim=1)
            + target.sum(dim=1)
            - intersection
        )

        iou = (
            intersection
            + smooth
        ) / (
            union
            + smooth
        )

        return iou

    @torch.inference_mode()
    def evaluate(
        self,
        dataset,
        batch_size=Config.BATCH_SIZE
    ):

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )

        self.model.eval()

        dice_values = []
        iou_values = []

        start_time = time.perf_counter()

        for images, masks in loader:

            images = images.to(
                self.device,
                non_blocking=True
            )

            masks = masks.to(
                self.device,
                non_blocking=True
            )

            with torch.cuda.amp.autocast(
                enabled=Config.AMP_ENABLED
            ):
                logits = self.model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= self.threshold
            ).float()

            dice = self.dice_score(
                predictions,
                masks
            )

            iou = self.iou_score(
                predictions,
                masks
            )

            dice_values.extend(
                dice.detach().cpu().numpy().tolist()
            )

            iou_values.extend(
                iou.detach().cpu().numpy().tolist()
            )

        evaluation_time = (
            time.perf_counter()
            - start_time
        )

        mean_dice = float(
            np.mean(dice_values)
        )

        mean_iou = float(
            np.mean(iou_values)
        )

        dice_std = float(
            np.std(dice_values)
        )

        return {
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "dice_std": dice_std,
            "evaluation_time": evaluation_time
        }


# ================================================================
# MODULE 15 — EXPERIMENT RUNNER
# ================================================================

class ExperimentRunner:
    """
    Runs:

        Experiment A:
            U-Net trained on 100% training data.

        Experiment B:
            U-Net trained on Louvain-pruned data.

    Both are evaluated on exactly the same test set.
    """

    def __init__(
        self,
        full_dataset,
        pruned_dataset,
        test_dataset
    ):
        self.full_dataset = full_dataset
        self.pruned_dataset = pruned_dataset
        self.test_dataset = test_dataset

    def create_model(self):
        """
        Creates a fresh U-Net so experiments are independent.
        """

        model = CompactUNet(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS
        )

        return model

    def run_full_experiment(self):

        print("\n")
        print("#" * 70)
        print("EXPERIMENT A — FULL TRAINING DATASET")
        print("#" * 70)

        # Fresh model.
        model = self.create_model()

        trainer = UNetTrainer(
            model=model
        )

        training_result = trainer.fit(
            self.full_dataset
        )

        evaluator = SegmentationEvaluator(
            training_result["model"]
        )

        metrics = evaluator.evaluate(
            self.test_dataset
        )

        return {
            **training_result,
            **metrics
        }

    def run_pruned_experiment(self):

        print("\n")
        print("#" * 70)
        print("EXPERIMENT B — PRUNED TRAINING DATASET")
        print("#" * 70)

        # Fresh model.
        model = self.create_model()

        trainer = UNetTrainer(
            model=model
        )

        training_result = trainer.fit(
            self.pruned_dataset
        )

        evaluator = SegmentationEvaluator(
            training_result["model"]
        )

        metrics = evaluator.evaluate(
            self.test_dataset
        )

        return {
            **training_result,
            **metrics
        }

    def run(self):

        full_result = self.run_full_experiment()

        # Free some GPU memory between experiments.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pruned_result = self.run_pruned_experiment()

        return full_result, pruned_result


# ================================================================
# MODULE 16 — RUN BOTH TRAINING EXPERIMENTS
# ================================================================

experiment_runner = ExperimentRunner(
    full_dataset=train_dataset,
    pruned_dataset=pruned_dataset,
    test_dataset=test_dataset
)

full_result, pruned_result = (
    experiment_runner.run()
)


# ================================================================
# MODULE 17 — FINAL METRIC CALCULATIONS
# ================================================================

class FinalReport:
    """
    Produces the final thesis-ready comparison.
    """

    def __init__(
        self,
        train_count,
        pruned_count,
        full_result,
        pruned_result,
        modularity,
        tau,
        retention_p
    ):
        self.train_count = train_count
        self.pruned_count = pruned_count

        self.full_result = full_result
        self.pruned_result = pruned_result

        self.modularity = modularity
        self.tau = tau
        self.retention_p = retention_p

    def calculate(self):

        reduction_ratio = (
            1.0
            - self.pruned_count
            / self.train_count
        ) * 100.0

        full_time = (
            self.full_result["training_time"]
        )

        pruned_time = (
            self.pruned_result["training_time"]
        )

        if full_time > 0:
            acceleration_delta = (
                (full_time - pruned_time)
                / full_time
            ) * 100.0
        else:
            acceleration_delta = float("nan")

        full_dice = (
            self.full_result["mean_dice"]
        )

        pruned_dice = (
            self.pruned_result["mean_dice"]
        )

        absolute_dice_delta = abs(
            full_dice - pruned_dice
        )

        return {
            "train_count": self.train_count,
            "pruned_count": self.pruned_count,
            "reduction_ratio": reduction_ratio,
            "full_time": full_time,
            "pruned_time": pruned_time,
            "acceleration_delta": acceleration_delta,
            "full_dice": full_dice,
            "pruned_dice": pruned_dice,
            "full_iou": self.full_result["mean_iou"],
            "pruned_iou": self.pruned_result["mean_iou"],
            "dice_delta": absolute_dice_delta,
            "full_dice_std": self.full_result["dice_std"],
            "pruned_dice_std": self.pruned_result["dice_std"],
            "modularity": self.modularity,
            "tau": self.tau,
            "retention_p": self.retention_p
        }

    def print_report(self):

        results = self.calculate()

        print("\n")
        print("=" * 78)
        print("FINAL THESIS EXPERIMENT SUMMARY")
        print("=" * 78)

        print("\n[GRAPH / PRUNING]")
        print(
            f"Similarity threshold (tau): "
            f"{results['tau']:.2f}"
        )

        print(
            f"Community retention budget (p): "
            f"{results['retention_p']:.2f}"
        )

        print(
            f"Louvain modularity Q: "
            f"{results['modularity']:.6f}"
        )

        print("\n[DATASET]")
        print(
            f"Original training samples : "
            f"{results['train_count']}"
        )

        print(
            f"Pruned training samples   : "
            f"{results['pruned_count']}"
        )

        print(
            f"Data reduction ratio      : "
            f"{results['reduction_ratio']:.2f}%"
        )

        print("\n[TRAINING COMPUTATION]")
        print(
            f"Full dataset training time   : "
            f"{results['full_time']:.2f} sec "
            f"({results['full_time'] / 60:.2f} min)"
        )

        print(
            f"Pruned dataset training time : "
            f"{results['pruned_time']:.2f} sec "
            f"({results['pruned_time'] / 60:.2f} min)"
        )

        print(
            f"Training acceleration delta   : "
            f"{results['acceleration_delta']:.2f}%"
        )

        print("\n[TEST-SET SEGMENTATION]")
        print(
            f"Full dataset mDSC   : "
            f"{results['full_dice']:.4f}"
        )

        print(
            f"Pruned dataset mDSC : "
            f"{results['pruned_dice']:.4f}"
        )

        print(
            f"Full dataset mIoU   : "
            f"{results['full_iou']:.4f}"
        )

        print(
            f"Pruned dataset mIoU : "
            f"{results['pruned_iou']:.4f}"
        )

        print(
            f"Absolute Dice delta : "
            f"{results['dice_delta']:.4f}"
        )

        print("\n[ADDITIONAL DICE DISPERSION]")
        print(
            f"Full dataset Dice SD   : "
            f"{results['full_dice_std']:.4f}"
        )

        print(
            f"Pruned dataset Dice SD : "
            f"{results['pruned_dice_std']:.4f}"
        )

        print("\n" + "=" * 78)

        return results


# ================================================================
# MODULE 18 — GENERATE FINAL REPORT
# ================================================================

report = FinalReport(
    train_count=len(train_pairs),
    pruned_count=len(pruned_train_pairs),
    full_result=full_result,
    pruned_result=pruned_result,
    modularity=pruning_result["modularity"],
    tau=Config.TAU,
    retention_p=Config.RETENTION_P
)

final_results = report.print_report()


# ================================================================
# MODULE 19 — OPTIONAL: SAVE EXPERIMENT ARTIFACTS
# ================================================================

class ExperimentArtifactManager:
    """
    Saves reproducibility artifacts to Colab local storage.
    """

    def __init__(
        self,
        output_dir="/content/kvasir_pruning/results"
    ):
        self.output_dir = output_dir
        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    def save_selected_indices(
        self,
        indices
    ):

        path = os.path.join(
            self.output_dir,
            "selected_training_indices.npy"
        )

        np.save(
            path,
            np.asarray(indices)
        )

        return path

    def save_feature_matrix(
        self,
        features
    ):

        path = os.path.join(
            self.output_dir,
            "dinov2_hybrid_features.npy"
        )

        np.save(
            path,
            features.numpy()
        )

        return path

    def save_models(
        self,
        full_model,
        pruned_model
    ):

        full_path = os.path.join(
            self.output_dir,
            "unet_full_dataset.pth"
        )

        pruned_path = os.path.join(
            self.output_dir,
            "unet_pruned_dataset.pth"
        )

        torch.save(
            full_model.state_dict(),
            full_path
        )

        torch.save(
            pruned_model.state_dict(),
            pruned_path
        )

        return full_path, pruned_path


artifact_manager = ExperimentArtifactManager()

indices_path = artifact_manager.save_selected_indices(
    selected_indices
)

features_path = artifact_manager.save_feature_matrix(
    train_features
)

print("\nArtifacts saved:")
print(indices_path)
print(features_path)

print("\nExperiment completed successfully.")