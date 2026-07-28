"""
dataset.py
==========
Dataset class, transforms, and seed/pool/test split logic for the
Chest X-Ray Pneumonia active learning pipeline.

Key invariants maintained:
- labeled_indices, unlabeled_indices, and test set never overlap.
- Test set is the original test/ folder — never touched by the AL loop.
- Augmentation is applied ONLY to labeled training images, never to
  unlabeled pool images during scoring or to the test set.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


# ── ImageNet normalization constants ──────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Transform factories ──────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Augmented transforms for labeled training images."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Deterministic transforms for evaluation and unlabeled pool scoring."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    Wraps a flat list of (image_path, label) pairs.

    Parameters
    ----------
    image_paths : list of str
        Absolute paths to image files.
    labels : list of int
        0 = NORMAL, 1 = PNEUMONIA.
    transform : torchvision.transforms.Compose or None
        Image transforms to apply.
    """

    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        assert len(image_paths) == len(labels), "paths and labels must match"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or get_eval_transforms()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        img = self.transform(img)
        return img, label


# ── Scanning helpers ──────────────────────────────────────────────────────────

def _scan_folder(root: str) -> Tuple[List[str], List[int]]:
    """
    Scan a directory with NORMAL/ and PNEUMONIA/ subfolders.
    Returns (image_paths, labels) sorted for reproducibility.
    """
    paths, labels = [], []
    for class_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected directory not found: {class_dir}")
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    return paths, labels


def get_test_dataset(data_root: str) -> ChestXrayDataset:
    """
    Load the fixed held-out test set.
    This MUST NEVER be used in the active learning loop.
    """
    test_dir = os.path.join(data_root, "test")
    paths, labels = _scan_folder(test_dir)
    print(f"[Dataset] Test set: {len(paths)} images "
          f"({labels.count(0)} NORMAL, {labels.count(1)} PNEUMONIA)")
    return ChestXrayDataset(paths, labels, transform=get_eval_transforms())


# ── Active Learning Pool ──────────────────────────────────────────────────────

class ActiveLearningPool:
    """
    Manages the labeled/unlabeled split over the training images.

    The pool scans train/ once, then maintains two non-overlapping index sets:
    - labeled_indices: images whose labels are "known" (oracle has revealed them)
    - unlabeled_indices: images still in the pool

    Parameters
    ----------
    data_root : str
        Path to chest_xray/ root containing train/, test/.
    seed : int
        Random seed for reproducible seed-set selection.
    seed_size : int
        Number of images in the initial labeled seed set (class-balanced).
    """

    def __init__(self, data_root: str, seed: int, seed_size: int = 100):
        self.seed = seed
        self.rng = random.Random(seed)

        # Scan training folder
        train_dir = os.path.join(data_root, "train")
        self.all_paths, self.all_labels = _scan_folder(train_dir)
        self.n_total = len(self.all_paths)

        # Separate indices by class
        normal_indices = [i for i, l in enumerate(self.all_labels) if l == 0]
        pneumonia_indices = [i for i, l in enumerate(self.all_labels) if l == 1]

        # Class-balanced seed selection
        half = seed_size // 2
        self.rng.shuffle(normal_indices)
        self.rng.shuffle(pneumonia_indices)

        seed_normal = normal_indices[:half]
        seed_pneumonia = pneumonia_indices[:half]

        self.labeled_indices: List[int] = sorted(seed_normal + seed_pneumonia)
        self.unlabeled_indices: List[int] = sorted(
            set(range(self.n_total)) - set(self.labeled_indices)
        )

        print(f"[Pool] Total training images: {self.n_total}")
        print(f"[Pool] Initial labeled set: {len(self.labeled_indices)} "
              f"({half} NORMAL + {half} PNEUMONIA)")
        print(f"[Pool] Unlabeled pool: {len(self.unlabeled_indices)}")

        # Sanity check: no overlap
        assert len(set(self.labeled_indices) & set(self.unlabeled_indices)) == 0

    def reveal_labels(self, indices: List[int]) -> None:
        """
        Oracle step: move indices from unlabeled to labeled.
        Simulates a human annotator revealing the true labels.
        """
        indices_set = set(indices)
        # Verify all requested indices are currently unlabeled
        assert indices_set.issubset(set(self.unlabeled_indices)), \
            "Tried to reveal labels for indices not in the unlabeled pool!"

        self.unlabeled_indices = [i for i in self.unlabeled_indices
                                  if i not in indices_set]
        self.labeled_indices = sorted(self.labeled_indices + list(indices_set))

    def get_labeled_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
    ) -> DataLoader:
        """DataLoader over labeled images with training augmentation."""
        dataset = ChestXrayDataset(
            image_paths=[self.all_paths[i] for i in self.labeled_indices],
            labels=[self.all_labels[i] for i in self.labeled_indices],
            transform=get_train_transforms(),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_unlabeled_loader(
        self,
        batch_size: int = 64,
        num_workers: int = 2,
    ) -> DataLoader:
        """
        DataLoader over unlabeled images with eval transforms (no augmentation).
        Returns (images, original_pool_indices) so we can map scores back.
        """
        dataset = _IndexedSubset(
            image_paths=[self.all_paths[i] for i in self.unlabeled_indices],
            labels=[self.all_labels[i] for i in self.unlabeled_indices],
            pool_indices=self.unlabeled_indices,
            transform=get_eval_transforms(),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_stats(self) -> Dict[str, int]:
        """Current pool statistics."""
        labeled_labels = [self.all_labels[i] for i in self.labeled_indices]
        return {
            "n_labeled": len(self.labeled_indices),
            "n_unlabeled": len(self.unlabeled_indices),
            "labeled_normal": labeled_labels.count(0),
            "labeled_pneumonia": labeled_labels.count(1),
        }


class _IndexedSubset(Dataset):
    """
    Like ChestXrayDataset but also returns the original pool index
    so acquisition scores can be mapped back to pool indices.
    """

    def __init__(self, image_paths, labels, pool_indices, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.pool_indices = pool_indices
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.pool_indices[idx]
