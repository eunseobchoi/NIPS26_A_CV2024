"""
Dataset loaders for capsule endoscopy TTA experiments.
Supports: Capsule Vision 2024, Kvasir-Capsule, Galar (subset).

Domain shift scenarios:
  1. Cross-source: Capsule Vision train → test (multi-hospital)
  2. Cross-device: Galar Olympus → PillCam (if available)
  3. Kvasir-Capsule official frame-list folds plus video-prefix stress tests
"""
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Capsule Vision 2024 classes
CV2024_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema",
    "Foreign Body", "Lymphangiectasia", "Normal",
    "Polyp", "Ulcer", "Worms"
]

# Standard transforms for DINOv2 (224x224, ImageNet normalization)
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CapsuleVision2024(Dataset):
    """Capsule Vision 2024 Challenge dataset.
    Directory structure: root/{split}/{class_name}/*.jpg
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = Path(root) / split
        self.transform = transform or EVAL_TRANSFORM
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path

    def get_class_distribution(self) -> Dict[str, int]:
        dist = {}
        for _, label, _ in self:
            cls = self.classes[label]
            dist[cls] = dist.get(cls, 0) + 1
        return dist


class KvasirCapsule(Dataset):
    """Kvasir-Capsule labelled-image loader for domain-shift diagnostics.

    Labeled images in: root/labelled_images/{class_name}/*.jpg
    """

    def __init__(self, root: str, video_ids: Optional[List[str]] = None,
                 transform=None):
        self.root = Path(root) / "labelled_images"
        self.transform = transform or EVAL_TRANSFORM

        if not self.root.exists():
            # Alternative structure
            self.root = Path(root)

        self.classes = sorted([d.name for d in self.root.iterdir()
                              if d.is_dir() and not d.name.startswith(".")])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


def create_domain_shift_loaders(
    data_root: str,
    dataset_name: str = "capsule_vision_2024",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create source (train), source (val), and target (test) loaders.

    For Capsule Vision 2024:
      source = training set (multi-source)
      target = test set (AIIMS hospital - different distribution)

    Returns: (train_loader, val_loader, test_loader)
    """
    if dataset_name == "capsule_vision_2024":
        train_ds = CapsuleVision2024(data_root, split="training",
                                      transform=TRAIN_TRANSFORM)
        val_ds = CapsuleVision2024(data_root, split="validation",
                                    transform=EVAL_TRANSFORM)
        test_ds = CapsuleVision2024(data_root, split="testing",
                                     transform=EVAL_TRANSFORM)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
