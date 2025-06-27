#!/usr/bin/env python3
"""
main.py (Updated)
✓ Automatically loads .pth checkpoint and class labels from Swin Transformer V2
✓ Classifies JPEG/JPG folders and copies them to a labeled destination folder
✓ Supports zipped or direct .pth checkpoints
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from timm import create_model
from transformers import AutoModelForImageClassification
from functools import lru_cache

import argparse
import shutil
from tqdm import tqdm

CLASS_NAMES: List[str] = [
    "Hazy",
    "Normal",
    "raining",
    "rainy but not raining",
    "snowing",
    "snowy but not snowing",
    "unclear",
]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

@lru_cache(maxsize=1)
def _load_model() -> torch.nn.Module:
    """Load Swin V2 classification model checkpoint lazily (singleton)."""



    # Project root (= two levels up from this file) → weather_classification/models/
    ckpt_dir = Path(__file__).resolve().parents[2] / "Downloads" / "filter-classify-data-main" / "models"
    ckpt_list = sorted(ckpt_dir.glob("*.pth"))
    if not ckpt_list:
        raise FileNotFoundError(
            f"No .pth checkpoint found in {ckpt_dir}. Please place your trained model there."  # noqa: E501
        )
    ckpt_path = ckpt_list[0]

    num_classes = len(CLASS_NAMES)

    ckpt = torch.load(ckpt_path, map_location=_DEVICE)
    state_dict = (
        ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt  # raw state-dict fallback
    )

    try:
        model = create_model("swinv2_large", pretrained=False, num_classes=num_classes)
    except RuntimeError:
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    model.load_state_dict(state_dict, strict=False)
    model.eval().to(_DEVICE)
    return model

# Utility

def has_jpeg(folder: Path) -> bool:
    return any(p.suffix.lower() in {'.jpg', '.jpeg'} for p in folder.iterdir() if p.is_file())

def classify_folder(folder: Path) -> int:
    """Predict weather class index for a folder of images.

    Parameters
    ----------
    folder : Path
        Directory containing JPEG images.

    Returns
    -------
    int
        Index 0‒6 corresponding to CLASS_NAMES.
    """
    model = _load_model()

    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    if not images:
        return CLASS_NAMES.index("unclear")

    probs_sum = torch.zeros(len(CLASS_NAMES), device=_DEVICE)

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        tensor = _PREPROCESS(img).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            output = model(tensor)
            logits = output.logits if hasattr(output, "logits") else output
            probs = F.softmax(logits, dim=1)[0]
            probs_sum += probs

    avg_probs = probs_sum / len(images)
    return int(avg_probs.argmax().item())

# Main

def main(src_root: Path, dst_root: Path, overwrite: bool):
    # Pre-create 7 class folders at destination
    for name in CLASS_NAMES:
        (dst_root / name).mkdir(parents=True, exist_ok=True)

    # Step 1: Iterate through 1-depth subfolders of source root
    for subdir in tqdm([p for p in src_root.iterdir() if p.is_dir()], desc="Folders", unit="dir"):
        if not has_jpeg(subdir):            # Skip folders without JPEG
            continue

        label = classify_folder(subdir)     # Predict 0-6 label
        dst_class_dir = dst_root / CLASS_NAMES[label]
        dst_path = dst_class_dir / subdir.name

        # Step 2: Copy folder (including metadata, Python≥3.8)
        if dst_path.exists() and not overwrite:
            print(f'‼️ Already exists → Skip: {dst_path}')
            continue
        shutil.copytree(subdir, dst_path, dirs_exist_ok=overwrite)
        print(f'✅ {subdir}  →  {dst_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JPEG folder classification and copying")
    parser.add_argument("--src", required=True, help="Source root folder")
    parser.add_argument("--dst", required=True, help="Destination root folder")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if folder already exists")
    args = parser.parse_args()

    main(Path(args.src), Path(args.dst), args.overwrite)
