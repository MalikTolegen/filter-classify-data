#!/usr/bin/env python3
"""
filter_and_copy.py

✓ Scan date/time named folders inside source root
✓ If there's at least one JPEG/JPG file, classify → copy
✓ Label 0-6 → ['Hazy', 'Normal', 'raining', 'rainy but not raining',
               'snowing', 'snowy but not snowing', 'unclear']
✓ Supports .pth checkpoints or Hugging Face model IDs
"""

from __future__ import annotations
from typing import List
from pathlib import Path
import argparse
import shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm import create_model
from transformers import AutoModelForImageClassification

# 0-6 label ↔ class folder names
CLASS_NAMES = [
    'Hazy',
    'Normal',
    'raining',
    'rainy but not raining',
    'snowing',
    'snowy but not snowing',
    'unclear'
]

# Image preprocessing
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
_preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

@torch.no_grad()
def _predict_one_image(img_path: str | Path, model: torch.nn.Module, device: torch.device, topk: int,
                       idx_to_class: list[str]) -> List[tuple[str, float]]:
    """Return top-k predictions as (label, prob) tuples."""
    img = Image.open(img_path).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0).to(device)
    output = model(tensor)
    logits = output.logits if hasattr(output, "logits") else output
    probs = F.softmax(logits, dim=1)[0]
    topk_prob, topk_idx = probs.topk(topk)
    results = []
    for p, idx in zip(topk_prob.tolist(), topk_idx.tolist()):
        label = idx_to_class[idx] if idx < len(idx_to_class) and idx_to_class[idx] is not None else str(idx)
        results.append((label, p))
    return results

def has_jpeg(folder: Path) -> bool:
    """Returns True if folder contains at least one .jpg/.jpeg file."""
    return any(p.suffix.lower() in {'.jpg', '.jpeg'} for p in folder.iterdir() if p.is_file())

def classify_folder(folder: Path, model, device) -> int:
    for file in folder.iterdir():
        if file.suffix.lower() in {'.jpg', '.jpeg'}:
            try:
                prob = _predict_one_image(file, model, device, topk=1, idx_to_class=CLASS_NAMES)
                label = prob[0][0]
                return CLASS_NAMES.index(label)
            except Exception as e:
                print(f"Error on {file.name}: {e}")
                break
    return 6  # fallback: "unclear"

def load_model_from_checkpoint(ckpt: str, device: torch.device, num_classes: int):
    ckpt_path = Path(ckpt)
    if ckpt_path.is_file() and ckpt_path.suffix == ".pth":
        print(f"📦 Loading .pth checkpoint: {ckpt}")
        model = create_model('swinv2_large_window12_192', pretrained=False, num_classes=num_classes)
        state_dict = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    else:
        print(f"🤗 Loading Hugging Face model: {ckpt}")
        model = AutoModelForImageClassification.from_pretrained(
            ckpt,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    return model.to(device).eval()

def main(src_root: Path, dst_root: Path, ckpt: str, overwrite: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(ckpt, device, num_classes=len(CLASS_NAMES))

    # Pre-create 7 class folders at destination
    for name in CLASS_NAMES:
        (dst_root / name).mkdir(parents=True, exist_ok=True)

    for subdir in [p for p in src_root.iterdir() if p.is_dir()]:
        if not has_jpeg(subdir):
            continue

        label = classify_folder(subdir, model, device)
        dst_class_dir = dst_root / CLASS_NAMES[label]
        dst_path = dst_class_dir / subdir.name

        if dst_path.exists() and not overwrite:
            print(f'‼️ Already exists → Skip: {dst_path}')
            continue

        shutil.copytree(subdir, dst_path, dirs_exist_ok=overwrite)
        print(f'✅ {subdir}  →  {dst_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JPEG folder classifier and copier (supports .pth or HuggingFace models)")
    parser.add_argument("--src", required=True, help="Source root path")
    parser.add_argument("--dst", required=True, help="Destination root path (creates 7 class folders)")
    parser.add_argument("--ckpt", required=True, help="Path to .pth file or Hugging Face model name")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing folders")
    args = parser.parse_args()

    main(Path(args.src).expanduser(), Path(args.dst).expanduser(), args.ckpt, args.overwrite)
