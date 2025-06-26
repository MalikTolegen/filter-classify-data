#!/usr/bin/env python3
"""
filter_and_copy.py

✓ Scan date/time named folders inside source root
✓ If there's at least one JPEG/JPG file, classify → copy
✓ Label 0-6 → ['Hazy', 'Normal', 'raining', 'rainy but not raining',
               'snowing', 'snowy but not snowing', 'unclear']
"""

import argparse
import shutil
from pathlib import Path

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

# ────────────────────────────── Utility Functions ──────────────────────────────
def has_jpeg(folder: Path) -> bool:
    """Returns True if folder contains at least one .jpg/.jpeg file."""
    return any(p.suffix.lower() in {'.jpg', '.jpeg'} for p in folder.iterdir() if p.is_file())

def classify_folder(folder: Path) -> int:
    
    # e.g., Apply desired logic like majority vote / average softmax after per-image prediction
    return 6

# ────────────────────────────── Main Logic ──────────────────────────────
def main(src_root: Path, dst_root: Path, overwrite: bool):
    # Pre-create 7 class folders at destination
    for name in CLASS_NAMES:
        (dst_root / name).mkdir(parents=True, exist_ok=True)

    # Step 1: Iterate through 1-depth subfolders of source root
    for subdir in [p for p in src_root.iterdir() if p.is_dir()]:
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

# ────────────────────────────── CLI Entry Point ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JPEG-containing folder filtering, classification, and copying script")
    parser.add_argument("--src", required=True, help="Source root path")
    parser.add_argument("--dst", required=True, help="Destination root path (contains/creates 7 class folders)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing folders")
    args = parser.parse_args()

    main(Path(args.src).expanduser(), Path(args.dst).expanduser(), args.overwrite)
