import os
from pathlib import Path
from src.proposals.selective_search import propose_regions
import cv2
import numpy as np

def resize_with_padding(img, size, pad_value=0):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    pad_w = size - new_w
    pad_h = size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    return padded

def extraxt_training_set():
    # Path ke folder images
    img_dir = Path(__file__).parent / "images" / "train_set_raw"
    out_dir = Path(__file__).parent / "images" / "train" / "1"
    out_dir.mkdir(exist_ok=True)
    
    crop_size = 128  # Ganti sesuai kebutuhan

    # Loop semua file gambar untuk objek (bounding box)
    for img_path in img_dir.glob("*.jpg"):
        print(f"\nProcessing {img_path.name} ...")
        regions = propose_regions(str(img_path), min_area=500, display=False)
        print(f"  → Found {len(regions)} regions")
        
        img = cv2.imread(str(img_path))
        for idx, (x, y, w, h) in enumerate(regions):
            crop = img[y:y+h, x:x+w]
            crop_padded = resize_with_padding(crop, crop_size, pad_value=0)
            crop_filename = f"{img_path.stem}_crop_{idx}.jpg"
            cv2.imwrite(str(out_dir / crop_filename), crop_padded)

    # Resize semua gambar laut saja di train/0
    sea_dir = Path(__file__).parent / "images" / "train" / "0"
    for sea_img_path in sea_dir.glob("*.jpg"):
        img = cv2.imread(str(sea_img_path))
        img_padded = resize_with_padding(img, crop_size, pad_value=0)
        cv2.imwrite(str(sea_img_path), img_padded)

extraxt_training_set()