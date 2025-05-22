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

def augment_image(img):
    # Flip horizontal
    flipped = cv2.flip(img, 1)
    # Rotate 15 degrees
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
    # Brightness up
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    return [flipped, rotated, bright]

def extract_training_set():
    crop_size = 128  # Ganti sesuai kebutuhan
    base_in = Path(__file__).parent / "images" / "train_source"
    base_out = Path(__file__).parent / "images" / "train_set"

    for class_label in ["0", "1"]:
        img_dir = base_in / class_label
        out_dir = base_out / class_label
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_dir.glob("*.jpg"):
            print(f"\nProcessing {img_path.name} (class {class_label}) ...")
            regions = propose_regions(str(img_path), min_area=500, display=False)
            print(f"  → Found {len(regions)} regions")

            img = cv2.imread(str(img_path))
            for idx, (x, y, w, h) in enumerate(regions):
                crop = img[y:y+h, x:x+w]
                crop_padded = resize_with_padding(crop, crop_size, pad_value=0)
                crop_filename = f"{img_path.stem}_crop_{idx}.jpg"
                cv2.imwrite(str(out_dir / crop_filename), crop_padded)

                # Augmentasi hanya untuk kapal (class 1)
                if class_label == "1":
                    aug_imgs = augment_image(crop_padded)
                    aug_names = ["flip", "rot", "bright"]
                    for aug, name in zip(aug_imgs, aug_names):
                        aug_filename = f"{img_path.stem}_crop_{idx}_{name}.jpg"
                        cv2.imwrite(str(out_dir / aug_filename), aug)

if __name__ == "__main__":
    extract_training_set()