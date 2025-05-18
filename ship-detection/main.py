import os
from pathlib import Path
from src.proposals.selective_search import propose_regions
import cv2
import numpy as np

def main():
    # Path ke folder images
    img_dir = Path(__file__).parent / "images"
    # Output folder (opsional) untuk simpan hasil visualisasi
    out_dir = Path(__file__).parent / "out" / "regions"
    out_dir.mkdir(exist_ok=True)

    # Loop semua file gambar
    for img_path in img_dir.glob("*.jpg"):
        print(f"\nProcessing {img_path.name} ...")
        # Dapatkan region proposals
        regions = propose_regions(str(img_path), min_area=500, display=False)
        print(f"  → Found {len(regions)} regions")
        
        # (Opsional) Simpan gambar dengan kotak
        img = cv2.imread(str(img_path))
        for (x, y, w, h) in regions:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), img)
        print(f"  → Saved annotated image to {save_path}")

if __name__ == "__main__":
    main()
