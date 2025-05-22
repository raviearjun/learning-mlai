import pandas as pd
from pathlib import Path

def main():
    # Path ke CSV dan folder test_set/1
    base = Path(__file__).parent
    csv_path = base / "train_ship_segmentations_v2.csv"
    test_dir = base / "images" / "test_set"
    out_csv  = base / "test_set_1_subset.csv"

    # 1) Ambil nama file (dengan ekstensi) di test_dir
    exts = {".jpg", ".jpeg", ".png"}
    image_names = {
        p.name.strip()
        for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    }
    print(f"[DEBUG] Found {len(image_names)} files:", image_names)

    # 2) Baca CSV utama dan strip ImageId
    df = pd.read_csv(csv_path, dtype=str)
    df["ImageId"] = df["ImageId"].str.strip()
    print(f"[DEBUG] Unique ImageIds in CSV: {df['ImageId'].nunique()}")

    # 3) Filter
    subset = df[df["ImageId"].isin(image_names)]
    print(f"[DEBUG] Rows after filter: {len(subset)}")

    # 4) Simpan hasil
    subset.to_csv(out_csv, index=False)
    print(f"Subset CSV saved to {out_csv}")

if __name__ == "__main__":
    main()
