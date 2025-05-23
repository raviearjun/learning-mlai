import csv
from pathlib import Path
import joblib
import numpy as np
import cv2

from src.proposals.selective_search import propose_regions
from src.feature_extraction.sift_features import extract_sift
from src.bovw.build_encode import encode_bovw
from src.utils.image_processing import resize_with_padding


def run_detection(image_path, kmeans, svm, region_params=None, resize_size=128):
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Cannot read {image_path}")

    regions = propose_regions(str(image_path), **(region_params or {}))
    pred_boxes = []
    score_thresh = 0.3

    for (x, y, w, h) in regions:
        patch = resize_with_padding(img[y:y+h, x:x+w], resize_size, 0)
        desc = extract_sift(patch)
        if desc is None or len(desc) == 0:
            continue

        fv = encode_bovw(desc, kmeans)
        score = float(svm.predict_proba([fv])[0][1])
        pred_boxes.append((x, y, x+w, y+h, score))

        # Annotate for visualization
        color = (0, 255, 0) if score >= score_thresh else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{score:.2f}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, pred_boxes


def detect_folder(test_dir, result_dir, pred_csv,
                  kmeans_path, svm_path,
                  region_params=None,
                  score_thresh=0.3):
    """
    Menjalankan deteksi pada semua gambar di `test_dir`, menyimpan citra ber-annotation di `result_dir`
    dan menulis file CSV `pred_csv` dengan filter score.

    Args:
        test_dir (Path): direktori input gambar .jpg/.png
        result_dir (Path): direktori output untuk gambar ber-annotasi
        pred_csv (Path): path file CSV untuk menyimpan prediksi
        kmeans_path (Path): path model kmeans (BOVW)
        svm_path (Path): path model SVM
        region_params (dict): parameter untuk propose_regions
        score_thresh (float): ambang minimum score untuk disimpan
    """
    # Load models
    kmeans = joblib.load(str(kmeans_path))
    svm = joblib.load(str(svm_path))

    # Siapkan direktori
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    Path(pred_csv).parent.mkdir(parents=True, exist_ok=True)

    # Tulis header CSV
    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "x0", "y0", "x1", "y1", "score"])

        # Loop setiap gambar
        for img_path in sorted(Path(test_dir).iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            # Jalankan deteksi region + fitur + klasifikasi
            img_annot, boxes = run_detection(
                img_path, kmeans, svm, region_params
            )

            # Filter berdasarkan score
            boxes = [b for b in boxes if b[4] >= score_thresh]

            # Simpan gambar ber-annotasi
            out_img = result_dir / f"{img_path.stem}_det.jpg"
            cv2.imwrite(str(out_img), img_annot)

            # Tulis baris prediksi ke CSV
            for x0, y0, x1, y1, score in boxes:
                writer.writerow([
                    img_path.name,
                    x0, y0, x1, y1,
                    f"{score:.4f}"
                ])

    print(f"[Done] Predictions saved to {pred_csv}")

