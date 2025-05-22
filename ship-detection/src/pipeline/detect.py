import cv2, csv
from pathlib import Path
import joblib
from src.proposals.selective_search import propose_regions
from src.feature_extraction.sift_features import extract_sift
from src.bovw.build_encode import encode_bovw
from src.utils.image_processing import resize_with_padding

def run_detection(image_path, kmeans, svm, region_params=None, resize_size=128):
    img = cv2.imread(str(image_path))
    if img is None: raise IOError(f"Cannot read {image_path}")
    regions = propose_regions(str(image_path), **(region_params or {}))
    pred_boxes = []
    for (x,y,w,h) in regions:
        patch = resize_with_padding(img[y:y+h, x:x+w], resize_size, 0)
        desc = extract_sift(patch)
        if desc is None or len(desc)==0: continue
        fv = encode_bovw(desc, kmeans)
        score = float(svm.predict_proba([fv])[0][1])
        pred_boxes.append((x, y, x+w, y+h, score))
        cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0) if score>0.5 else (0,0,255), 2)
        cv2.putText(img, f"{score:.2f}", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if score>0.5 else (0,0,255), 2)
    return img, pred_boxes

def detect_folder(test_dir, result_dir, pred_csv, kmeans_path, svm_path, region_params=None):
    kmeans = joblib.load(str(kmeans_path))
    svm    = joblib.load(str(svm_path))
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId","x0","y0","x1","y1","score"])
        for img_path in sorted(test_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
            img_annot, boxes = run_detection(img_path, kmeans, svm, region_params)
            out_img = result_dir / f"{img_path.stem}_det.jpg"
            cv2.imwrite(str(out_img), img_annot)
            for x0,y0,x1,y1,score in boxes:
                writer.writerow([img_path.name, x0, y0, x1, y1, f"{score:.4f}"])
    print(f"[Done] Predictions → {pred_csv}")
