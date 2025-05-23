import cv2
import numpy as np
import joblib

from src.feature_extraction.sift_features import extract_sift
from src.bovw.build_encode import encode_bovw

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
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_value)
    return padded

def propose_regions_from_frame_np(image, min_area=500):
    """
    Region proposals langsung dari ndarray (frame BGR).
    Replikasi fungsi selective_search.propose_regions tanpa file I/O.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        proposals.append((x, y, w, h))

    # Jika tidak ada region, coba CLAHE
    if not proposals:
        gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))\
                       .apply(blurred)
        edges2 = cv2.Canny(gray_clahe, 30, 90)
        dilated2 = cv2.dilate(edges2, kernel, iterations=2)
        contours2, _ = cv2.findContours(dilated2, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours2:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            proposals.append((x, y, w, h))

    return proposals

def main():
    # Load models
    kmeans = joblib.load("out/models/bovw_kmeans.pkl")
    svm    = joblib.load("out/models/svm_model.pkl")

    video_path = "test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Gagal membuka video:", video_path)
        return

    # Setup video writer
    import os
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_video_path = os.path.join(out_dir, "detected_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    print("✅ Memulai deteksi pada video:", video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        regions = propose_regions_from_frame_np(frame, min_area=500)

        for (x, y, w, h) in regions:
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            crop_padded = resize_with_padding(crop, 128, pad_value=0)
            desc = extract_sift(crop_padded)
            if desc is None or len(desc) == 0:
                continue

            bovw = encode_bovw(desc, kmeans)
            pred = svm.predict([bovw])[0]
            proba = svm.predict_proba([bovw])[0][pred]

            if pred == 1 and proba > 0.3:
                color = (0, 255, 0)
                label = f"Ship {proba:.2f}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Live Ship Detection", frame)
        writer.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"🛑 Video selesai dianalisis. Hasil disimpan di {out_video_path}")

if __name__ == "__main__":
    main()
