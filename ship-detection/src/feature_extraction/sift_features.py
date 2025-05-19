import os
import cv2

def extract_sift(image_gray, nfeatures=500):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    _, descriptors = sift.detectAndCompute(image_gray, None)
    return descriptors

def load_descriptors(train_dir):
    all_desc = []
    labels = []
    total = 0
    gagal_baca = 0
    kosong = 0

    for cls in ['0', '1']:
        folder = os.path.join(train_dir, cls)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            total += 1

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Gagal membaca {path}")
                gagal_baca += 1
                continue

            desc = extract_sift(img)
            if desc is None:
                print(f"[INFO] Tidak ada fitur SIFT pada {path}")
                kosong += 1
                continue

            all_desc.append(desc)
            labels.append(int(cls))

    print(f"\n[SUMMARY]")
    print(f"Total gambar         : {total}")
    print(f"Berhasil diekstraksi : {len(all_desc)}")
    print(f"Gagal dibaca         : {gagal_baca}")
    print(f"Tidak ada fitur SIFT : {kosong}\n")

    return all_desc, labels
