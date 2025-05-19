# src/model/train_svm.py
# Pastikan package terdeteksi dengan menambahkan file __init__.py di setiap folder src, feature_extraction, bovw, model
import os
import sys
# Tambahkan src ke path jika perlu saat menjalankan langsung
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import joblib
import numpy as np
from sklearn.svm import SVC
from src.feature_extraction.sift_features import load_descriptors
from src.bovw.build_encode import build_vocabulary, encode_dataset

if __name__ == '__main__':
    # 1. Muat dan ekstrak deskriptor beserta label
    desc_list, labels = load_descriptors('images/train')
    if len(desc_list) == 0:
        raise ValueError("Deskriptor kosong. Pastikan gambar ada dan bisa dibaca.")


    # 2. Bangun vocabulary dan simpan
    k = 100
    kmeans = build_vocabulary(desc_list, k=k)
    joblib.dump(kmeans, 'out/models/bovw_kmeans.pkl')

    # 3. Encode dataset ke BoVW
    X = encode_dataset(desc_list, kmeans)
    y = np.array(labels)

    # 4. Latih SVM dan simpan model
    model = SVC(kernel='rbf', C=1.0, probability=True)
    model.fit(X, y)
    joblib.dump(model, 'out/models/svm_model.pkl')

    print(f"Training selesai: dataset size={X.shape}, labels={np.unique(y)}")