import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from src.feature_extraction.sift_features import load_descriptors
from src.bovw.build_encode import build_vocabulary, encode_dataset

def train_and_save_model(model, X, y, out_path):
    model.fit(X, y)
    joblib.dump(model, out_path)
    print(f"Model saved to {out_path}")

if __name__ == '__main__':
    # 1. Muat dan ekstrak deskriptor beserta label
    desc_list, labels = load_descriptors('images/train_set')
    if len(desc_list) == 0:
        raise ValueError("Deskriptor kosong. Pastikan gambar ada dan bisa dibaca.")

    # 2. Bangun vocabulary dan simpan
    k = 100
    kmeans = build_vocabulary(desc_list, k=k)
    Path('out/models').mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, 'out/models/bovw_kmeans.pkl')

    # 3. Encode dataset ke BoVW
    X = encode_dataset(desc_list, kmeans)
    y = np.array(labels)

    # 4. Latih dan simpan model SVM
    svm = SVC(kernel='rbf', C=1.0, probability=True)
    train_and_save_model(svm, X, y, 'out/models/svm_model.pkl')

    # 5. Latih dan simpan model Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_save_model(rf, X, y, 'out/models/rf_model.pkl')

    # 6. Latih dan simpan model AdaBoost
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    train_and_save_model(ada, X, y, 'out/models/ada_model.pkl')

    print(f"Training selesai: dataset size={X.shape}, labels={np.unique(y)}")