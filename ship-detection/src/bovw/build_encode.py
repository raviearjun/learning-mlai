# src/bovw/build_encode.py
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Bangun vocabulary BoVW
def build_vocabulary(descriptor_list, k=100):
    all_descriptors = np.vstack(descriptor_list)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=k*10)
    kmeans.fit(all_descriptors)
    return kmeans

# Encode deskriptor setiap citra ke histogram BoVW
def encode_bovw(descriptors, kmeans):
    words = kmeans.predict(descriptors)
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters+1))
    return hist.astype(float) / hist.sum()

# Proses seluruh dataset
def encode_dataset(descriptor_list, kmeans):
    return np.array([encode_bovw(desc, kmeans) for desc in descriptor_list])