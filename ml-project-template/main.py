## File: main.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.data_loader import load_image, load_dataset
from src.face_detector import detect_faces, crop_faces
from src.features import resize_and_flatten
from src.model import build_pipeline
from src.utils import draw_result

# Load and process dataset
dataset_dir = 'images'
images, labels = load_dataset(dataset_dir)
X, y = [], []
for image, label in zip(images, labels):
    faces = detect_faces(image)
    cropped, _ = crop_faces(image, faces)
    if cropped:
        face_flattened = resize_and_flatten(cropped[0])
        X.append(face_flattened)
        y.append(label)
X, y = np.array(X), np.array(y)

print(f'Loaded {len(images)} images with labels: {np.unique(labels)}')

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train model
pipe = build_pipeline()
pipe.fit(X_train, y_train)

# Save model
with open('outputs/eigenface_pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Evaluate
from sklearn.metrics import classification_report
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualize eigenfaces
n_components = len(pipe[1].components_)
eigenfaces = pipe[1].components_.reshape((n_components, X_train.shape[1]))
face_size = (128, 128)
ncol = 4
nrow = (n_components + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.5 * nrow), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    if i < n_components:
        ax.imshow(eigenfaces[i].reshape(face_size), cmap='gray')
        ax.set_title(f'Eigenface {i+1}')
plt.tight_layout()
plt.savefig('outputs/visualizations/eigenfaces.png')
plt.show()