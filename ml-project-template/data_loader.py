## Directory: src/data_loader.py
import os
import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print('None')
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, gray

def load_dataset(dataset_dir):
    images, labels = [] , []
    for root, _, files in os.walk(dataset_dir):
        if not files:
            continue
        for f in files:
            _, gray = load_image(os.path.join(root, f))
            if gray is not None:
                images.append(gray)
                labels.append(root.split('/')[-1])
    return images, labels