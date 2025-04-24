## Directory: src/features.py
import cv2

face_size = (128, 128)

def resize_and_flatten(face):
    face_resized = cv2.resize(face, face_size)
    return face_resized.flatten()

