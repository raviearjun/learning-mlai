# src/utils/image.py
import cv2

def resize_with_padding(img, size, pad_value=0):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (new_w,new_h))
    pad_w = size - new_w
    pad_h = size - new_h
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    return cv2.copyMakeBorder(
      resized, top, bottom, left, right,
      cv2.BORDER_CONSTANT, value=pad_value
    )
