import cv2
import numpy as np
import matplotlib.pyplot as plt
from .nms import non_max_suppression

def propose_regions(image_path, min_area=500, display=False):
    """
    Detect region proposals in the input image using edge detection and contour finding.
    If no region is found, enhance contrast with CLAHE and try again.
    """
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- Proses awal tanpa CLAHE ---
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    proposals = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        proposals.append((x, y, w, h))
    
    # --- Jika tidak ada region (kontras rendah) coba dengan CLAHE ---
    if len(proposals) == 0:
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16)).apply(blurred)

        edges = cv2.Canny(gray_clahe, 40, 100)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            proposals.append((x, y, w, h))
    
    # Optionally display results
    if display:
        disp_img = image.copy()
        for (x, y, w, h) in proposals:
            cv2.rectangle(disp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return proposals

