import cv2
import numpy as np
import matplotlib.pyplot as plt

def propose_regions(image_path, min_area=500, display=False):
    """
    Detect region proposals in the input image using edge detection and contour finding.
    
    Args:
        image_path (str): Path to the input image.
        min_area (int): Minimum contour area to be considered a proposal.
        display (bool): If True, will display the image with proposals drawn.
        
    Returns:
        List of bounding boxes [(x, y, w, h), ...].
    """
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Smooth and detect edges
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    proposals = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # Approximate contour to polygon and get bounding box
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

# Example usage:
# regions = propose_regions('/mnt/data/0006c52e8.jpg', min_area=1000, display=True)

