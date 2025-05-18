import numpy as np
def non_max_suppression(boxes, overlap_thresh=0.3):
    """
    Apply Non-Maximum Suppression (NMS) to suppress overlapping bounding boxes.
    
    Args:
        boxes (list of tuples): List of bounding boxes [(x, y, w, h), ...].
        overlap_thresh (float): Threshold for overlapping. Lower = more strict.
        
    Returns:
        List of filtered bounding boxes after NMS.
    """
    if len(boxes) == 0:
        return []

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    boxes_np = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes], dtype=np.float32)
    
    # If there's just one box, return it
    if len(boxes_np) == 1:
        return boxes
    
    # Extract coordinates
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    
    # Calculate areas for each box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # For this application, we'll use area as our "score"
    # Sort by area (largest first)
    order = np.argsort(areas)[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Find overlapping boxes with the current largest box
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Compute the width and height of the intersection area
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # Compute IoU (Intersection over Union)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with overlap less than the threshold
        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]
        
    return [boxes[i] for i in keep]