import numpy as np
from sklearn.metrics import classification_report

def iou(boxA, boxB):
    xA, yA = max(boxA[0],boxB[0]), max(boxA[1],boxB[1])
    xB, yB = min(boxA[2],boxB[2]), min(boxA[3],boxB[3])
    inter = max(0, xB-xA+1) * max(0, yB-yA+1)
    areaA = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    areaB = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    union = float(areaA + areaB - inter)
    return inter/union if union>0 else 0

def match_detections(pred_boxes, gt_boxes, iou_thresh=0.5):
    preds = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    matched, tp, fp = set(), 0, 0
    y_true, y_pred = [], []
    for (x0,y0,x1,y1,s) in preds:
        best_i, best_iou = None, 0
        for j, gt in enumerate(gt_boxes):
            if j in matched: continue
            iou_val = iou((x0,y0,x1,y1), gt)
            if iou_val>best_iou: best_iou, best_i = iou_val, j
        if best_iou>=iou_thresh:
            tp +=1; matched.add(best_i)
            y_true.append(1); y_pred.append(1)
        else:
            fp +=1
            y_true.append(0); y_pred.append(1)
    fn = len(gt_boxes) - len(matched)
    for _ in range(fn):
        y_true.append(1); y_pred.append(0)
    return tp, fp, fn, y_true, y_pred

def classification_report_from_matches(y_true, y_pred):
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred,
          target_names=["background","ship"], digits=4))
