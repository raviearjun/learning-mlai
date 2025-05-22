#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

from src.pipeline.detect import detect_folder
from src.evaluation.metrics import match_detections, classification_report_from_matches

# 1) Paths
PROJECT = Path(__file__).parent
TEST_DIR    = PROJECT/"images"/"test_set"
RESULT_DIR  = PROJECT/"images"/"test_result"
PRED_CSV    = PROJECT/"out"/"predictions"/"predictions.csv"
KM_PATH     = PROJECT/"out"/"models"/"bovw_kmeans.pkl"
SVM_PATH    = PROJECT/"out"/"models"/"rf_model.pkl"
GT_CSV      = PROJECT/"test_set_1_subset.csv"  # sesuaikan path

# 2) Deteksi dan simpan predictions.csv
detect_folder(TEST_DIR, RESULT_DIR, PRED_CSV, KM_PATH, SVM_PATH,
              region_params={"min_area":500,"display":False})

# 3) Decode GT RLE → gt_boxes dict
def rle_to_boxes(df):
    boxes = {}
    for img, g in df.groupby("ImageId"):
        mask = np.zeros(768*768, bool)
        for r in g.EncodedPixels.dropna():
            arr = np.array(r.split(), int).reshape(-1,2)
            for start,length in arr:
                mask[start-1:start-1+length] = True
        m = mask.reshape((768,768), order='F')
        ys, xs = np.where(m)
        if len(xs):
            boxes[img] = [(xs.min(), ys.min(), xs.max(), ys.max())]
    return boxes

gt_df = pd.read_csv(GT_CSV)
gt_boxes = rle_to_boxes(gt_df)

# 4) Load pred_boxes from CSV
import csv
pred_boxes = {}
with open(PRED_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        img = row["ImageId"]
        pred_boxes.setdefault(img, []).append(
            (int(row["x0"]),int(row["y0"]),int(row["x1"]),int(row["y1"]), float(row["score"]))
        )

# 5) Match & kumpulkan labels
all_true, all_pred = [], []
tot_tp = tot_fp = tot_fn = 0
for img, gtb in gt_boxes.items():
    pb = pred_boxes.get(img, [])
    tp, fp, fn, y_true, y_pred = match_detections(pb, gtb, iou_thresh=0.5)
    tot_tp+=tp; tot_fp+=fp; tot_fn+=fn
    all_true.extend(y_true); all_pred.extend(y_pred)

print(f"\n=== Summary ===\nTP={tot_tp}, FP={tot_fp}, FN={tot_fn}\n")
classification_report_from_matches(all_true, all_pred)

print("Total GT boxes:", sum(len(v) for v in gt_boxes.values()))
print("Total pred boxes:", sum(len(v) for v in pred_boxes.values()))
print("Total y_true:", len(all_true))
print("Total y_pred:", len(all_pred))
