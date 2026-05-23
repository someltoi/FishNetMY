# =============================================================================
# FIFTYONE EVALUATION - COMPLETE CELL FOR NOTEBOOK
# Copy this entire cell into your Faster_R_CNN_script 51.ipynb
# =============================================================================

# Install fiftyone if not already installed (uncomment if needed)
# !pip install fiftyone fiftyone-db

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
import fiftyone as fo
import fiftyone.utils.coco as fouc

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR = ROOT_DIR  # Uses the same ROOT_DIR from your notebook
SPLIT = "valid"  # Change to "test" for final evaluation
CHECKPOINT_PATH = CHECKPOINT_PATH  # Uses existing checkpoint path
RESULTS_DIR = os.path.join(RESULTS_DIR, "fiftyone_eval")
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}")

# =============================================================================
# LOAD MODEL (uses existing model from notebook or loads from checkpoint)
# =============================================================================

# Option 1: Use the already loaded model from your notebook
# (model should already be in memory from training)

# Option 2: Load from checkpoint (uncomment if needed)
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
"""

print(f"Model device: {next(model.parameters()).device}")

# =============================================================================
# RUN INFERENCE
# =============================================================================

data_path = os.path.join(DATASET_DIR, SPLIT)
labels_path = os.path.join(DATASET_DIR, SPLIT, "_annotations.coco.json")

def transform_func(image, target):
    return F.to_tensor(image), target

eval_dataset = CocoDetection(root=data_path, annFile=labels_path, transforms=transform_func)

print(f"\nRunning inference on {len(eval_dataset)} images...")

predictions = []
with torch.no_grad():
    for idx in range(len(eval_dataset)):
        img, _ = eval_dataset[idx]
        output = model([img.to(device)])[0]
        image_id = int(eval_dataset.ids[idx])

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        # Filter by confidence
        keep = scores > CONF_THRESHOLD

        for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
            x1, y1, x2, y2 = box
            coco_cat_id = MODEL_LABEL_TO_COCO_ID.get(int(label), int(label))
            predictions.append({
                "image_id": image_id,
                "category_id": int(coco_cat_id),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(eval_dataset)} images")

print(f"Total predictions: {len(predictions)}")

# Save predictions
pred_json_path = os.path.join(RESULTS_DIR, "predictions.json")
with open(pred_json_path, "w") as f:
    json.dump(predictions, f)
print(f"Predictions saved to: {pred_json_path}")

# =============================================================================
# CREATE FIFTYONE DATASET
# =============================================================================

dataset_name = f"fasterrcnn_v7_{SPLIT}"

# Delete existing dataset
if fo.dataset_exists(dataset_name):
    print(f"\nDeleting existing dataset: {dataset_name}")
    fo.delete_dataset(dataset_name)

# Load COCO dataset into FiftyOne
print("\nLoading COCO dataset into FiftyOne...")
fo_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
    name=dataset_name,
    label_field="ground_truth",
)

# Add predictions
print("Adding model predictions to FiftyOne dataset...")
fouc.add_coco_labels(
    fo_dataset,
    "predictions",
    predictions,
    fo_dataset.default_classes,
)

print(f"FiftyOne dataset '{dataset_name}' created with {len(fo_dataset)} samples")

# =============================================================================
# EVALUATION - Get F1, Precision, Recall
# =============================================================================

print("\n" + "="*60)
print("FIFTYONE EVALUATION RESULTS")
print("="*60)

# Run COCO evaluation
results = fo_dataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    eval_key="eval",
    method="coco",
    iou=IOU_THRESHOLD,
)

# Print COCO metrics
print("\nCOCO METRICS:")
results.print_report()

# Get detailed report with precision, recall, F1
print(f"\n\nDETAILED METRICS REPORT (IoU={IOU_THRESHOLD}):")
report = results.get_report(iou=IOU_THRESHOLD)
print(report)

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

cm = results.confusion_matrix(iou=IOU_THRESHOLD)
print(cm)

# =============================================================================
# SAVE RESULTS
# =============================================================================

# Save text report
report_path = os.path.join(RESULTS_DIR, "fiftyone_metrics_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("FIFTYONE EVALUATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"IoU Threshold: {IOU_THRESHOLD}\n")
    f.write(f"Confidence Threshold: {CONF_THRESHOLD}\n")
    f.write(f"Classes: {CLASS_NAMES}\n\n")
    f.write(str(report))
print(f"\nMetrics report saved to: {report_path}")

# Save confusion matrix CSV
cm_csv_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
np.savetxt(cm_csv_path, cm, delimiter=",", fmt="%d")
print(f"Confusion matrix CSV saved to: {cm_csv_path}")

# Plot confusion matrix
plt.figure(figsize=(12, 10))

# Get labels for confusion matrix
cm_labels = CLASS_NAMES if len(CLASS_NAMES) == cm.shape[0] else [f"Class_{i}" for i in range(cm.shape[0])]

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=cm_labels,
    yticklabels=cm_labels,
    annot_kws={"size": 12}
)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Ground Truth", fontsize=14)
plt.title(f"Faster R-CNN Confusion Matrix (IoU={IOU_THRESHOLD})", fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Confusion matrix plot saved to: {cm_plot_path}")

# =============================================================================
# LAUNCH FIFTYONE APP (INTERACTIVE VISUALIZATION)
# =============================================================================

print("\n" + "="*60)
print("LAUNCHING FIFTYONE APP")
print("="*60)
print("""
The FiftyOne app will open in your browser.

You can:
  - Visualize predictions (blue) vs ground truth (green)
  - Filter by confidence threshold
  - Inspect false positives and false negatives
  - View per-class metrics
  - Export samples for analysis

To close the app, press Ctrl+C or run: fo.close_app()
""")

# Launch the app (this will block until closed)
session = fo.launch_app(fo_dataset)

print("\nEvaluation complete! All results saved to:", RESULTS_DIR)
