"""
FiftyOne Evaluation Script for Faster R-CNN (Fixed Version)
============================================================
This script evaluates your trained Faster R-CNN model using FiftyOne (Voxel51).
Fixed version that properly handles COCO dataset loading and label fields.

Usage:
    python fiftyone_evaluation_fixed.py
"""

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
import fiftyone.core.labels as fol

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SETUP
# =============================================================================

# Dataset paths (for WSL2, use Linux paths)
DATASET_DIR = "/home/somel/code/FYP_Project/Dataset/COCO/version7.coco"
SPLIT = "valid"  # Options: "train", "valid", "test"

# Model checkpoint path
CHECKPOINT_PATH = "/home/somel/code/FYP_Project/Training_checkpoints/FasterRCNN/fastrcnn_run14/best.pth"

# Output directory for results
RESULTS_DIR = "/home/somel/code/FYP_Project/results/fiftyone_eval"

# Inference confidence threshold
CONF_THRESHOLD = 0.5

# IoU threshold for evaluation
IOU_THRESHOLD = 0.5

# =============================================================================


def setup_configuration():
    """Create necessary directories and validate paths."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Validate paths
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print(f"Results will be saved to: {RESULTS_DIR}")


def load_model_and_class_mapping(device):
    """Load the trained model and class mappings."""

    # Load COCO annotations to get class mapping
    labels_path = os.path.join(DATASET_DIR, SPLIT, "_annotations.coco.json")
    coco_gt = COCO(labels_path)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cats = sorted(cats, key=lambda x: x["id"])

    # Create mappings
    COCO_ID_TO_MODEL_LABEL = {c["id"]: i + 1 for i, c in enumerate(cats)}
    MODEL_LABEL_TO_COCO_ID = {v: k for k, v in COCO_ID_TO_MODEL_LABEL.items()}
    CLASS_NAMES = [c["name"] for c in cats]
    num_classes = len(cats) + 1  # +1 for background

    print(f"Classes: {CLASS_NAMES}")
    print(f"Number of classes: {len(CLASS_NAMES)}")

    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from: {CHECKPOINT_PATH}")

    return model, COCO_ID_TO_MODEL_LABEL, MODEL_LABEL_TO_COCO_ID, CLASS_NAMES


def run_inference(model, dataset, device, COCO_ID_TO_MODEL_LABEL, MODEL_LABEL_TO_COCO_ID):
    """Run model inference on the dataset and return predictions in COCO format."""

    predictions = []
    print(f"Running inference on {len(dataset)} images...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            img, _ = dataset[idx]
            output = model([img.to(device)])[0]
            image_id = int(dataset.ids[idx])

            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            # Filter by confidence threshold
            keep = scores > CONF_THRESHOLD

            for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
                x1, y1, x2, y2 = box
                # Convert model label back to COCO category ID
                coco_cat_id = MODEL_LABEL_TO_COCO_ID.get(int(label), int(label))

                predictions.append({
                    "image_id": image_id,
                    "category_id": int(coco_cat_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)} images")

    print(f"Total predictions: {len(predictions)}")
    return predictions


def create_fiftyone_dataset_with_ground_truth(data_path, labels_path, predictions, split_name, class_names):
    """
    Create a FiftyOne dataset by manually loading images and annotations.
    This ensures proper ground_truth field setup.
    """

    dataset_name = f"fasterrcnn_v7_{split_name}"

    # Delete existing dataset with same name
    if fo.dataset_exists(dataset_name):
        print(f"Deleting existing dataset: {dataset_name}")
        fo.delete_dataset(dataset_name)

    # Create new dataset
    fo_dataset = fo.Dataset(dataset_name)

    # Load COCO annotations
    with open(labels_path, 'r') as f:
        coco_data = json.load(f)

    # Build image_id to annotations mapping
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Build category mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    print(f"Loading {len(coco_data['images'])} images into FiftyOne...")

    # Add samples to dataset
    samples = []
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_path = os.path.join(data_path, img_info['file_name'])

        # Check if image exists
        if not os.path.exists(img_path):
            print(f"  Warning: Image not found: {img_path}")
            continue

        # Create FiftyOne sample
        sample = fo.Sample(filepath=img_path)

        # Add ground truth detections
        ground_truth_detections = []
        if img_id in img_to_anns:
            for ann in img_to_anns[img_id]:
                bbox_coco = ann['bbox']  # [x, y, w, h]
                category_id = ann['category_id']
                label = cat_id_to_name.get(category_id, f"class_{category_id}")

                # Convert COCO bbox to FiftyOne format [top-left-x, top-left-y, width, height]
                # FiftyOne expects normalized coordinates [0-1]
                img_width = img_info['width']
                img_height = img_info['height']

                det = fol.Detection(
                    label=label,
                    bounding_box=[
                        bbox_coco[0] / img_width,
                        bbox_coco[1] / img_height,
                        bbox_coco[2] / img_width,
                        bbox_coco[3] / img_height
                    ],
                    confidence=1.0  # Ground truth has 100% confidence
                )
                ground_truth_detections.append(det)

        if ground_truth_detections:
            sample["ground_truth"] = fol.Detections(detections=ground_truth_detections)

        samples.append(sample)

        if len(samples) % 100 == 0:
            print(f"  Loaded {len(samples)} images")

    # Add all samples to dataset
    fo_dataset.add_samples(samples)

    print(f"Dataset created with {len(fo_dataset)} samples")

    # Now add predictions
    print("Adding model predictions...")

    # Build image_id to predictions mapping
    pred_by_img_id = {}
    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in pred_by_img_id:
            pred_by_img_id[img_id] = []
        pred_by_img_id[img_id].append(pred)

    # Add predictions to samples
    for sample in fo_dataset:
        # Get image_id from filepath
        img_filename = os.path.basename(sample.filepath)

        # Find matching image_id in coco_data
        img_id = None
        for img_info in coco_data['images']:
            if img_info['file_name'] == img_filename:
                img_id = img_info['id']
                break

        if img_id is None:
            continue

        img_width = None
        img_height = None
        for img_info in coco_data['images']:
            if img_info['id'] == img_id:
                img_width = img_info['width']
                img_height = img_info['height']
                break

        pred_detections = []
        if img_id in pred_by_img_id:
            for pred in pred_by_img_id[img_id]:
                bbox_coco = pred['bbox']
                category_id = pred['category_id']
                score = pred['score']
                label = cat_id_to_name.get(category_id, f"class_{category_id}")

                det = fol.Detection(
                    label=label,
                    bounding_box=[
                        bbox_coco[0] / img_width,
                        bbox_coco[1] / img_height,
                        bbox_coco[2] / img_width,
                        bbox_coco[3] / img_height
                    ],
                    confidence=score
                )
                pred_detections.append(det)

        if pred_detections:
            sample["predictions"] = fol.Detections(detections=pred_detections)
            sample.save()

    print(f"Predictions added to {len(fo_dataset)} samples")

    return fo_dataset


def evaluate_and_get_metrics(fo_dataset, iou_threshold=0.5):
    """Run evaluation and extract F1, Precision, Recall, and Confusion Matrix."""

    print(f"\nRunning FiftyOne evaluation at IoU={iou_threshold}...")

    # Run detection evaluation
    results = fo_dataset.evaluate_detections(
        pred_field="predictions",
        gt_field="ground_truth",
        eval_key="eval",
        method="coco",
        iou=iou_threshold,
    )

    # Print COCO metrics report
    print("\n" + "="*60)
    print("COCO METRICS REPORT")
    print("="*60)
    results.print_report()

    # Get detailed report with precision, recall, F1
    print("\n" + "="*60)
    print(f"DETAILED METRICS (IoU={iou_threshold})")
    print("="*60)

    try:
        report = results.get_report(iou=iou_threshold)
        print(report)
    except Exception as e:
        print(f"Could not get detailed report: {e}")
        report = None

    # Get confusion matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    try:
        cm = results.confusion_matrix(iou=iou_threshold)
        print(cm)
    except Exception as e:
        print(f"Could not get confusion matrix: {e}")
        cm = None

    return results, report, cm


def save_metrics_report(report, cm, class_names, results_dir):
    """Save the evaluation report and confusion matrix."""

    # Save text report
    report_path = os.path.join(results_dir, "fiftyone_metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FIFTYONE EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"IoU Threshold: {IOU_THRESHOLD}\n")
        f.write(f"Confidence Threshold: {CONF_THRESHOLD}\n")
        f.write(f"Classes: {class_names}\n\n")
        if report is not None:
            f.write(str(report))
        else:
            f.write("Report not available\n")
    print(f"Metrics report saved to: {report_path}")

    # Save confusion matrix as CSV if available
    if cm is not None:
        cm_csv_path = os.path.join(results_dir, "confusion_matrix.csv")
        np.savetxt(cm_csv_path, cm, delimiter=",", fmt="%d")
        print(f"Confusion matrix CSV saved to: {cm_csv_path}")

        # Plot and save confusion matrix heatmap
        plt.figure(figsize=(12, 10))

        # Use class names for labels
        labels = class_names if len(class_names) == cm.shape[0] else [f"Class_{i}" for i in range(cm.shape[0])]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 12}
        )
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Ground Truth", fontsize=14)
        plt.title(f"Faster R-CNN Confusion Matrix (IoU={IOU_THRESHOLD})", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix plot saved to: {cm_plot_path}")
    else:
        print("Skipping confusion matrix (not available)")


def main():
    """Main evaluation pipeline."""

    print("="*60)
    print("FIFTYONE MODEL EVALUATION - FASTER R-CNN")
    print("="*60)

    # Setup
    setup_configuration()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, COCO_ID_TO_MODEL_LABEL, MODEL_LABEL_TO_COCO_ID, CLASS_NAMES = load_model_and_class_mapping(device)

    # Setup data paths
    data_path = os.path.join(DATASET_DIR, SPLIT)
    labels_path = os.path.join(DATASET_DIR, SPLIT, "_annotations.coco.json")

    # Load dataset for inference
    def transform_func(image, target):
        return F.to_tensor(image), target

    dataset = CocoDetection(root=data_path, annFile=labels_path, transforms=transform_func)

    # Run inference
    predictions = run_inference(model, dataset, device, COCO_ID_TO_MODEL_LABEL, MODEL_LABEL_TO_COCO_ID)

    # Save predictions to JSON
    pred_json_path = os.path.join(RESULTS_DIR, "predictions.json")
    with open(pred_json_path, "w") as f:
        json.dump(predictions, f)
    print(f"Predictions saved to: {pred_json_path}")

    # Create FiftyOne dataset with proper ground truth
    fo_dataset = create_fiftyone_dataset_with_ground_truth(
        data_path, labels_path, predictions, SPLIT, CLASS_NAMES
    )

    # Evaluate and get metrics
    results, report, cm = evaluate_and_get_metrics(fo_dataset, iou_threshold=IOU_THRESHOLD)

    # Save metrics
    save_metrics_report(report, cm, CLASS_NAMES, RESULTS_DIR)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"All results saved to: {RESULTS_DIR}")
    print(f"\nFiftyOne dataset name: {fo_dataset.name}")
    print("\nTo launch FiftyOne app later, run:")
    print(f"  fiftyone app {fo_dataset.name}")


if __name__ == "__main__":
    main()
