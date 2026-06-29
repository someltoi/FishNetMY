# FishNetMY 🐟
### Deep Learning-Based Detection of Malaysian Fish Species for Smart Aquaculture Applications

> **Bachelor of Computer Engineering (Honours) Final Year Project**
> Samuel Tai Zi Wei — Universiti Teknikal Malaysia Melaka (UTeM), 2026

---

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Results](#results)
- [Deployment — FishNetMY Desktop Application](#deployment--fishnetmy-desktop-application)

---

## Introduction

Fish are a cornerstone of global food security, serving as a primary source of animal protein for over 3.3 billion people worldwide. In Malaysia, fish are a dietary staple and a critical pillar of the national food supply chain.

However, the Malaysian aquaculture sector is facing a compounding socio-economic crisis. The industry's workforce is ageing rapidly — studies in Perak reported an average farmer age of 55.43 years, with over 33% aged above 60. Coupled with the migration of younger generations to urban areas, the sector has grown increasingly dependent on manual, labour-intensive practices that are unsustainable in the long term.

While modern technologies such as AI, automation, and IoT offer viable solutions, their adoption remains low due to high initial costs, technical complexity, and a lack of local expertise. Even where affordable sensors are deployed, they often fall short in accuracy and require complex calibration.

**FishNetMY** addresses these challenges by applying deep learning-based object detection to automate the identification of three commercially significant Malaysian fish species — **Tilapia**, **Grouper**, and **Seabass** — under real-world aquaculture conditions. Three architectures were benchmarked: **YOLOv11**, **Faster R-CNN**, and **Mask R-CNN**, with the best-performing model deployed through a cross-platform desktop application supporting multiple inference modes.

---

## Problem Statement

Three core challenges motivate this project:

**1. Scarcity of Local Dataset**
There is a critical lack of standardized, high-quality image datasets tailored to Malaysian aquaculture species. Most public datasets exhibit regional bias, featuring species not native to local farms. Real-world aquaculture images are also degraded by water turbidity, low lighting, and light scattering, which significantly hampers standard computer vision algorithms.

**2. Lab-to-Field Performance Gap**
Many existing CNN architectures perform well on clean, pre-processed datasets but struggle to generalize in dynamic aquaculture environments. Complex background noise, varying water conditions, and the presence of sediment or algae introduce noise not present during training, causing accuracy to degrade in practical deployment.

**3. Pose and Orientation Limitations**
Conventional detection models rely heavily on clear side-view poses to extract features effectively. In natural aquatic environments, fish move freely in three-dimensional space, resulting in complex poses, substantial occlusion, and rotation. Standard models often fail to correctly detect or classify fish viewed from the front, rear, or at acute angles.

---

## Objectives

1. **Dataset Development** — Build a comprehensive, annotated image dataset of Tilapia, Grouper, and Seabass by collecting, standardizing, and augmenting images from multiple sources under varied environmental conditions.

2. **Architecture Design & Training** — Design and implement multiple deep learning architectures (YOLOv11, Faster R-CNN, Mask R-CNN) capable of detecting the three target species using the established dataset.

3. **Performance Evaluation** — Benchmark the selected models using standard object detection metrics: **Precision**, **Recall**, **F1-Score**, and **mean Average Precision (mAP)**.

---

## Results

### Dataset

A total of **1,800 raw images** were collected across three classes (600 per species). After a stratified 80:10:10 split, the training set was augmented 3× to yield **4,320 training images**, while validation and test sets remained at 180 images each of unseen, raw data.

| Class   | Raw Images | Augmented Training (80%) | Validation (10%) | Test (10%) |
|---------|-----------|--------------------------|------------------|------------|
| Tilapia | 600       | 1,440                    | 60               | 60         |
| Grouper | 600       | 1,440                    | 60               | 60         |
| Seabass | 600       | 1,440                    | 60               | 60         |
| **Total** | **1,800** | **4,320**              | **180**          | **180**    |

<img width="720" height="433" alt="image" src="https://github.com/user-attachments/assets/c3dd9396-5f08-45c8-bc3a-f7e043681e71" />


---

### Training Convergence

All three architectures demonstrated stable convergence with no significant overfitting, validating the effectiveness of the augmentation pipeline and hyperparameter configurations.

- **YOLOv11** trained over 300 epochs, with localization loss stabilizing around epoch 200 and mAP plateauing after epoch 150.
- **Faster R-CNN** (80 epochs) and **Mask R-CNN** (70 epochs) converged much faster, with loss values plateauing within the first 20–40 epochs — typical of two-stage detectors.

Loss Curve Graph for a) YOLOv11, b) Faster R-CNN, c) Mask R-CNN
<img width="586" height="216" alt="image" src="https://github.com/user-attachments/assets/7fad767e-38e7-4c7a-980c-473b5e506ebb" />

The figure below shows only YOLOv11 mAP@50 and mAP@50-95 curve graphs
<img width="634" height="306" alt="image" src="https://github.com/user-attachments/assets/585e57ec-9dea-4077-a50f-cc02e196e2cd" />

---

### Quantitative Performance

| Model         | Precision (%) | Recall (%) | F1-Score (%) | mAP@0.5 (%) | mAP@0.5-0.95 (%) |
|---------------|--------------|------------|--------------|-------------|-----------------|
| **YOLOv11**   | **96.17**    | 96.16      | 96.17        | **98.26**   | **90.68**       |
| Faster R-CNN  | 90.73        | 96.88      | 93.68        | 97.22       | 74.92           |
| Mask R-CNN    | 94.50        | **97.93**  | **96.18**    | 97.31       | 79.66           |

**YOLOv11** achieved the highest Precision (96.17%), mAP@0.5 (98.26%), and strict mAP@0.5-0.95 (90.68%), making it the clear winner for deployment. Faster R-CNN recorded competitive recall but the weakest bounding box regression under strict IoU. Mask R-CNN achieved the highest recall but traded off precision with a higher false positive rate.

F1-Curve Graph for a) YOLOv11, b) Faster R-CNN, c) Mask R-CNN
<img width="584" height="202" alt="image" src="https://github.com/user-attachments/assets/72d0e4a2-8fb2-4599-b937-91e53108287c" />


---

### Qualitative Analysis (YOLOv11 only)

**Confusion Matrix** results showed YOLOv11 achieved per-class accuracy of **0.98 for Seabass**, **0.97 for Tilapia**, and **0.94 for Grouper**, with a low false positive rate against background.

<img width="854" height="420" alt="image" src="https://github.com/user-attachments/assets/79d17261-9691-4d4b-8db0-bcf0f056460a" />

**Inference Samples** demonstrated strong detection across varied lighting, turbidity, and fish orientation. The primary failure cases observed were heavily occluded fish and extreme non-standard poses.

<img width="854" height="575" alt="image" src="https://github.com/user-attachments/assets/6ff75895-94a4-4a29-86cb-e4e6244d1ca0" />


<img width="424" height="423" alt="image" src="https://github.com/user-attachments/assets/8cd4a519-ae12-4907-90f1-3f8062d959bf" />


---

### Computational Efficiency

| Model        | Parameters  | Model Size (MB) | FPS     | Peak VRAM (MB) |
|--------------|-------------|-----------------|---------|----------------|
| **YOLOv11**  | 2.59M       | **5.29**        | **222.73** | 71.96       |
| Faster R-CNN | 41.37M      | 314.15          | 24.36   | ~542           |
| Mask R-CNN   | 43.99M      | 334.94          | 28.10   | ~549           |

YOLOv11 is **~8× faster** than R-CNN variants and occupies less than **2% of the storage space** of Mask R-CNN. Its low VRAM footprint (71.96 MB) makes it suitable for edge deployment on NVIDIA Jetson and lower-end GPUs.

**YOLOv11 was selected as the primary architecture** for FishNetMY — offering the optimal balance of precision, real-time speed, and lightweight deployment footprint.

---

## Deployment — FishNetMY Desktop Application

The trained YOLOv11 model was integrated into a **cross-platform desktop application** built with Python and Tkinter. The application supports four inference modalities, a centralized model management system, and a non-blocking threading architecture to keep the UI responsive during heavy inference workloads.

---

### Main Interface (Idle State)

The main window provides a top navigation bar with direct access to four modes: **Live Camera**, **Single Image**, **Folder Batch**, and **Video Inference**. A dropdown dynamically populates registered `.pt` and `.pth` model checkpoints, and a status indicator provides real-time CPU/GPU utilization feedback.

<img width="696" height="512" alt="image" src="https://github.com/user-attachments/assets/3ab327fb-fb16-4f98-826b-7d6542f44b5c" />


---

### Model Configuration

The **Model Configuration Dialog** allows users to assign display aliases, edit class labels, and set custom overlay colours per class for visual differentiation. It supports bulk import via COCO JSON or YAML files, with configurations persisted to a JSON registry for consistent deployment across workstations.

<img width="703" height="478" alt="image" src="https://github.com/user-attachments/assets/7ef2ef1a-1177-424e-b980-df7bb6edfa81" />


---

### Live Camera Inference

The **Live Inference** mode streams real-time video from a connected camera device. A dedicated threading worker captures frames, applies user-defined confidence thresholds, and renders bounding boxes without blocking the main UI thread. Compatible with virtual camera drivers (OBS, Camo Studio) via DirectShow on Windows.

<img width="636" height="641" alt="image" src="https://github.com/user-attachments/assets/fe9525f7-9402-4be5-93f6-596f0e030109" />


---

### Single Image Inference

The **Single Image** mode is designed for diagnostic validation and quick field assessments. Users upload a static image via file dialog; the system runs a synchronous forward pass and renders detection outputs. Results can be exported as PNG/JPEG or launched directly in the system's default image viewer.

<img width="710" height="382" alt="image" src="https://github.com/user-attachments/assets/5ffb0f8c-40d2-41d4-9a30-006d9232c909" />


---

### Folder Batch Inference

The **Folder Batch** mode processes large datasets or archived monitoring footage. The interface uses a paginated grid (~30 images per page) with responsive tile scaling. Inference runs asynchronously with a real-time progress bar. Post-processing supports individual or bulk export with auto-appended suffixes to prevent overwriting.

<img width="854" height="548" alt="image" src="https://github.com/user-attachments/assets/3f493939-d0bb-4b98-a128-42994320e812" />


---

### Video Inference

The **Video Inference** mode processes sequential video frames for temporal analysis. Supported formats include MP4, AVI, MOV, MKV, and WMV. A letterbox preprocessing mechanism preserves aspect ratios at 1280×720 inference resolution. Processed frames are buffered in memory and exported as a complete annotated video retaining the original frame rate.

<img width="854" height="550" alt="image" src="https://github.com/user-attachments/assets/f6750c02-82cb-4686-adf3-20f2c306b238" />


---

## Tech Stack

- **Model Training:** Python, PyTorch, Ultralytics YOLOv11, Torchvision (Faster R-CNN / Mask R-CNN)
- **Dataset Management:** Roboflow (annotation & augmentation)
- **Desktop Application:** Python, Tkinter
- **Hardware:** NVIDIA RTX 5060 8GB GPU
- **Evaluation:** Precision, Recall, F1-Score, mAP@0.5, mAP@0.5-0.95

---

## Author

**Samuel Tai Zi Wei**
Faculty of Electronics and Computer Technology and Engineering
Universiti Teknikal Malaysia Melaka (UTeM)
