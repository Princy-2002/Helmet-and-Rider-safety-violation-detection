# Data Management & Annotation Audit

## 1. Dataset Overview
This project uses the **Helmet Detection Dataset (Kaggle)** consisting of approximately 760 traffic images containing two-wheeler riders (motorcycles and scooters) in varied urban and highway conditions.

### Target Classes
| Class ID | Class Label | Description | Enforcement Action |
| :---: | :---: | :---: | :---: |
| **0** | `With Helmet` | Rider or pillion passenger wearing a protective safety helmet | Compliant |
| **1** | `Without Helmet` | Rider or pillion passenger not wearing a helmet | **Violation** |

---

## 2. Week 1 Annotation Audit & Data Cleaning Insights
During the initial Data Analytics & EDA phase in `notebooks/InternshipProjectHelmetDetection_ (1).ipynb`, several critical data quality issues and statistical patterns were discovered:

### 1. Bounding Box Coordinate Corruption Fix
- **Issue:** 17 annotations contained coordinates exceeding the image boundary dimensions (e.g., coordinate values over 100,000 due to legacy formatting bugs in raw XML).
- **Audit Result:** Investigating `BikesHelmets444.xml` and similar files revealed coordinates were erroneously scaled by 1,000.
- **Resolution:** Dividing invalid coordinates by `1000` precisely aligned all 17 boxes inside image dimensions (`xmin >= 0`, `xmax <= width`, `ymin >= 0`, `ymax <= height`), preserving valid training data with 0 data loss.

### 2. Class Balance & Size Distribution
- **Class Imbalance:** `With Helmet` instances outnumber `Without Helmet` instances (approx. 57% vs. 43%).
- **Relative Bounding Box Area:** `With Helmet` annotations average larger pixel areas compared to `Without Helmet`.
- **Rider & Pillion Occlusion:** Multiple riders per bike (pillion passengers) frequently exhibit smaller bounding boxes and partial occlusion behind the primary rider, presenting a key challenge for model recall.

---

## 3. Recommended Directory Structure
```
data/
├── raw/                      # Original raw Kaggle images & Pascal VOC XML files
│   ├── images/               # Raw .png / .jpg images
│   └── annotations/          # Raw .xml annotations
├── processed/                # Normalized YOLOv8 dataset splits
│   ├── images/
│   │   ├── train/            # ~70% training split (~532 images)
│   │   ├── val/              # ~20% validation split (~152 images)
│   │   └── test/             # ~10% held-out test split (~77 images)
│   ├── labels/
│   │   ├── train/            # Normalized YOLO coordinates (class x_c y_c w h)
│   │   ├── val/
│   │   └── test/
│   └── data.yaml             # Dataset configuration file for Ultralytics
└── README.md
```

---

## 4. Dataset Configuration (`data.yaml`)
```yaml
path: /path/to/data/processed
train: images/train
val: images/val
test: images/test

names:
  0: With Helmet
  1: Without Helmet
```

---

## 5. Privacy & Ethics Protocol
In alignment with privacy-safe traffic analytics standards:
- **No License Plate Tracking:** Raw license plates are never cataloged, indexed, or stored in violation logs.
- **No Facial Biometrics:** Cropped facial representations are intentionally excluded from log storage.
- **Aggregation:** Detection logs store only spatial bounding boxes, confidence scores, and aggregate temporal counts.
