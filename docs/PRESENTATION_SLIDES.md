# Final Project Presentation Slide Deck Outline
**AEGIS-AI: Automated Real-Time Two-Wheeler Safety Violation Detection, Temporal Analytics & Compliance Dashboard**  
*Trackgenesis Combined Batch: Data Science & Data Analytics + AI/ML (October 2026)*  
*Project Topic 5 Final Review*

---

## Slide 1: Title Slide
- **Project Title:** AEGIS-AI | Helmet & Rider-Safety Violation Detection with Analytics
- **Subtitle:** An End-to-End Deep Learning & Spatial-Temporal Analytics Platform for Traffic Safety Enforcement
- **Track:** Combined Batch (DS & DA + AI/ML) – October 2026
- **Technology Stack:** Ultralytics YOLOv8, OpenCV, PyTorch, ByteTrack MOT, Streamlit, Plotly, Pandas
- **Presenters:** Project Team 5

---

## Slide 2: Problem Statement & Motivation
- **Context:** Dense two-wheeler traffic across urban India (especially Kerala) results in disproportionate road accident fatalities.
- **The Core Problem:**
  - Manual roadside policing is unsafe, inconsistent, and causes traffic congestion.
  - Basic image classification fails on multi-rider bikes (driver vs pillion).
  - Pillion riders without helmets are frequently missed.
- **Project Mission:** Build an automated computer vision pipeline for real-time unhelmeted rider detection paired with an operational temporal analytics dashboard—without infringing on citizen privacy.

---

## Slide 3: System Architecture & Workflow
- **Pipeline:**
  1. **Video Ingestion:** Live CCTV / Webcam / Video feed at 30+ FPS.
  2. **YOLOv8 Inference Engine:** Detects `With Helmet` (green) and `Without Helmet` (red).
  3. **ByteTrack MOT Tracker:** Assigns persistent track IDs to eliminate multi-frame duplicate counting.
  4. **Violation Logger:** Structured event ledger storing timestamps, sector IDs, counts, confidences.
  5. **Analytics & GIS Hotspots:** Hourly trends, camera perspective comparisons, and 2D severity heatmaps.
  6. **Privacy Guard:** Zero-PII aggregated exports (no facial biometrics, no license plates).

---

## Slide 4: Milestone Progress (Weeks 1 to 4)
| Milestone | Focus Domain | Key Accomplishments |
| :--- | :--- | :--- |
| **Week 1** | **Data Audit & Baseline Model** | Pascal VOC annotation audit, repaired 17 corrupted `/1000` box coordinates, trained baseline YOLOv8n (50 epochs, mAP@50 = 87.59%). |
| **Week 2** | **Violation Logging & Analytics** | Created structured logging schema (`src/violation_logger.py`), temporal analytics engine, peak hour identification, and 2D severity heatmaps. |
| **Week 3** | **Interactive Streamlit MVP** | Built multi-tab dashboard (`app.py`), dynamic confidence & IoU threshold sliders, live webcam inference, and evidence vault. |
| **Week 4** | **Full Audit & Final Release** | Automated Zero-PII privacy compliance audit suite, multi-model benchmark report, patrol shift analytics, and final demo packaging. |

---

## Slide 5: AI/ML Engineering & Model Benchmarks
- **Dataset Partition:** 70% Train (532 img) | 20% Val (152 img) | 10% Test (76 img).
- **Test Set Evaluation (156 Ground-Truth Instances):**
  - **Overall mAP@50:** `89.20%` (Calibrated Production Model)
  - **Overall Precision:** `84.50%`
  - **Overall Recall:** `79.80%`
  - **Inference Speed:** `6.8 ms` (RTX 4060 GPU) / `12.4 ms` (Tesla T4) / `48.2 ms` (CPU)
- **Per-Class Breakdown:**
  - *With Helmet:* Precision = 85.9%, Recall = 88.8%, mAP@50 = 91.6%
  - *Without Helmet (Violation):* Precision = 82.3%, Recall = 69.4% (Baseline) ➔ **79.8%** (Calibrated)

---

## Slide 6: Diagnostic Case Study: Closing the Pillion Recall Gap
- **The Challenge:** Baseline violation recall lagged at 69.4% because rear pillion heads are smaller (<2% pixel area) and occluded by drivers' shoulders.
- **Experimentation Findings:**
  - *Heavy Synthetic Augmentation (Mosaic, MixUp):* Degraded boundary precision on distant pillion heads (mAP dropped to 83.67%).
  - *Confidence Threshold Calibration (0.23 – 0.25):* Provided the optimal operating point, increasing violation recall to ~80% while retaining high precision.
- **Interactive UI Solution:** Real-time confidence sliders in the Streamlit HUD allow operators to tune detection sensitivity dynamically.

---

## Slide 7: Data Analytics & Temporal Hotspot Intelligence
- **Operational Shift Breakdown:**
  - *Morning Rush (08:00 – 11:00):* High volume, moderate compliance (78.4%).
  - *Evening Peak (16:00 – 20:00):* **Highest Risk Window (Peak violations at 17:00–19:00, 68.2% compliance).**
  - *Night Patrol (20:00 – 08:00):* Lower rider volume, severe infractions.
- **Perspective Comparisons:** High-angle flyover CCTV perspectives achieved 8.4% higher rider localization accuracy compared to ground-level street cameras.
- **Smart City GIS Radar:** Interactive spatial tracking of active surveillance nodes across Kochi, Trivandrum, Kozhikode, and Thrissur.

---

## Slide 8: Privacy-by-Design Governance (Zero-PII)
- **Ethical Compliance:**
  - ❌ **No ALPR / Number Plates:** Vehicle registration details are neither captured nor stored.
  - ❌ **No Facial Biometrics:** Driver and pillion facial images are never extracted or fingerprinted.
  - ✅ **Transient In-Memory Processing:** Video frames exist only in RAM during YOLO inference and are purged immediately.
  - ✅ **Aggregated Export Schema:** Public reports are strictly grouped by camera node and hour of day.
- **Verification:** 100% PASS rate across all automated checks in `scripts/run_privacy_audit.py` (Cert: `AEGIS-GOV-2026-W4-CERT`).

---

## Slide 9: Live Product Demonstration
*(Transition to live Streamlit App at `http://localhost:8501`)*
1. **Live Detection Tab:** Real-time video processing with dynamic green/red HUD bounding boxes.
2. **Interactive Controls:** Demonstrating confidence slider sensitivity tuning live.
3. **Evidence Vault:** Automated incident logging and Section 129 Kerala MVD E-Challan generation.
4. **Temporal Dashboard:** Interactive Plotly curves, 2D heatmaps, and GIS node telemetry.
5. **Audited Event Ledger:** Real-time ledger filtering and Zero-PII compliance export.

---

## Slide 10: Conclusion & Impact
- **Accomplishments:**
  - Delivered a robust, high-FPS object detection system fine-tuned on realistic Indian road traffic.
  - Solved pillion rider recall through dynamic confidence calibration.
  - Created an operational analytics suite that informs targeted police patrol scheduling.
  - Successfully audited and certified Zero-PII privacy compliance.
- **Ready for Deployment:** Lightweight YOLOv8 architecture enables edge deployment on roadside CCTV hardware or centralized smart city control centers.

---

## Slide 11: Q&A
- *Thank you!*
- Open for questions and feedback from mentors and evaluators.
