# Final Demo & Video Walkthrough Guide
**AEGIS-AI | Helmet & Rider-Safety Violation Detection with Analytics**  
*Trackgenesis Combined Batch (DS & DA + AI/ML) – October 2026*  
*Milestone: Week 4 Deliverable*

---

## 🎬 Video Demo Structure (5–7 Minute Recording)

This choreography is optimized for screen recording (OBS, Loom, or Zoom) and live mentor walkthroughs.

```
⏱️ TIMELINE BREAKDOWN:
00:00 - 01:00  👉 Scene 1: Executive Overview & Problem Context
01:00 - 02:15  👉 Scene 2: Live Inference & Dynamic Sensitivity Tuning
02:15 - 03:30  👉 Scene 3: Evidence Vault & Kerala E-Challan Issuance
03:30 - 04:45  👉 Scene 4: Temporal Analytics, Hotspots & Patrol Shifts
04:45 - 05:45  👉 Scene 5: Zero-PII Privacy Audit & Ledger Export
05:45 - 06:45  👉 Scene 6: AI/ML Benchmarks & Pillion Recall Resolution
06:45 - 07:00  👉 Scene 7: Conclusion & Production Readiness
```

---

## 📽️ Scene-by-Scene Script & Action Guide

### Scene 1: Introduction & High-Level Problem (0:00 – 1:00)
- **Visual:** Browser on `http://localhost:8501`, top hero banner visible.
- **Narrator Speech:**
  > *"Welcome. Today we present the final production demo for **AEGIS-AI: Helmet & Rider-Safety Violation Detection with Analytics**.*  
  > *Two-wheeler traffic is exceptionally dense across Indian cities, particularly in Kerala, where unhelmeted motorcycle collisions account for the largest share of preventable fatalities.*  
  > *Rather than relying on human traffic officers or static image classification, AEGIS-AI combines real-time YOLOv8 object detection, multi-object tracking, and automated temporal analytics to empower traffic enforcement agencies while guaranteeing citizen privacy."*

---

### Scene 2: Live Inference & Dynamic Slider Tuning (1:00 – 2:15)
- **Visual:** Tab 1 (`🎥 Live Video & AI Detection`).
- **Actions:**
  1. Point cursor to the Sidebar: Show `models/best.pt` selected as the active neural network.
  2. Select **"Upload Video"** or click **"Simulate Sample Frame"** / **"Live Webcam"**.
  3. Show the dynamic bounding boxes:
     - 🟢 **Emerald Green:** `With Helmet` (Compliant rider).
     - 🔴 **Crimson Red:** `Without Helmet` (Safety violation).
  4. Point out the **Live Feed Telemetry** panel updating in real time: Active Riders, Violations, Compliant Riders, and Inference FPS.
  5. **Demonstrate Slider Tuning:** Move the **Confidence Threshold** slider from `0.35` down to `0.23`.
  - **Narrator Speech:**
    > *"Notice our dynamic sensitivity sliders. Unlike rigid systems where thresholds are hardcoded, AEGIS-AI allows operators to adjust confidence live. Lowering confidence to 0.23 dramatically boosts recall for partially occluded pillion passengers without triggering false positives."*

---

### Scene 3: Evidence Vault & Kerala MVD E-Challan (2:15 – 3:30)
- **Visual:** Switch to Tab 2 (`📸 Evidence Vault & E-Challan`).
- **Actions:**
  1. Scroll through the cropped violation cards in the vault.
  2. Select an incident from the dropdown.
  3. Click **"Download Official Challan Receipt (TXT)"**.
- **Narrator Speech:**
  > *"Under Tab 2, our Evidence Vault records verified infraction snapshots with timestamp, camera perspective, and confidence metrics. Field officers can instantly generate standardized legal traffic notices under Section 129 of the Motor Vehicles Act."*

---

### Scene 4: Temporal Analytics, Kerala GIS & Patrol Shifts (3:30 – 4:45)
- **Visual:** Switch to Tab 3 (`📊 Unified Analytics & GIS Hotspots`).
- **Actions:**
  1. Show Executive KPI cards (Total Events, Total Riders, Overall Compliance Rate, Peak Window).
  2. Point out **Chart 1:** Hourly Violation Curve vs. Compliance Rate.
  3. Point out **Chart 2:** Multi-Camera Angle Breakdown (flyover vs street crossing).
  4. Point out **Chart 3:** 2D Temporal Heatmap (Hour of Day vs Day of Week).
  5. Point out **Chart 4:** Interactive Kerala Smart City GIS Radar (Kochi, Trivandrum, Kozhikode).
  6. Point out **Chart 5:** Police Deployment Shift Breakdown & Multi-Day Compliance Trajectory.
- **Narrator Speech:**
  > *"In Tab 3, our Data Analytics engine turns raw detections into operational intelligence. We highlight peak violation windows (such as evening commute hours between 17:00 and 19:00), allowing authorities to deploy targeted patrols rather than conducting untargeted checks."*

---

### Scene 5: Zero-PII Privacy Audit & Ledger Export (4:45 – 5:45)
- **Visual:** Switch to Tab 4 (`📋 Violation Event Logs & Privacy Export`).
- **Actions:**
  1. Show the interactive filtered ledger.
  2. Filter by camera sector.
  3. Click **"Export Anonymized Compliance Report (CSV)"**.
  4. Briefly show the passing test run of `python scripts/run_privacy_audit.py` in the terminal.
- **Narrator Speech:**
  > *"Privacy is baked into our system by design. AEGIS-AI does not collect facial crops or vehicle license plates. All exported compliance summaries are strictly aggregated by hour and camera sector, complying fully with the Digital Personal Data Protection Act of India."*

---

### Scene 6: AI/ML Benchmarks & Resolving Pillion Recall (5:45 – 6:45)
- **Visual:** Switch to Tab 5 (`🧠 AI/ML Model Diagnostics & Benchmarks`).
- **Actions:**
  1. Highlight test-split metrics: **87.59% mAP@50**, **84.11% Precision**, **79.10% Recall**.
  2. Review the per-class comparison and multi-model augmentation bars.
- **Narrator Speech:**
  > *"In Tab 5, we present empirical evaluation metrics across 156 test instances. We identified the Pillion Rider Recall Gap and engineered a threshold-calibrated solution that boosted violation recall to ~80% while retaining over 84% precision."*

---

### Scene 7: Conclusion & Sign-Off (6:45 – 7:00)
- **Visual:** Return to main dashboard view.
- **Narrator Speech:**
  > *"With full real-time inference, privacy compliance, temporal hotspot intelligence, and verified model benchmarks, AEGIS-AI fulfills all requirements for the Trackgenesis combined curriculum. Thank you."*
