"""
violation_logger.py
-------------------
Structured event logging and privacy-safe data export module for
Helmet & Rider-Safety Violation Detection.

Features:
- Structured violation schema capturing timestamps, counts, confidence, and metadata.
- Export to CSV, JSON, and Pandas DataFrames.
- Privacy-safe aggregated report generator ensuring no facial/plate identifiers are stored.
- Synthetic traffic violation log generator for development and dashboard testing.
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random
import pandas as pd
import numpy as np


class ViolationLogger:
    """
    Manages recording, indexing, and exporting helmet safety violation events.
    """

    DEFAULT_LOG_COLUMNS = [
        "event_id",
        "timestamp",
        "date",
        "time",
        "hour",
        "camera_id",
        "frame_id",
        "total_riders",
        "with_helmet_count",
        "without_helmet_count",
        "compliance_rate",
        "avg_confidence",
        "max_confidence",
        "severity_level",
        "has_violation",
        "raw_detections_count"
    ]

    def __init__(self, log_dir: str = "data/logs", auto_save: bool = False):
        """
        Initialize logger with storage directory and session buffer.
        """
        self.log_dir = log_dir
        self.auto_save = auto_save
        os.makedirs(self.log_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def _determine_severity(self, violation_count: int) -> str:
        """Categorize violation severity based on count of unhelmeted riders."""
        if violation_count == 0:
            return "NONE"
        elif violation_count == 1:
            return "LOW"
        elif violation_count == 2:
            return "MEDIUM"
        else:
            return "HIGH"

    def log_detection(
        self,
        frame_id: int,
        detections: List[Dict[str, Any]],
        camera_id: str = "CAM-01",
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse detections from a frame and record a violation event.

        Args:
            frame_id: Sequential video frame index or timestamp sequence.
            detections: List of detection dictionaries:
                        [{'class_id': 0 or 1, 'class_name': 'With Helmet'/'Without Helmet',
                          'confidence': 0.85, 'box': [x1, y1, x2, y2]}]
            camera_id: Identifier for camera source / perspective.
            timestamp: Event datetime (defaults to now).

        Returns:
            The created log record dictionary.
        """
        if timestamp is None:
            timestamp = datetime.now()

        with_helmet = sum(1 for d in detections if d.get("class_id") == 0 or d.get("class_name") == "With Helmet")
        without_helmet = sum(1 for d in detections if d.get("class_id") == 1 or d.get("class_name") == "Without Helmet")
        total_riders = with_helmet + without_helmet

        confidences = [d.get("confidence", 0.0) for d in detections]
        avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
        max_conf = round(float(np.max(confidences)), 4) if confidences else 0.0

        compliance_rate = round(with_helmet / total_riders, 4) if total_riders > 0 else 1.0
        severity = self._determine_severity(without_helmet)

        record = {
            "event_id": f"EVT-{timestamp.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "hour": timestamp.hour,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "total_riders": total_riders,
            "with_helmet_count": with_helmet,
            "without_helmet_count": without_helmet,
            "compliance_rate": compliance_rate,
            "avg_confidence": avg_conf,
            "max_confidence": max_conf,
            "severity_level": severity,
            "has_violation": without_helmet > 0,
            "raw_detections_count": len(detections)
        }

        self.records.append(record)

        if self.auto_save:
            daily_file = os.path.join(self.log_dir, f"violations_{record['date']}.csv")
            self.to_csv(daily_file)

        return record

    def get_dataframe(self) -> pd.DataFrame:
        """Convert current records in memory to a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=self.DEFAULT_LOG_COLUMNS)
        df = pd.DataFrame(self.records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    to_dataframe = get_dataframe

    def to_csv(self, filepath: Optional[str] = None) -> str:
        """Export current log events to a CSV file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"violation_logs_{datetime.now().strftime('%Y%m%d')}.csv")
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
        return filepath

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export current log events to a JSON file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"violation_logs_{datetime.now().strftime('%Y%m%d')}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, default=str)
        return filepath

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load violation events from an existing CSV log."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Log file not found: {filepath}")
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        self.records = df.to_dict(orient="records")
        return df

    def generate_privacy_safe_export(
        self,
        filepath: Optional[str] = None,
        input_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate a strictly privacy-safe aggregated report.
        Excludes individual frame identifiers, specific timestamps down to the second,
        and ensures zero driver/vehicle PII is included.
        Aggregates by Date, Hour, and Camera Location.
        """
        df = input_df if input_df is not None else self.get_dataframe()
        if df.empty:
            summary_cols = ["date", "hour", "camera_id", "total_riders", "violations", "compliant", "compliance_rate_pct"]
            return pd.DataFrame(columns=summary_cols)

        agg_df = df.groupby(["date", "hour", "camera_id"]).agg(
            total_riders=("total_riders", "sum"),
            violations=("without_helmet_count", "sum"),
            compliant=("with_helmet_count", "sum"),
            avg_detection_confidence=("avg_confidence", "mean")
        ).reset_index()

        agg_df["compliance_rate_pct"] = (
            (agg_df["compliant"] / agg_df["total_riders"].replace(0, 1)) * 100
        ).round(2)
        agg_df["avg_detection_confidence"] = agg_df["avg_detection_confidence"].round(4)

        if filepath is not None:
            agg_df.to_csv(filepath, index=False)

        return agg_df

    def generate_synthetic_logs(
        self,
        num_events: int = 250,
        days: int = 7,
        camera_ids: Optional[List[str]] = None,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic traffic violation logs matching real-world Kerala road patterns:
        - Morning peak (8:00 - 10:30 AM)
        - Evening peak (17:00 - 20:00 PM)
        - Lower compliance observed in evening commutes
        - Multiple camera perspectives (Junction, Flyover, Suburb)
        """
        if camera_ids is None:
            camera_ids = ["CAM-Kochi-MG-Road", "CAM-Trivandrum-East-Fort", "CAM-Calicut-Beach-Junction"]

        base_time = datetime.now() - timedelta(days=days)

        for i in range(num_events):
            # Pick a random day and hour with realistic rush-hour weighting
            day_offset = random.randint(0, days - 1)
            # Biased hour selection towards peak rush hours
            hour_weights = [
                1, 1, 1, 1, 2, 4,       # 00:00 - 05:00 (quiet)
                6, 9, 14, 12, 8, 7,     # 06:00 - 11:00 (morning rush)
                6, 6, 6, 7, 10, 15,     # 12:00 - 17:00 (afternoon build up)
                14, 12, 9, 6, 4, 2      # 18:00 - 23:00 (evening peak)
            ]
            hour = random.choices(range(24), weights=hour_weights)[0]
            minute = random.randint(0, 59)
            second = random.randint(0, 59)

            event_time = base_time + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
            camera = random.choice(camera_ids)

            # Determine number of riders in frame (1 to 4)
            num_riders = random.choices([1, 2, 3, 4], weights=[0.60, 0.30, 0.08, 0.02])[0]
            
            # Evening hours slightly higher violation probability
            violation_prob = 0.35 if (17 <= hour <= 21) else 0.22

            detections = []
            for _ in range(num_riders):
                is_violation = random.random() < violation_prob
                conf = round(random.uniform(0.72, 0.96), 2)
                if is_violation:
                    detections.append({
                        "class_id": 1,
                        "class_name": "Without Helmet",
                        "confidence": conf
                    })
                else:
                    detections.append({
                        "class_id": 0,
                        "class_name": "With Helmet",
                        "confidence": conf
                    })

            self.log_detection(
                frame_id=i + 1,
                detections=detections,
                camera_id=camera,
                timestamp=event_time
            )

        df = self.get_dataframe().sort_values("timestamp").reset_index(drop=True)

        if output_csv:
            df.to_csv(output_csv, index=False)

        return df


if __name__ == "__main__":
    print("Testing ViolationLogger...")
    test_logger = ViolationLogger(log_dir="data/logs")
    synth_df = test_logger.generate_synthetic_logs(
        num_events=150,
        days=5,
        output_csv="data/logs/sample_violation_logs.csv"
    )
    print(f"Generated {len(synth_df)} synthetic violation records.")
    print("Columns:", list(synth_df.columns))
    print("\nSample records:")
    print(synth_df[["timestamp", "camera_id", "total_riders", "without_helmet_count", "compliance_rate", "severity_level"]].head())

    privacy_safe_df = test_logger.generate_privacy_safe_export(filepath="data/logs/privacy_safe_summary.csv")
    print("\nPrivacy-Safe Aggregated Sample:")
    print(privacy_safe_df.head())
