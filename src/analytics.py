"""
analytics.py
------------
Data Science & Analytics module for Helmet & Rider-Safety Violation Detection.

Features:
- Temporal violation analytics (hourly patterns, peak violation hours, daily compliance trends).
- Severity heatmaps (Day of week vs. Hour of day).
- Multi-camera perspective comparisons (road angles, junctions vs. flyovers).
- Interactive Plotly visualizers for dashboard integration.
- Bounding-box annotation auditing (class balance, box size, aspect ratios).
- Executive KPI summary generation and privacy-safe reporting.
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ViolationAnalytics:
    """
    Core analytics engine for processing helmet violation event logs
    and annotation audit datasets.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize with a violation log DataFrame or leave None to supply dynamically.
        """
        self.df = data.copy() if data is not None else None
        if self.df is not None and not self.df.empty:
            self._prepare_data()

    def _prepare_data(self):
        """Ensure date, time, hour, and day_of_week fields are correctly formatted."""
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.df["date"] = self.df["timestamp"].dt.date
            self.df["hour"] = self.df["timestamp"].dt.hour
            self.df["day_name"] = self.df["timestamp"].dt.day_name()
            self.df["day_of_week"] = self.df["timestamp"].dt.dayofweek

    def load_data(self, df: pd.DataFrame):
        """Load and prepare a new DataFrame."""
        self.df = df.copy()
        self._prepare_data()

    # =========================================================================
    # 1. SUMMARY KPIS & TEMPORAL METRICS
    # =========================================================================

    def get_summary_kpis(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Compute high-level executive KPIs for dashboards and reports.
        """
        data = df if df is not None else self.df
        if data is None or data.empty:
            return {
                "total_events": 0,
                "total_riders": 0,
                "total_violations": 0,
                "overall_compliance_rate_pct": 100.0,
                "peak_violation_hour": "N/A",
                "highest_risk_camera": "N/A",
                "avg_detection_confidence": 0.0,
                "high_severity_events": 0
            }

        total_riders = int(data["total_riders"].sum())
        total_violations = int(data["without_helmet_count"].sum())
        total_compliant = int(data["with_helmet_count"].sum())
        compliance_pct = round((total_compliant / total_riders * 100), 2) if total_riders > 0 else 100.0

        hourly_viols = data.groupby("hour")["without_helmet_count"].sum()
        peak_hour_val = hourly_viols.idxmax() if not hourly_viols.empty else 0
        peak_hour_str = f"{peak_hour_val:02d}:00 - {(peak_hour_val+1)%24:02d}:00"

        camera_viols = data.groupby("camera_id")["without_helmet_count"].sum()
        worst_cam = camera_viols.idxmax() if not camera_viols.empty else "N/A"

        avg_conf = round(float(data["avg_confidence"].mean()), 3) if "avg_confidence" in data.columns else 0.0
        high_sev = int((data["severity_level"] == "HIGH").sum()) if "severity_level" in data.columns else 0

        return {
            "total_events": len(data),
            "total_riders": total_riders,
            "total_violations": total_violations,
            "overall_compliance_rate_pct": compliance_pct,
            "peak_violation_hour": peak_hour_str,
            "highest_risk_camera": worst_cam,
            "avg_detection_confidence": avg_conf,
            "high_severity_events": high_sev
        }

    def get_hourly_trends(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Aggregate total riders, violations, and compliance rates across 24 hours.
        """
        data = df if df is not None else self.df
        if data is None or data.empty:
            return pd.DataFrame()

        hourly = data.groupby("hour").agg(
            total_riders=("total_riders", "sum"),
            violations=("without_helmet_count", "sum"),
            compliant=("with_helmet_count", "sum"),
            avg_confidence=("avg_confidence", "mean")
        ).reindex(range(24), fill_value=0).reset_index()

        hourly["compliance_rate_pct"] = (
            (hourly["compliant"] / hourly["total_riders"].replace(0, 1)) * 100
        ).round(2)
        hourly["avg_confidence"] = hourly["avg_confidence"].round(3)
        return hourly

    def get_daily_trends(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Aggregate metrics day-by-day to measure long-term compliance trajectory.
        """
        data = df if df is not None else self.df
        if data is None or data.empty:
            return pd.DataFrame()

        daily = data.groupby("date").agg(
            total_events=("event_id", "count"),
            total_riders=("total_riders", "sum"),
            violations=("without_helmet_count", "sum"),
            compliant=("with_helmet_count", "sum")
        ).reset_index()

        daily["compliance_rate_pct"] = (
            (daily["compliant"] / daily["total_riders"].replace(0, 1)) * 100
        ).round(2)
        return daily

    def get_camera_perspective_analysis(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Evaluate variation in detection performance and violation frequency
        across multiple camera perspectives and road locations.
        """
        data = df if df is not None else self.df
        if data is None or data.empty:
            return pd.DataFrame()

        cam_stats = data.groupby("camera_id").agg(
            events=("event_id", "count"),
            total_riders=("total_riders", "sum"),
            violations=("without_helmet_count", "sum"),
            compliant=("with_helmet_count", "sum"),
            avg_confidence=("avg_confidence", "mean")
        ).reset_index()

        cam_stats["compliance_rate_pct"] = (
            (cam_stats["compliant"] / cam_stats["total_riders"].replace(0, 1)) * 100
        ).round(2)
        cam_stats["violation_rate_pct"] = (100 - cam_stats["compliance_rate_pct"]).round(2)
        cam_stats["avg_confidence"] = cam_stats["avg_confidence"].round(3)

        return cam_stats.sort_values(by="violations", ascending=False)

    # =========================================================================
    # 2. PLOTLY INTERACTIVE VISUALIZATIONS
    # =========================================================================

    def plot_hourly_trends(self, df: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Generate a dual-axis interactive chart:
        - Bar chart: Violations vs Compliant riders
        - Line chart: Compliance Rate (%) across hours
        """
        hourly = self.get_hourly_trends(df)
        if hourly.empty:
            return go.Figure()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=hourly["hour"],
                y=hourly["violations"],
                name="Violations (No Helmet)",
                marker_color="#EF4444",
                opacity=0.85
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Bar(
                x=hourly["hour"],
                y=hourly["compliant"],
                name="Compliant (With Helmet)",
                marker_color="#10B981",
                opacity=0.75
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=hourly["hour"],
                y=hourly["compliance_rate_pct"],
                name="Compliance Rate (%)",
                mode="lines+markers",
                line=dict(color="#3B82F6", width=3),
                marker=dict(size=6)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=dict(text="Hourly Traffic Violations vs. Compliance Rate", x=0.01, y=0.98),
            barmode="stack",
            xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode="linear", tick0=0, dtick=2),
            yaxis=dict(title="Number of Riders"),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
            margin=dict(t=65),
            template="plotly_white",
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Compliance Rate (%)", secondary_y=True, range=[0, 105])

        return fig

    def plot_severity_heatmap(self, df: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Generate a 2D Heatmap (Day of Week vs. Hour of Day)
        identifying violation peak clusters and high-risk temporal windows.
        """
        data = df if df is not None else self.df
        if data is None or data.empty:
            return go.Figure()

        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        matrix = data.pivot_table(
            index="day_name",
            columns="hour",
            values="without_helmet_count",
            aggfunc="sum",
            fill_value=0
        ).reindex(days_order, fill_value=0).reindex(columns=range(24), fill_value=0)

        fig = px.imshow(
            matrix,
            labels=dict(x="Hour of Day", y="Day of Week", color="Violations"),
            x=[f"{h:02d}:00" for h in range(24)],
            y=days_order,
            color_continuous_scale="Reds",
            title="Temporal Violation Hotspot Heatmap (Day of Week vs. Hour of Day)"
        )

        fig.update_layout(
            template="plotly_white",
            xaxis=dict(tickangle=-45)
        )
        return fig

    def plot_camera_comparison(self, df: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Bar chart comparing compliance rate and violation count across cameras/perspectives.
        """
        cam_stats = self.get_camera_perspective_analysis(df)
        if cam_stats.empty:
            return go.Figure()

        fig = px.bar(
            cam_stats,
            x="camera_id",
            y=["violations", "compliant"],
            barmode="group",
            labels={"value": "Riders Count", "camera_id": "Camera Location / Angle", "variable": "Status"},
            title="Safety Compliance Across Camera Locations / Road Angles",
            color_discrete_map={"violations": "#EF4444", "compliant": "#10B981"}
        )
        fig.update_layout(template="plotly_white")
        return fig

    def plot_confidence_distribution(self, df: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Histogram of detection confidence scores to inspect model stability.
        """
        data = df if df is not None else self.df
        if data is None or data.empty or "avg_confidence" not in data.columns:
            return go.Figure()

        fig = px.histogram(
            data,
            x="avg_confidence",
            color="has_violation",
            nbins=20,
            barmode="overlay",
            title="Detection Confidence Distribution (Violations vs. Normal Traffic)",
            labels={"avg_confidence": "Detection Confidence", "has_violation": "Contains Violation"},
            color_discrete_map={True: "#EF4444", False: "#10B981"},
            opacity=0.7
        )
        fig.update_layout(template="plotly_white")
        return fig

    def get_rush_hour_breakdown(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Segment 24 hours into 4 actionable operational shifts:
        - Morning Rush (08:00 - 11:00)
        - Midday (11:00 - 16:00)
        - Evening Peak (16:00 - 20:00)
        - Night Patrol (20:00 - 08:00)
        """
        data = df if df is not None else self.df
        if data is None or data.empty:
            return pd.DataFrame()

        def assign_shift(hour):
            if 8 <= hour < 11:
                return "Morning Rush (08:00 - 11:00)"
            elif 11 <= hour < 16:
                return "Midday (11:00 - 16:00)"
            elif 16 <= hour < 20:
                return "Evening Peak (16:00 - 20:00)"
            else:
                return "Night Patrol (20:00 - 08:00)"

        df_copy = data.copy()
        df_copy["shift"] = df_copy["hour"].apply(assign_shift)

        shifts = df_copy.groupby("shift").agg(
            total_riders=("total_riders", "sum"),
            violations=("without_helmet_count", "sum"),
            compliant=("with_helmet_count", "sum")
        ).reset_index()

        shifts["compliance_rate_pct"] = (
            (shifts["compliant"] / shifts["total_riders"].replace(0, 1)) * 100
        ).round(2)
        return shifts

    def plot_daily_compliance_trajectory(self, df: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Plot rolling daily compliance trajectory over multi-day period.
        """
        daily = self.get_daily_trends(df)
        if daily.empty:
            return go.Figure()

        fig = px.line(
            daily,
            x="date",
            y="compliance_rate_pct",
            markers=True,
            title="Daily Traffic Compliance Trajectory (% Helmet Wearing)",
            labels={"date": "Date", "compliance_rate_pct": "Compliance Rate (%)"}
        )
        fig.update_traces(line_color="#10B981", line_width=3, marker=dict(size=8, color="#38BDF8"))
        fig.update_layout(yaxis=dict(range=[0, 105]))
        return fig

    # =========================================================================
    # 3. BOUNDING BOX ANNOTATION AUDIT TOOL
    # =========================================================================

    @staticmethod
    def audit_bounding_boxes(df_annotations: pd.DataFrame) -> Dict[str, Any]:
        """
        Audit bounding box properties from EDA / Pascal VOC / YOLO tables:
        - Class balance
        - Bounding box relative area distribution
        - Aspect ratios (width / height)
        - Identification of extreme coordinates
        """
        report = {}
        report["total_boxes"] = len(df_annotations)
        report["class_distribution"] = df_annotations["class"].value_counts().to_dict()

        if "box_area" in df_annotations.columns:
            report["box_area_stats"] = df_annotations.groupby("class")["box_area"].describe().to_dict()

        if "aspect_ratio" in df_annotations.columns:
            report["aspect_ratio_stats"] = df_annotations.groupby("class")["aspect_ratio"].describe().to_dict()

        return report

    @staticmethod
    def plot_annotation_class_balance(class_counts: Dict[str, int]) -> go.Figure:
        """Plotly Donut chart for class distribution in dataset."""
        labels = list(class_counts.keys())
        values = list(class_counts.values())

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45, marker_colors=["#10B981", "#EF4444"])])
        fig.update_layout(
            title="Dataset Annotation Audit: Class Balance (With Helmet vs. Without Helmet)",
            template="plotly_white"
        )
        return fig


if __name__ == "__main__":
    from violation_logger import ViolationLogger

    print("Testing ViolationAnalytics...")
    logger = ViolationLogger()
    sample_df = logger.generate_synthetic_logs(num_events=200, days=7)

    analytics = ViolationAnalytics(sample_df)
    kpis = analytics.get_summary_kpis()
    print("\nExecutive KPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    hourly = analytics.get_hourly_trends()
    print("\nHourly Trends (Top 5 peak hours):")
    print(hourly.sort_values(by="violations", ascending=False).head(5))

    cam_analysis = analytics.get_camera_perspective_analysis()
    print("\nCamera Perspective Analysis:")
    print(cam_analysis)
