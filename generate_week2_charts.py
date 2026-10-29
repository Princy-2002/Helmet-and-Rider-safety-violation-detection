"""
generate_week2_charts.py
------------------------
Utility script to generate Week 2 interactive Plotly analytics charts
and save them to data/logs/ as standalone HTML files.
"""

import os
import sys

# Ensure src/ is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from analytics import ViolationAnalytics

def main():
    log_file = "data/logs/sample_violation_logs.csv"
    if not os.path.exists(log_file):
        print(f"[ERROR] Sample logs not found at {log_file}.")
        return

    print(f"[INFO] Loading violation logs from '{log_file}'...")
    df = pd.read_csv(log_file)
    analytics = ViolationAnalytics(df)

    kpis = analytics.get_summary_kpis()
    print("\n--- Key Performance Indicators ---")
    print(f"Total Traffic Events Logged: {kpis['total_events']}")
    print(f"Total Riders Audited:        {kpis['total_riders']}")
    print(f"Total Violations Detected:   {kpis['total_violations']}")
    print(f"Overall Compliance Rate:     {kpis['overall_compliance_rate_pct']}%")
    print(f"Peak Violation Window:       {kpis['peak_violation_hour']}")
    print(f"Highest Risk Camera:         {kpis['highest_risk_camera']}\n")

    out_dir = "data/logs"
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Generating interactive charts...")
    # 1. Hourly trends chart
    fig_hourly = analytics.plot_hourly_trends()
    hourly_path = os.path.join(out_dir, "hourly_trends.html")
    fig_hourly.write_html(hourly_path, include_plotlyjs='cdn')
    print(f"  -> Saved hourly trends to: {hourly_path}")

    # 2. 2D Severity Heatmap
    fig_heatmap = analytics.plot_severity_heatmap()
    heatmap_path = os.path.join(out_dir, "severity_heatmap.html")
    fig_heatmap.write_html(heatmap_path, include_plotlyjs='cdn')
    print(f"  -> Saved severity heatmap to: {heatmap_path}")

    # 3. Camera perspective comparison
    fig_camera = analytics.plot_camera_comparison()
    camera_path = os.path.join(out_dir, "camera_comparison.html")
    fig_camera.write_html(camera_path, include_plotlyjs='cdn')
    print(f"  -> Saved camera comparison to: {camera_path}")

    print("\n[SUCCESS] All Week 2 analytics charts generated successfully!")

if __name__ == "__main__":
    main()
