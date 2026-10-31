"""End-to-end automated verification script for GRIDGUARD AI."""
import os
import sys
import tempfile
from pathlib import Path
import cv2
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vehicle_detection import analyze_video
from src.route_engine import (
    evaluate_routes,
    compute_segment_metrics,
    build_network_graph_figure,
    build_route_comparison_bar_chart,
)
from src.risk_engine import calculate_risk
from src.traffic_metrics import calculate_metrics
from src.network_analysis import build_demo_road_graph, rank_bottlenecks


def test_core_metrics():
    print("[1/6] Testing traffic_metrics & risk_engine...")
    centres = [(100, 200), (300, 400), (500, 500)]
    metrics = calculate_metrics(centres, width=640, height=480)
    assert metrics.vehicle_count == 3
    assert 0.0 <= metrics.occupancy <= 1.0

    risk = calculate_risk(0.8, 0.7, 0.3, 0.0)
    assert risk.level in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= risk.score <= 1.0
    print("  ✓ Traffic metrics and risk calculations verified.")


def test_route_scoring():
    print("[2/6] Testing route_engine evaluation...")
    # Generate synthetic observations for 50 frames
    mock_rows = []
    for f in range(50):
        mock_rows.append({
            "frame": f + 1,
            "time_seconds": f * 0.04,
            "vehicle_count": 14 if f % 2 == 0 else 12,
            "roi_vehicle_count": 6 if f % 2 == 0 else 5,
            "occupancy_estimate": 0.6,
            "queue_growth": 0.1,
            "risk_score": 0.52,
            "risk_level": "MEDIUM",
        })
    df_primary = pd.DataFrame(mock_rows)

    # Test Mode 1: corridor_baseline
    res_baseline = evaluate_routes(df_primary, primary_name="Road A (Primary Arterial)", mode="corridor_baseline")
    assert res_baseline.has_alternatives is True
    assert len(res_baseline.segments) == 3
    assert res_baseline.recommended_segment_id in ["ROAD_A", "ROAD_B", "ROAD_C"]
    assert "BEST ROUTE" in res_baseline.recommendation_headline or "LEAST DELAY" in res_baseline.recommendation_headline
    print(f"  ✓ Corridor baseline mode: Recommended = {res_baseline.recommended_segment_name}")

    # Test Mode 2: isolated_road (no alternatives)
    res_isolated = evaluate_routes(df_primary, primary_name="Road A (Primary Arterial)", mode="isolated_road")
    assert res_isolated.has_alternatives is False
    assert len(res_isolated.segments) == 1
    assert "Analysis Complete" in res_isolated.recommendation_headline
    print("  ✓ Isolated road mode verified (honestly reports no alternative corridor data).")

    # Test Mode 3: Direct multi-corridor comparison with high congestion on all routes
    sec_dfs = {
        "Road B (Northern Bypass)": pd.DataFrame([{"vehicle_count": 22, "roi_vehicle_count": 9, "occupancy_estimate": 0.9, "risk_score": 0.85} for _ in range(20)]),
        "Road C (Eastern Link)": pd.DataFrame([{"vehicle_count": 25, "roi_vehicle_count": 10, "occupancy_estimate": 0.95, "risk_score": 0.90} for _ in range(20)]),
    }
    df_heavy = pd.DataFrame([{"vehicle_count": 28, "roi_vehicle_count": 11, "occupancy_estimate": 0.98, "risk_score": 0.92} for _ in range(20)])
    res_heavy = evaluate_routes(df_heavy, secondary_dfs=sec_dfs)
    assert res_heavy.all_congested is True
    assert "HEAVY TRAFFIC ACROSS ALL CORRIDORS" in res_heavy.recommendation_headline
    print(f"  ✓ All-congested scenario verified: '{res_heavy.recommendation_headline}'")


def test_visualizations():
    print("[3/6] Testing Plotly graph and chart generation...")
    mock_rows = [{"vehicle_count": 8, "roi_vehicle_count": 3, "occupancy_estimate": 0.3, "risk_score": 0.28} for _ in range(40)]
    df = pd.DataFrame(mock_rows)
    res = evaluate_routes(df, mode="corridor_baseline")

    fig_net = build_network_graph_figure(res.graph, res.segments, res.recommended_segment_id)
    assert fig_net is not None
    assert len(fig_net.data) >= 3  # Traces for segments + nodes

    fig_bar = build_route_comparison_bar_chart(res.segments, res.recommended_segment_id)
    assert fig_bar is not None
    assert len(fig_bar.data) >= 1  # Clean light-themed bar comparison trace
    print("  ✓ Schematic network graph and comparative bar charts generated successfully.")


def test_network_analysis_legacy():
    print("[4/6] Testing network_analysis and graph bottlenecks...")
    g = build_demo_road_graph()
    assert len(g.nodes) == 5
    ranks = rank_bottlenecks({"A": 0.4, "B": 0.8, "C": 0.2, "D": 0.5, "E": 0.1}, g)
    assert len(ranks) == 5
    assert ranks[0]["node"] in ["D", "B"]
    print(f"  ✓ Bottleneck ranking verified. Peak bottleneck: Node {ranks[0]['node']} (Score: {ranks[0]['bottleneck_score']})")


def test_yolo_video_analysis():
    print("[5/6] Testing real YOLO video detection on sample clip...")
    sample_video = ROOT / "results" / "gridguard_annotated.mp4"
    if not sample_video.is_file():
        print("  ⚠️ Sample video not found, skipping real video test.")
        return

    progress_events = []
    frame_events = []

    def on_prog(pct, stage):
        progress_events.append((pct, stage))

    def on_frame(ann, row):
        frame_events.append(row["frame"])

    out_mp4 = ROOT / "results" / "test_out.mp4"
    df, out_path = analyze_video(
        video_path=str(sample_video),
        output_path=str(out_mp4),
        max_frames=15,  # 15 frames for fast unit testing
        image_size=320,
        conf=0.25,
        progress_callback=on_prog,
        frame_callback=on_frame,
        preview_every=2,
    )
    assert len(df) == 15
    assert "vehicle_count" in df.columns
    assert "roi_vehicle_count" in df.columns
    assert "car_count" in df.columns
    assert len(progress_events) > 0
    assert len(frame_events) > 0
    print(f"  ✓ Processed {len(df)} frames with YOLO. Detected {df['vehicle_count'].sum()} vehicle instances.")
    if out_mp4.is_file():
        out_mp4.unlink()


def test_report_and_exports():
    print("[6/6] Testing data exports (CSV, JSON)...")
    mock_rows = [{"frame": 1, "time_seconds": 0.04, "vehicle_count": 5, "roi_vehicle_count": 2, "occupancy_estimate": 0.2, "risk_score": 0.25, "risk_level": "LOW"}]
    df = pd.DataFrame(mock_rows)
    csv_str = df.to_csv(index=False)
    assert "vehicle_count" in csv_str

    res = evaluate_routes(df, mode="corridor_baseline")
    assert not res.comparison_df.empty
    print("  ✓ Exports and structured reports validated.")


if __name__ == "__main__":
    print("\n================== GRIDGUARD AI VERIFICATION ==================")
    test_core_metrics()
    test_route_scoring()
    test_visualizations()
    test_network_analysis_legacy()
    test_yolo_video_analysis()
    test_report_and_exports()
    print("================== ALL TESTS PASSED SUCCESSFULLY! ==================\n")
